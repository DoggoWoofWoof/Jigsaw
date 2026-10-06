"""run_external: plans, records, witness checks, NONE certification, resume, shards and the sealed gate.

The solvers are the fakes (fake_subgraphmatching.py, fake_gnnpe.py) on the conftest synthetic store.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.canonical import backends, classical, gnnpe_adapter, manifest, protocol, queries, seal, verify
from scripts.canonical import run_experiment as rx
from scripts.canonical import run_external as rex
from scripts.canonical import store as store_mod

HERE = Path(__file__).resolve().parent
FAKE_SM = [sys.executable, str(HERE / "fake_subgraphmatching.py")]
FAKE_GNNPE = [sys.executable, str(HERE / "fake_gnnpe.py")]
MASTERS = {"dev": "20260607", "test": "sealed-for-tests"}
SIZE = 5


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No git status in manifests (slow, and irrelevant here); the fakes start in their default modes."""
    monkeypatch.setattr(manifest, "git_info", lambda *args, **kwargs: {"available": False})
    monkeypatch.delenv("FAKE_SM_MODE", raising=False)
    monkeypatch.delenv("FAKE_GNNPE_MODE", raising=False)


def _bfs(gstore, start: int, size: int) -> np.ndarray | None:
    order, seen = [start], {start}
    for u in order:
        for w in gstore.indices[gstore.indptr[u]:gstore.indptr[u + 1]].tolist():
            if w not in seen and len(order) < size:
                seen.add(w)
                order.append(w)
    return np.asarray(order, dtype=np.int64) if len(order) == size else None


def _induced(gstore, nodes: np.ndarray) -> np.ndarray:
    local = {int(v): i for i, v in enumerate(nodes.tolist())}
    rows = [(local[v], local[w]) for v in nodes.tolist()
            for w in gstore.indices[gstore.indptr[v]:gstore.indptr[v + 1]].tolist() if w in local]
    return queries.normalize_edges(np.asarray(rows, dtype=np.int64))


def _query_set(gstore, regime: str) -> list[queries.Query]:
    """Four connected planted queries and one whose first label the data graph never holds."""
    out: list[queries.Query] = []
    for start in range(0, gstore.num_nodes, 41):
        nodes = _bfs(gstore, start, SIZE)
        if nodes is not None:
            out.append(queries.Query(uid=f"synth-{regime}-k_hop-{len(out)}", family="k_hop", size=SIZE,
                                     kind="positive", regime=regime, edges=_induced(gstore, nodes),
                                     payload=queries.payload_for(gstore, nodes, regime), planted=nodes, seed=start,
                                     attempt=0))
        if len(out) == 4:
            break
    first = out[0]
    payload = {k: np.array(v, copy=True) for k, v in first.payload.items()}
    if regime == "R0":
        payload["attrs"][0] = 1e6
    else:
        payload["classes"][0] = 99
    out.append(queries.Query(uid=f"synth-{regime}-neg_absent_label-0", family="neg_absent_label", size=SIZE,
                             kind="negative", regime=regime, edges=first.edges, payload=payload,
                             planted=first.planted, seed=first.seed, attempt=0, certified="none"))
    return out


@pytest.fixture(scope="module")
def ext(synthetic_store, tmp_path_factory):
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth", mmap=False)
    base = tmp_path_factory.mktemp("external")
    sets = {}
    for regime in ("R0", "R1"):
        sets[regime] = _query_set(gstore, regime)
        for split, master in MASTERS.items():
            queries.save_query_set(base / "queries" / split / f"{regime}.npz", sets[regime],
                                   {"dataset": "synth", "split": split, "master": master})
    membership = base / "membership.npy"
    np.save(membership, np.arange(gstore.num_nodes, dtype=np.int64) % 2)
    # the data graph files and the GNN-PE index live outside the shared store
    return SimpleNamespace(root=root, store=gstore, base=base, cache=base / "cache", sets=sets,
                           membership=membership)


def _argv(ext, out: Path, method: str = "classical_gql", *extra: str, split: str = "dev",
          regimes: tuple[str, ...] = ("R0",), T: float | None = 20.0, binary: list[str] | None = None) -> list[str]:
    binary = binary or (FAKE_GNNPE if method == "gnnpe" else FAKE_SM)
    argv = ["--method", method, "--store-root", str(ext.root), "--dataset", "synth", "--queries",
            str(ext.base / "queries" / split), "--out", str(out), "--binary", *binary, "--cache-dir", str(ext.cache)]
    if regimes:
        argv += ["--regimes", *regimes]
    if T is not None:
        argv += ["--T", str(T)]
    if method == "gnnpe":
        argv += ["--gnnpe-partitions", "2", "--gnnpe-membership", str(ext.membership)]
    return [*argv, *extra]


def _records(out: Path, method: str = "classical_gql", regime: str = "R0", shard: str = "") -> list[dict]:
    recs = rx.read_records(out / "records" / regime / f"{method}{shard}.jsonl")
    return sorted(recs, key=lambda r: r["query_index"])


def _manifest(out: Path) -> dict:
    return json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))


def _check_common(r: dict, method: str, T: float, attempt: int = 1) -> None:
    """The run_experiment record fields every external record carries."""
    assert r["schema"] == rx.SCHEMA and r["arm"] == r["method"] == method and r["arm_settings"] == {}
    assert r["train_seed"] is None and r["model_id"] is None and r["arm_slot"] == 0
    assert r["mode"] == "resident" and r["cache"] == "none" and r["attempt"] == attempt
    assert r["backend"] == rex.backend_name(method) and r["dataset"] == "synth" and r["T"] == T
    assert r["decided_step"] is None and r["steps"] == [] and isinstance(r["io"], dict)
    assert 0 <= r["elapsed"] <= T and r["wall_seconds"] >= 0 and "query_labels" in r["stages"]
    spec = protocol.METHODS[method]
    assert r["external"]["tool"] == spec["external"] and r["external"]["binary_sha256"]
    assert r["external"]["algorithm"] == spec.get("algorithm", spec["external"])
    assert isinstance(r["external"]["none_certified"], bool)


# ----------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", list(protocol.DATASETS))
def test_plans_follow_the_protocol(dataset):
    assert set(rex.METHODS) == {m for m, spec in protocol.METHODS.items() if spec.get("external")}
    assert set(rex.CLASSICAL.values()) == set(classical.DEFAULT_ALGORITHMS)
    for name in ("E2", "E7"):  # the queries, T and regimes of run_experiment's Jigsaw side
        ours, theirs = rex.experiment_plan(name, dataset), rx.experiment_plan(name, dataset)
        assert ours["methods"] == list(rex.CLASSICAL)
        assert (ours["select"], ours["T"], ours["regimes"]) == (theirs["select"], theirs["T"], theirs["regimes"])
    assert rex.experiment_plan("E2", dataset)["caches"] == rx.experiment_plan("E2", dataset)["caches"]
    e12 = rex.experiment_plan("E12", dataset)
    assert e12["methods"] == ["gnnpe"] and e12["T"] == protocol.TIME["run_T_max_seconds"]["E1"]
    assert e12["select"] == {"per_cell": 50}
    assert e12["regimes"] == (["R1"] if "R1" in protocol.regimes_for(dataset) else [])
    if dataset in protocol.SERVING["scale_out"]["datasets"]:
        ours, theirs = rex.experiment_plan("E10", dataset), rx.experiment_plan("E10", dataset)
        assert ours["methods"] == list(rex.CLASSICAL) and ours["caches"] == theirs["caches"]
        assert (ours["select"], ours["T"], ours["regimes"]) == (theirs["select"], theirs["T"], theirs["regimes"])
    else:
        with pytest.raises(ValueError, match="E10 runs on"):
            rex.experiment_plan("E10", dataset)
    with pytest.raises(ValueError, match="unknown experiment"):
        rex.experiment_plan("E1", dataset)


# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------
def test_classical_found_is_verified_and_recorded_like_run_experiment(ext, tmp_path):
    out = tmp_path / "run"
    assert rex.main(_argv(ext, out)) == 0
    recs, qs = _records(out), ext.sets["R0"]
    assert [r["uid"] for r in recs] == [q.uid for q in qs]
    for r, q in zip(recs, qs):
        _check_common(r, "classical_gql", 20.0)
        assert (r["family"], r["size"], r["kind"], r["certified"]) == (q.family, q.size, q.kind, q.certified)
        if q.kind == "positive":
            assert r["status"] == backends.FOUND, r.get("error")
            assert r["witness_check"] == "ok" and r["time_found"] == r["time_decided"] == r["elapsed"]
            assert verify.check_embedding(ext.store, q.payload, q.edges, np.asarray(r["witness"]), "R0").ok
            assert {"write_query", "load", "solve"} <= set(r["stages"]) and r["witness_check_seconds"] >= 0
            assert r["external"]["load_seconds"] == r["stages"]["load"] and r["external"]["returncode"] == 0
            assert not r["external"]["none_certified"] and r.get("peak_rss_bytes", 1) > 0
        else:  # the absent label: a sound whole-graph NONE, decided without running the binary
            assert r["status"] == backends.NONE and r["external"]["none_certified"]
            assert r["time_decided"] == r["elapsed"] and r["time_found"] is None
            assert "trivial" in r["external"]["detail"] and r["external"]["load_seconds"] is None
    info = _manifest(out)
    assert info["status"] == "complete" and info["experiment"] == "external" and info["split"] == "dev"
    config = info["config"]
    assert config["arms"] == ["classical_gql"] and config["mode"] == "resident" and config["T"] == 20.0
    assert config["external"]["preset"] == list(classical.PRESETS["gql"])
    assert config["external"]["data_format"] == "text" and config["query_sets"]["R0"]["selected"] == len(qs)
    assert info["solvers"]["classical_gql"]["available"]
    assert {"store_meta", "queries/R0", "data_graph/R0"} <= set(info["inputs"])
    assert info["extra"]["attempt"] == 1 and "reused" in info["extra"]["prepare"]["R0"]


def test_rejected_witnesses_timeouts_and_crashes_are_recorded(ext, tmp_path, monkeypatch):
    first = ext.sets["R0"][0]
    with monkeypatch.context() as m:
        m.setattr(verify, "check_embedding", lambda *args, **kwargs: verify.CheckResult(False, "test rejection"))
        rex.main(_argv(ext, tmp_path / "rejected", "classical_gql", "--limit", "1"))
    [r] = _records(tmp_path / "rejected")
    assert r["uid"] == first.uid and r["status"] == backends.ERROR, r
    assert r["error"] == "witness rejected: test rejection" and r["witness_check"] == "test rejection"
    assert r["time_found"] is None and r["time_decided"] is None and len(r["witness"]) == first.n

    monkeypatch.setenv("FAKE_SM_MODE", "sleep")
    t0 = time.perf_counter()
    rex.main(_argv(ext, tmp_path / "timeout", "classical_gql", "--limit", "1", T=1.5))
    assert time.perf_counter() - t0 < 25  # the 30 s sleep is killed at the cap
    [r] = _records(tmp_path / "timeout")
    assert r["status"] == backends.TIMEOUT and r["external"]["killed_at_cap"] and r["elapsed"] <= 1.5
    assert r["time_found"] is None and r["time_decided"] is None and not r["external"]["none_certified"]

    monkeypatch.setenv("FAKE_SM_MODE", "crash")
    rex.main(_argv(ext, tmp_path / "crash", "classical_gql", "--limit", "1"))
    [r] = _records(tmp_path / "crash")
    assert r["status"] == backends.ERROR and r["error"] == "the solver failed (exit code 139)"
    assert r["external"]["returncode"] == 139 and "output_tail" in r["external"]["detail"]

    def broken(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(classical.FullGraphClassicalRunner, "solve", broken)
    rex.main(_argv(ext, tmp_path / "broken"))
    recs = _records(tmp_path / "broken")
    assert len(recs) == len(ext.sets["R0"])  # an exception is recorded and the run goes on
    assert all(r["status"] == backends.ERROR and r["error"] == "RuntimeError: boom" and "boom" in r["traceback"]
               and r["time_decided"] is None for r in recs)
    assert _manifest(tmp_path / "broken")["status"] == "complete"


def test_classical_none_is_a_proof_and_a_gnnpe_none_is_not(ext, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SM_MODE", "none")
    rex.main(_argv(ext, tmp_path / "classical"))
    for r in _records(tmp_path / "classical"):
        assert r["status"] == backends.NONE and r["external"]["none_certified"]
        assert r["time_decided"] == r["elapsed"] and r["time_found"] is None

    monkeypatch.setenv("FAKE_GNNPE_MODE", "none")
    rex.main(_argv(ext, tmp_path / "gnnpe_none", "gnnpe", regimes=("R1",)))
    recs, qs = _records(tmp_path / "gnnpe_none", "gnnpe", "R1"), ext.sets["R1"]
    assert [r["uid"] for r in recs] == [q.uid for q in qs]
    for r, q in zip(recs, qs):
        _check_common(r, "gnnpe", 20.0)
        assert r["status"] == backends.NONE and r["time_found"] is None
        if q.kind == "positive":  # a search NONE of GNN-PE is no certificate
            assert not r["external"]["none_certified"] and r["time_decided"] is None
            assert r["external"]["detail"]["none_unverified"] and {"load", "solve"} <= set(r["stages"])
        else:  # the absent label is refused before the binary runs, which is sound
            assert r["external"]["none_certified"] and r["time_decided"] == r["elapsed"]

    monkeypatch.delenv("FAKE_GNNPE_MODE")
    rex.main(_argv(ext, tmp_path / "gnnpe_found", "gnnpe", regimes=("R1",)))
    for r, q in zip(_records(tmp_path / "gnnpe_found", "gnnpe", "R1"), qs):
        if q.kind == "positive":
            assert r["status"] == backends.FOUND and r["witness_check"] == "ok", r.get("error")
            assert r["time_found"] == r["time_decided"] == r["elapsed"]
            assert verify.check_embedding(ext.store, q.payload, q.edges, np.asarray(r["witness"]), "R1").ok
    info = _manifest(tmp_path / "gnnpe_found")
    settings = info["config"]["external"]
    assert settings["partitions"] == 2 and settings["path_length"] == gnnpe_adapter.PATH_LENGTH
    assert settings["membership_source"] == gnnpe_adapter._membership_sha256(np.load(ext.membership))
    assert info["extra"]["prepare"]["R1"]["reused"]  # the offline stage of the first run is reused


# ----------------------------------------------------------------------------
# Resume, shards and setup failures
# ----------------------------------------------------------------------------
def test_resume_skips_recorded_queries_and_refuses_other_settings(ext, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SM_MODE", "none")
    out = tmp_path / "run"
    argv = _argv(ext, out)
    rex.main(argv)
    path = out / "records" / "R0" / "classical_gql.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + lines[-1][:15])  # the last record torn by a crash
    summary = rex.execute(rex.build_parser().parse_args(argv))
    regime = summary["regimes"]["R0"]
    assert summary["attempt"] == 2 and regime["arms"]["classical_gql"] == {"run": 1, "recorded_before": len(lines) - 1}
    assert regime["recovered_bytes"] == {"classical_gql": 15}
    recs = _records(out)
    assert len(recs) == len(lines) and [r["attempt"] for r in recs] == [1] * (len(lines) - 1) + [2]
    assert (out / "manifest.attempt1.json").exists() and _manifest(out)["extra"]["attempt"] == 2
    with pytest.raises(SystemExit, match="other settings"):
        rex.main(_argv(ext, out, T=7.0))


def test_shards_prepare_only_and_bad_setups(ext, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SM_MODE", "none")
    out = tmp_path / "run"
    rex.main(_argv(ext, out, "classical_gql", "--shard", "1/2"))
    recs = _records(out, shard=".shard1of2")
    assert [r["uid"] for r in recs] == [q.uid for q in ext.sets["R0"]][1::2]
    assert [r["query_index"] for r in recs] == list(range(len(recs)))
    assert _manifest(out)["config"]["shard"] == [1, 2]
    assert rex.record_name("classical_gql", (1, 2)) == "classical_gql.shard1of2.jsonl"
    assert rex.record_name("gnnpe", None) == "gnnpe.jsonl"

    fresh = tmp_path / "fresh"
    summary = rex.execute(rex.build_parser().parse_args(_argv(ext, fresh, "classical_gql", "--prepare-only")))
    assert summary["prepared"]["R0"]["reused"] and not fresh.exists()
    with pytest.raises(SystemExit, match="0 <= i < n"):
        rex.main(_argv(ext, fresh, "classical_gql", "--shard", "2/2"))
    with pytest.raises(SystemExit, match="not found; build it"):
        rex.main(_argv(ext, fresh, binary=[str(tmp_path / "missing")]))
    with pytest.raises(SystemExit, match="give --T and --regimes"):
        rex.main(_argv(ext, fresh, T=None))
    with pytest.raises(SystemExit, match="--method gnnpe only"):
        rex.main(_argv(ext, fresh, "classical_gql", "--gnnpe-partitions", "2"))
    with pytest.raises(SystemExit, match="classical methods only"):
        rex.main(_argv(ext, fresh, "gnnpe", "--data-format", "csr", regimes=("R1",)))
    with pytest.raises(SystemExit, match="no regime on papers100m"):
        rex.main(["--experiment", "E12", "--method", "gnnpe", "--store-root", str(ext.root), "--dataset",
                  "papers100m", "--queries", str(ext.base), "--out", str(fresh)])
    if not hasattr(os, "posix_fadvise"):
        with pytest.raises(SystemExit, match="cold runs"):
            rex.main(_argv(ext, fresh, "classical_gql", "--cache", "cold"))
    assert not fresh.exists()


def test_sealed_query_sets_run_only_as_a_registered_plan_through_the_gate(ext, tmp_path, monkeypatch):
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    monkeypatch.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6))
    out = tmp_path / "run"
    common = ["--store-root", str(ext.root), "--dataset", "synth", "--queries", str(ext.base / "queries" / "test"),
              "--out", str(out), "--cache-dir", str(ext.cache)]
    e7 = ["--experiment", "E7", "--method", "classical_gql", "--regimes", "R0", "--binary", *FAKE_SM, *common]
    e12 = ["--experiment", "E12", "--method", "gnnpe", "--binary", *FAKE_GNNPE, *common]
    with pytest.raises(seal.SealError, match="--unseal"):
        rex.main(e7)
    with pytest.raises(seal.SealError, match="not frozen"):  # the registered plan reaches the gate
        rex.main([*e7, "--unseal"])
    with pytest.raises(seal.SealError, match="not frozen"):
        rex.main([*e12, "--gnnpe-membership", str(ext.membership), "--unseal"])
    # anything but the registered plan is refused before the gate, so nothing reaches the unseal log
    refused = {("--limit", "1"): "--limit", ("--kinds", "positive"): "--kinds", ("--families", "k_hop"): "--families",
               ("--sizes", "5"): "--sizes", ("--per-cell", "1"): "--per-cell", ("--uids", "u.txt"): "--uids",
               ("--data-format", "csr"): "--data-format", ("--T", "30"): r"--T 30 \(the plan's is 120\)",
               ("--cache", "warm"): "--cache warm", ("--method", "gnnpe"): "--method gnnpe is outside E7"}
    for extra, match in refused.items():
        with pytest.raises(SystemExit, match=match):
            rex.main([*e7, *extra, "--unseal"])
    with pytest.raises(SystemExit, match="--gnnpe-partitions"):
        rex.main([*e12, "--gnnpe-partitions", "2", "--unseal"])
    with pytest.raises(SystemExit, match="--method classical_gql is outside E12"):
        rex.main([*e12, "--method", "classical_gql", "--unseal"])
    with pytest.raises(SystemExit, match=r"regimes outside the plan: \['R1'\]"):
        rex.main(["--experiment", "E2", "--method", "classical_cfl", "--regimes", "R1", "--cache", "warm",
                  "--binary", *FAKE_SM, *common, "--unseal"])
    with pytest.raises(SystemExit, match="ad-hoc runs"):
        rex.main(_argv(ext, out, "classical_gql", "--unseal", split="test"))
    assert not log.exists() and not out.exists()
