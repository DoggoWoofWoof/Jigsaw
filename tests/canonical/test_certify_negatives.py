"""E14: every negative decided by complete search (FilterAll, then classical GQL), witnesses checked."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import backends, cascade, classical, manifest, queries, seal, verify
from scripts.canonical import certify_negatives as cn
from scripts.canonical import store as store_mod

HERE = Path(__file__).resolve().parent
FAKE_SM = [sys.executable, str(HERE / "fake_subgraphmatching.py")]


@pytest.fixture(scope="module")
def cstore(clustered_store, tmp_path_factory):
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("certify")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    cascade.build_label_index(gstore, "R0")
    for split, master in (("dev", "20260607"), ("test", "sealed-for-tests")):
        queries.SplitBuilder(gstore, "synth", split, master, sizes=[6, 8], max_per_cell=2).build(
            ["R0"], copy / "queries" / split, master_tag=master)
    qs = queries.QuerySet.load(copy / "queries" / "dev" / "R0.npz")
    return gstore, copy, qs


@pytest.fixture(autouse=True)
def _fake_mode(monkeypatch):
    monkeypatch.delenv("FAKE_SM_MODE", raising=False)


def _has_embedding(gstore, q) -> bool:
    """VF2 on the label-filtered whole graph, independently of the module."""
    import networkx as nx
    from networkx.algorithms import isomorphism

    codes = gstore.sigma("R0").query_codes(q.payload, gstore)
    if (np.asarray(codes) < 0).any():
        return False
    cand = gstore.candidate(range(gstore.num_blocks), codes, "R0")
    gt, gq = nx.Graph(), nx.Graph()
    for g, h in ((cand, gt), (q.local_graph(codes), gq)):
        for i, lab in enumerate(np.asarray(g.labels).tolist()):
            h.add_node(i, label=lab)
        h.add_edges_from(map(tuple, g.edge_list().tolist()))
    matcher = isomorphism.GraphMatcher(gt, gq, node_match=lambda a, b: a["label"] == b["label"])
    return next(matcher.subgraph_monomorphisms_iter(), None) is not None


def _stage2(gstore, tmp_path):
    return classical.FullGraphClassicalRunner(gstore, "R0", "gql", cache_dir=tmp_path / "xcache", binary=FAKE_SM)


def _negatives(qs, gstore, extra_positive: bool = True):
    negs = [q for q in qs.queries if q.kind == "negative"]
    if extra_positive:  # a "negative" that holds a match: certification must find it
        pos = next(q for q in qs.queries if q.kind == "positive")
        negs.append(dataclasses.replace(pos, uid=pos.uid + "-as-negative", kind="negative", certified="uncertified"))
    return negs


# ----------------------------------------------------------------------------
# One query
# ----------------------------------------------------------------------------
def test_both_stages_agree_with_brute_force(cstore, tmp_path):
    gstore, _, qs = cstore
    stage1 = cn.stage1_cascade(gstore, "R0", backends.NetworkXBackend())
    assert stage1.options()["component_cap"] is None and stage1.budgets == [stage1.num_blocks]
    stage2 = _stage2(gstore, tmp_path)
    by = {"stage1": 0, "stage2": 0}
    found = 0
    for q in _negatives(qs, gstore):
        truth = _has_embedding(gstore, q)
        for T1 in (60.0, 1e-9):  # a vanishing stage 1 leaves every search to stage 2
            e = cn.certify_query(q, stage1, stage2, T1=T1, T2=60.0)
            assert e["certified"] == ("found" if truth else "none"), (q.uid, T1, e)
            assert not e["rejected"] and not e.get("errors")
            by[e["by"]] += 1
            if truth:
                found += 1
                assert e["witness_check"] == "ok"
                assert verify.check_embedding(gstore, q.payload, q.edges, e["witness"], "R0").ok
            if T1 < 1 and e["certificate"] is None:
                assert e["by"] == "stage2" and e["status"]["stage1"] == backends.TIMEOUT
    assert by["stage1"] and by["stage2"] and found >= 2


class Rejecting:
    """A stage-1 solver that claims a wrong embedding on every target large enough."""
    name = "fake_wrong"

    def solve(self, query, target, *, time_limit, max_solutions=1):
        if target.n < query.n:
            return backends.SolveOutcome(backends.NONE, [], 0.0, self.name, {})
        return backends.SolveOutcome(backends.FOUND, [np.arange(query.n)[::-1].copy()], 0.0, self.name, {})


class Slow:
    name = "fake_timeout"

    def solve(self, query, target, *, time_limit, max_solutions=1):
        return backends.SolveOutcome(backends.TIMEOUT, [], time_limit, self.name, {})


def test_rejected_witnesses_and_open_searches_decide_nothing(cstore, tmp_path, monkeypatch):
    gstore, _, qs = cstore
    negs = [q for q in _negatives(qs, gstore) if cn.stage1_cascade(gstore, "R0", backends.NetworkXBackend())
            .run(q, time_limit=60.0).get("certificate") is None]
    q = negs[-1]  # the positive posing as a negative
    wrong = cn.stage1_cascade(gstore, "R0", Rejecting())
    e = cn.certify_query(q, wrong, _stage2(gstore, tmp_path), T1=10.0, T2=60.0)
    assert e["rejected"] and e["rejected"][0]["stage"] == "stage1"
    assert e["certified"] == "found" and e["by"] == "stage2"  # stage 2 still decides it
    slow = cn.stage1_cascade(gstore, "R0", Slow())
    assert cn.certify_query(q, slow, None, T1=1.0, T2=1.0)["certified"] == "undecided"
    monkeypatch.setenv("FAKE_SM_MODE", "sleep")
    e = cn.certify_query(q, slow, _stage2(gstore, tmp_path), T1=1.0, T2=1.0)
    assert e["certified"] == "undecided" and e["status"]["stage2"] == backends.TIMEOUT and e["by"] is None
    monkeypatch.setenv("FAKE_SM_MODE", "invalid")
    e = cn.certify_query(q, slow, _stage2(gstore, tmp_path), T1=1.0, T2=30.0)
    assert e["certified"] == "undecided" and e["witness"] is None
    monkeypatch.setenv("FAKE_SM_MODE", "limit")  # a search the solver itself stopped proves nothing
    e = cn.certify_query(q, slow, _stage2(gstore, tmp_path), T1=1.0, T2=30.0)
    assert e["certified"] == "undecided" and e["status"] == {"stage1": backends.TIMEOUT, "stage2": backends.TIMEOUT}


# ----------------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------------
def _run(copy: Path, out: Path, *extra: str, split: str = "dev") -> list[str]:
    return ["run", "--store-root", str(copy), "--dataset", "synth", "--queries", str(copy / "queries" / split),
            "--out", str(out), *extra]


FAST = ("--backend", "networkx", "--classical-binary", *FAKE_SM, "--stage1-T", "30", "--stage2-T", "30")


def test_runner_writes_the_certification_and_resumes(cstore, tmp_path):
    gstore, copy, qs = cstore
    out = tmp_path / "cert"
    argv = _run(copy, out, *FAST, "--cache-dir", str(tmp_path / "xc"))
    assert cn.main(argv) == 0
    doc = cn.load_certification(out / "R0.json")
    negatives = [q for q in qs.queries if q.kind == "negative"]
    assert doc["schema"] == cn.SCHEMA and doc["complete"] and doc["split"] == "dev" and doc["regime"] == "R0"
    assert set(doc["queries"]) == {q.uid for q in negatives}
    counts = doc["counts"]
    assert counts["none"] + counts["found"] + counts["undecided"] == len(negatives) == counts["negatives"]
    assert counts["undecided"] == 0 and counts["rejected_witnesses"] == 0
    for q in negatives:
        e = doc["queries"][q.uid]
        assert e["certified"] == ("found" if _has_embedding(gstore, q) else "none")
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert info["status"] == "complete" and info["experiment"] == "E14"
    sha = info["inputs"]["queries/R0"]["sha256"]
    assert doc["query_set"]["sha256"] == sha == manifest.sha256_file(qs.path)["sha256"]
    assert [s["name"] for s in doc["stages"]] == ["stage1", "stage2"] and doc["stages"][0]["T"] == 30.0
    assert doc["stages"][1]["method"] == "classical_gql" and doc["stages"][1]["options"]["preset"]
    with pytest.raises(ValueError, match="another query set"):
        cn.load_certification(out / "R0.json", query_set_sha256="0" * 64)
    lines = (out / "R0.progress.jsonl").read_text(encoding="utf-8").splitlines()
    summary = cn.execute(cn.build_parser().parse_args(argv))
    assert summary["attempt"] == 2 and summary["regimes"]["R0"]["recorded_before"] == len(lines)
    assert (out / "R0.progress.jsonl").read_text(encoding="utf-8").splitlines() == lines
    with pytest.raises(SystemExit, match="other settings"):
        cn.main(_run(copy, out, *FAST[:-1], "20", "--cache-dir", str(tmp_path / "xc")))


def test_shards_merge_only_with_equal_settings(cstore, tmp_path):
    _, copy, qs = cstore
    whole = tmp_path / "whole"
    cn.main(_run(copy, whole, *FAST, "--skip-stage2", "--cache-dir", str(tmp_path / "xc")))
    out = tmp_path / "sharded"
    first = cn.execute(cn.build_parser().parse_args(_run(copy, out, *FAST, "--skip-stage2", "--shard", "0/2")))
    assert first["regimes"]["R0"]["certification"]["complete"] is False and not (out / "R0.json").exists()
    assert (out / "shard0of2" / manifest.MANIFEST_NAME).exists()
    second = cn.execute(cn.build_parser().parse_args(_run(copy, out, *FAST, "--skip-stage2", "--shard", "1/2")))
    assert second["regimes"]["R0"]["certification"]["complete"] is True
    merged = cn.load_certification(out / "R0.json")
    alone = cn.load_certification(whole / "R0.json")
    assert {u: e["certified"] for u, e in merged["queries"].items()} == \
        {u: e["certified"] for u, e in alone["queries"].items()}
    assert [s["name"] for s in merged["stages"]] == ["stage1"]
    (out / "R0.json").unlink()
    assert cn.main(["finalize", "--queries", str(copy / "queries" / "dev"), "--out", str(out)]) == 0
    assert cn.load_certification(out / "R0.json")["counts"] == merged["counts"]
    # a shard run with another stage 1 time cannot join
    mixed = tmp_path / "mixed"
    cn.main(_run(copy, mixed, *FAST, "--skip-stage2", "--shard", "0/2"))
    with pytest.raises(SystemExit, match="other stages"):
        cn.main(_run(copy, mixed, *FAST[:-3], "29", "--stage2-T", "30", "--skip-stage2", "--shard", "1/2"))
    # a partial certification is written only on request, and never loads
    part = tmp_path / "part"
    cn.main(_run(copy, part, *FAST, "--skip-stage2", "--limit", "2"))
    assert not (part / "R0.json").exists()
    cn.main(["finalize", "--queries", str(copy / "queries" / "dev"), "--out", str(part), "--allow-partial"])
    with pytest.raises(ValueError, match="partial"):
        cn.load_certification(part / "R0.json")


def test_sealed_sets_use_the_protocol_stages(cstore, tmp_path, monkeypatch):
    _, copy, _ = cstore
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    out = tmp_path / "cert"
    with pytest.raises(seal.SealError, match="--unseal"):
        cn.main(_run(copy, out, split="test"))
    with pytest.raises(seal.SealError, match="not frozen"):
        cn.main(_run(copy, out, "--unseal", split="test"))
    refused = {("--stage1-T", "5"): "--stage1-T", ("--stage2-T", "5"): "--stage2-T",
               ("--backend", "networkx"): "--backend", ("--limit", "3"): "--limit",
               ("--families", "neg_structure"): "--families", ("--skip-stage2",): "--skip-stage2",
               ("--partition", "kahip_fast"): "--partition kahip_fast"}
    for extra, match in refused.items():
        with pytest.raises(SystemExit, match=match):
            cn.main(_run(copy, out, *extra, "--unseal", split="test"))
    with pytest.raises(SystemExit, match="sealed query sets only"):
        cn.main(_run(copy, out, *FAST, "--unseal"))
    assert not log.exists() and not out.exists()


def test_certification_files_are_checked_on_load(tmp_path):
    base = {"schema": cn.SCHEMA, "complete": True, "query_set": {"sha256": "a" * 64},
            "queries": {"q": {"certified": "found", "witness_check": "ok"}}}
    path = tmp_path / "R0.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    assert cn.load_certification(path, query_set_sha256="a" * 64)["queries"]["q"]["certified"] == "found"
    for change, match in (({"schema": "other"}, "schema"), ({"complete": False}, "partial"),
                          ({"queries": {"q": {"certified": "found", "witness_check": None}}}, "checked witness")):
        path.write_text(json.dumps({**base, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            cn.load_certification(path)
