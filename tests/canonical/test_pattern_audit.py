"""Pattern-overlap audit (E20): regeneration checks, isomorphic lookup, served regimes and the audit file."""

from __future__ import annotations

import concurrent.futures
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.canonical import manifest, protocol, queries, sigma  # noqa: E402
from scripts.canonical import pattern_audit as pa  # noqa: E402
from scripts.canonical import train_encoder as te  # noqa: E402
from scripts.canonical.store import GraphStore  # noqa: E402

SEED = 7301
SMOKE = ["--epochs", "2", "--steps-per-epoch", "3", "--layers", "2", "--hidden", "16", "--output", "8",
         "--val-every", "1", "--device", "cpu"]
MASTERS = tuple(protocol.SEEDS["validation"])


@pytest.fixture(scope="module")
def synth(clustered_store, tmp_path_factory):
    """The clustered store as dataset "synth", its three validation masters (R0, R2-512) in the canonical
    layout, and smoke encoders of R0 and R2-512 that hold out all three masters (as train_matrix's rows)."""
    root, _ = clustered_store
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6, ladder=[1, 2, 3]))
        mp.setitem(protocol.TRAINING["loss"]["topk_barrier_K"], "synth", 3)
        mp.setitem(protocol.TRAINING["loss"]["live_positives"], "synth", 4)
        store = GraphStore(root, "synth", mmap=True)
        qroot = tmp_path_factory.mktemp("queries")
        taken: list[str] = []
        for master in MASTERS:
            out = qroot / "synth" / "validation" / master
            queries.SplitBuilder(store, "synth", "validation", master, sizes=[8, 12], max_per_cell=2,
                                 exclude=queries.read_exclusions(taken)).build(["R0", "R2-512"], out, master_tag=master)
            taken.append(str(out))
        base = tmp_path_factory.mktemp("runs")
        runs = {}
        for regime in ("R0", "R2-512"):
            run_dir = base / "train" / te.model_id_for("synth", regime, "graphsage", SEED)
            val = qroot / "synth" / "validation" / te.SELECTION_MASTER / f"{regime}.npz"
            assert te.main(["--dataset", "synth", "--regime", regime, "--seed", str(SEED), "--store-root", str(root),
                            "--val-queries", str(val), "--run-dir", str(run_dir), "--models-root", str(base / "models"),
                            "--held-out", *taken, *SMOKE]) == 0
            runs[regime] = run_dir
        yield SimpleNamespace(root=root, store=store, qroot=qroot, base=base, runs=runs)


def first_training_query(synth, regime: str) -> te.TrainQuery:
    """The first training query of the run (epoch 1, step 0), regenerated with the run's held-out node sets."""
    config = pa.read_run(synth.runs[regime], "synth", allow_smoke=True)
    held, _ = pa.held_out_of(config)
    sampler = te.TrainSampler(synth.store, "synth", config["regime"], SEED, te.sampler_settings(), held_out=held)
    item = sampler.batch(1, 0, int(config["schedule"]["batch_size"]))[0]
    lines = (synth.runs[regime] / te.TRAIN_NODESETS).read_text(encoding="ascii").split()
    assert item.nodeset == lines[0]
    return item


def permuted(store, item, regime: str, rng, uid: str) -> queries.Query:
    """The training query with its nodes in another order: an isomorphic positive of the same node set."""
    n = item.planted.shape[0]
    perm = rng.permutation(n)  # new node j is old node perm[j]
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    planted = item.planted[perm]
    return queries.Query(uid, item.family, item.size, "positive", regime, queries.normalize_edges(inv[item.edges]),
                         queries.payload_for(store, planted, regime), planted, 0, 0, {})


def absent(store, item, regime: str, rng, uid: str) -> queries.Query:
    """The training query with one label no data node carries: no training query can be isomorphic to it."""
    pq = queries.PlantedQuery(item.family, item.size, item.planted, item.edges)
    payload, info = queries.absent_label(store, regime, pq, queries.payload_for(store, item.planted, regime), rng)
    return queries.Query(uid, "neg_absent_label", item.size, "negative", regime, item.edges, payload, item.planted,
                         0, 0, info, certified="uncertified")


@pytest.fixture(scope="module")
def audited(synth, tmp_path_factory):
    """A dev query directory (R0, R2-512, R2-32) with seen, aliased and novel queries, and its audit."""
    rng = np.random.default_rng(0)
    qdir = tmp_path_factory.mktemp("audited") / "synth" / "dev" / "999"
    meta = {"dataset": "synth", "split": "dev", "master": "999", "protocol_hash": protocol.protocol_hash()}
    r0 = first_training_query(synth, "R0")
    queries.save_query_set(qdir / "R0.npz", [permuted(synth.store, r0, "R0", rng, "seen-a"),
                                             permuted(synth.store, r0, "R0", rng, "seen-b"),
                                             absent(synth.store, r0, "R0", rng, "novel")], meta)
    r2 = first_training_query(synth, "R2-512")
    for regime in ("R2-512", "R2-32"):
        queries.save_query_set(qdir / f"{regime}.npz", [permuted(synth.store, r2, regime, rng, "seen-r2")], meta)
    out = tmp_path_factory.mktemp("audit") / "out"
    doc = pa.run_audit(store_root=synth.root, query_dir=qdir, out=out, run_dirs=list(synth.runs.values()),
                       allow_smoke=True, argv=["test"])
    return SimpleNamespace(qdir=qdir, out=out, doc=doc, r0=r0, r2=r2)


# ----------------------------------------------------------------------------- lookup and regimes
def test_seen_aliased_and_novel_queries(synth, audited):
    doc = audited.doc
    model_r0 = te.model_id_for("synth", "R0", "graphsage", SEED)
    entry = doc["regimes"]["R0"]
    assert entry["uids"] == ["seen-a", "seen-b", "novel"]
    assert entry["encoders"] == [model_r0]
    assert set(entry["seen"]) == {"seen-a", "seen-b"}  # isomorphic queries of one set are seen together
    first = entry["seen"]["seen-a"][model_r0]
    assert first["first"] == [1, 0, audited.r0.index] and first["hits"] >= 1
    assert entry["seen"]["seen-b"] == entry["seen"]["seen-a"]
    assert (entry["queries"], entry["positives"], entry["negatives"]) == (3, 2, 1)
    assert (entry["seen_positives"], entry["seen_negatives"]) == (2, 0)
    enc = doc["encoders"][model_r0]
    lines = (synth.runs["R0"] / te.TRAIN_NODESETS).read_text(encoding="ascii").split()
    assert enc["accepted"] == enc["train_nodesets_lines"] == len(lines) > 0
    assert enc["draws"] == enc["accepted"] + enc["failed"] + enc["held_out_skipped"]
    assert enc["serves"] == ["R0"] and enc["smoke"] and (enc["epochs"], enc["steps_per_epoch"]) == (2, 3)
    assert doc["schema"] == pa.SCHEMA and doc["complete"] and (doc["dataset"], doc["split"]) == ("synth", "dev")


def test_the_r2_512_encoder_serves_the_coarse_regimes(synth, audited):
    model_r2 = te.model_id_for("synth", "R2-512", "graphsage", SEED)
    for regime in ("R2-512", "R2-32"):
        entry = audited.doc["regimes"][regime]
        assert entry["encoders"] == [model_r2]
        assert entry["seen"]["seen-r2"][model_r2]["first"] == [1, 0, audited.r2.index]
    assert audited.doc["encoders"][model_r2]["serves"] == ["R2-512", "R2-32"]  # the regimes the directory holds
    assert pa.served_regimes("corafull", "R2-512") == ["R2-512", "R2-128", "R2-32"]
    assert pa.served_regimes("corafull", "R0") == ["R0"]


def test_coarse_buckets_are_a_function_of_the_512_buckets():
    digests = np.random.default_rng(1).integers(0, 256, size=(5000, 32), dtype=np.uint8)
    fine = sigma.hashed_buckets(digests, 512)
    for k in (128, 32):
        assert np.array_equal(sigma.hashed_buckets(digests, k), fine % k)


def test_the_manifest_loses_the_family_order(synth):
    # why the audit samples with the protocol's settings: the manifest's table has sorted keys
    config = pa.read_run(synth.runs["R0"], "synth", allow_smoke=True)
    stored = list(config["sampler"]["family_probabilities"])
    assert stored == sorted(stored)
    assert list(te.sampler_settings()["family_probabilities"]) != stored  # the protocol's order is not sorted


def test_main_rows_are_train_matrix_main_rows():
    rows = pa.main_rows("corafull")
    assert len(rows) == len(te.trained_regimes_for("corafull")) * len(protocol.SEEDS["training"])
    assert ("R0", 7301, "corafull-R0-graphsage-seed7301-main") in rows
    assert {r for r, _, _ in rows} == set(te.trained_regimes_for("corafull"))


# ----------------------------------------------------------------------------- the audit file
def test_audit_file_and_manifest(audited):
    on_disk = json.loads((audited.out / pa.OUTPUT).read_text(encoding="utf-8"))
    assert on_disk == audited.doc
    record = json.loads((audited.out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert record["status"] == "complete" and record["experiment"] == pa.EXPERIMENT and record["split"] == "dev"
    assert manifest.verify_manifest(audited.out) == []
    assert {"queries/R0", "queries/R2-512", "queries/R2-32"} <= set(record["inputs"])
    assert record["extra"]["summary"]["regimes"]["R0"]["seen_positives"] == 2
    for regime, entry in audited.doc["regimes"].items():
        assert entry["query_set"]["sha256"] == manifest.sha256_file(audited.qdir / f"{regime}.npz")["sha256"]


def test_cli_writes_once(synth, audited, tmp_path, capsys):
    args = ["--store-root", str(synth.root), "--queries", str(audited.qdir), "--out", str(tmp_path / "cli"),
            "--runs", *map(str, synth.runs.values()), "--allow-smoke"]
    assert pa.main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["regimes"]["R0"]["seen_positives"] == 2 and printed["regimes"]["R0"]["seen_negatives"] == 0
    assert json.loads((tmp_path / "cli" / pa.OUTPUT).read_text(encoding="utf-8"))["regimes"] == audited.doc["regimes"]
    assert pa.main(args) == 2  # an existing audit is never overwritten
    assert "exists" in capsys.readouterr().err


def test_workers_give_the_sequential_result(synth, audited, tmp_path, monkeypatch):
    # threads stand in for processes: a spawned worker would not see the test's protocol.DATASETS["synth"]
    monkeypatch.setattr(pa.concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    doc = pa.run_audit(store_root=synth.root, query_dir=audited.qdir, out=tmp_path / "w", workers=2,
                       run_dirs=list(synth.runs.values()), allow_smoke=True, argv=["test"])
    assert doc["regimes"] == audited.doc["regimes"]
    strip = lambda e: {k: v for k, v in e.items() if k != "seconds"}  # noqa: E731
    assert {m: strip(e) for m, e in doc["encoders"].items()} == {m: strip(e) for m, e in audited.doc["encoders"].items()}


# ----------------------------------------------------------------------------- refusals
def _copy_run(synth, tmp_path, regime="R0"):
    dst = tmp_path / "train" / synth.runs[regime].name
    shutil.copytree(synth.runs[regime], dst)
    return dst


def test_a_changed_training_log_is_refused(synth, audited, tmp_path):
    run = _copy_run(synth, tmp_path)
    path = run / te.TRAIN_NODESETS
    lines = path.read_text(encoding="ascii").split()
    lines[3] = "0" * 32
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(pa.AuditError, match="training query 3 .* regenerates as"):
        pa.run_audit(store_root=synth.root, query_dir=audited.qdir, out=tmp_path / "a",
                     run_dirs=[run, synth.runs["R2-512"]], allow_smoke=True, argv=["test"])
    assert not (tmp_path / "a" / pa.OUTPUT).exists()
    assert json.loads((tmp_path / "a" / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))["status"] == "failed"
    path.write_text("\n".join((synth.runs["R0"] / te.TRAIN_NODESETS).read_text(encoding="ascii").split()
                              + ["f" * 32]) + "\n", encoding="ascii")
    with pytest.raises(pa.AuditError, match="lines in"):
        pa.run_audit(store_root=synth.root, query_dir=audited.qdir, out=tmp_path / "b",
                     run_dirs=[run, synth.runs["R2-512"]], allow_smoke=True, argv=["test"])


def test_a_test_split_is_audited_only_through_the_seal(synth, audited, tmp_path, monkeypatch):
    monkeypatch.setattr(pa.seal, "UNSEAL_LOG", tmp_path / "UNSEAL_LOG.jsonl")
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", None)
    qdir = tmp_path / "synth" / "test"
    qdir.mkdir(parents=True)
    for path in audited.qdir.glob("*.npz"):
        query_set = queries.QuerySet.load(path)
        queries.save_query_set(qdir / path.name, list(query_set), {**query_set.meta, "split": "test"})
    for unseal, match in ((False, "explicit --unseal"), (True, "protocol is not frozen")):
        out = tmp_path / f"out-{unseal}"
        with pytest.raises(pa.AuditError, match=match):
            pa.run_audit(store_root=synth.root, query_dir=qdir, out=out, run_dirs=list(synth.runs.values()),
                         allow_smoke=True, unseal=unseal, reason="refused", argv=["test"])
        assert not out.exists()  # refused before the manifest
    assert not (tmp_path / "UNSEAL_LOG.jsonl").exists()


def test_held_out_node_sets_must_match_the_run(synth, tmp_path):
    config = pa.read_run(synth.runs["R0"], "synth", allow_smoke=True)
    held, digest = pa.held_out_of(config)
    assert len(held) == config["held_out"]["nodesets"] and digest == config["held_out"]["sha256"]
    assert pa.held_out_of(config, held_out_root=synth.qroot) == (held, digest)  # the canonical layout agrees
    other = tmp_path / "moved"
    shutil.copytree(synth.qroot, other)
    with open(other / "synth" / "validation" / MASTERS[1] / "nodesets.txt", "a", encoding="ascii") as handle:
        handle.write("a" * 32 + "\n")
    with pytest.raises(pa.AuditError, match="rebuild to"):
        pa.held_out_of(config, held_out_root=other)
    with pytest.raises(pa.AuditError, match="cannot rebuild"):
        pa.held_out_of(config, held_out_root=tmp_path / "nowhere")


def test_runs_the_audit_refuses(synth, audited, tmp_path):
    with pytest.raises(pa.AuditError, match="smoke run"):
        pa.read_run(synth.runs["R0"], "synth")
    with pytest.raises(pa.AuditError, match="dataset"):
        pa.read_run(synth.runs["R0"], "corafull", allow_smoke=True)
    run = _copy_run(synth, tmp_path)
    record = json.loads((run / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    record["status"] = "running"
    (run / manifest.MANIFEST_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(pa.AuditError, match="status 'running'"):
        pa.read_run(run, "synth", allow_smoke=True)
    record["status"] = "complete"
    record["config"]["sampler"]["size_jitter"] += 1
    (run / manifest.MANIFEST_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(pa.AuditError, match="sampler settings differ"):
        pa.read_run(run, "synth", allow_smoke=True)
    with pytest.raises(pa.AuditError, match="main encoders of synth not found"):
        pa.plan_tasks(store_root=synth.root, query_dir=audited.qdir, run_root=tmp_path / "empty")
    with pytest.raises(pa.AuditError, match="audited twice"):
        pa.plan_tasks(store_root=synth.root, query_dir=audited.qdir, allow_smoke=True,
                      run_dirs=[synth.runs["R0"], _copy_run(synth, tmp_path / "again"), synth.runs["R2-512"]])
    with pytest.raises(pa.AuditError, match=r"no audited encoder serves \['R2-32', 'R2-512'\]"):
        pa.plan_tasks(store_root=synth.root, query_dir=audited.qdir, run_dirs=[synth.runs["R0"]], allow_smoke=True)
