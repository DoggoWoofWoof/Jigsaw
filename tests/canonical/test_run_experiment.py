"""The experiment runner: arms, plans, selection, records, resume and the sealed gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.canonical import backends, cascade, encoder_numpy, features, manifest, protocol, queries, rankers
from scripts.canonical import run_experiment as rx
from scripts.canonical import prepare_data, seal, serving
from scripts.canonical import store as store_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
DECIDED = {backends.FOUND, backends.NONE}
SMOKE_FAMILIES = ["single", "k_hop", "neg_absent_label", "neg_swapped_label", "neg_structure"]


@pytest.fixture(scope="module")
def rstore(clustered_store, tmp_path_factory):
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("runner")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    rankers.build_artifacts(gstore, ["R0"])
    cascade.build_label_index(gstore, "R0")
    for split, master in (("dev", "20260607"), ("test", "sealed-for-tests")):
        queries.SplitBuilder(gstore, "synth", split, master, sizes=[8], max_per_cell=1).build(
            ["R0"], copy / "queries" / split, master_tag=master)
    return gstore, copy


def _argv(copy: Path, out: Path, *arms: str, split: str = "dev", T: float = 2.0, extra: tuple[str, ...] = (),
          budgets: tuple[str, ...] = ("1", "2")):
    return ["--arms", *arms, "--store-root", str(copy), "--dataset", "synth", "--queries",
            str(copy / "queries" / split), "--regimes", "R0", "--T", str(T),
            *(("--budgets", *budgets) if budgets else ()),
            "--backend", "networkx", "--families", *SMOKE_FAMILIES, "--out", str(out), *extra]


def _records(out: Path, arm: str) -> list[dict]:
    return rx.read_records(out / "records" / "R0" / f"{arm}.jsonl")


# ----------------------------------------------------------------------------
# Arms and plans
# ----------------------------------------------------------------------------
def test_arm_names_are_canonical():
    assert rx.Arm.parse("jigsaw_index").name == "jigsaw_index"
    arm = rx.Arm.parse("jigsaw@7301+halo=0+component_cap=50")
    assert arm.name == "jigsaw@7301+component_cap=50+halo=0" and arm.seed == 7301
    assert rx.Arm.parse(arm.name) == arm
    assert arm.get("halo") is False and arm.get("component_cap") == 50
    assert rx.Arm.parse("filterall+backend=gql").name == "filterall"  # the default verifier is implied
    assert rx.Arm.parse("filterall+backend=vf2").name == "filterall+backend=networkx"
    assert rx.Arm.parse("filterall").with_setting("backend", "glasgow").name == "filterall+backend=glasgow"
    assert rx.Arm.parse("filterall+backend=cfl").with_setting("backend", "glasgow").name == "filterall+backend=cfl"
    # E16 settings: the protocol's own values are implied
    assert rx.Arm.parse("jigsaw@7301+seed_count=20+policy=hybrid").name == "jigsaw@7301"
    e16 = rx.Arm.parse("jigsaw@7301+seed_count=40+halo_hops=2")
    assert e16.name == "jigsaw@7301+halo_hops=2+seed_count=40" and rx.Arm.parse(e16.name) == e16
    assert rx.Arm.parse("jigsaw_index+policy=topk").get("policy") == "topk"
    bad = {"classical_gql": "not a cascade method", "jigsaw": "encoder", "jigsaw_index@7301": "no encoder",
           "filterall+halo": "bad arm setting", "filterall+foo=1": "bad arm setting", "filterall+halo=2": "boolean",
           "filterall+component_mode=all": "component mode", "filterall+halo=0+halo=1": "twice",
           "filterall+backend=ullmann": "unknown backend", "filterall+policy=topk": "selects no blocks",
           "direct+seed_count=10": "selects no blocks", "jigsaw_index+policy=topk+seed_count=10": "top-k",
           "jigsaw_index+halo_hops=1": "2 or more", "jigsaw_index+halo=1+halo_hops=2": "both set",
           "jigsaw_index+seed_count=0": "at least one", "jigsaw_index+policy=greedy": "unknown selection policy"}
    for text, match in bad.items():
        with pytest.raises(ValueError, match=match):
            rx.Arm.parse(text)


def test_arms_name_the_serving_model_ids():
    # test_train_encoder checks that serving.model_id_for names the trainer's exports
    for dataset in ("arxiv", "mag"):
        arch = protocol.DATASETS[dataset]["encoder"]
        for regime in protocol.regimes_for(dataset):
            trained = protocol.TRAINED_REGIMES[regime]
            assert rx.Arm.parse("jigsaw_learned@7302").model_id(dataset, regime) == \
                serving.model_id_for(dataset, trained, arch, 7302)
    assert rx.Arm.parse("jigsaw_learned@7302").model_id("mag", "R2-32") == "mag-R2-512-rgcn-seed7302-main"
    assert rx.Arm.parse("mean_rrf+model=abc").model_id("mag", "R0") == "abc"
    assert rx.Arm.parse("filterall").model_id("mag", "R0") is None


@pytest.mark.parametrize("dataset", list(protocol.DATASETS))
def test_experiment_plans_are_well_formed(dataset):
    seeds = [int(s) for s in protocol.SEEDS["training"]]
    for name in rx.EXPERIMENTS:
        if name == "E10" and dataset not in protocol.SERVING["scale_out"]["datasets"]:
            with pytest.raises(ValueError, match="E10 runs on"):
                rx.experiment_plan(name, dataset)
            continue
        plan = rx.experiment_plan(name, dataset)
        arms = [rx.Arm.parse(a) for a in plan["arms"]]
        assert len({a.name for a in arms}) == len(arms)
        assert set(plan["regimes"]) <= set(protocol.regimes_for(dataset)) and plan["regimes"]
        assert all(a.seed in (None, *seeds) for a in arms)
        for mode_arms in plan.get("modes", {}).values():
            assert {rx.Arm.parse(a).name for a in mode_arms} <= {a.name for a in arms}
        for part_arms in plan.get("partitions", {}).values():
            assert {rx.Arm.parse(a).name for a in part_arms} <= {a.name for a in arms}
    e2 = rx.experiment_plan("E2", dataset)
    assert set(e2["modes"]) == {"streaming", "resident"} and e2["caches"] == ["cold", "warm"]
    assert e2["memory_caps_gb"] == protocol.SERVING["memory_caps_gb"] and e2["T"] == protocol.TIME["primary_T_seconds"]
    e1 = [rx.Arm.parse(a) for a in rx.experiment_plan("E1", dataset)["arms"]]
    assert sum(a.learned for a in e1) == len(rx.LEARNED_METHODS) * len(seeds)
    assert {a.method for a in e1} == set(rx.LEARNED_METHODS) | set(rx.FIXED_METHODS)
    assert rx.experiment_plan("E1", dataset)["T"] == protocol.TIME["run_T_max_seconds"]["E1"]
    assert rx.experiment_plan("E8", dataset)["T"] == max(protocol.TIME["run_T_max_seconds"]["E8"])
    e9 = [rx.Arm.parse(a) for a in rx.experiment_plan("E9", dataset)["arms"]]
    assert {a.method for a in e9} == {"jigsaw"} and len(e9) == len(rx.ABLATIONS)
    assert rx.experiment_plan("E9", dataset)["T"] == protocol.TIME["primary_T_seconds"]
    e16 = rx.experiment_plan("E16", dataset)
    assert set(e16["partitions"]) == {"primary", *prepare_data.ALTERNATIVES}
    assert e16["partitions"]["primary"] == e16["arms"] and e16["T"] == protocol.TIME["primary_T_seconds"]
    settings = {k for a in e16["arms"] for k, _ in rx.Arm.parse(a).settings}
    assert settings == {"halo", "halo_hops", "seed_count", "policy"}
    if dataset in protocol.SERVING["scale_out"]["datasets"]:
        e10 = rx.experiment_plan("E10", dataset)
        assert set(e10["modes"]) == {"streaming"} and e10["caches"] == ["cold"] and e10["regimes"] == ["R0"]
        assert e10["memory_caps_gb"] == protocol.SERVING["scale_out"]["memory_caps_gb"]
        assert e10["repeats"] == protocol.SERVING["scale_out"]["repeats"]
    with pytest.raises(ValueError, match="unknown experiment"):
        rx.experiment_plan("E99", dataset)


def test_sensitivity_partitions_are_the_prepared_ones():
    assert rx.sensitivity_partitions() == list(prepare_data.ALTERNATIVES)


# ----------------------------------------------------------------------------
# Selection and order
# ----------------------------------------------------------------------------
def _fake(uid, family, size, kind="positive"):
    return SimpleNamespace(uid=uid, family=family, size=size, kind=kind)


def test_selection_is_balanced_and_sharded():
    qs = [_fake(f"a{i}", "a", 20) for i in range(5)] + [_fake("b0", "b", 20)] + \
         [_fake(f"c{i}", "c", 50) for i in range(3)] + [_fake(f"n{i}", "neg", 20, "negative") for i in range(4)]
    picked = rx.balanced([q for q in qs if q.kind == "positive"], 6)
    assert [q.uid for q in picked] == ["a0", "a1", "a2", "b0", "c0", "c1"]
    assert [q.uid for q in rx.select_queries(qs, per_cell=1)] == ["a0", "b0", "c0", "n0"]
    chosen = rx.select_queries(qs, totals={"positive": 4, "negative": 2})
    assert [q.uid for q in chosen] == ["a0", "a1", "b0", "c0", "n0", "n1"]
    assert [q.uid for q in rx.select_queries(qs, uids=["c2", "a1"])] == ["a1", "c2"]
    assert [q.uid for q in rx.select_queries(qs, kinds=["negative"], limit=3)] == ["n0", "n1", "n2"]
    shards = [rx.select_queries(qs, shard=(i, 3)) for i in range(3)]
    assert sorted(q.uid for s in shards for q in s) == sorted(q.uid for q in qs)
    assert sum(len(s) for s in shards) == len(qs)
    slots = np.zeros((4, 4), dtype=int)
    for position in range(8):
        order = rx.arm_order(4, position)
        assert sorted(order) == [0, 1, 2, 3]
        for slot, arm in enumerate(order):
            slots[arm, slot] += 1
    assert (slots == 2).all()


def test_record_file_cuts_a_torn_line(tmp_path):
    path = tmp_path / "records" / "arm.jsonl"
    f = rx.RecordFile(path)
    f.write({"uid": "q1", "status": "FOUND", "value": np.float32(0.5)})
    f.write({"uid": "q2", "status": "NONE", "n": np.int64(3)})
    f.close()
    with open(path, "ab") as handle:
        handle.write(b'{"uid": "q3", "sta')
    again = rx.RecordFile(path)
    assert again.done == {"q1", "q2"} and again.recovered_bytes == len(b'{"uid": "q3", "sta')
    again.write({"uid": "q3", "status": "TIMEOUT"})
    again.close()
    assert [r["uid"] for r in rx.read_records(path)] == ["q1", "q2", "q3"]


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
def test_smoke_run_is_sound_across_arms(rstore, tmp_path):
    gstore, copy = rstore
    arms = ["filterall", "jigsaw_index", "random", "jigsaw_index+halo=0", "jigsaw_index+prune_passes=0",
            "jigsaw_index+component_mode=none", "jigsaw_index+component_cap=1", "jigsaw_index+label_filter=0",
            "jigsaw_index+halo_hops=2", "jigsaw_index+policy=topk", "jigsaw_index+seed_count=1", "direct"]
    out = tmp_path / "run"
    assert rx.main(_argv(copy, out, *arms)) == 0
    qs = rx.select_queries(queries.QuerySet.load(copy / "queries" / "dev" / "R0.npz").queries,
                           families=SMOKE_FAMILIES)
    assert qs and {q.kind for q in qs} == {"positive", "negative"}
    decided: dict[str, dict[str, str]] = {}
    for name in arms:
        arm = rx.Arm.parse(name).with_setting("backend", "networkx")
        recs = _records(out, arm.name)
        assert [r["uid"] for r in sorted(recs, key=lambda r: r["query_index"])] == [q.uid for q in qs]
        for r in recs:
            assert r["status"] != backends.ERROR, r.get("error")
            assert r["arm"] == arm.name and r["backend"] == "networkx_vf2" and r["attempt"] == 1
            assert isinstance(r["io"], dict) and all(isinstance(v, int) for v in r["io"].values())
            if sys.platform.startswith("linux"):
                assert {"read_bytes", "rchar", "major_faults", "children_major_faults"} <= set(r["io"])
            if r["kind"] == "positive":
                assert r["status"] not in (backends.NONE, cascade.INCOMPLETE)
            if r["status"] == backends.FOUND:
                assert r["witness_check"] == "ok"
            if r["status"] == cascade.INCOMPLETE:
                assert r["options"]["component_cap"] == 1 and r["time_decided"] is None
            if r["status"] in DECIDED:
                decided.setdefault(r["uid"], {})[arm.name] = r["status"]
        slots = [r["arm_slot"] for r in recs]
        assert len(set(slots)) > 1 or len(qs) == 1
    for uid, answers in decided.items():
        assert len(set(answers.values())) == 1, (uid, answers)
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert info["status"] == "complete" and info["split"] == "dev"
    assert info["config"]["arms"] == [rx.Arm.parse(a).with_setting("backend", "networkx").name for a in arms]
    options = info["config"]["cascade_options"]["R0"]
    assert options["jigsaw_index+backend=networkx+halo_hops=2"]["halo"] == 2
    assert options["jigsaw_index+backend=networkx+policy=topk"]["policy"] == "topk"
    assert options["jigsaw_index+backend=networkx+seed_count=1"]["policy_params"] == {"seed_count": 1}
    assert options["jigsaw_index+backend=networkx"]["policy"] == "hybrid"
    assert info["extra"]["threads"]["threads"] == protocol.SERVING["threads"]
    assert "queries/R0" in info["inputs"] and "label_index/R0" in info["inputs"]


def test_resume_skips_recorded_pairs_and_refuses_other_settings(rstore, tmp_path):
    _, copy = rstore
    out = tmp_path / "run"
    argv = _argv(copy, out, "filterall", "jigsaw_index")
    rx.main(argv)
    path = out / "records" / "R0" / "jigsaw_index+backend=networkx.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + lines[-1][:15])  # the last record torn by a crash
    summary = rx.execute(rx.build_parser().parse_args(argv))
    counts = summary["regimes"]["R0"]["arms"]
    assert summary["attempt"] == 2
    assert counts["jigsaw_index+backend=networkx"] == {"run": 1, "recorded_before": len(lines) - 1}
    assert counts["filterall+backend=networkx"]["run"] == 0
    assert summary["regimes"]["R0"]["recovered_bytes"] == {"jigsaw_index+backend=networkx": 15}
    assert len(rx.read_records(path)) == len(lines)
    assert (out / "manifest.attempt1.json").exists()
    with pytest.raises(SystemExit, match="other settings"):
        rx.main(_argv(copy, out, "filterall", "jigsaw_index", T=7.0))


def test_learned_arms_serve_the_numpy_encoder(rstore, tmp_path):
    gstore, copy = rstore
    models = tmp_path / "models"
    model_id = "synth-R0-graphsage-seed1-main"
    in_dim = features.input_dim("R0", int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    config = {"arch": "graphsage", "in_dim": in_dim, "hidden": 16, "output": 8, "layers": 2, "residual": True,
              "layer_norm": True, "dropout": 0.0}
    rng = np.random.default_rng(0)
    state = {k: (rng.normal(size=shape) / np.sqrt(shape[-1])).astype(np.float32)
             for k, shape in encoder_numpy.state_shapes(config).items()}
    encoder_numpy.save_numpy_model(models / model_id, config, state,
                                   extra={"model_id": model_id, "dataset": "synth", "regime": "R0",
                                          "selected_epoch": 1})
    serving.build_block_embeddings(gstore, "R0", models / model_id)
    out = tmp_path / "run"
    arms = [f"jigsaw_learned+model={model_id}", f"jigsaw+model={model_id}", f"mean_rrf+model={model_id}"]
    with pytest.raises(ValueError, match="router threshold"):
        rx.main(_argv(copy, out, *arms, extra=("--models-root", str(models))))
    assert not out.exists()  # the arms are assembled before anything is written
    rx.main(_argv(copy, out, *arms, extra=("--models-root", str(models), "--router-tau", "1.0")))
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    identity = info["config"]["models"][f"R0/jigsaw_learned+backend=networkx+model={model_id}"]
    assert identity["model_id"] == model_id and identity["state_sha256"]
    assert f"embeddings/{model_id}/R0" in info["inputs"]
    router = _records(out, f"jigsaw+backend=networkx+model={model_id}")
    assert all(r["model_id"] == model_id and r["status"] != backends.ERROR for r in router)
    assert {r["ranker_info"]["route"] for r in router if "ranker_info" in r} <= {"index", "fused_learned_index"}


def test_sealed_query_sets_need_the_gate(rstore, tmp_path, monkeypatch):
    _, copy = rstore
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    monkeypatch.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6))
    out = tmp_path / "run"
    sealed = ["--experiment", "E8", "--store-root", str(copy), "--dataset", "synth",
              "--queries", str(copy / "queries" / "test"), "--out", str(out)]
    with pytest.raises(seal.SealError, match="--unseal"):
        rx.main([*sealed, "--arms", "filterall"])
    with pytest.raises(seal.SealError, match="not frozen"):
        rx.main([*sealed, "--arms", "filterall", "--unseal"])
    # anything but the registered experiment is refused before the gate, so nothing reaches the unseal log
    refused = {("--budgets", "1", "2"): "--budgets", ("--limit", "1"): "--limit", ("--router-tau", "0"): "--router-tau",
               ("--families", "single"): "--families", ("--per-cell", "1"): "--per-cell", ("--T", "30"): "--T 30",
               ("--arms", "jigsaw_index"): "arms outside E8", ("--backend", "networkx"): "arms outside E8",
               ("--mode", "streaming"): "--mode streaming", ("--cache", "warm"): "--cache warm",
               ("--partition", "kahip_fast"): "--partition kahip_fast"}
    for extra, match in refused.items():
        arms = () if "--arms" in extra else ("--arms", "filterall")
        with pytest.raises(SystemExit, match=match):
            rx.main([*sealed, *arms, *extra, "--unseal"])
    # E16 registers a shorter arm list on each sensitivity partition
    e16 = ["--experiment", "E16", *sealed[2:], "--unseal"]
    for arm in ("jigsaw_index+halo_hops=2", "jigsaw_index+policy=topk"):
        with pytest.raises(SystemExit, match="arms outside E16 in resident mode on partition kahip_fast"):
            rx.main([*e16, "--partition", "kahip_fast", "--arms", arm])
    with pytest.raises(SystemExit, match="--partition nowhere"):
        rx.main([*e16, "--partition", "nowhere", "--arms", "jigsaw_index"])
    with pytest.raises(seal.SealError, match="not frozen"):  # a registered arm passes the plan check
        rx.main([*e16, "--partition", "kahip_fast", "--arms", "jigsaw_index"])
    with pytest.raises(SystemExit, match="ad-hoc"):
        rx.main(_argv(copy, out, "filterall", split="test", budgets=(), extra=("--unseal",)))
    assert not log.exists() and not out.exists()


def test_bad_setups_fail_before_writing(rstore, tmp_path):
    _, copy = rstore
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="0 <= i < n"):
        rx.main(_argv(copy, out, "filterall", extra=("--shard", "2/2")))
    with pytest.raises(SystemExit, match="listed twice"):
        rx.main(_argv(copy, out, "filterall", "filterall+backend=networkx"))
    with pytest.raises(SystemExit, match="--arms, --T and --regimes"):
        rx.main(["--store-root", str(copy), "--dataset", "synth", "--queries", str(copy), "--out", str(out)])
    with pytest.raises(FileNotFoundError):
        rx.main(_argv(copy, out, "jigsaw_learned+model=missing", extra=("--models-root", str(tmp_path))))
    if not hasattr(__import__("os"), "posix_fadvise"):
        with pytest.raises(SystemExit, match="cold runs"):
            rx.main(_argv(copy, out, "filterall", extra=("--mode", "streaming", "--cache", "cold")))
    assert not out.exists()


def test_runner_imports_no_torch():
    code = "import sys, scripts.canonical.run_experiment; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_escalate_is_a_ladder_setting():
    arm = rx.Arm.parse("jigsaw_index+escalate=4")
    assert arm.name == "jigsaw_index+escalate=4.0" and rx.Arm.parse(arm.name) == arm
    for bad in ("filterall+escalate=4", "jigsaw_index+escalate=0", "jigsaw_index+escalate=-1"):
        with pytest.raises(ValueError):
            rx.Arm.parse(bad)


def test_skip_below_is_a_ladder_setting():
    arm = rx.Arm.parse("jigsaw_index+skip_below=5000")
    assert arm.name == "jigsaw_index+skip_below=5000" and rx.Arm.parse(arm.name) == arm
    both = rx.Arm.parse("jigsaw_index+skip_below=5000+escalate=3")
    assert both.get("skip_below") == 5000 and both.get("escalate") == 3.0
    for bad in ("filterall+skip_below=5000", "jigsaw_index+skip_below=0", "jigsaw_index+skip_below=-4",
                "jigsaw_index+skip_below=2.5"):
        with pytest.raises(ValueError):
            rx.Arm.parse(bad)
    off = rx.Arm.parse("jigsaw_index+skip_below=off+escalate=off")
    assert off.name == "jigsaw_index+escalate=off+skip_below=off" and rx.Arm.parse(off.name) == off
    assert off.get("skip_below") == "off" and off.get("escalate") == "off"
    with pytest.raises(ValueError):
        rx.Arm.parse("filterall+escalate=off")
