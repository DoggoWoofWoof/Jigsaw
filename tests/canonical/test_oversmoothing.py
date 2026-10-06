"""Layerwise encoder analysis (E6): metrics, the sample, model checks, the run CLI and summarize."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.canonical import encoder_numpy, encoders, features, manifest, protocol, serving  # noqa: E402
from scripts.canonical import oversmoothing as ov  # noqa: E402
from scripts.canonical import train_encoder as te  # noqa: E402
from scripts.canonical.store import GraphStore  # noqa: E402

ARCHS = ("graphsage", "gin", "gat", "rgcn")
LAYERS = 3
SEED_A, SEED_B = (int(s) for s in protocol.SEEDS["training"][:2])
MAX_BLOCKS, MAX_NODES = 4, 64
SECONDS = re.compile(rb'"seconds": [^,\n]+')


def brute_force_cosine(x: np.ndarray) -> float:
    """Mean cosine over ordered pairs i != j with its own normalisation, one pair at a time."""
    x = np.asarray(x, dtype=np.float64)
    units = [row / max(float(np.sqrt(row @ row)), 1e-12) for row in x]
    n = len(units)
    total = sum(float(units[i] @ units[j]) for i in range(n) for j in range(n) if i != j)
    return total / (n * (n - 1))


def entropy_rank(values) -> float:
    p = np.asarray(values, dtype=np.float64)
    p = p / p.sum()
    return math.exp(-float(np.sum(p * np.log(p))))


# ----------------------------------------------------------------------------- metrics
def test_identical_rows_have_cosine_one_and_rank_one():
    x = np.tile(np.array([[0.5, -2.0, 3.0]]), (5, 1))
    m = ov.node_metrics(x)
    assert m["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-12)
    assert m["mean_average_distance"] == pytest.approx(0.0, abs=1e-12)
    assert m["effective_rank"] == pytest.approx(1.0, abs=1e-12)
    assert m["zero_rows"] == 0
    assert ov.mean_pairwise_cosine(x * np.arange(1.0, 6.0)[:, None]) == pytest.approx(1.0, abs=1e-12)


def test_orthonormal_rows_have_cosine_zero_and_full_rank():
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))
    m = ov.node_metrics(q)
    assert m["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-12)
    assert m["mean_average_distance"] == pytest.approx(1.0, abs=1e-12)
    assert m["effective_rank"] == pytest.approx(6.0, rel=1e-9)
    scales = np.arange(1.0, 7.0)
    assert ov.mean_pairwise_cosine(q * scales[:, None]) == pytest.approx(0.0, abs=1e-12)
    assert ov.effective_rank(q * scales[:, None]) == pytest.approx(entropy_rank(scales), rel=1e-9)


def test_a_zero_row_has_cosine_zero_with_everything_and_is_counted():
    x = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    m = ov.node_metrics(x)
    assert m["zero_rows"] == 1
    assert m["mean_pairwise_cosine"] == pytest.approx(1 / 3)  # (0, 1) and (1, 0) give 1; the 4 pairs with row 2 give 0
    assert m["mean_average_distance"] == pytest.approx(2 / 3)
    assert m["effective_rank"] == 1.0
    np.testing.assert_array_equal(ov.unit_rows(x)[2], [0.0, 0.0])
    assert ov.zero_rows(np.array([[-0.0, 0.0], [1.0, 0.0]])) == 1
    assert ov.node_metrics(np.zeros((4, 3))) == {"mean_pairwise_cosine": 0.0, "mean_average_distance": 1.0,
                                                 "effective_rank": 0.0, "zero_rows": 4}


def test_closed_form_cosine_matches_a_brute_force_loop():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 7))
    x[[3, 17]] = 0.0
    x[5] *= 1e-14  # below the 1e-12 floor: divided by the floor, not by its norm
    for data in (x, np.abs(x), np.maximum(x, 0.0)):
        assert ov.mean_pairwise_cosine(data) == pytest.approx(brute_force_cosine(data), abs=1e-12)
        assert ov.mean_average_distance(data) == pytest.approx(1 - brute_force_cosine(data), abs=1e-12)
    assert ov.mean_pairwise_cosine(x[:1]) is None and ov.mean_average_distance(x[:0]) is None
    with pytest.raises(ValueError, match="matrix"):
        ov.mean_pairwise_cosine(x[0])


def test_effective_rank_follows_roy_and_vetterli():
    assert ov.effective_rank(np.diag([3.0, 1.0])) == pytest.approx(entropy_rank([0.75, 0.25]), rel=1e-12)
    assert ov.effective_rank(np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0]])) == 1.0
    assert ov.effective_rank(np.zeros((3, 4))) == 0.0
    assert ov.effective_rank(np.zeros((0, 4))) == 0.0


def test_block_metrics_on_hand_built_summaries():
    summaries = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = ov.block_metrics(summaries)
    assert m["block_separation"] == pytest.approx(1 - math.sqrt(2) / 3, rel=1e-12)
    assert m["block_effective_rank"] == pytest.approx(entropy_rank([math.sqrt(3), 1.0]), rel=1e-12)
    assert ov.block_metrics(summaries[:1]) == {"block_effective_rank": 1.0, "block_separation": None}
    rep = np.array([[1, 2], [3, 5]], dtype=np.float32)
    assert ov.block_summary(rep).dtype == np.float64
    np.testing.assert_array_equal(ov.block_summary(rep), [2.0, 3.5])
    np.testing.assert_array_equal(ov.block_summary(np.zeros((0, 3), dtype=np.float32)), np.zeros(3))


# ----------------------------------------------------------------------------- sample and settings
def test_node_quotas_stratify_equally_and_draws_are_seeded():
    assert ov.node_quotas([10, 3, 50, 7], 20) == [6, 3, 6, 5]
    assert ov.node_quotas([10, 3, 50, 7], 70) == [10, 3, 50, 7]
    assert ov.node_quotas([4, 4, 4], 7) == [3, 2, 2]
    assert ov.node_quotas([5, 0, 9], 7) == [4, 0, 3]
    for sizes, budget in (([10, 3, 50, 7], 20), ([100] * 7, 64), ([1, 2, 3, 400], 50)):
        quotas = ov.node_quotas(sizes, budget)
        assert sum(quotas) == min(budget, sum(sizes)) and all(0 <= q <= s for q, s in zip(quotas, sizes))
    assert ov.choose_blocks(6, 6, dataset="synth", regime="R0", seed=0) == list(range(6))
    drawn = ov.choose_blocks(100, 8, dataset="synth", regime="R0", seed=0)
    assert drawn == sorted(set(drawn)) and len(drawn) == 8
    assert drawn == ov.choose_blocks(100, 8, dataset="synth", regime="R0", seed=0)
    assert drawn != ov.choose_blocks(100, 8, dataset="synth", regime="R1", seed=0)
    pos = ov.sample_positions(50, 10, dataset="synth", regime="R0", seed=0, block=3)
    assert pos.tolist() == sorted(set(pos.tolist())) and len(pos) == 10 and 0 <= pos.min() and pos.max() < 50
    np.testing.assert_array_equal(pos, ov.sample_positions(50, 10, dataset="synth", regime="R0", seed=0, block=3))
    np.testing.assert_array_equal(ov.sample_positions(5, 9, dataset="synth", regime="R0", seed=0, block=3),
                                  np.arange(5))


def test_settings_and_metric_names_come_from_the_protocol(monkeypatch):
    listed = protocol.ARCHITECTURE_ABLATION.get("layerwise_sample") or {}
    assert ov.sample_settings() == {**ov.DEFAULT_SAMPLE, **{k: int(v) for k, v in listed.items()}}
    monkeypatch.setitem(protocol.ARCHITECTURE_ABLATION, "layerwise_sample", {"seed": 5, "max_nodes": 128})
    assert ov.sample_settings() == {"seed": 5, "max_blocks": ov.DEFAULT_SAMPLE["max_blocks"], "max_nodes": 128}
    monkeypatch.setitem(protocol.ARCHITECTURE_ABLATION, "layerwise_sample", {"sead": 5})
    with pytest.raises(ValueError, match="unknown layerwise_sample"):
        ov.sample_settings()
    names = ov.metric_names()
    assert names[:4] == protocol.ARCHITECTURE_ABLATION["layerwise_metrics"]
    assert sorted(names) == sorted(ov.NODE_METRICS + ov.BLOCK_METRICS)
    monkeypatch.setitem(protocol.ARCHITECTURE_ABLATION, "layerwise_metrics", ["effective_rank", "dirichlet_energy"])
    with pytest.raises(ValueError, match="dirichlet_energy"):
        ov.metric_names()


# ----------------------------------------------------------------------------- end to end
def _export(gstore: GraphStore, models_root: Path, arch: str, residual: bool, seed: int, regime: str = "R0") -> str:
    """A random hidden-16, 3-layer encoder written by train_encoder.export_model; returns its model id."""
    in_dim = features.input_dim(regime, int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    config = {"arch": arch, "in_dim": in_dim, "hidden": 16, "output": 8, "layers": LAYERS, "residual": residual,
              "layer_norm": True, "dropout": 0.0}
    if arch == "rgcn":
        config["num_relations"] = int(features.store_relation_table(gstore).max()) + 1
    torch.manual_seed(protocol.derive_seed(seed, arch, residual, regime))
    model = encoders.build_encoder(config)
    arm = te.arch_arm_name({"arch": arch, "layers": LAYERS, "residual": residual})
    model_id = te.model_id_for("synth", regime, arch, seed, arm)
    plan = {"model_id": model_id, "dataset": "synth", "regime": regime, "requested_regime": regime, "arch": arch,
            "seed": seed, "arm": arm, "arm_kind": "architecture", "smoke": {}}
    selection = {"epoch": 1, "metric": 0.0, "checkpoint": "ckpt_epoch001.pt", "fullcov_at": {}}
    te.export_model(model, gstore, plan, models_root / model_id, selection=selection)
    return model_id


def _run_argv(env: SimpleNamespace, out: Path, models, *extra: str, regime: str = "R0") -> list[str]:
    return ["run", "--store-root", str(env.stores), "--dataset", "synth", "--regime", regime, "--models-root",
            str(env.models), "--models", *models, "--out", str(out), *extra]


def _records(run_dir: Path) -> dict[str, dict]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.json"))
            if p.name != manifest.MANIFEST_NAME}


@pytest.fixture(scope="module")
def env(clustered_store, tmp_path_factory):
    """Nine exported encoders (4 archs x residual on/off, plus a second seed of graphsage) and two identical runs."""
    root, _ = clustered_store
    base = tmp_path_factory.mktemp("layerwise")
    stores = base / "stores"
    shutil.copytree(root / "synth", stores / "synth")
    gstore = GraphStore(stores, "synth", mmap=True)
    models = base / "models"
    ids = {}
    for arch in ARCHS:
        for residual in (True, False):
            ids[arch, residual, SEED_A] = _export(gstore, models, arch, residual, SEED_A)
    ids["graphsage", True, SEED_B] = _export(gstore, models, "graphsage", True, SEED_B)
    ns = SimpleNamespace(base=base, stores=stores, store=gstore, models=models, ids=ids, runs={},
                         threads=torch.get_num_threads())
    for name in ("out1", "out2"):
        argv = _run_argv(ns, base / name, list(ids.values()), "--max-blocks", str(MAX_BLOCKS), "--max-nodes",
                         str(MAX_NODES), "--knockout-residual")
        assert ov.main(argv) == 0
        ns.runs[name] = base / name
    ns.threads_after = torch.get_num_threads()
    return ns


def test_run_writes_one_self_describing_record_per_model_and_knockout(env):
    out = env.runs["out1"]
    records = _records(out)
    by_id = {model_id: key for key, model_id in env.ids.items()}
    residual_ids = [model_id for (_, residual, _), model_id in env.ids.items() if residual]
    assert set(records) == ({f"{m}.json" for m in env.ids.values()} | {f"{m}+knockout.json" for m in residual_ids})
    assert manifest.verify_manifest(out) == []
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (info["experiment"], info["split"], info["status"]) == ("E6_layerwise", "none", "complete")
    assert sorted(info["extra"]["knockout_skipped"]) == sorted(set(env.ids.values()) - set(residual_ids))
    assert {"store_meta", "partition_json"} <= set(info["inputs"])
    assert {f"{m}/torch_state.pt" for m in env.ids.values()} | {f"{m}/model.json" for m in env.ids.values()} \
        <= set(info["inputs"])
    assert env.threads_after == env.threads
    samples = set()
    for name, record in records.items():
        ov.validate_record(record)
        arch, residual, seed = by_id[record["model_id"]]
        assert (record["schema"], record["dataset"], record["regime"], record["partition"]) == \
            ("jigsaw-layerwise-v1", "synth", "R0", "primary")
        assert record["knockout_residual"] is name.endswith("+knockout.json")
        assert record["model"] == {
            "arch": arch, "layers": LAYERS, "hidden": 16, "residual": residual, "layer_norm": True,
            "train_seed": seed, "arm": te.arch_arm_name({"arch": arch, "layers": LAYERS, "residual": residual}),
            "trained_regime": "R0",
            "torch_state_sha256": manifest.sha256_file(env.models / record["model_id"] / "torch_state.pt")["sha256"]}
        assert [e["layer"] for e in record["layers"]] == ["input", "proj", "1", "2", "3", "readout"]
        assert [e["dim"] for e in record["layers"]] == [14, 16, 16, 16, 16, 8]
        for entry in record["layers"]:
            assert set(entry) == {"layer", "dim", *ov.NODE_METRICS, *ov.BLOCK_METRICS}
            assert 1.0 <= entry["block_effective_rank"] <= MAX_BLOCKS and entry["block_separation"] is not None
            if entry["layer"] == "readout":
                assert all(entry[k] is None for k in ov.NODE_METRICS)
                continue
            assert -1.0 <= entry["mean_pairwise_cosine"] <= 1.0
            assert entry["mean_average_distance"] == pytest.approx(1 - entry["mean_pairwise_cosine"], abs=1e-15)
            assert 1.0 <= entry["effective_rank"] <= entry["dim"] and 0 <= entry["zero_rows"] <= MAX_NODES
        assert set(record["definitions"]) >= {"layer", "knockout_residual", *ov.NODE_METRICS, *ov.BLOCK_METRICS}
        assert record["protocol_hash"] == protocol.protocol_hash() and record["seconds"] >= 0
        samples.add(protocol.canonical_json(record["sample"]))
    assert len(samples) == 1  # every model and variant is measured on the same nodes
    sample = json.loads(samples.pop())
    assert sample["blocks"] == ov.choose_blocks(env.store.num_blocks, MAX_BLOCKS, dataset="synth", regime="R0",
                                                seed=ov.sample_settings()["seed"])
    assert len(sample["blocks"]) == MAX_BLOCKS < env.store.num_blocks
    assert sample["nodes_per_block"] == ov.node_quotas(sample["block_sizes"], MAX_NODES) == [16] * MAX_BLOCKS
    assert (sample["max_blocks"], sample["max_nodes"], sample["seed"]) == \
        (MAX_BLOCKS, MAX_NODES, ov.sample_settings()["seed"])


def test_knockout_changes_only_the_layers_after_the_projection(env):
    records = _records(env.runs["out1"])
    for (_, residual, _), model_id in env.ids.items():
        if not residual:
            assert f"{model_id}+knockout.json" not in records
            continue
        trained, knocked = records[f"{model_id}.json"], records[f"{model_id}+knockout.json"]
        assert (trained["knockout_residual"], knocked["knockout_residual"]) == (False, True)
        assert knocked["model"] == trained["model"] and knocked["model"]["residual"] is True
        for a, b in zip(trained["layers"], knocked["layers"]):
            same = [a[k] == b[k] for k in ov.NODE_METRICS + ov.BLOCK_METRICS]
            assert all(same) if a["layer"] in ("input", "proj") else not all(same), (model_id, a["layer"])


def test_a_rerun_is_byte_identical_apart_from_seconds(env):
    one, two = env.runs["out1"], env.runs["out2"]
    assert sorted(_records(one)) == sorted(_records(two))
    for name in _records(one):
        a, b = (one / name).read_bytes(), (two / name).read_bytes()
        assert SECONDS.search(a) and b"\r\n" not in a
        assert SECONDS.sub(b'"seconds": 0', a) == SECONDS.sub(b'"seconds": 0', b), name


@pytest.mark.parametrize("arch", ARCHS)
def test_readout_equals_the_served_block_embeddings(env, arch):
    for residual in (True, False):
        model_dir = env.models / env.ids[arch, residual, SEED_A]
        record, z = ov.analyse_model(env.store, "R0", model_dir, max_blocks=MAX_BLOCKS, max_nodes=MAX_NODES)[0]
        served = np.load(Path(serving.build_block_embeddings(env.store, "R0", model_dir)["dir"]) / "block_emb.npy")
        assert z.shape == (MAX_BLOCKS, 8) and np.abs(z - served[record["sample"]["blocks"]]).max() < 1e-5
        cli = json.loads((env.runs["out1"] / f"{record['model_id']}.json").read_text(encoding="utf-8"))
        api = json.loads(json.dumps(record))
        assert {k: v for k, v in api.items() if k != "seconds"} == {k: v for k, v in cli.items() if k != "seconds"}
    record, z = ov.analyse_model(env.store, "R0", model_dir, max_blocks=64, max_nodes=MAX_NODES)[0]
    assert record["sample"]["blocks"] == list(range(env.store.num_blocks)) and np.abs(z - served).max() < 1e-5


def test_the_r2_512_encoder_serves_r2_32(env):
    model_id = _export(env.store, env.models, "graphsage", True, SEED_A, regime="R2-512")
    record, z = ov.analyse_model(env.store, "R2-32", env.models / model_id, max_blocks=MAX_BLOCKS,
                                 max_nodes=MAX_NODES)[0]
    assert (record["regime"], record["model"]["trained_regime"], record["model_id"]) == ("R2-32", "R2-512", model_id)
    served = np.load(Path(serving.build_block_embeddings(env.store, "R2-32", env.models / model_id)["dir"])
                     / "block_emb.npy")
    assert np.abs(z - served[record["sample"]["blocks"]]).max() < 1e-5
    with pytest.raises(ValueError, match="trained for R2-512, R0 is served by the R0 encoder"):
        ov.analyse_model(env.store, "R0", env.models / model_id)


def test_a_regime_the_model_was_not_trained_for_is_refused_before_out_exists(env):
    out = env.base / "refused"
    models = [env.ids["graphsage", True, SEED_A]]
    with pytest.raises(ValueError, match="trained for R0, R1 is served by the R1 encoder"):
        ov.main(_run_argv(env, out, models, regime="R1"))
    with pytest.raises(ValueError, match="unknown regime 'R7'"):
        ov.main(_run_argv(env, out, models, regime="R7"))
    assert not out.exists()


def _edit_json(path: Path, update) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_model_identity_and_integrity_are_checked(env, tmp_path):
    source = env.models / env.ids["graphsage", True, SEED_A]

    def copy(case: str, name: str | None = None) -> Path:
        target = tmp_path / case / (name or source.name)
        shutil.copytree(source, target)
        return target

    assert ov.load_model(copy("ok")).identity == {"model_id": source.name, "dataset": "synth", "regime": "R0",
                                                  "arch": "graphsage", "seed": SEED_A,
                                                  "arm": "arch-graphsage-L3-res"}
    with pytest.raises(ValueError, match="the directory renamed holds model"):
        ov.load_model(copy("renamed", "renamed"))
    seed = copy("seed")
    _edit_json(seed / "model.json", lambda p: p["extra"].update(seed=7399))
    with pytest.raises(ValueError, match="the seed differs"):
        ov.load_model(seed)
    both = copy("both")
    for name in ("model.json", "export.json"):
        _edit_json(both / name, lambda p: (p.get("extra") or p).update(seed=7399))
    with pytest.raises(ValueError, match="does not match its identity fields"):
        ov.load_model(both)
    sha = copy("sha")
    _edit_json(sha / "export.json", lambda p: p.update(torch_state_sha256="0" * 64))
    with pytest.raises(ValueError, match="torch_state_sha256"):
        ov.load_model(sha)
    weights = copy("weights")
    model_json = json.loads((weights / "model.json").read_text(encoding="utf-8"))
    state = {k: np.full(shape, 0.01, dtype=np.float32)
             for k, shape in encoder_numpy.state_shapes(model_json["config"]).items()}
    encoder_numpy.save_numpy_model(weights, model_json["config"], state, extra=model_json["extra"])
    with pytest.raises(ValueError, match="holds other weights"):
        ov.load_model(weights)
    missing = copy("missing")
    (missing / "export.json").unlink()
    _edit_json(missing / "model.json", lambda p: p["extra"].pop("arm"))
    with pytest.raises(ValueError, match="records the arm"):
        ov.load_model(missing)
    nores = ov.load_model(env.models / env.ids["gin", False, SEED_A])
    sample = ov.draw_sample(env.store, "R0", max_blocks=MAX_BLOCKS, max_nodes=MAX_NODES, seed=0)
    with pytest.raises(ValueError, match="trained without the residual"):
        ov.analyse(env.store, "R0", nores, sample, None, knockout=True)


def test_run_refuses_stray_files_repeated_models_missing_models_and_tiny_samples(env, tmp_path):
    model_id = env.ids["gin", True, SEED_A]
    out = tmp_path / "stray"
    out.mkdir()
    (out / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="would not write"):
        ov.main(_run_argv(env, out, [model_id]))
    assert [p.name for p in out.iterdir()] == ["notes.txt"]
    with pytest.raises(ValueError, match="more than once"):
        ov.main(_run_argv(env, tmp_path / "twice", [model_id, model_id]))
    with pytest.raises(ValueError, match="not an export_model directory"):
        ov.main(_run_argv(env, tmp_path / "absent", ["synth-R0-gin-seed7301-no-such-arm"]))
    with pytest.raises(ValueError, match="at least 2"):
        ov.main(_run_argv(env, tmp_path / "tiny", [model_id], "--max-nodes", "1"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stray"]


def test_summarize_aggregates_training_seeds(env, tmp_path):
    path = tmp_path / "summary.csv"
    rows = ov.summarize([env.runs["out1"]], path)
    with open(path, encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert list(table[0]) == ov.csv_columns() and len(table) == len(rows) == 12 * (LAYERS + 3)
    records = _records(env.runs["out1"])
    a, b = (records[f"{env.ids['graphsage', True, seed]}.json"] for seed in (SEED_A, SEED_B))
    group = [r for r in table if (r["arch"], r["residual"], r["knockout"]) == ("graphsage", "true", "false")]
    assert [r["layer"] for r in group] == ["input", "proj", "1", "2", "3", "readout"]
    for row, ea, eb in zip(group, a["layers"], b["layers"]):
        assert (row["n_seeds"], row["seeds"], row["seeds_complete"]) == ("2", f"{SEED_A};{SEED_B}", "false")
        assert (row["dataset"], row["regime"], row["partition"], row["trained_regime"], row["dim"]) == \
            ("synth", "R0", "primary", "R0", str(ea["dim"]))
        assert (row["sample_blocks"], row["sample_nodes"]) == (str(MAX_BLOCKS), str(MAX_NODES))
        for metric in ov.metric_names():
            if ea[metric] is None:
                assert row[f"{metric}_mean"] == row[f"{metric}_sd"] == ""
                continue
            assert float(row[f"{metric}_mean"]) == pytest.approx((ea[metric] + eb[metric]) / 2, rel=1e-12, abs=1e-15)
            assert float(row[f"{metric}_sd"]) == pytest.approx(abs(ea[metric] - eb[metric]) / math.sqrt(2),
                                                               rel=1e-9, abs=1e-15)
    single = [r for r in table if r["arch"] == "gin"]
    assert {r["n_seeds"] for r in single} == {"1"} and {r["effective_rank_sd"] for r in single} == {""}
    knocked = [r for r in table if r["knockout"] == "true"]
    assert {r["arch"] for r in knocked} == set(ARCHS) and {r["residual"] for r in knocked} == {"true"}
    assert {r["knockout"] for r in table if r["residual"] == "false"} == {"false"}
    order = {name: i for i, name in enumerate(ov.layer_names(LAYERS))}
    keys = [(r["dataset"], r["regime"], r["arch"], int(r["layers"]), r["residual"] == "true",
             r["knockout"] == "true", r["arm"], order[r["layer"]]) for r in table]
    assert keys == sorted(keys)
    assert ov.main(["summarize", "--inputs", str(env.runs["out1"]), "--out", str(tmp_path / "cli.csv")]) == 0
    assert (tmp_path / "cli.csv").read_bytes() == path.read_bytes()
    # all three protocol seeds make the row complete
    third = json.loads(json.dumps(b))
    third["model"]["train_seed"] = int(protocol.SEEDS["training"][2])
    third["model_id"] = serving.model_id_for("synth", "R0", "graphsage", third["model"]["train_seed"],
                                             third["model"]["arm"])
    complete = [r for r in ov.summary_rows([a, b, third]) if not r["knockout"]]
    assert {(r["n_seeds"], r["seeds_complete"]) for r in complete} == {(3, True)}


def test_summarize_refuses_repeated_seeds_tampered_runs_and_mixed_samples(env, tmp_path):
    with pytest.raises(ValueError, match="appear more than once"):
        ov.summarize([env.runs["out1"], env.runs["out2"]], tmp_path / "twice.csv")
    tampered = tmp_path / "tampered"
    shutil.copytree(env.runs["out1"], tampered)
    victim = tampered / f"{env.ids['gat', True, SEED_A]}.json"
    victim.write_bytes(victim.read_bytes().replace(b'"layer_norm": true', b'"layer_norm": false'))
    with pytest.raises(ValueError, match="changed output"):
        ov.summarize([tampered], tmp_path / "tampered.csv")
    with pytest.raises(ValueError, match="no manifest.json"):
        ov.read_run(tmp_path)
    records = _records(env.runs["out1"])
    a, b = (json.loads(json.dumps(records[f"{env.ids['graphsage', True, seed]}.json"])) for seed in (SEED_A, SEED_B))
    b["sample"]["max_nodes"] = 2 * MAX_NODES
    with pytest.raises(ValueError, match="differ in sample"):
        ov.summary_rows([a, b])
    b["model"]["train_seed"] = 7399
    with pytest.raises(ValueError, match="does not match its identity fields"):
        ov.validate_record(b)
    with pytest.raises(ValueError, match="not a jigsaw-layerwise-v1 record"):
        ov.validate_record({"schema": "jigsaw-layerwise-v0"})
    assert not any((tmp_path / name).exists() for name in ("twice.csv", "tampered.csv"))
