"""Manifests of the offline builds (E17): label index, ranker inputs and block embeddings."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from scripts.canonical import cascade, encoder_numpy, features, manifest, offline_build, rankers, serving
from scripts.canonical import store as store_mod


@pytest.fixture
def ostore(clustered_store, tmp_path):
    root, _ = clustered_store
    shutil.copytree(root / "synth", tmp_path / "stores" / "synth")
    return tmp_path / "stores", store_mod.GraphStore(tmp_path / "stores", "synth", mmap=True)


def _model(gstore, model_dir):
    in_dim = features.input_dim("R0", int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    config = {"arch": "graphsage", "in_dim": in_dim, "hidden": 16, "output": 8, "layers": 2, "residual": True,
              "layer_norm": True, "dropout": 0.0}
    rng = np.random.default_rng(0)
    state = {k: (rng.normal(size=shape) / np.sqrt(shape[-1])).astype(np.float32)
             for k, shape in encoder_numpy.state_shapes(config).items()}
    encoder_numpy.save_numpy_model(model_dir, config, state, extra={"model_id": "synth-R0-graphsage-test",
                                                                     "dataset": "synth", "regime": "R0",
                                                                     "selected_epoch": 1})
    return model_dir


def _check_run(run_dir, gstore, stage, prefixes):
    doc = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert doc["experiment"] == offline_build.EXPERIMENT and doc["status"] == "complete"
    assert doc["config"]["stage"] == stage and doc["config"]["dataset"] == "synth"
    assert manifest.verify_manifest(run_dir) == []
    assert list(doc["outputs"]) == [offline_build.ARTIFACTS]
    entries = json.loads((run_dir / offline_build.ARTIFACTS).read_text(encoding="utf-8"))
    on_disk = {p.relative_to(gstore.dir).as_posix(): p.stat().st_size
               for prefix in prefixes for p in (gstore.dir / prefix).rglob("*") if p.is_file()}
    assert {k: v["bytes"] for k, v in entries.items()} == on_disk and on_disk
    extra = doc["extra"]
    assert extra["stage"] == stage and extra["artifacts"] == len(on_disk)
    assert extra["artifact_bytes"] == sum(on_disk.values())
    assert extra["wall_seconds"] >= 0 and isinstance(extra["peak_memory"], dict)
    return doc


def test_the_three_builds_record_their_artifacts(ostore, tmp_path, capsys):
    root, gstore = ostore
    runs = tmp_path / "offline"
    assert cascade.main(["build-label-index", "--root", str(root), "--dataset", "synth", "--regimes", "R0", "R2-32",
                         "--run-dir", str(runs / "label_index")]) == 0
    doc = _check_run(runs / "label_index", gstore, "label_index", ["label_index/R0", "label_index/R2-32"])
    assert doc["config"]["regimes"] == ["R0", "R2-32"] and "sigma/R0" in doc["inputs"]
    assert rankers.main(["build", "--root", str(root), "--dataset", "synth", "--regimes", "R0",
                         "--run-dir", str(runs / "rankers")]) == 0
    doc = _check_run(runs / "rankers", gstore, "rankers", ["part/primary/rankers"])
    assert doc["config"]["partition"] == "primary"
    assert set(p.name for p in (gstore.dir / "part/primary/rankers").iterdir()) == {"R0", "topology"}
    model_dir = _model(gstore, tmp_path / "model")
    assert serving.main(["embed-blocks", "--root", str(root), "--dataset", "synth", "--model-dir", str(model_dir),
                         "--regimes", "R0", "--run-dir", str(runs / "embeddings")]) == 0
    doc = _check_run(runs / "embeddings", gstore, "embeddings", ["part/primary/embeddings"])
    assert doc["config"]["model_dir"] == str(model_dir) and doc["inputs"]["model_dir"]["files"] > 0
    capsys.readouterr()


def test_without_a_run_dir_no_manifest_is_written(ostore, tmp_path):
    root, gstore = ostore
    before = {p for p in tmp_path.rglob("*")}
    assert cascade.main(["build-label-index", "--root", str(root), "--dataset", "synth", "--regimes", "R0"]) == 0
    new = {p for p in tmp_path.rglob("*")} - before
    assert new and all("label_index" in p.relative_to(gstore.dir).as_posix() for p in new)
    assert not any(p.name == manifest.MANIFEST_NAME for p in new)
    with pytest.raises(SystemExit):  # check is a read: it takes no run dir
        serving.main(["check", "--root", str(root), "--dataset", "synth", "--model-dir", str(tmp_path),
                      "--run-dir", str(tmp_path / "x")])


def test_a_failed_build_is_recorded_as_failed_without_an_artifact_list(ostore, tmp_path):
    _, gstore = ostore
    run_dir = tmp_path / "run"
    with offline_build.build_run(run_dir, stage="label_index", store_dir=gstore.dir, dataset="synth",
                                 config={}) as written:
        written.append(cascade.label_index_dir(gstore, "R0"))
        cascade.build_label_index(gstore, "R0")
    assert (run_dir / offline_build.ARTIFACTS).exists()
    with pytest.raises(RuntimeError, match="boom"):
        with offline_build.build_run(run_dir, stage="label_index", store_dir=gstore.dir, dataset="synth",
                                     config={}):
            raise RuntimeError("boom")
    doc = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert doc["status"] == "failed" and "boom" in doc["extra"]["error"]
    assert not (run_dir / offline_build.ARTIFACTS).exists() and doc["outputs"] == {}


def test_refusals(ostore, tmp_path):
    _, gstore = ostore
    with pytest.raises(ValueError, match="unknown offline build stage"):
        with offline_build.build_run(tmp_path / "r", stage="training", store_dir=gstore.dir, dataset="synth",
                                     config={}):
            pass
    with pytest.raises(FileNotFoundError, match="no such artifact"):
        offline_build.artifact_entries(gstore.dir, [gstore.dir / "label_index" / "missing"])
    with offline_build.build_run(None, stage="anything", store_dir=gstore.dir, dataset="synth", config={}) as w:
        assert w == []  # without a run dir nothing is checked or written
    assert not (tmp_path / "r" / offline_build.ARTIFACTS).exists()


# ----------------------------------------------------------------------------- offline_cost (E17)
def test_names_match_the_modules_they_come_from():
    from scripts.canonical import prepare_data, train_encoder

    assert offline_build.PREPARE_STAGES == prepare_data.STAGES
    assert offline_build.TRAIN_LOGS == (train_encoder.TRAIN_LOG, train_encoder.VAL_METRICS)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_prepare_cost_comes_from_the_stage_markers_and_store_files(tmp_path):
    run = tmp_path / "_prepare_run"
    (run / "stages").mkdir(parents=True)
    for name, wall, peak in (("load", 10.0, 100), ("partition", 30.5, 400), ("sigma", 2.0, None)):
        (run / "stages" / f"{name}.json").write_text(json.dumps({"stage": name, "wall_seconds": wall,
                                                                 "peak_rss_bytes": peak}), encoding="utf-8")
    files = {"meta.json": 5, "prepare_report.json": 7, "graph/indptr.npy": 1000, "graph/attrs.npy": 2000,
             "sigma/R0/codes.npy": 300, "part/primary/node_block.npy": 40, "part/metis_x0.5/node_block.npy": 41,
             "part/primary/rankers/R0/postings_ptr.npy": 999, "part/primary/embeddings/m/R0/block_emb.npy": 999}
    doc = {"experiment": "prepare_data", "extra": {"wall_seconds": 0.4,  # a resumed call: every stage skipped
                                                   "store_files": {k: {"bytes": v} for k, v in files.items()}}}
    cost = offline_build.offline_cost("prepare", run, doc)
    assert cost["wall_seconds"] == 42.5 and cost["peak_bytes"] == 400
    assert cost["parts"] == {"load": {"wall_seconds": 10.0, "peak_bytes": 100},
                             "partition": {"wall_seconds": 30.5, "peak_bytes": 400},
                             "sigma": {"wall_seconds": 2.0, "peak_bytes": None}}
    assert cost["disk"] == {"meta": 12, "graph": 3000, "sigma": 300, "partition/primary": 40,
                            "partition/metis_x0.5": 41}
    assert cost["bytes"] == 3393
    (run / "stages" / "load.json").unlink()
    for name in ("partition", "sigma"):
        (run / "stages" / f"{name}.json").unlink()
    with pytest.raises(ValueError, match="without stage markers"):
        offline_build.offline_cost("prepare", run, doc)


def test_training_cost_sums_the_kept_epochs_and_validations(tmp_path):
    _jsonl(tmp_path / "train_log.jsonl", [{"epoch": 1, "seconds": 100.0}, {"epoch": 2, "seconds": 110.5}])
    _jsonl(tmp_path / "val_metrics.jsonl", [{"epoch": 2, "seconds": 9.5}])
    doc = {"experiment": "train_encoder", "extra": {"wall_seconds": 12.0}}  # the last segment of a resumed run
    cost = offline_build.offline_cost("training", tmp_path, doc)
    assert cost["wall_seconds"] == 220.0 and cost["bytes"] is None
    (tmp_path / "val_metrics.jsonl").unlink()
    with pytest.raises(ValueError, match="without train_log.jsonl and val_metrics.jsonl"):
        offline_build.offline_cost("training", tmp_path, doc)


def test_quiet_training_times_the_epochs_without_other_work(tmp_path):
    # other load = host_busy_cpus - cpu_seconds / seconds: 0.5, 5.0, 0.0, 0.9, 1.0 (not below 1), then unmeasured
    epochs = [(10.0, 40.0, 4.5), (20.0, 80.0, 9.0), (12.0, 48.0, 4.0), (11.0, 44.0, 4.9), (30.0, 120.0, 5.0)]
    lines = [{"epoch": i + 1, "seconds": s, "cpu_seconds": c, "host_busy_cpus": b} for i, (s, c, b) in enumerate(epochs)]
    _jsonl(tmp_path / "train_log.jsonl", lines + [{"epoch": 6, "seconds": 13.0}])
    _jsonl(tmp_path / "val_metrics.jsonl", [{"epoch": 6, "seconds": 2.0}])
    q = offline_build.quiet_training(tmp_path)
    assert (q["epochs"], q["measured"], q["quiet"]) == (6, 5, 3)
    assert q["epoch_seconds_median"] == 11.0 and q["other_cpus_median"] == pytest.approx(0.9)
    assert q["wall_seconds"] is None  # 3 quiet epochs < QUIET_MIN_EPOCHS
    lines += [{"epoch": 7 + i, "seconds": 9.0, "cpu_seconds": 36.0, "host_busy_cpus": 4.0} for i in range(2)]
    _jsonl(tmp_path / "train_log.jsonl", lines)
    q = offline_build.quiet_training(tmp_path)
    assert (q["epochs"], q["quiet"], q["epoch_seconds_median"]) == (7, 5, 10.0)
    assert q["wall_seconds"] == 10.0 * 7 + 2.0
    doc = {"experiment": "train_encoder", "extra": {}}
    assert offline_build.offline_cost("training", tmp_path, doc)["quiet"] == q
    (tmp_path / "train_log.jsonl").write_text("", encoding="utf-8")
    assert offline_build.quiet_training(tmp_path) == {"epochs": 0, "measured": 0, "quiet": 0, "other_cpus_median": None,
                                                     "epoch_seconds_median": None, "wall_seconds": None}


def test_build_and_other_costs(ostore, tmp_path):
    root, gstore = ostore
    run_dir = tmp_path / "li"
    assert cascade.main(["build-label-index", "--root", str(root), "--dataset", "synth", "--regimes", "R0",
                         "--run-dir", str(run_dir)]) == 0
    doc = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    cost = offline_build.offline_cost("label_index", run_dir, doc)
    assert cost["wall_seconds"] == doc["extra"]["wall_seconds"] and cost["bytes"] == doc["extra"]["artifact_bytes"]
    assert cost["peak_bytes"] == offline_build.peak_bytes(doc["extra"]["peak_memory"])
    with pytest.raises(ValueError, match="offline_build manifest"):
        offline_build.offline_cost("rankers", run_dir, {**doc, "experiment": "router_fit"})
    other = {"experiment": "router_fit", "extra": {"wall_seconds": 3.5},
             "outputs": {"router_fit.json": {"bytes": 10}, "router_fit.queries.jsonl": {"bytes": 32}}}
    assert offline_build.offline_cost("router_fit", tmp_path, other)["bytes"] == 42
    assert offline_build.peak_bytes({"maxrss_bytes": 5, "peak_wset_bytes": 9, "flag": True}) == 9
    assert offline_build.peak_bytes(None) is None and offline_build.peak_bytes(7.0) == 7
