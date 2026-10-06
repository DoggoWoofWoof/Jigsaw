"""Canonical encoder trainer: objective terms, sampler, checkpoints, resume, export and the matrix."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.canonical import encoders, manifest, protocol, queries  # noqa: E402
from scripts.canonical import train_encoder as te  # noqa: E402
from scripts.canonical.launch import train_matrix as tm  # noqa: E402
from scripts.canonical.store import GraphStore  # noqa: E402

F = torch.nn.functional
SEED = 7301
SMOKE = ["--epochs", "2", "--steps-per-epoch", "3", "--layers", "2", "--hidden", "16", "--output", "8",
         "--val-every", "1", "--device", "cpu"]


def softplus(x: float) -> float:
    return math.log1p(math.exp(x))


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synth(clustered_store, tmp_path_factory):
    """The clustered store as a protocol dataset "synth" (6 blocks, ladder 1 2 3) and its validation set."""
    root, _ = clustered_store
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6, ladder=[1, 2, 3]))
        mp.setitem(protocol.TRAINING["loss"]["topk_barrier_K"], "synth", 3)
        mp.setitem(protocol.TRAINING["loss"]["live_positives"], "synth", 4)
        store = GraphStore(root, "synth", mmap=True)
        val_dir = tmp_path_factory.mktemp("validation")
        queries.SplitBuilder(store, "synth", "validation", "31415", sizes=[8, 12], max_per_cell=2).build(
            ["R0"], val_dir, master_tag="31415")
        yield SimpleNamespace(root=root, store=store, val=val_dir / "R0.npz")


def _args(synth, run_dir, models_root, *extra):
    return ["--dataset", "synth", "--regime", "R0", "--seed", str(SEED), "--store-root", str(synth.root),
            "--val-queries", str(synth.val), "--run-dir", str(run_dir), "--models-root", str(models_root),
            *SMOKE, *extra]


@pytest.fixture(scope="module")
def trained(synth, tmp_path_factory):
    base = tmp_path_factory.mktemp("trained")
    run_dir, models = base / "run", base / "models"
    assert te.main(_args(synth, run_dir, models)) == 0
    model_id = te.model_id_for("synth", "R0", "graphsage", SEED)
    return SimpleNamespace(base=base, run=run_dir, models=models, model_dir=models / model_id, model_id=model_id)


# ----------------------------------------------------------------------------- objective terms
def test_block_infonce_by_hand():
    zq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])  # the second query has no positive
    # logits of query 0: 1, 0, -1; each positive against the single negative
    expected = (softplus(-1 - 1) + softplus(-1 - 0)) / 2
    assert te.block_infonce(zq, keys, mask, temperature=1.0).item() == pytest.approx(expected, rel=1e-6)
    everything = torch.tensor([[True, True, True], [False, False, False]])
    assert te.block_infonce(zq, keys, everything, temperature=1.0).item() == 0.0  # no negative, no term
    assert te.block_infonce(zq, keys, torch.zeros(2, 3, dtype=torch.bool)).item() == 0.0


def test_fine_nce_by_hand():
    zq = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    mask = torch.tensor([[True, False, False, True]])
    # T = 0.5: logits 2, 0, -2, 0; negatives {0, -2}
    neg = math.log(math.exp(0.0) + math.exp(-2.0))
    expected = (softplus(neg - 2.0) + softplus(neg - 0.0)) / 2
    assert te.fine_nce(zq, keys, mask, temperature=0.5).item() == pytest.approx(expected, rel=1e-6)


def test_node_nce_by_hand():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]], requires_grad=True)
    p = torch.tensor([[0.0, 1.0], [5.0, 5.0], [1.0, 0.0], [3.0, 3.0]])
    # ids 5 and 7 are shared; the first row of each id is used on both sides
    loss = te.node_nce(q, np.array([5, 7, 5]), p, np.array([7, 9, 5, 7]), temperature=0.5)
    assert loss.item() == pytest.approx(softplus(-2.0), rel=1e-6)  # logits [[2, 0], [0, 2]]
    loss.backward()
    assert torch.all(q.grad[2] == 0)
    none = te.node_nce(q, np.array([1, 2, 3]), p, np.array([3, 4, 5, 6]))
    assert none.item() == 0.0 and none.requires_grad


def test_aggregate_positive_terms():
    terms = torch.tensor([[1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]])
    mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
    assert te.aggregate_positive_terms(terms, mask, "mean").item() == pytest.approx(2.5)
    assert te.aggregate_positive_terms(terms, mask, "cvar", cvar_fraction=0.5).item() == pytest.approx(3.5)
    assert te.aggregate_positive_terms(terms, mask, "cvar", cvar_fraction=0.25).item() == pytest.approx(4.0)
    smooth = 0.1 * math.log(sum(math.exp(v / 0.1) for v in (1, 2, 3, 4))) - 0.1 * math.log(4)
    assert te.aggregate_positive_terms(terms, mask, "smoothmax", smoothmax_temperature=0.1).item() == pytest.approx(
        smooth, rel=1e-5)
    two = torch.tensor([[True, False, True, False], [False, True, False, False]])
    assert te.aggregate_positive_terms(terms, two, "cvar", cvar_fraction=0.5).item() == pytest.approx((3 + 7) / 2)
    assert te.aggregate_positive_terms(terms, torch.zeros_like(mask), "mean").item() == 0.0
    with pytest.raises(ValueError):
        te.aggregate_positive_terms(terms, mask, "median")


def test_coverage_terms_by_hand():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    mask = torch.tensor([[True, False, True, False]])
    lse = math.log(sum(math.exp(v) for v in (2.0, 1.0, 0.0, -1.0)))
    ce = te.coverage_ce_terms(logits)
    assert ce[0, 0].item() == pytest.approx(lse - 2.0) and ce[0, 2].item() == pytest.approx(lse)
    margin = te.hardest_negative_margin_terms(logits, mask)  # hardest negative logit 1
    assert margin[0, 0].item() == pytest.approx(softplus(-1.0)) and margin[0, 2].item() == pytest.approx(softplus(1.0))
    no_negative = te.hardest_negative_margin_terms(logits, torch.ones_like(mask))
    assert torch.all(no_negative == 0)
    # K = 2 >= 2 positives: the threshold is the largest negative (1)
    barrier, rows = te.topk_barrier_terms(logits, mask, target_topk=2, bucket_size=10, margin=0.0)
    assert torch.equal(rows, mask)
    assert barrier[0, 0].item() == pytest.approx(softplus(-1.0))
    assert barrier[0, 2].item() == pytest.approx(softplus(1.0))
    # K = 1 < 2 positives: k rounds up to the bucket (10), past the last negative (-1)
    barrier, _ = te.topk_barrier_terms(logits, mask, target_topk=1, bucket_size=10, margin=0.5)
    assert barrier[0, 0].item() == pytest.approx(softplus(-1.0 + 0.5 - 2.0))
    assert te.topk_barrier_terms(logits, torch.zeros_like(mask), 2) == (None, None)
    # the whole loss: mean aggregation, margin weight 0.25, barrier weight 0.35 at K = 2
    zq, keys = torch.ones(1, 1), logits[0].unsqueeze(1)  # zq keys^T = logits
    got = te.coverage_loss(zq, keys, mask, temperature=1.0, aggregation="mean", topk=2, topk_weight=0.35)
    first = ((lse - 2.0 + 0.25 * softplus(-1.0)) + (lse + 0.25 * softplus(1.0))) / 2
    assert got.item() == pytest.approx(first + 0.35 * (softplus(-1.0) + softplus(1.0)) / 2, rel=1e-6)


@pytest.mark.parametrize("aggregation", ["mean", "cvar", "smoothmax"])
@pytest.mark.parametrize("barrier", [0.0, 0.35])
def test_coverage_loss_matches_the_legacy_function(aggregation, barrier):
    legacy = pytest.importorskip("scripts.coverage_losses")
    gen = torch.Generator().manual_seed(3)
    zq0 = F.normalize(torch.randn(5, 8, generator=gen), dim=1)
    bank = F.normalize(torch.randn(12, 8, generator=gen), dim=1)
    live0 = F.normalize(torch.randn(3, 8, generator=gen), dim=1)
    live_idx = torch.tensor([1, 4, 7])
    ids = [[1, 2], [], [4, 5, 6, 7, 8], list(range(11)), [0]]  # includes |P| > K and a query without targets
    kwargs = dict(positive_aggregation=aggregation, cvar_fraction=0.5, smoothmax_temperature=0.1)
    zq_a, live_a = zq0.clone().requires_grad_(), live0.clone().requires_grad_()
    ref = legacy.partition_coverage_loss(zq_a, bank, ids, temperature=0.07, target_topk=2, topk_bucket_size=3,
                                         topk_weight=barrier, topk_margin=0.1, live_partition_indices=live_idx,
                                         live_partition_embeddings=live_a, **kwargs)
    ref.backward()
    zq_b, live_b = zq0.clone().requires_grad_(), live0.clone().requires_grad_()
    keys = te.merge_live(bank, live_idx, live_b)
    got = te.coverage_loss(zq_b, keys, te.positive_mask(ids, 12), temperature=0.07, margin_weight=0.25,
                           aggregation=aggregation, cvar_fraction=0.5, smoothmax_temperature=0.1, topk=2,
                           topk_weight=barrier, topk_bucket=3, topk_margin=0.1)
    got.backward()
    assert got.item() == pytest.approx(ref.item(), rel=1e-6, abs=1e-7)
    assert torch.allclose(zq_a.grad, zq_b.grad, atol=1e-6) and torch.allclose(live_a.grad, live_b.grad, atol=1e-6)


def test_combine_objective():
    terms = {"fine_nce": 1.0, "block_infonce": 2.0, "node_nce": 3.0, "coverage": 4.0}
    cfg = {"fine_nce_weight_alpha": 0.2, "node_nce_weight_beta": 0.1, "coverage_weight_gamma": 1.0}
    assert te.combine_objective(terms, cfg) == pytest.approx(0.2 * 1 + 0.7 * 2 + 0.1 * 3 + 1.0 * 4)
    default = te.loss_config("arxiv")
    assert te.combine_objective(terms, default) == pytest.approx(0.2 * 1 + 0.8 * 2 + 0.0 * 3 + 1.0 * 4)


# ----------------------------------------------------------------------------- live set, bank, schedule
def test_select_live_blocks():
    positives = np.array([[1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=bool)
    assert te.select_live_blocks(positives, None, 2).tolist() == [2, 0]  # by count, then id
    scores = np.array([[0.9, 0.3, 0.0, 0.8, 0.1, 0.5], [0.0, 0.2, 0.0, 0.0, 0.3, 0.5]])
    # free slots: the hardest negatives by the best score any query gives them (block 0 is a target)
    assert te.select_live_blocks(positives, scores, 5).tolist() == [2, 0, 3, 5, 1]
    assert te.select_live_blocks(positives, scores, 6).tolist() == [2, 0, 3, 5, 1, 4]
    assert te.select_live_blocks(positives, scores, 0).tolist() == []


def test_merge_live_and_ema():
    bank = torch.zeros(4, 2)
    live = torch.ones(2, 2, requires_grad=True)
    keys = te.merge_live(bank, torch.tensor([1, 3]), live)
    assert torch.all(bank == 0) and keys[1].tolist() == [1.0, 1.0] and keys[0].tolist() == [0.0, 0.0]
    keys.sum().backward()
    assert torch.all(live.grad == 1)
    model, key = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = [p.detach().clone() for p in key.parameters()]
    te.ema_update(key, model, 0.5)
    for k, m, b in zip(key.parameters(), model.parameters(), before):
        assert torch.allclose(k, 0.5 * b + 0.5 * m)
    te.ema_update(key, model, 0.0)
    assert all(torch.equal(k, m) for k, m in zip(key.parameters(), model.parameters()))


def test_lr_schedule():
    kw = dict(lr=1e-3, lr_final=1e-4, warmup=10)
    assert te.lr_at(0, 1000, **kw) == pytest.approx(1e-4)
    assert te.lr_at(9, 1000, **kw) == pytest.approx(1e-3) and te.lr_at(10, 1000, **kw) == pytest.approx(1e-3)
    assert te.lr_at(999, 1000, **kw) == pytest.approx(1e-4)
    assert te.lr_at(10 + 989 // 2, 1000, **kw) == pytest.approx(5.5e-4, rel=1e-2)
    after = [te.lr_at(t, 1000, **kw) for t in range(10, 1000)]
    assert all(a >= b for a, b in zip(after, after[1:]))


# ----------------------------------------------------------------------------- sampler
def test_sampler_is_deterministic(synth):
    a = te.TrainSampler(synth.store, "synth", "R0", SEED)
    b = te.TrainSampler(synth.store, "synth", "R0", SEED)
    first, again = a.batch(1, 0, 8), b.batch(1, 0, 8)
    assert [q.nodeset for q in first] == [q.nodeset for q in again] and len(first) == 8
    assert [q.nodeset for q in a.batch(1, 1, 8)] != [q.nodeset for q in first]
    assert [q.nodeset for q in te.TrainSampler(synth.store, "synth", "R0", SEED + 1).batch(1, 0, 8)] != \
        [q.nodeset for q in first]
    # draw i of (epoch, step) uses its own derived stream
    rng = np.random.default_rng(protocol.derive_seed(SEED, "synth", "R0", "train", 1, 0, 0))
    spec = protocol.TRAINING["sampler"]
    family = list(spec["family_probabilities"])[int(rng.choice(5, p=list(spec["family_probabilities"].values())))]
    base = spec["sizes"][int(rng.integers(3))]
    size = max(5, base + int(rng.integers(-spec["size_jitter"], spec["size_jitter"] + 1)))
    assert first[0].index == 0 and first[0].family == family
    planted = queries.QueryContext(synth.store).sample(family, size, rng)
    assert np.array_equal(first[0].planted, planted.planted)
    item = first[0]
    assert item.targets == queries.greedy_cover(synth.store, item.planted)
    assert queries.covers(synth.store, item.planted, item.targets)
    assert item.nodeset == queries.nodeset_hash(item.planted)
    assert int(item.fine_counts.sum()) == item.planted.shape[0]


def test_sampler_skips_and_counts_failed_draws(synth):
    sampler = te.TrainSampler(synth.store, "synth", "R0", SEED)
    real = sampler.ctx.sample
    calls = {"n": 0}

    def flaky(family, size, rng):
        calls["n"] += 1
        return None if calls["n"] <= 3 else real(family, size, rng)

    sampler.ctx.sample = flaky
    items = sampler.batch(1, 0, 4)
    assert [q.index for q in items] == [3, 4, 5, 6]
    assert sampler.stats["failed"] == 3 and sampler.stats["draws"] == 7
    assert sum(sampler.stats["failed_by_family"].values()) == 3
    sampler.ctx.sample = lambda family, size, rng: None
    assert sampler.batch(1, 1, 2) == [] and sampler.stats["short_batches"] == 1
    assert sampler.stats["draws"] == 7 + 2 * sampler.max_draws_per_query


def test_sampler_never_trains_on_held_out_node_sets(synth):
    first = te.TrainSampler(synth.store, "synth", "R0", SEED).batch(1, 0, 8)
    held_out = {first[0].nodeset, first[3].nodeset}
    sampler = te.TrainSampler(synth.store, "synth", "R0", SEED, held_out=held_out)
    items = sampler.batch(1, 0, 8)
    assert len(items) == 8 and not held_out & {q.nodeset for q in items}
    stats = sampler.stats
    assert stats["held_out"] >= 2 and stats["draws"] == len(items) + stats["held_out"] + stats["failed"]
    by_index = {q.index: q.nodeset for q in first}
    assert all(by_index.get(q.index, q.nodeset) == q.nodeset for q in items)  # a skip shifts no other draw


def test_held_out_node_sets_cover_every_validation_master(synth, tmp_path):
    val_set = te.load_validation_queries(synth.val, "synth", "R0")
    own = queries.read_exclusions([synth.val.parent])
    assert {queries.nodeset_hash(np.asarray(q.planted, dtype=np.int64)) for q in val_set} <= own
    assert te.held_out_nodesets(synth.val, val_set) == own
    other = tmp_path / "27182"
    other.mkdir()
    (other / "nodesets.txt").write_text("a1\nb2\n", encoding="ascii")
    assert te.held_out_nodesets(synth.val, val_set, [other, synth.val.parent]) == own | {"a1", "b2"}
    with pytest.raises(FileNotFoundError, match="held-out"):
        te.held_out_nodesets(synth.val, val_set, [other, tmp_path / "16180"])


def test_trainer_step_with_every_term(synth):
    plan = te.resolve_plan("synth", "R0", SEED, smoke={"layers": 2, "hidden": 16, "output": 8})
    plan["loss"]["node_nce_weight_beta"] = 0.1
    trainer = te.Trainer(plan, synth.store, device="cpu")
    before = [p.detach().clone() for p in trainer.model.parameters()]
    record = trainer.train_step(1, 0)
    assert not record["skipped"] and record["queries"] == 8 and record["live_blocks"] == 4
    assert all(math.isfinite(v) and v > 0 for v in record["terms"].values())
    assert record["total"] == pytest.approx(te.combine_objective(record["terms"], plan["loss"]), rel=1e-5)
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    assert trainer.bank is not None and tuple(trainer.bank.shape) == (6, 8) and trainer.global_step == 1


def test_rgcn_on_class_labels_trains_and_exports(synth, tmp_path):
    plan = te.resolve_plan("synth", "R1", SEED, arch="rgcn", smoke={"layers": 2, "hidden": 16, "output": 8})
    trainer = te.Trainer(plan, synth.store, device="cpu")
    assert trainer.model.config["num_relations"] == int(trainer.inputs.rel_table.max()) + 1
    record = trainer.train_step(1, 0)
    assert not record["skipped"] and all(math.isfinite(v) for v in record["terms"].values())
    selection = {"epoch": 1, "metric": 0.0, "checkpoint": "ckpt_epoch001.pt", "fullcov_at": {}}
    export = te.export_model(trainer.model, synth.store, plan, tmp_path / plan["model_id"], selection=selection)
    assert export["block_emb"]["shape"] == [6, 8] and export["regime"] == "R1" and export["arch"] == "rgcn"
    if export["numpy_export"] != "pending":
        assert export["numpy_parity"]["max_abs_diff"] < 1e-4


# ----------------------------------------------------------------------------- a full run
def test_run_writes_checkpoints_metrics_selection_and_export(synth, trained):
    run, model_dir = trained.run, trained.model_dir
    assert [p.name for _, p in te.list_checkpoints(run)] == ["ckpt_epoch001.pt", "ckpt_epoch002.pt"]
    records = te.read_jsonl(run / te.VAL_METRICS)
    assert [r["epoch"] for r in records] == [1, 2]
    for r in records:
        assert r["ladder"] == [1, 2, 3] and 0.0 <= r["metric"] <= 1.0
        assert r["metric"] == pytest.approx(np.mean(list(r["fullcov_at"].values())))
        assert r["checkpoint_sha256"] == manifest.sha256_file(run / r["checkpoint"])["sha256"]
    ckpt = te.read_checkpoint(run / "ckpt_epoch001.pt")
    assert {"config", "state_dict", "momentum_state_dict", "optimizer", "epoch", "metric", "protocol_hash",
            "train_args"} <= set(ckpt)
    assert ckpt["epoch"] == 1 and ckpt["protocol_hash"] == protocol.protocol_hash()
    assert ckpt["config"]["output"] == 8 and ckpt["config"]["layers"] == 2 and ckpt["train_args"]["arm"] == "main"
    selection = json.loads((run / te.SELECTION).read_text(encoding="utf-8"))
    assert selection["epoch"] == te.select_checkpoint(records)["epoch"]
    block_emb = np.load(model_dir / "block_emb.npy")
    fine_emb = np.load(model_dir / "fine_emb.npy")
    assert block_emb.shape == (6, 8) and block_emb.dtype == np.float32 and fine_emb.shape == (18, 8)
    export = json.loads((model_dir / te.EXPORT).read_text(encoding="utf-8"))
    assert export == json.loads((run / te.EXPORT).read_text(encoding="utf-8"))
    assert export["model_id"] == trained.model_id == "synth-R0-graphsage-seed7301-main"
    assert (export["dataset"], export["regime"], export["arch"], export["seed"], export["arm"]) == \
        ("synth", "R0", "graphsage", SEED, "main")
    assert export["selected_epoch"] == selection["epoch"] and export["metric"] == selection["metric"]
    assert export["protocol_hash"] == protocol.protocol_hash() and export["partition"] == "primary"
    assert export["store_meta_sha256"] == manifest.sha256_file(synth.store.dir / "meta.json")["sha256"]
    assert export["block_emb"]["sha256"] == manifest.sha256_file(model_dir / "block_emb.npy")["sha256"]
    assert export["fine_emb"]["sha256"] == manifest.sha256_file(model_dir / "fine_emb.npy")["sha256"]
    assert (model_dir / "torch_state.pt").exists()
    if export["numpy_export"] != "pending":
        assert export["numpy_export"]["status"] == "written" and (model_dir / "weights.npz").exists()
        assert export["numpy_parity"]["max_abs_diff"] < 1e-4
    run_manifest = json.loads((run / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert run_manifest["status"] == "complete" and run_manifest["split"] == "validation"
    assert run_manifest["config"]["smoke"]["epochs"] == 2 and "ckpt_epoch002.pt" in run_manifest["outputs"]
    assert run_manifest["config"]["cuda_cache_cap"] is None  # a CPU run sets no GPU cap


def test_cuda_cache_is_held_to_the_card(monkeypatch):
    """On the GPU the allocator may not reserve past the card's memory; off it nothing is set."""
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: calls.append((f, d)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=24 << 30))
    assert te.cap_cuda_cache(torch.device("cpu")) is None and calls == []
    assert te.cap_cuda_cache(torch.device("cuda")) == {"memory_fraction": 1.0, "device_total_bytes": 24 << 30}
    assert te.cap_cuda_cache(torch.device("cuda", 1))["memory_fraction"] == 1.0
    assert calls == [(1.0, 0), (1.0, 1)]


def test_train_nodesets_log_every_trained_query(synth, trained):
    logged = (trained.run / te.TRAIN_NODESETS).read_text(encoding="ascii").split()
    sampler = te.TrainSampler(synth.store, "synth", "R0", SEED)
    expected = [q.nodeset for epoch in (1, 2) for step in range(3) for q in sampler.batch(epoch, step, 8)]
    assert logged == expected and len(logged) == 48
    summaries = te.read_jsonl(trained.run / te.TRAIN_LOG)
    assert sum(s["queries"] for s in summaries) == 48 and [s["epoch"] for s in summaries] == [1, 2]


def test_epoch_log_records_the_machine_load(trained, monkeypatch):
    """Every epoch line carries its end, this process's CPU seconds and the machine's mean busy CPUs."""
    summaries = te.read_jsonl(trained.run / te.TRAIN_LOG)
    assert summaries[0]["utc"] <= summaries[1]["utc"]
    for s in summaries:  # a smoke epoch is too short to bound the busy CPUs by the CPU count (timer ticks)
        assert s["seconds"] > 0 and s["cpu_seconds"] >= 0 and s["host_busy_cpus"] >= 0
    for r in summaries + te.read_jsonl(trained.run / te.VAL_METRICS):  # a CPU run has no GPU peaks
        assert r["host_peak_bytes"] > 0 and r["gpu_peak_allocated_bytes"] is None
        assert r["gpu_peak_reserved_bytes"] is None
    busy = te.host_busy_seconds()
    assert busy is not None and te.host_busy_seconds() >= busy
    # without psutil the load is unknown: None, never a guess
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert te.host_busy_seconds() is None
    trainer = SimpleNamespace(global_step=3, sampler=SimpleNamespace(stats={"failed": 0}), bank_refreshes=0)
    summary = te._epoch_summary(1, [], trainer, 2.0, 0)
    assert summary["cpu_seconds"] is None and summary["host_busy_cpus"] is None
    assert te._epoch_summary(1, [], trainer, 2.0, 0, cpu_seconds=1.5, host_busy=6.0)["host_busy_cpus"] == 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_run_records_the_gpu_peaks(synth, tmp_path):
    """The GPU path of the smoke run: each epoch and validation records the GPU peaks since the last one."""
    args = [a if a != "cpu" else "cuda" for a in _args(synth, tmp_path / "run", tmp_path / "models")]
    assert te.main(args) == 0
    lines = te.read_jsonl(tmp_path / "run" / te.TRAIN_LOG) + te.read_jsonl(tmp_path / "run" / te.VAL_METRICS)
    assert len(lines) == 4
    for r in lines:
        assert 0 < r["gpu_peak_allocated_bytes"] <= r["gpu_peak_reserved_bytes"] and r["host_peak_bytes"] > 0
    cap = json.loads((tmp_path / "run" / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))["config"]["cuda_cache_cap"]
    assert cap["memory_fraction"] == 1.0
    assert max(r["gpu_peak_reserved_bytes"] for r in lines) <= cap["device_total_bytes"]


def test_run_records_and_audits_the_held_out_node_sets(synth, trained, tmp_path):
    own = queries.read_exclusions([synth.val.parent])
    ckpt = te.read_checkpoint(trained.run / "ckpt_epoch001.pt")
    assert ckpt["train_args"]["held_out"] == {
        "nodesets": len(own), "sha256": hashlib.sha256("\n".join(sorted(own)).encode("ascii")).hexdigest()}
    assert ckpt["train_args"]["paths"]["held_out"] == []
    assert not own & set((trained.run / te.TRAIN_NODESETS).read_text(encoding="ascii").split())
    export = json.loads((trained.model_dir / te.EXPORT).read_text(encoding="utf-8"))
    assert export["held_out_nodesets"] == len(own) and export["train_val_nodeset_overlap"] == 0
    # a resume refuses another held-out set; a moved directory with the same node sets is not a change
    run = tmp_path / "run"
    shutil.copytree(trained.run, run)
    (run / "ckpt_epoch002.pt").unlink()
    other = tmp_path / "27182"
    other.mkdir()
    (other / "nodesets.txt").write_text("a1\n", encoding="ascii")
    with pytest.raises(ValueError, match="held_out"):
        te.main(_args(synth, run, tmp_path / "models", "--resume", "--held-out", str(other)))
    moved = tmp_path / "moved"
    shutil.copytree(synth.val.parent, moved)
    assert te.main(_args(synth, run, tmp_path / "models", "--resume", "--held-out", str(moved))) == 0


def test_selected_checkpoint_loads_strictly_and_reproduces_the_export(synth, trained, tmp_path):
    selection = json.loads((trained.run / te.SELECTION).read_text(encoding="utf-8"))
    path = trained.run / selection["checkpoint"]
    model, ckpt = te.load_checkpoint(path, expect_config=json.loads(
        (trained.model_dir / te.EXPORT).read_text(encoding="utf-8"))["config"])
    emb = encoders.encode_blocks(model, synth.store, "R0", batch_size=32)
    assert np.array_equal(emb.astype(np.float32), np.load(trained.model_dir / "block_emb.npy"))
    with pytest.raises(ValueError):
        te.load_checkpoint(path, expect_config={**ckpt["config"], "output": 16})
    state = dict(ckpt["state_dict"])
    state.pop(next(iter(state)))
    te.save_checkpoint(tmp_path / "missing_key.pt", {**ckpt, "state_dict": state})
    with pytest.raises(RuntimeError):
        te.load_checkpoint(tmp_path / "missing_key.pt")
    with pytest.raises(FileExistsError):
        te.save_checkpoint(path, {"epoch": 0})


def test_existing_checkpoints_need_resume(synth, trained):
    with pytest.raises(FileExistsError):
        te.main(_args(synth, trained.run, trained.base / "other_models"))


def test_resume_continues_the_run_exactly(synth, trained, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(trained.run, run)
    (run / "ckpt_epoch002.pt").unlink()
    (run / te.SELECTION).unlink()
    lines = (run / te.VAL_METRICS).read_text(encoding="utf-8").splitlines()
    (run / te.VAL_METRICS).write_text(lines[0] + "\n", encoding="utf-8")
    assert te.main(_args(synth, run, tmp_path / "models", "--resume")) == 0
    original = te.read_checkpoint(trained.run / "ckpt_epoch002.pt")
    resumed = te.read_checkpoint(run / "ckpt_epoch002.pt")
    assert resumed["global_step"] == original["global_step"] == 6
    for key, value in original["state_dict"].items():
        assert torch.allclose(value, resumed["state_dict"][key], atol=1e-6), key
    for key, value in original["momentum_state_dict"].items():
        assert torch.allclose(value, resumed["momentum_state_dict"][key], atol=1e-6), key
    assert (run / te.TRAIN_NODESETS).read_text(encoding="ascii") == \
        (trained.run / te.TRAIN_NODESETS).read_text(encoding="ascii")
    assert [r["epoch"] for r in te.read_jsonl(run / te.VAL_METRICS)] == [1, 2]
    assert [r["epoch"] for r in te.read_jsonl(run / te.TRAIN_LOG)] == [1, 2]
    new_sel = json.loads((run / te.SELECTION).read_text(encoding="utf-8"))
    old_sel = json.loads((trained.run / te.SELECTION).read_text(encoding="utf-8"))
    assert new_sel["epoch"] == old_sel["epoch"]
    assert (run / "manifest.previous-1.json").exists()
    run_manifest = json.loads((run / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert run_manifest["status"] == "complete" and run_manifest["config"]["resumed_from"]["epoch"] == 1


def test_validation_set_must_match_the_run(synth, tmp_path):
    assert len(te.load_validation_queries(synth.val, "synth", "R0")) > 0
    assert len(te.load_validation_queries(synth.val.parent, "synth", "R0")) > 0
    with pytest.raises(ValueError):
        te.load_validation_queries(synth.val, "corafull", "R0")
    with pytest.raises((ValueError, FileNotFoundError)):
        te.load_validation_queries(synth.val.parent, "synth", "R1")
    assert te.is_smoke_query_set({"max_per_cell": None, "sizes": [8, 12]})
    assert not te.is_smoke_query_set({"max_per_cell": None, "sizes": list(protocol.QUERY_SIZES)})
    with pytest.raises(ValueError, match="smoke query set"):  # a canonical run refuses a capped validation set
        te.run(te.resolve_plan("synth", "R0", SEED), store_root=synth.root, val_queries=synth.val,
               run_dir=tmp_path / "run", models_root=tmp_path / "models", device="cpu")
    assert not (tmp_path / "run").exists()


def test_numpy_export_is_pending_without_the_module(monkeypatch, tmp_path):
    import scripts.canonical as package

    monkeypatch.delattr(package, "encoder_numpy", raising=False)
    monkeypatch.setitem(sys.modules, "scripts.canonical.encoder_numpy", None)
    model = encoders.GraphEncoder("graphsage", 4, hidden=8, output=4, layers=2)
    assert te._numpy_export(model, tmp_path, {}) == "pending"
    assert not (tmp_path / "weights.npz").exists()


# ----------------------------------------------------------------------------- arms, plans and the matrix
def test_arms_and_plans():
    assert len(te.arms("arxiv")) == 1 + 8 + 1 + 8 and len(te.arms("mag")) == 1 + 6 + 3 and len(te.arms("corafull")) == 1
    plan = te.resolve_plan("arxiv", "R2-32", SEED)
    assert plan["regime"] == "R2-512" and plan["model_id"] == "arxiv-R2-512-graphsage-seed7301-main"
    assert plan["loss"]["topk_barrier_K"] == 50 and plan["loss"]["live_positives"] == 64
    assert (plan["loss"]["temperature"], plan["loss"]["coverage_temperature"]) == (0.1, 0.05)
    assert (plan["schedule"]["optimizer"], plan["schedule"]["weight_decay"], plan["schedule"]["grad_clip_norm"]) \
        == ("adamw", 0.0, 1.0)
    from scripts.canonical import serving  # serving restates model_id_for so it never imports torch
    for dataset in protocol.DATASETS:
        arch = protocol.DATASETS[dataset]["encoder"]
        for regime in te.trained_regimes_for(dataset):
            assert serving.model_id_for(dataset, regime, arch, SEED) == te.model_id_for(dataset, regime, arch, SEED)
            assert serving.default_model_id(dataset, regime, SEED) == te.model_id_for(dataset, regime, arch, SEED)
    grid = te.resolve_plan("arxiv", "R0", SEED, arm="grid-mean-barrier0-live0")
    assert (grid["loss"]["coverage_aggregation"], grid["loss"]["topk_barrier_weight"], grid["loss"]["live_positives"]) \
        == ("mean", 0.0, 0)
    assert grid["overrides"]["loss"] == {"coverage_aggregation": "mean", "topk_barrier_weight": 0.0,
                                         "live_positives": 0}
    assert te.resolve_plan("mag", "R0", SEED, arm="no_momentum")["loss"]["negative_encoder_momentum"] == 0.0
    arch = te.resolve_plan("arxiv", "R0", SEED, arm="arch-gat-L6-res")
    assert arch["arch"] == "gat" and arch["model"]["layers"] == 6 and arch["model_id"].endswith("-arch-gat-L6-res")
    nores = te.resolve_plan("mag", "R0", SEED, arm="arch-rgcn-L6-nores")
    assert nores["arch"] == "rgcn" and nores["model"]["residual"] is False
    assert plan["init_seed"] == te.resolve_plan("arxiv", "R2-512", SEED, arm="infonce_only")["init_seed"]
    with pytest.raises(ValueError):
        te.resolve_plan("arxiv", "R0", SEED, arm="no_such_arm")
    with pytest.raises(ValueError):
        te.resolve_plan("arxiv", "R0", SEED, arch="graphsage", arm="arch-gat-L6-res")
    with pytest.raises(ValueError):
        te.resolve_plan("papers100m", "R1", SEED)


def test_train_matrix_counts_and_commands(monkeypatch, tmp_path):
    monkeypatch.delitem(protocol.DATASETS, "synth", raising=False)  # the module fixture adds it
    seeds = len(protocol.SEEDS["training"])
    rows = tm.enumerate_matrix(store_root="S", queries_root="Q", run_root="R", models_root="M")
    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in tm.KINDS}
    main_rows = seeds * sum(len(te.trained_regimes_for(d)) for d in protocol.DATASETS)
    grid = math.prod(len(v) for v in protocol.FACTORIAL["arxiv"]["factors"].values())
    factorial = seeds * (grid + len(protocol.FACTORIAL["arxiv"]["extra_arms"]) + len(protocol.FACTORIAL["mag"]["arms"]))
    architecture = seeds * (len(protocol.ARCHITECTURE_ABLATION["arxiv"]) + len(protocol.ARCHITECTURE_ABLATION["mag"]))
    assert kinds == {"main": main_rows, "factorial": factorial, "architecture": architecture}
    assert (main_rows, factorial, architecture) == (39, 45, 33)
    assert len({r["model_id"] for r in rows}) == len(rows)
    duplicates = {r["model_id"]: r["duplicate_of"] for r in rows if r["duplicate_of"]}
    assert len(duplicates) == 4 * seeds
    assert duplicates["arxiv-R0-graphsage-seed7301-grid-cvar-barrier0.35-live64"] == "arxiv-R0-graphsage-seed7301-main"
    assert duplicates["mag-R0-rgcn-seed7302-full"] == "mag-R0-rgcn-seed7302-main"
    for r in rows:
        assert r["est_gpu_hours"] > 0 and "estimate" in r["est_basis"]
        argv = shlex.split(r["command"])
        assert argv[argv.index("--arm") + 1] == r["arm"] and argv[argv.index("--run-dir") + 1] == r["run_dir"]
        assert r["run_dir"].endswith(r["model_id"]) and r["val_queries"].endswith(f"/31415/{r['regime']}.npz")
        held = argv[argv.index("--held-out") + 1:argv.index("--device")]
        assert held == r["held_out"] == [f"Q/{r['dataset']}/validation/{m}" for m in protocol.SEEDS["validation"]]
        assert len(held) == 3 and r["val_queries"].rsplit("/", 1)[0] in held
    arxiv = tm.enumerate_matrix(store_root="S", queries_root="Q", run_root="R", models_root="M", filters=["arxiv/*"])
    assert {r["dataset"] for r in arxiv} == {"arxiv"} and len(arxiv) == seeds * (3 + 9 + 8)
    mag_no = tm.enumerate_matrix(store_root="S", queries_root="Q", run_root="R", models_root="M", filters=["mag/no_*"])
    assert sorted({r["arm"] for r in mag_no}) == ["no_barrier", "no_cvar", "no_live", "no_momentum"]
    out = tmp_path / "matrix.json"
    assert tm.main(["--store-root", str(tmp_path / "store"), "--queries-root", "Q", "--run-root",
                    str(tmp_path / "runs"), "--models-root", str(tmp_path / "models"), "--filter", "corafull/*",
                    "--out", str(out)]) == 0
    listed = json.loads(out.read_text(encoding="utf-8"))
    assert len(listed) == 3 * seeds and {r["dataset"] for r in listed} == {"corafull"}
    assert not (tmp_path / "runs").exists() and not (tmp_path / "models").exists()  # a dry run by default
    assert {"model_id", "command", "run_dir", "est_gpu_hours"} <= set(listed[0])


def test_train_matrix_execute_skips_only_complete_rows(monkeypatch, tmp_path):
    def row(name):
        return {"model_id": name, "duplicate_of": None, "model_dir": str(tmp_path / "models" / name),
                "run_dir": str(tmp_path / "runs" / name), "argv": ["trainer", name]}

    done, half, fresh = row("done"), row("half"), row("fresh")
    for entry, status in ((done, "complete"), (half, "interrupted")):
        model_dir, run_dir = tmp_path / "models" / entry["model_id"], tmp_path / "runs" / entry["model_id"]
        model_dir.mkdir(parents=True)
        run_dir.mkdir(parents=True)
        (model_dir / te.EXPORT).write_text("{}", encoding="utf-8")
        (run_dir / manifest.MANIFEST_NAME).write_text(json.dumps({"status": status}), encoding="utf-8")
    (tmp_path / "runs" / "done" / te.EXPORT).write_text("{}", encoding="utf-8")
    te.checkpoint_path(tmp_path / "runs" / "half", 90).write_bytes(b"")  # stopped after the model export
    calls = []
    monkeypatch.setattr(tm.subprocess, "run", lambda argv, check=False: calls.append(argv) or SimpleNamespace(returncode=0))
    assert tm.execute([done, half, fresh]) == 0
    assert calls == [["trainer", "half", "--resume"], ["trainer", "fresh"]]
    assert tm.row_complete(done) and not tm.row_complete(half) and not tm.row_complete(fresh)
    (tmp_path / "runs" / "half" / te.EXPORT).write_text("{}", encoding="utf-8")
    assert not tm.row_complete(half)  # both exports, but the manifest never closed


def _jobs_rows(tmp_path, names):
    return [{"model_id": name, "kind": kind, "duplicate_of": None, "model_dir": str(tmp_path / "models" / name),
             "run_dir": str(tmp_path / "runs" / name), "argv": ["trainer", name]} for name, kind in names]


def _fake_popen(starts, failing=()):
    live = set()

    class Proc:  # running for one poll, then exits (non-zero when failing)
        def __init__(self, argv, stdout, stderr, env):
            self.name, self.polls = argv[1], 0
            starts.append((self.name, frozenset(live), None if env is None else env["OMP_NUM_THREADS"],
                           None if env is None else {env[v] for v in tm.THREAD_VARS}))
            live.add(self.name)

        def poll(self):
            self.polls += 1
            if self.polls < 2:
                return None
            live.discard(self.name)
            return 3 if self.name in failing else 0

    return Proc


def test_train_matrix_jobs_share_the_gpu_and_keep_main_rows_alone(monkeypatch, tmp_path):
    names = [("m1", "main"), ("f1", "factorial"), ("f2", "factorial"), ("f3", "factorial"), ("m2", "main"),
             ("a1", "architecture")]
    starts = []
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(tm.subprocess, "Popen", _fake_popen(starts, failing={"f2"}))
    assert tm.execute(_jobs_rows(tmp_path, names), jobs=3, keep_going=True, poll_seconds=0) == 1
    # a main row starts on an idle machine with the job's threads, and nothing starts beside it;
    # ablation rows share: three at once, 8 // 3 = 2 threads each
    assert starts == [("m1", frozenset(), None, None), ("f1", frozenset(), "2", {"2"}),
                      ("f2", frozenset({"f1"}), "2", {"2"}), ("f3", frozenset({"f1", "f2"}), "2", {"2"}),
                      ("m2", frozenset(), None, None), ("a1", frozenset(), "2", {"2"})]
    assert all((tmp_path / "runs" / f"{name}.log").is_file() for name, _ in names)


def test_train_matrix_jobs_stop_after_a_failure(monkeypatch, tmp_path, capsys):
    starts = []
    monkeypatch.setattr(tm.subprocess, "Popen", _fake_popen(starts, failing={"f1"}))
    rows = _jobs_rows(tmp_path, [("f1", "factorial"), ("f2", "factorial"), ("f3", "factorial"), ("f4", "factorial")])
    assert tm.execute(rows, jobs=2, poll_seconds=0) == 1
    assert [s[0] for s in starts] == ["f1", "f2"]  # f2 finishes; f3 and f4 never start
    assert "not started: 2 rows" in capsys.readouterr().err
