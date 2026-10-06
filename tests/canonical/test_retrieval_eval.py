"""Retrieval evaluation (E4, E5, E6) and the router-threshold fit."""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.canonical import cascade, encoder_numpy, features, fit_router, manifest, protocol, queries, rankers
from scripts.canonical import retrieval_eval as rev
from scripts.canonical import run_experiment as rx
from scripts.canonical import seal, serving, sigma
from scripts.canonical import store as store_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [int(s) for s in protocol.SEEDS["training"]]
DEV = str(protocol.SEEDS["dev"][0])
SEALED = "sealed-for-tests"
TRAINED = ("R0", "R1", "R2-512")  # the regimes with their own encoder; the dev and test sets cover them
DOCUMENTED = set(re.findall(r"^  ([a-z_]+)\s{2,}", rev.__doc__, flags=re.MULTILINE))  # docstring record fields


def save_model(gstore, models: Path, model_id: str, trained: str, *, export: bool = False, **settings) -> Path:
    """A random numpy encoder of the store's input width (weights differ per model id)."""
    in_dim = features.input_dim(trained, int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    config = {"arch": "graphsage", "in_dim": in_dim, "hidden": 16, "output": 8, "layers": 2, "residual": True,
              "layer_norm": True, "dropout": 0.0, **settings}
    rng = np.random.default_rng(list(model_id.encode("utf-8")))
    state = {k: (rng.normal(size=shape) / np.sqrt(shape[-1])).astype(np.float32)
             for k, shape in encoder_numpy.state_shapes(config).items()}
    encoder_numpy.save_numpy_model(models / model_id, config, state,
                                   extra={"model_id": model_id, "dataset": "synth", "regime": trained,
                                          "selected_epoch": 1})
    if export:
        (models / model_id / rev.EXPORT).write_text(json.dumps({"model_id": model_id, "smoke": {}}), encoding="utf-8")
    return models / model_id


@pytest.fixture(scope="module")
def env(clustered_store, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6, ladder=[1, 2, 3]))
    # the 6 blocks form a complete block graph: with the protocol's 20 seeds and model weight 0.5 hybrid is
    # topk there; 2 seeds and weight 0.2 reorder most rankings, so the greedy growth is exercised
    mp.setitem(protocol.SELECTION["hybrid"], "seed_count", 2)
    mp.setitem(protocol.SELECTION["hybrid"], "model_weight", 0.2)
    try:
        root, _ = clustered_store
        copy = tmp_path_factory.mktemp("retrieval")
        shutil.copytree(root / "synth", copy / "synth")
        sigma.build_store_regimes(copy / "synth", ["R2-128"])
        gstore = store_mod.GraphStore(copy, "synth", mmap=True)
        regimes = protocol.regimes_for("synth")
        assert regimes == ["R0", "R1", "R2-512", "R2-128", "R2-32"]
        rankers.build_artifacts(gstore, regimes)
        models = copy / "models"
        for trained in TRAINED:
            for seed in SEEDS:
                save_model(gstore, models, serving.model_id_for("synth", trained, "graphsage", seed), trained)
        for regime in regimes:
            for seed in SEEDS:
                serving.build_block_embeddings(gstore, regime, models / serving.default_model_id("synth", regime, seed))
        qroot = copy / "queries"
        dirs = {"dev": qroot / "synth" / "dev" / DEV, "test": qroot / "synth" / "test" / SEALED,
                "validation": qroot / "synth" / "validation" / fit_router.router_master()}
        queries.SplitBuilder(gstore, "synth", "dev", DEV, sizes=[8, 12], max_per_cell=2).build(
            list(TRAINED), dirs["dev"], master_tag=DEV)
        queries.SplitBuilder(gstore, "synth", "validation", fit_router.router_master(), sizes=[8, 12],
                             max_per_cell=2).build(regimes, dirs["validation"], master_tag=fit_router.router_master())
        queries.SplitBuilder(gstore, "synth", "test", SEALED, sizes=[8], max_per_cell=1).build(
            list(TRAINED), dirs["test"], master_tag=SEALED)
        sets = {(split, r): queries.QuerySet.load(d / f"{r}.npz") for split, d in dirs.items()
                for r in (regimes if split == "validation" else TRAINED)}
        yield SimpleNamespace(store=gstore, copy=copy, models=models, qroot=qroot, dirs=dirs, sets=sets,
                              regimes=regimes)
    finally:
        mp.undo()


def positives(env, regime: str, split: str = "dev") -> list[queries.Query]:
    return [q for q in env.sets[(split, regime)].queries if q.kind == "positive"]


def main_model(env, regime: str, seed: int = SEEDS[0], router_tau: float | None = None) -> rev.LoadedModel:
    return rev.load_model(env.store, regime, env.models, serving.default_model_id("synth", regime, seed),
                          router_tau=router_tau)


def brute_cost(gstore, regime: str, order: np.ndarray, q: queries.Query) -> int | None:
    """Smallest K whose candidate (blocks, halo, sigma filter) holds every planted node, by building it."""
    codes = gstore.sigma(regime).query_codes(q.payload, gstore)
    for k in range(1, order.shape[0] + 1):
        if np.isin(q.planted, gstore.candidate(order[:k], codes, regime).global_ids).all():
            return k
    return None


def brute_home_cost(gstore, order: np.ndarray, q: queries.Query) -> int | None:
    home = set(np.asarray(gstore.part.node_block[q.planted]).tolist())
    for k in range(1, order.shape[0] + 1):
        if home <= set(order[:k].tolist()):
            return k
    return None


def median_tau(env, regime: str, qs: list[queries.Query]) -> float:
    bank = rankers.RankerBank(env.store, regime)
    return float(np.median([bank.selectivity(env.store.sigma(regime).query_codes(q.payload, env.store)) for q in qs]))


def run_argv(env, out: Path, *extra: str, split: str = "dev") -> list[str]:
    return ["run", "--store-root", str(env.copy), "--dataset", "synth", "--queries", str(env.dirs[split]),
            "--models-root", str(env.models), "--out", str(out), *extra]


def with_flag(argv: list[str], flag: str, *values: str) -> list[str]:
    """argv with `flag` set to `values` (replaced when present)."""
    out = list(argv)
    if flag in out:
        i = out.index(flag) + 1
        while i < len(out) and not out[i].startswith("--"):
            del out[i]
        out[i:i] = list(values)
        return out
    return [*out, flag, *values]


# ----------------------------------------------------------------------------
# The measure
# ----------------------------------------------------------------------------
def test_costs_match_a_brute_force_prefix_search(env):
    gstore = env.store
    budgets = rev.ladder_budgets("synth", gstore.num_blocks)
    assert budgets == [1, 2, 3, 6]
    hybrid_grew, costs = False, set()
    for regime in TRAINED:
        model, base = main_model(env, regime), rankers.RankerBank(gstore, regime)
        for q in positives(env, regime)[::3]:  # every family, sm_* too
            for ranker in ("index", "random", "topo_feature", "learned"):
                learned = ranker in rankers.NEEDS_EMBEDDING
                kwargs = {"bank": model.bank if learned else base, "ranker": ranker, "embed": model.embed if learned
                          else None, "rel_table": model.rel_table if learned else None}
                orders = {}
                for policy in ("hybrid", "topk"):
                    rec = rev.query_retrieval(gstore, regime, q, policy=policy, budgets=budgets, **kwargs)
                    r = rev.retrieve(gstore, regime, q, policy=policy, **kwargs)
                    order = r.selector.prefix(gstore.num_blocks)
                    assert sorted(order.tolist()) == list(range(gstore.num_blocks))
                    fresh = rankers.make_selector(policy, r.ranking, gstore.part)
                    for k in (3, 1, 6, 2, 5, 4):  # prefix-consistent in any call order
                        assert np.array_equal(fresh.prefix(k), order[:k])
                    assert "error" not in rec and (rec["num_blocks"], rec["planted"]) == (6, q.n)
                    assert rec["cost"] == brute_cost(gstore, regime, order, q)
                    assert rec["home_cost"] == brute_home_cost(gstore, order, q)
                    assert rec["cost"] <= rec["home_cost"]
                    flags = [rec["fullcov_at"][str(k)] for k in budgets]
                    assert flags == [rec["cost"] <= k for k in budgets] and flags == sorted(flags)
                    assert rec["route"] is None and rec["tau"] is None
                    assert set(rec["seconds"]) == {"inputs", "embed", "rank", "select"}
                    orders[policy] = order
                    costs |= {(rec["cost"], rec["home_cost"])}
                hybrid_grew |= not np.array_equal(orders["hybrid"], orders["topk"])
    assert hybrid_grew and len({c for c, _ in costs}) >= 4 and any(c < h for c, h in costs)


def test_rankings_equal_the_cascade_bit_for_bit(env):
    gstore = env.store
    rgcn = serving.model_id_for("synth", "R0", "rgcn", SEEDS[1])  # MAG's architecture: relations enter the inputs
    save_model(gstore, env.models, rgcn, "R0", arch="rgcn",
               num_relations=int(features.store_relation_table(gstore).max()) + 1)
    serving.build_block_embeddings(gstore, "R0", env.models / rgcn)
    methods = {"jigsaw_learned": "learned", "jigsaw_fused": "fused_learned_index", "mean_rrf": "fused_learned_mean",
               "jigsaw": "router"}
    for regime, model_id in (("R0", serving.default_model_id("synth", "R0", SEEDS[1])), ("R0", rgcn),
                             ("R2-128", serving.default_model_id("synth", "R2-128", SEEDS[1]))):  # an R2-512 encoder
        qs = positives(env, regime, "validation")
        tau = median_tau(env, regime, qs)
        model = rev.load_model(gstore, regime, env.models, model_id, router_tau=tau)
        assert (model.rel_table is not None) == (model_id == rgcn)
        routes = set()
        for method, ranker in methods.items():
            assert protocol.METHODS[method]["ranker"] == ranker
            c = serving.make_cascade(gstore, regime, method, None, model_dir=env.models / model_id, router_tau=tau)
            assert rev.ladder_budgets("synth", gstore.num_blocks) == c.budgets
            for q in qs:
                expected, info = c.bank.rank(ranker, c.query_inputs(q, c.table.query_codes(q.payload, c.store)))
                r = rev.retrieve(gstore, regime, q, bank=model.bank, ranker=ranker, policy=c.spec["policy"],
                                 embed=model.embed, rel_table=model.rel_table, budgets=c.budgets)
                assert r.ranking.dtype == expected.dtype and np.array_equal(r.ranking, expected)
                assert r.info == info
                theirs = rankers.make_selector(c.spec["policy"], expected, gstore.part, **c.policy_params)
                assert np.array_equal(r.selector.prefix(6), theirs.prefix(6))
                routes.add(info.get("route"))
        assert {"index", "fused_learned_index"} <= routes


def test_router_cost_is_the_cost_of_its_route(env):
    gstore, regime = env.store, "R0"
    qs = positives(env, regime)
    budgets = rev.ladder_budgets("synth", gstore.num_blocks)
    tau = median_tau(env, regime, qs)
    model, base = main_model(env, regime, router_tau=tau), rankers.RankerBank(gstore, regime)
    learned = {"bank": model.bank, "embed": model.embed, "rel_table": model.rel_table, "policy": "hybrid"}
    routes = set()
    for q in qs:
        rec = rev.query_retrieval(gstore, regime, q, ranker="router", budgets=budgets, **learned)
        assert rec["tau"] == tau and (rec["selectivity"] <= tau) == (rec["route"] == "index")
        routes.add(rec["route"])
        if rec["route"] == "fused_learned_index":
            fused = rev.query_retrieval(gstore, regime, q, ranker="fused_learned_index", budgets=budgets, **learned)
            assert (rec["cost"], rec["home_cost"], rec["fullcov_at"]) == (fused["cost"], fused["home_cost"],
                                                                           fused["fullcov_at"])
        else:  # index with ties broken by the learned score
            r = rev.retrieve(gstore, regime, q, ranker="router", **learned)
            index, score = model.bank.scores("index", r.inputs), model.bank.scores("learned", r.inputs)
            ranking = np.lexsort((np.arange(6), -score, -index))
            assert np.array_equal(r.ranking, ranking)
            order = rankers.make_selector("hybrid", ranking, gstore.part).prefix(6)
            assert rec["cost"] == brute_cost(gstore, regime, order, q)
        plain = rev.query_retrieval(gstore, regime, q, bank=base, ranker="index", policy="hybrid", budgets=budgets)
        assert plain["selectivity"] == rec["selectivity"] and plain["route"] is None
    assert routes == {"index", "fused_learned_index"}
    negative = next(q for q in env.sets[("dev", regime)].queries if q.kind == "negative")
    with pytest.raises(ValueError, match="only a positive"):
        rev.query_retrieval(gstore, regime, negative, bank=base, ranker="index", policy="hybrid", budgets=budgets)


def test_budgets_are_the_cascade_ladder():
    """ladder_budgets() against Cascade.__post_init__ on the primary and on sensitivity partitions."""
    for partition, blocks in (("primary", 200), ("kahip_fast", 200), ("kahip_fast", 100), ("bfs_random", 400),
                              ("kahip_fast", 7), ("bfs_random", 1)):
        fake = SimpleNamespace(dataset="arxiv", num_blocks=blocks, partition_name=partition, sigma=lambda r: None)
        c = cascade.Cascade(fake, "R0", "jigsaw_index", None, bank=object())
        assert rev.ladder_budgets("arxiv", blocks, partition) == c.budgets
    assert rev.ladder_budgets("arxiv", 100, "kahip_fast") == [3, 5, 10, 25, 50, 100]


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
def test_e4_run_records_resume_and_summary(env, tmp_path, capsys):
    out = tmp_path / "e4"
    argv = run_argv(env, out, "--experiment", "E4", "--regimes", "R0", "--router-tau", "1.0", "--limit", "4")
    assert rev.main(argv) == 0
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (info["status"], info["experiment"], info["split"]) == ("complete", "E4", "dev")
    config = info["config"]
    assert config["rankers"] == list(rankers.RANKER_NAMES) and config["policies"] == ["hybrid", "topk"]
    assert (config["seeds"], config["budgets"], config["router_tau"]) == (SEEDS, [1, 2, 3, 6], 1.0)
    names = config["arms"]["R0"]
    assert len(names) == 5 * 2 + 4 * 2 * len(SEEDS)  # rankers without an encoder, then one arm per seed
    assert sorted(p.stem for p in (out / "records" / "R0").glob("*.jsonl")) == sorted(names)
    main_ids = {serving.default_model_id("synth", "R0", s) for s in SEEDS}
    assert all(f"embeddings/{m}/R0" in info["inputs"] and f"model/{m}" in info["inputs"] for m in main_ids)
    groups = rev.load_records(out)
    assert len(groups) == len(names)
    uids = [q.uid for q in positives(env, "R0")[:4]]
    for (regime, ranker, policy, model_id), recs in groups.items():
        assert list(recs) == uids  # every arm holds the same positives, so records pair by uid
        for rec in recs.values():
            assert set(rec) | {"error", "traceback"} == DOCUMENTED
            assert (rec["schema"], rec["experiment"], rec["dataset"], rec["partition"], rec["split"], rec["master"],
                    rec["regime"], rec["kind"]) == (rev.RECORD_SCHEMA, "E4", "synth", "primary", "dev", DEV, "R0",
                                                    "positive")
            assert rec["arm"] == f"{ranker}+{policy}" + ("" if model_id is None else f"+model={model_id}")
            assert "error" not in rec and 1 <= rec["cost"] <= rec["home_cost"] <= 6 and rec["attempt"] == 1
            if ranker in rankers.NEEDS_EMBEDDING:
                assert model_id in main_ids and rec["evaluated_model_id"] == model_id
                assert (rec["model_kind"], rec["train_arm"], rec["train_regime"], rec["train_arch"]) == (
                    "main", "main", "R0", "graphsage")
                assert model_id == serving.default_model_id("synth", "R0", rec["train_seed"])
            else:
                assert model_id is None and rec["train_seed"] is None and rec["model_kind"] is None
            assert (rec["route"] is not None) == (ranker == "router")
    summary = rev.summarize(out)
    row = summary["R0/index+hybrid"]
    assert (row["n"], row["errors"], row["fullcov_rate"]["6"]) == (4, 0, 1.0)
    assert list(row["fullcov_rate"]) == ["1", "2", "3", "6"]
    capsys.readouterr()
    assert rev.main(["summarize", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(summary))

    path = out / "records" / "R0" / "index+hybrid.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + lines[-1][:20])  # the last record torn by a crash
    again = rev.execute(rev.build_parser().parse_args(argv))
    counts = again["regimes"]["R0"]["arms"]
    assert again["attempt"] == 2 and counts["index+hybrid"] == {"run": 1, "recorded_before": 3, "errors": 0}
    assert all(c["run"] == 0 for name, c in counts.items() if name != "index+hybrid")
    assert again["regimes"]["R0"]["recovered_bytes"] == {"index+hybrid": 20}
    redone, first = rx.read_records(path)[-1], groups[("R0", "index", "hybrid", None)][uids[-1]]
    assert (out / "manifest.attempt1.json").exists() and redone["attempt"] == 2
    same = ("uid", "cost", "home_cost", "fullcov_at", "selectivity", "query_index")
    assert [redone[k] for k in same] == [first[k] for k in same]
    for flag, values in (("--limit", ("5",)), ("--router-tau", ("2.0",)), ("--policies", ("hybrid",))):
        with pytest.raises(SystemExit, match="other settings"):
            rev.main(with_flag(argv, flag, *values))


def matrix_row(kind: str, arm: str, seed: int, duplicate_of: str | None = None, regime: str = "R0",
               dataset: str = "synth") -> dict:
    return {"model_id": serving.model_id_for(dataset, regime, "graphsage", seed, arm), "kind": kind, "dataset": dataset,
            "regime": regime, "arch": "graphsage", "arm": arm, "seed": seed, "duplicate_of": duplicate_of,
            "command": "", "argv": [], "run_dir": "", "val_queries": "", "held_out": [], "model_dir": "",
            "est_gpu_hours": 0.0, "est_basis": ""}


def test_matrix_experiments_evaluate_duplicates_only_without_an_export(env, tmp_path):
    seed = SEEDS[0]
    main_id = serving.default_model_id("synth", "R0", seed)
    own = matrix_row("factorial", "infonce_only", seed)
    dup = matrix_row("factorial", "grid-cvar-barrier0.35-live64", seed, duplicate_of=main_id)
    dup_own = matrix_row("factorial", "no_live", seed, duplicate_of=main_id)
    arch = matrix_row("architecture", "arch-graphsage-L2-res", seed)
    for row in (own, dup_own, arch):
        save_model(env.store, env.models, row["model_id"], "R0", export=True)
        serving.build_block_embeddings(env.store, "R0", env.models / row["model_id"])
    rows = [own, dup, dup_own, arch, matrix_row("factorial", "infonce_only", seed, dataset="arxiv")]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(rows), encoding="utf-8")

    out = tmp_path / "e5"
    assert rev.main(run_argv(env, out, "--experiment", "E5", "--matrix", str(matrix), "--limit", "3")) == 0
    groups = rev.load_records(out)
    assert {k[3] for k in groups} == {main_id, own["model_id"], dup["model_id"], dup_own["model_id"]}
    assert {(k[0], k[1], k[2]) for k in groups} == {("R0", "learned", "hybrid"), ("R0", "learned", "topk")}
    expect = {main_id: ("main", main_id, None), own["model_id"]: ("factorial", own["model_id"], None),
              dup["model_id"]: ("factorial", main_id, main_id), dup_own["model_id"]: ("factorial", dup_own["model_id"],
                                                                                    main_id)}
    for (_, _, policy, model_id), recs in groups.items():
        for rec in recs.values():
            assert (rec["model_kind"], rec["evaluated_model_id"], rec["duplicate_of"]) == expect[model_id]
            assert rec["train_seed"] == seed and rec["experiment"] == "E5"
            assert rec["train_arm"] == rev.parse_model_id(model_id, "synth")["arm"]
        reference = groups[("R0", "learned", policy, main_id)]
        same = groups[("R0", "learned", policy, dup["model_id"])]
        assert [(r["cost"], r["fullcov_at"]) for r in same.values()] == [
            (r["cost"], r["fullcov_at"]) for r in reference.values()]
    models = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))["config"]["models"]["R0"]
    assert [m["model_id"] for m in models] == [main_id, own["model_id"], dup["model_id"], dup_own["model_id"]]

    plan = rev.experiment_plan("E6", "synth", matrix=rows, models_root=env.models)
    assert [(m.model_id, m.kind) for m in plan.models["R0"]] == [(main_id, "main"), (arch["model_id"], "architecture")]
    assert (plan.regimes, plan.rankers, plan.policies) == (["R0"], ["learned"], ["hybrid", "topk"])
    with pytest.raises(SystemExit, match="evaluates R0 encoders"):
        rev.experiment_plan("E5", "synth", matrix=[matrix_row("factorial", "no_live", seed, regime="R1")],
                            models_root=env.models)
    with pytest.raises(SystemExit, match="no architecture rows"):
        rev.experiment_plan("E6", "synth", matrix=[own], models_root=env.models)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([dict(own, arm="no_live")]), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not name"):
        rev.load_matrix(bad)
    with pytest.raises(SystemExit, match="belongs to E5 and E6"):
        rev.main(run_argv(env, tmp_path / "x", "--experiment", "E4", "--matrix", str(matrix)))


def test_sealed_sets_run_only_the_whole_plan(env, tmp_path, monkeypatch):
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    out = tmp_path / "run"
    sealed = run_argv(env, out, "--experiment", "E4", split="test")
    with pytest.raises(SystemExit, match="no router threshold"):  # the plan has the router; no frozen tau yet
        rev.main([*sealed, "--unseal"])
    monkeypatch.setitem(protocol.RANKERS, "router_tau", {"synth": 1.0})
    with pytest.raises(seal.SealError, match="--unseal"):
        rev.main(sealed)
    with pytest.raises(seal.SealError, match="not frozen"):
        rev.main([*sealed, "--unseal"])
    refused = {("--limit", "1"): "--limit", ("--router-tau", "0.5"): "--router-tau",
               ("--partition", "other"): "--partition other", ("--rankers", "index"): "rankers other",
               ("--policies", "hybrid"): "policies other", ("--seeds", str(SEEDS[0])): "seeds other",
               ("--regimes", "R0"): "regimes other",
               ("--models", serving.default_model_id("synth", "R0", SEEDS[0])): "models other"}
    for extra, match in refused.items():
        with pytest.raises(SystemExit, match=match):
            rev.main([*sealed, *extra, "--unseal"])
    with pytest.raises(SystemExit, match="free-form"):
        rev.main(run_argv(env, out, "--rankers", "index", "--regimes", "R0", "--unseal", split="test"))

    monkeypatch.setitem(protocol.FACTORIAL, "synth", {"arms": {"infonce_only": {}, "no_live": {}}})
    matrix = tmp_path / "matrix.json"
    rows = [matrix_row("factorial", arm, s) for arm in ("infonce_only", "no_live") for s in SEEDS]
    matrix.write_text(json.dumps(rows[:2]), encoding="utf-8")  # one arm of two, for two seeds of three
    e5 = run_argv(env, out, "--experiment", "E5", "--matrix", str(matrix), "--unseal", split="test")
    with pytest.raises(SystemExit, match="incomplete factorial matrix"):
        rev.main(e5)
    matrix.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(seal.SealError, match="not frozen"):
        rev.main(e5)
    assert not log.exists() and not out.exists()


def test_bad_setups_fail_before_writing(env, tmp_path):
    out = tmp_path / "run"
    model_id = serving.model_id_for("synth", "R0", "graphsage", 9999)
    save_model(env.store, env.models, model_id, "R0")
    argv = run_argv(env, out, "--rankers", "learned", "--regimes", "R0", "--seeds", "9999", "--limit", "2")
    with pytest.raises(FileNotFoundError, match="no block embeddings"):
        rev.main(argv)
    with pytest.raises(SystemExit, match="no router threshold"):
        rev.main(with_flag(argv, "--rankers", "router"))
    with pytest.raises(FileNotFoundError, match="no model"):
        rev.main(with_flag(argv, "--seeds", "1"))
    refused = [(with_flag(argv, "--rankers", "index"), "none of the rankers"),
               (with_flag(argv, "--models", model_id), "drop --seeds"),
               (with_flag(argv, "--limit", "0"), "N >= 1"),
               (with_flag(with_flag(with_flag(argv, "--regimes", "R0", "R1"), "--experiment", "E4"), "--seeds", "1"),
                "seeds outside the plan")]
    for args, match in refused:
        with pytest.raises(SystemExit, match=match):
            rev.main(args)
    with pytest.raises(SystemExit, match="--rankers and --regimes"):
        rev.main(run_argv(env, out, "--rankers", "index"))
    assert not out.exists()
    assert rev.main([*argv, "--build-embeddings"]) == 0
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert info["extra"]["embeddings_built"] == [f"{model_id}/R0"]
    recs = rx.read_records(out / "records" / "R0" / f"learned+topk+model={model_id}.jsonl")
    assert [(r["train_seed"], r["model_kind"]) for r in recs] == [(9999, "main"), (9999, "main")]


def test_a_failing_query_is_recorded_not_dropped(env, tmp_path, monkeypatch):
    uids = [q.uid for q in positives(env, "R0")[:3]]
    real = rev.query_retrieval

    def flaky(store, regime, q, **kwargs):
        if q.uid == uids[1]:
            raise RuntimeError("injected")
        return real(store, regime, q, **kwargs)

    monkeypatch.setattr(rev, "query_retrieval", flaky)
    out = tmp_path / "run"
    assert rev.main(run_argv(env, out, "--rankers", "index", "--regimes", "R0", "--policies", "topk", "--limit",
                             "3")) == 0
    recs = rx.read_records(out / "records" / "R0" / "index+topk.jsonl")
    assert [r["uid"] for r in recs] == uids and set(recs[1]) == DOCUMENTED
    assert recs[1]["error"] == "RuntimeError: injected" and "injected" in recs[1]["traceback"]
    assert (recs[1]["cost"], recs[1]["home_cost"], recs[1]["fullcov_at"], recs[1]["experiment"]) == (None,) * 4
    assert "error" not in recs[0] and "error" not in recs[2]
    row = rev.summarize(out)["R0/index+topk"]
    assert (row["n"], row["errors"], row["mean_cost"]) == (3, 1, float(np.mean([recs[0]["cost"], recs[2]["cost"]])))
    for k, rate in row["fullcov_rate"].items():  # the error is uncovered and stays in the denominator
        assert rate == sum(bool(recs[i]["fullcov_at"][k]) for i in (0, 2)) / 3
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert info["extra"]["regimes"]["R0"]["arms"]["index+topk"] == {"run": 3, "recorded_before": 0, "errors": 1}


# ----------------------------------------------------------------------------
# Router fit
# ----------------------------------------------------------------------------
def fit_argv(env, out: Path, *extra: str, queries_root: Path | None = None) -> list[str]:
    return ["fit", "--store-root", str(env.copy), "--dataset", "synth", "--queries-root",
            str(queries_root or env.qroot), "--models-root", str(env.models), "--out", str(out), *extra]


def test_fit_router_minimises_the_summed_cost(env, tmp_path):
    out = tmp_path / "fit" / "router_fit.json"
    with pytest.raises(SystemExit, match="--allow-smoke"):  # the test sets are smoke sets
        fit_router.main(fit_argv(env, out))
    assert not out.parent.exists()
    assert fit_router.main(fit_argv(env, out, "--allow-smoke")) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in fit_router.sidecar_path(out).read_text(encoding="utf-8").splitlines()]
    assert (result["schema"], result["dataset"], result["split"], result["master"], result["policy"]) == (
        fit_router.FIT_SCHEMA, "synth", "validation", fit_router.router_master(), "hybrid")
    assert result["smoke"] and result["seeds"] == SEEDS and result["protocol_hash"] == protocol.protocol_hash()
    assert {r: v["queries"] for r, v in result["per_regime"].items()} == {
        r: len(positives(env, r, "validation")) for r in env.regimes}
    assert len(rows) == sum(v["queries"] for v in result["per_regime"].values())
    r2 = serving.default_model_id("synth", "R2-32", SEEDS[0])  # the R2-512 encoder serves every R2-* regime
    assert set(result["embeddings"][r2]) == {"R2-512", "R2-128", "R2-32"}

    sel = np.asarray([row["selectivity"] for row in rows])
    ci = np.asarray([row["cost_index"] for row in rows], dtype=np.float64)
    cf = np.asarray([row["cost_fused"] for row in rows])
    best = min(ci[sel <= t].sum() + cf[sel > t].sum() for t in np.concatenate([[-np.inf], np.unique(sel)]))
    tau = result["tau"]
    assert np.isclose(ci[sel <= tau].sum() + cf[sel > tau].sum(), best) and np.isclose(result["fit"]["cost"], best)
    assert not np.isin(tau, sel) and result["fit"]["routed_to_index"] == int((sel <= tau).sum())
    assert 0 < result["fit"]["routed_to_index"] < len(rows)  # both routes, so the realized router ran both
    assert np.isclose(sum(v["cost_router"] for v in result["per_regime"].values()), best)
    for row in rows:
        assert row["route"] == ("index" if row["selectivity"] <= tau else "fused_learned_index")
        assert set(row["cost_fused_by_seed"]) == {str(s) for s in SEEDS}
        assert row["cost_fused"] == float(np.mean(list(row["cost_fused_by_seed"].values())))
        if row["route"] == "fused_learned_index":
            assert row["cost_router_realized_by_seed"] == row["cost_fused_by_seed"]
    # the costs are retrieval_eval's, so E4 and the fit agree
    first = next(q for q in positives(env, rows[0]["regime"], "validation") if q.uid == rows[0]["uid"])
    budgets = rev.ladder_budgets("synth", 6)
    model = main_model(env, rows[0]["regime"], SEEDS[2])
    again = rev.query_retrieval(env.store, rows[0]["regime"], first, bank=model.bank, ranker="fused_learned_index",
                                policy="hybrid", embed=model.embed, rel_table=model.rel_table, budgets=budgets)
    assert again["cost"] == rows[0]["cost_fused_by_seed"][str(SEEDS[2])]
    info = json.loads((out.parent / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (info["status"], info["experiment"], info["split"]) == ("complete", "router_fit", "validation")
    assert {out.name, fit_router.sidecar_path(out).name} <= set(info["outputs"])


def test_fit_router_refuses_other_sets_and_setups(env, tmp_path):
    master = fit_router.router_master()
    out = tmp_path / "fit" / "router_fit.json"
    wrong = tmp_path / "wrong"
    shutil.copytree(env.dirs["dev"], wrong / "synth" / "validation" / master)  # dev sets on the router path
    with pytest.raises(SystemExit, match="not the router validation set.*split 'dev'"):
        fit_router.main(fit_argv(env, out, "--allow-smoke", queries_root=wrong))
    partial = tmp_path / "partial"
    (partial / "synth" / "validation" / master).mkdir(parents=True)
    shutil.copy(env.dirs["validation"] / "R0.npz", partial / "synth" / "validation" / master / "R0.npz")
    with pytest.raises(SystemExit, match="no router validation set"):
        fit_router.main(fit_argv(env, out, "--allow-smoke", queries_root=partial))
    with pytest.raises(FileNotFoundError, match="no model"):
        fit_router.main(fit_argv(env, out, "--allow-smoke", "--seeds", "1"))
    assert not out.parent.exists()
    out.parent.mkdir()
    (out.parent / "notes.txt").write_text("another run", encoding="utf-8")
    with pytest.raises(SystemExit, match="directory of its own"):
        fit_router.main(fit_argv(env, out, "--allow-smoke"))
    assert sorted(p.name for p in out.parent.iterdir()) == ["notes.txt"]


def test_merge_builds_the_protocol_literal(tmp_path, capsys):
    def fit_file(name: str, dataset: str, tau: float, **changes) -> Path:
        path = tmp_path / f"{name}.json"
        body = {"schema": fit_router.FIT_SCHEMA, "dataset": dataset, "tau": tau, "protocol_hash": "h1",
                "smoke": False, "split": "validation", "master": fit_router.router_master(), **changes}
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    arxiv, mag = fit_file("a", "arxiv", 1.25), fit_file("m", "mag", 0.1 + 0.2)
    out = tmp_path / "taus.json"
    assert fit_router.main(["merge", "--inputs", str(mag), str(arxiv), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"arxiv": 1.25, "mag": 0.1 + 0.2}
    line = capsys.readouterr().out.strip()
    assert line == 'RANKERS["router_tau"] = {"arxiv": 1.25, "mag": 0.30000000000000004}'
    assert ast.literal_eval(line.split(" = ", 1)[1]) == {"arxiv": 1.25, "mag": 0.1 + 0.2}
    with pytest.raises(SystemExit, match="different protocols"):
        fit_router.merge([arxiv, fit_file("h", "mag", 1.0, protocol_hash="h2")])
    with pytest.raises(SystemExit, match="more than once"):
        fit_router.merge([arxiv, fit_file("t", "arxiv", 2.0)])
    smoke = fit_file("s", "mag", 1.0, smoke=True)
    with pytest.raises(SystemExit, match="--allow-smoke"):
        fit_router.merge([arxiv, smoke])
    assert fit_router.merge([arxiv, smoke], allow_smoke=True) == {"arxiv": 1.25, "mag": 1.0}
    with pytest.raises(SystemExit, match="another master"):
        fit_router.merge([fit_file("o", "mag", 1.0, master="31415")])
    with pytest.raises(SystemExit, match="not a router fit"):
        fit_router.merge([fit_file("x", "mag", 1.0, schema="other")])


# ----------------------------------------------------------------------------
# Names and imports
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dataset,regime,arch,seed,arm", [
    ("arxiv", "R0", "graphsage", 7301, "grid-cvar-barrier0.35-live64"), ("arxiv", "R2-512", "graphsage", 7302, "main"),
    ("mag", "R0", "rgcn", 7303, "arch-rgcn-L6-nores"), ("arxiv", "R0", "gin", 7301, "arch-gin-L6-res"),
    ("mag", "R2-128", "rgcn", 7301, "no_cvar")])
def test_model_ids_parse_back(dataset, regime, arch, seed, arm):
    model_id = serving.model_id_for(dataset, regime, arch, seed, arm)
    assert rev.parse_model_id(model_id, dataset) == {"regime": regime, "arch": arch, "seed": seed, "arm": arm}
    assert rev.parse_model_id(model_id, "other") is None


def test_restated_trainer_names_match_the_trainer():
    tree = ast.parse((REPO_ROOT / "scripts" / "canonical" / "train_encoder.py").read_text(encoding="utf-8"))
    constants = {t.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) for t in node.targets if isinstance(t, ast.Name)}
    assert (constants["EXPORT"], constants["MAIN_ARM"]) == (rev.EXPORT, rev.MAIN_ARM)
    theirs = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_smoke_query_set")
    ours = ast.parse(inspect.getsource(fit_router.is_smoke_set)).body[0]
    assert [ast.dump(s) for s in theirs.body[1:]] == [ast.dump(s) for s in ours.body[1:]]  # past the docstrings


def test_modules_import_no_torch():
    code = ("import sys, scripts.canonical.retrieval_eval, scripts.canonical.fit_router; "
            "print('torch' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
