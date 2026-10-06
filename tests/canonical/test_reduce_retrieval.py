"""reduce_retrieval.py on synthetic runs: hand-computed statistics, refusals, duplicate rows, layerwise curves.

Records are written in the exact retrieval_eval schema (the field table of its docstring, checked below)
through run_experiment.RecordFile inside a manifest.RunManifest, as retrieval_eval.run_regime writes them;
layerwise records go through oversmoothing.write_json. No GPU, no torch, no store.
"""

from __future__ import annotations

import __future__
import ast
import csv
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.canonical import build_paper, manifest, oversmoothing, protocol, serving
from scripts.canonical import numbers_json as nj
from scripts.canonical import reduce_retrieval as rr
from scripts.canonical import retrieval_eval as rev
from scripts.canonical import run_experiment as rx
from scripts.canonical.store import PRIMARY

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "arxiv"
REGIME = "R0"
BUDGETS = [1, 2, 5]  # the ladder; P = 5 blocks
P = BUDGETS[-1]
SEEDS = (7301, 7302)
Z = 1.959963984540054
ERR = None  # a query whose record carries an error
DOCUMENTED = set(re.findall(r"^  ([a-z_]+)\s{2,}", rev.__doc__, flags=re.MULTILINE))

X_ARM = "grid-mean-barrier0-live0"
CONTROL = "grid-mean-barrier0.35-live64"
DUP = "grid-cvar-barrier0.35-live64"
GIN = "arch-gin-L6-res"
NORES = "arch-graphsage-L6-nores"

# E4: ten planted queries, then three standard-SM ones
E4_FAMILIES = ["single"] * 4 + ["k_hop"] * 3 + ["random_walk"] * 3 + ["sm_any", "sm_any", "sm_dense"]
N4 = 10
E4_INDEX = [1, 1, 2, 5, 3, 1, 4, 2, 5, ERR, 2, 5, 1]
E4_LEARNED = {7301: [1, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 2, 1], 7302: [1, 1, 2, 1, 2, ERR, 3, 1, 1, 2, 1, 1, 2]}
E4_FUSED = {7301: [1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1], 7302: [2, 1, 1, 1, 3, 1, 2, 2, 1, 1, 1, 2, 2]}

# E5 / E6: six planted queries, then two standard-SM ones
E5_FAMILIES = ["single"] * 3 + ["k_hop"] * 3 + ["sm_any", "sm_sparse"]
N5 = 6
E5_MAIN = {7301: [2, 2, 3, 1, 4, 5, 1, 2], 7302: [2, 4, 1, 1, 4, 3, 2, 2]}
E5_X = {7301: [1, 3, 2, 2, 5, 4, 1, 1], 7302: [3, 1, 2, 2, 3, 4, 2, 1]}
E5_CONTROL = {7301: [3, 3, 3, 2, 5, 5, 2, 2], 7302: [2, 4, 2, 1, 5, 4, 3, 3]}
E5_OWN = {7301: [2, 2, 2, 2, 4, 4, 1, 1], 7302: [2, 3, 1, 1, 4, 4, 2, 2]}
E6_NORES = {7301: [3, 3, 4, 2, 5, 5, 2, 3], 7302: [3, 5, 2, 2, 5, 4, 3, 3]}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No git, machine or library probes in the manifests; few resamples (the arithmetic is unchanged)."""
    monkeypatch.setattr(manifest, "git_info", lambda *args, **kwargs: {"available": False})
    monkeypatch.setattr(manifest, "machine_info", lambda *args, **kwargs: {})
    monkeypatch.setattr(manifest, "library_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(rr, "N_BOOT", 400)
    monkeypatch.setattr(rr, "N_PERM", 2000)
    monkeypatch.setattr(rr, "N_HIER", 200)


# ----------------------------------------------------------------------------
# Synthetic runs
# ----------------------------------------------------------------------------
def query_list(split: str, families: list[str]) -> list[tuple[str, str, int]]:
    return [(f"{DATASET}/{split}/{family}/20/{i:04d}", family, i) for i, family in enumerate(families)]


def record(experiment: str, split: str, arm: rev.Arm, query: tuple[str, str, int], cost: int | None) -> dict:
    """One record as retrieval_eval.evaluate and run_regime write it (cost ERR: an exception record)."""
    uid, family, index = query
    rec: dict[str, Any] = {
        "schema": rev.RECORD_SCHEMA, "experiment": experiment, "dataset": DATASET, "partition": PRIMARY,
        "split": split, "master": "20260607", "regime": REGIME, "uid": uid, "family": family, "size": 20,
        "kind": "positive", "query_index": index, "arm": arm.name, "ranker": arm.ranker, "policy": arm.policy,
        "num_blocks": P, "planted": 20, "attempt": 1,
    }
    rec.update(arm.model.describe() if arm.model is not None else rev.NO_MODEL)
    if cost is ERR:
        rec.update({"cost": None, "home_cost": None, "fullcov_at": None, "selectivity": None, "route": None,
                    "tau": None, "seconds": None, "error": "RuntimeError: synthetic failure",
                    "traceback": "Traceback (most recent call last): ..."})
    else:
        rec.update({"cost": cost, "home_cost": min(cost + 1, P), "fullcov_at": {str(k): cost <= k for k in BUDGETS},
                    "selectivity": 1.25, "route": None, "tau": None,
                    "seconds": {"inputs": 0.001, "embed": 0.0, "rank": 0.002, "select": 0.001}})
    return rec


def write_run(path: Path, experiment: str, split: str, arms: dict, queries: list, *, complete: bool = True,
              mutate=None, planned_extra: tuple[str, ...] = (), inputs: dict | None = None) -> str:
    config = {"schema": rev.RECORD_SCHEMA, "experiment": experiment, "dataset": DATASET, "partition": PRIMARY,
              "regimes": [REGIME], "budgets": list(BUDGETS),
              "arms": {REGIME: [arm.name for arm in arms] + list(planned_extra)}}

    def records() -> None:
        for arm, costs in arms.items():
            handle = rx.RecordFile(Path(path) / "records" / REGIME / rev.record_name(arm))
            try:
                for query, cost in zip(queries, costs, strict=True):
                    rec = record(experiment, split, arm, query, cost)
                    handle.write(mutate(rec) if mutate else rec)
            finally:
                handle.close()

    if complete:
        with manifest.RunManifest(path, experiment=experiment, config=config, split=split, inputs=inputs):
            records()
    else:
        manifest.write_manifest(path, experiment=experiment, config=config, split=split, inputs=inputs)
        records()
    return str(path)


def shifted(costs: list, policy: str) -> list:
    """topk costs one block more than hybrid, up to P."""
    return [c if c is ERR or policy == "hybrid" else min(c + 1, P) for c in costs]


def learned_arm(policy: str, seed: int, arm: str = rev.MAIN_ARM, *, kind: str = "factorial",
                arch: str = "graphsage", duplicate_of: str | None = None, own: bool = False,
                ranker: str = "learned") -> rev.Arm:
    if arm == rev.MAIN_ARM:
        return rev.Arm(ranker, policy, rev.main_model(DATASET, REGIME, seed))
    model_id = serving.model_id_for(DATASET, REGIME, arch, seed, arm)
    target = None if duplicate_of is None else serving.model_id_for(DATASET, REGIME, "graphsage", seed, duplicate_of)
    load_id = model_id if target is None or own else target
    return rev.Arm(ranker, policy, rev.ModelSpec(model_id, load_id, seed, arm, kind, target, REGIME, arch))


def e4_arms(*, fused_seeds=SEEDS, policies=("hybrid", "topk")) -> dict:
    arms: dict = {}
    for policy in policies:
        arms[rev.Arm("index", policy)] = shifted(E4_INDEX, policy)
        for seed in SEEDS:
            arms[learned_arm(policy, seed)] = shifted(E4_LEARNED[seed], policy)
        for seed in fused_seeds:
            arms[learned_arm(policy, seed, ranker="fused_learned_mean")] = shifted(E4_FUSED[seed], policy)
    return arms


def write_e4(tmp_path: Path, name: str = "e4", split: str = "test", **kwargs) -> str:
    arms = kwargs.pop("arms", None) or e4_arms()
    return write_run(tmp_path / name, "E4", split, arms, query_list(split, E4_FAMILIES), **kwargs)


def e5_arms(*, dup: str | None = None, x_seeds=SEEDS, policies=("hybrid", "topk")) -> dict:
    """dup: "weights" (evaluated on main's weights), "own" (its own export), "differ" (main's weights, other records)."""
    arms: dict = {}
    for policy in policies:
        for seed in SEEDS:
            arms[learned_arm(policy, seed)] = shifted(E5_MAIN[seed], policy)
            if seed in x_seeds:
                arms[learned_arm(policy, seed, X_ARM)] = shifted(E5_X[seed], policy)
            arms[learned_arm(policy, seed, CONTROL)] = shifted(E5_CONTROL[seed], policy)
            if dup is not None:
                costs = E5_MAIN[seed] if dup == "weights" else E5_OWN[seed]
                arms[learned_arm(policy, seed, DUP, duplicate_of=rev.MAIN_ARM, own=dup == "own")] = shifted(costs, policy)
    return arms


def write_e5(tmp_path: Path, name: str = "e5", split: str = "test", **kwargs) -> str:
    return write_run(tmp_path / name, "E5", split, e5_arms(**kwargs), query_list(split, E5_FAMILIES))


def write_e6(tmp_path: Path, name: str = "e6") -> str:
    arms: dict = {}
    for seed in SEEDS:
        arms[learned_arm("hybrid", seed)] = E5_MAIN[seed]
        arms[learned_arm("hybrid", seed, GIN, kind="architecture", arch="gin")] = E5_MAIN[seed]  # the same outcomes
        arms[learned_arm("hybrid", seed, NORES, kind="architecture")] = E6_NORES[seed]
    return write_run(tmp_path / name, "E6", "test", arms, query_list("test", E5_FAMILIES))


def layer_record(arm: str, seed: int, *, arch: str, residual: bool, knockout: bool, layers: int = 2) -> dict:
    """A jigsaw-layerwise-v1 record; seed 7302 adds 0.1 and a knockout 0.2 to the distance metrics."""
    shift = 0.1 * SEEDS.index(seed) + (0.2 if knockout else 0.0)
    entries = []
    for i, name in enumerate(oversmoothing.layer_names(layers)):
        node = name != "readout"
        entries.append({"layer": name, "dim": 8 if name == "input" else 16,
                        "mean_average_distance": 0.1 * i + shift if node else None,
                        "mean_pairwise_cosine": 1.0 - (0.1 * i + shift) if node else None,
                        "effective_rank": 12.0 - i if node else None,
                        "zero_rows": 0 if node else None,
                        "block_effective_rank": 4.0 - 0.5 * i + shift,
                        "block_separation": 0.5 + 0.01 * i + shift})
    return {"schema": oversmoothing.SCHEMA, "model_id": serving.model_id_for(DATASET, REGIME, arch, seed, arm),
            "dataset": DATASET, "regime": REGIME, "partition": PRIMARY, "knockout_residual": knockout,
            "model": {"arch": arch, "layers": layers, "hidden": 16, "residual": residual, "layer_norm": True,
                      "train_seed": seed, "arm": arm, "trained_regime": REGIME, "torch_state_sha256": "0" * 64},
            "sample": {"blocks": [0, 1, 2], "nodes_per_block": [4, 4, 4], "block_sizes": [10, 12, 9],
                       "max_blocks": 64, "max_nodes": 4096, "seed": 0, "nodes_sha256": "a" * 64},
            "layers": entries, "definitions": dict(oversmoothing.DEFINITIONS),
            "protocol_hash": protocol.protocol_hash(), "seconds": 1.0}


LAYERWISE_MODELS = [(rev.MAIN_ARM, "graphsage", True, False, SEEDS), (rev.MAIN_ARM, "graphsage", True, True, SEEDS),
                    ("arch-graphsage-L2-nores", "graphsage", False, False, SEEDS),
                    ("arch-gin-L2-res", "gin", True, False, (7301,))]


def write_layerwise(tmp_path: Path, name: str = "layerwise") -> str:
    run = tmp_path / name
    with manifest.RunManifest(run, experiment=oversmoothing.EXPERIMENT, config={"dataset": DATASET, "regime": REGIME},
                              split=oversmoothing.SPLIT):
        for arm, arch, residual, knockout, seeds in LAYERWISE_MODELS:
            for seed in seeds:
                rec = layer_record(arm, seed, arch=arch, residual=residual, knockout=knockout)
                oversmoothing.write_json(run / oversmoothing.output_name(rec["model_id"], knockout), rec)
    return str(run)


def reduce_to(tmp_path: Path, **inputs) -> tuple[dict, dict, Path]:
    out = tmp_path / "numbers"
    summary = rr.reduce(out=str(out), **inputs)
    return summary, nj.load(out / "numbers.json"), out


def wilson(successes: int, n: int) -> tuple[float, float]:
    """The Wilson score interval in percent, written out."""
    p = successes / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def holm_by_hand(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        out[key] = running
    return out


def covered(costs: list, k: int) -> np.ndarray:
    return np.array([0.0 if c is ERR else float(c <= k) for c in costs])


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------
# Schema of the synthetic records
# ----------------------------------------------------------------------------
def test_synthetic_records_have_exactly_the_documented_fields():
    query = query_list("test", ["single"])[0]
    ok = record("E4", "test", rev.Arm("index", "hybrid"), query, 2)
    failed = record("E5", "test", learned_arm("hybrid", 7301, X_ARM), query, ERR)
    assert set(ok) == DOCUMENTED - {"error", "traceback"}
    assert set(failed) == DOCUMENTED
    assert rev.parse_model_id(failed["model_id"], DATASET) == {"regime": REGIME, "arch": "graphsage", "seed": 7301,
                                                                "arm": X_ARM}


# ----------------------------------------------------------------------------
# E4
# ----------------------------------------------------------------------------
def test_e4_fullcov_rates_have_wilson_bounds_and_count_errors_as_not_covered(tmp_path):
    _, doc, _ = reduce_to(tmp_path, e4=[write_e4(tmp_path)])
    num = doc["numbers"]
    base = "E4.arxiv.planted.R0.hybrid.index"
    # index over the planted queries: costs 1 1 2 5 3 1 4 2 5 and one error
    assert wilson(3, 10) == pytest.approx((10.7791267, 60.3221853))
    for k, hits in ((1, 3), (2, 5), (5, 9)):
        entry = num[f"{base}.fullcov.K{k}"]
        assert entry["unit"] == "pct" and entry["n"] == 10
        assert entry["value"] == pytest.approx(100 * hits / 10)
        assert entry["ci"] == pytest.approx(list(wilson(hits, 10)))
    assert num[f"{base}.n"]["value"] == 10
    assert (num[f"{base}.errors"]["value"], num[f"{base}.errors"]["n"]) == (1, 10)
    assert num[f"{base}.cost.mean"]["value"] == pytest.approx(24 / 9) and num[f"{base}.cost.mean"]["n"] == 9
    assert num[f"{base}.cost.median"]["value"] == 2 and num[f"{base}.cost.median"]["n"] == 9
    assert f"{base}.seeds" not in num and f"{base}.cost.gm_ratio_p1_vs_index" not in num
    # the standard-SM queries are reduced apart, never pooled with the planted ones
    assert num["E4.arxiv.sm.R0.hybrid.index.n"]["value"] == 3
    assert num["E4.arxiv.sm.R0.hybrid.index.fullcov.K1"]["value"] == pytest.approx(100 / 3)


def test_e4_learned_rankers_average_the_seeds_per_query(tmp_path):
    _, doc, out = reduce_to(tmp_path, e4=[write_e4(tmp_path)])
    num, tables = doc["numbers"], doc["tables"]
    base = "E4.arxiv.planted.R0.hybrid.learned"
    s1, s2 = E4_LEARNED[7301][:N4], E4_LEARNED[7302][:N4]
    for k in BUDGETS:
        entry = num[f"{base}.fullcov.K{k}"]
        assert entry["value"] == pytest.approx(100 * np.mean((covered(s1, k) + covered(s2, k)) / 2))
        assert entry["n"] == N4 and entry["ci"][0] <= entry["ci"][1]
    assert num[f"{base}.fullcov.K1"]["value"] == pytest.approx(55.0)
    assert (num[f"{base}.errors"]["value"], num[f"{base}.errors"]["n"]) == (1, 2 * N4)
    assert num[f"{base}.seeds"]["value"] == 2
    # the query with an error in seed 7302 leaves the cost of both seeds, and is counted
    keep = [i for i in range(N4) if ERR not in (s1[i], s2[i])]
    assert num[f"{base}.cost.mean"]["value"] == pytest.approx(np.mean([(s1[i] + s2[i]) / 2 for i in keep]))
    assert num[f"{base}.cost.mean"]["value"] == pytest.approx(14 / 9) and num[f"{base}.cost.mean"]["n"] == 9
    # one seed alone: Wilson over the queries, mean cost over its queries without an error
    assert num[f"{base}.seed7301.fullcov.K1"]["value"] == pytest.approx(60.0)
    assert num[f"{base}.seed7301.fullcov.K1"]["ci"] == pytest.approx(list(wilson(6, 10)))
    assert num[f"{base}.seed7302.fullcov.K1"]["value"] == pytest.approx(50.0)
    assert (num[f"{base}.seed7301.cost.mean"]["value"], num[f"{base}.seed7301.cost.mean"]["n"]) == (1.5, 10)
    assert num[f"{base}.seed7302.cost.mean"]["value"] == pytest.approx(14 / 9)
    assert num[f"{base}.seed7302.cost.mean"]["n"] == 9
    # (cost + 1) ratio against index over the queries with a cost on both sides
    index = E4_INDEX[:N4]
    pairs = [((s1[i] + s2[i]) / 2, index[i]) for i in keep if index[i] is not ERR]
    entry = num[f"{base}.cost.gm_ratio_p1_vs_index"]
    assert entry["value"] == pytest.approx(math.exp(np.mean([math.log((a + 1) / (b + 1)) for a, b in pairs])))
    assert entry["n"] == len(pairs) == 8 and entry["unit"] == "ratio"
    # tables follow rankers.RANKER_NAMES; the seeds table holds the encoder rankers only
    assert [row[0] for row in tables["E4.arxiv.planted.R0.hybrid.fullcov"]["rows"]] == \
        ["learned", "index", "fused_learned_mean"]
    assert [row[:2] for row in tables["E4.arxiv.planted.R0.hybrid.seeds"]["rows"]] == \
        [["learned", "7301"], ["learned", "7302"], ["fused_learned_mean", "7301"], ["fused_learned_mean", "7302"]]
    cost_rows = {row[0]: row for row in tables["E4.arxiv.planted.R0.hybrid.cost"]["rows"]}
    assert cost_rows["index"][-1] == "-"
    assert cost_rows["learned"][-1] == {"key": f"{base}.cost.gm_ratio_p1_vs_index", "ci": True}
    # the curve reads the same numbers, on a log axis
    rows = read_csv(out / "figures" / "E4.arxiv.planted.R0.hybrid.fullcov_curve.csv")
    learned = [row for row in rows if row["series"] == "learned"]
    assert [float(row["x"]) for row in learned] == BUDGETS
    assert float(learned[0]["y"]) == pytest.approx(55.0)
    assert doc["figures"]["E4.arxiv.planted.R0.hybrid.fullcov_curve"]["log_x"] is True


def test_pe5_pairs_rankers_by_query_with_holm_within_each_table(tmp_path):
    summary, doc, _ = reduce_to(tmp_path, e4=[write_e4(tmp_path)])
    num, tables = doc["numbers"], doc["tables"]
    q = "PE5.arxiv.planted.R0.learned_vs_index"
    s1, s2, index = E4_LEARNED[7301][:N4], E4_LEARNED[7302][:N4], E4_INDEX[:N4]
    keep = [i for i in range(N4) if ERR not in (s1[i], s2[i], index[i])]
    assert num[f"{q}.cost.gm_ratio"]["value"] == pytest.approx(
        math.exp(np.mean([math.log((s1[i] + s2[i]) / 2 / index[i]) for i in keep])))
    assert num[f"{q}.cost.gm_ratio"]["n"] == len(keep) == 8
    assert num[f"{q}.cost.excluded"]["value"] == 2 and num[f"{q}.n"]["value"] == N4
    diff = np.mean((covered(s1, 1) + covered(s2, 1)) / 2 - covered(index, 1))
    assert num[f"{q}.fullcov.K1.diff"]["value"] == pytest.approx(100 * diff) == pytest.approx(25.0)
    assert num[f"{q}.fullcov.K1.diff"]["unit"] == "pp" and num[f"{q}.fullcov.K1.diff"]["n"] == N4
    assert f"{q}.fullcov.K2.diff" not in num  # PE5 reports FullCov at the first budget only
    # the run lacks fused_learned_index and router: three comparisons, the others named in warnings
    comparisons = ["learned_vs_index", "learned_vs_fused_learned_mean", "fused_learned_mean_vs_index"]
    tested = sorted(k for k in num if k.startswith("PE5.arxiv.planted.") and k.endswith(".cost.p"))
    assert tested == sorted(f"PE5.arxiv.planted.R0.{c}.cost.p" for c in comparisons)
    assert any("router" in w for w in summary["warnings"])
    assert any("fused_learned_index" in w for w in summary["warnings"])
    for family, table in (("cost", "PE5.arxiv.planted.cost"), ("fullcov.K1", "PE5.arxiv.planted.fullcov")):
        pvalues = {c: num[f"PE5.arxiv.planted.R0.{c}.{family}.p"]["value"] for c in comparisons}
        for c, adjusted in holm_by_hand(pvalues).items():
            assert num[f"PE5.arxiv.planted.R0.{c}.{family}.p_holm"]["value"] == pytest.approx(adjusted)
        assert [row[1] for row in tables[table]["rows"]] == [c.replace("_vs_", " vs ") for c in comparisons]
    assert "PE5.arxiv.sm.R0.learned_vs_index.cost.gm_ratio" in num
    assert "Not a pre-registered endpoint" in tables["PE5.arxiv.sm.cost"]["notes"]


def test_holm_is_applied_in_p_order_within_a_table():
    nums = nj.Numbers(split="test", frozen=False, protocol_hash="h" * 64)
    rr.write_tests(nums, [rr.Test("T.a", 0.01, 10, []), rr.Test("T.b", 0.04, 10, []), rr.Test("T.c", 0.03, 10, [])])
    # sorted 0.01, 0.03, 0.04: 3 x 0.01 = 0.03; 2 x 0.03 = 0.06; max(0.06, 1 x 0.04) = 0.06
    assert {k: nums.numbers[f"T.{k}.p_holm"]["value"] for k in "abc"} == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert nums.numbers["T.b.p"]["value"] == 0.04 and nums.numbers["T.b.p_holm"]["n"] == 10


# ----------------------------------------------------------------------------
# E5, E6 and PE6
# ----------------------------------------------------------------------------
def test_e5_averages_seeds_per_query_before_pairing(tmp_path):
    _, doc, _ = reduce_to(tmp_path, e5=[write_e5(tmp_path)])
    num = doc["numbers"]
    q = f"E5.arxiv.planted.R0.hybrid.vs_main.{rr.segment(X_ARM)}"
    arm = np.array([E5_X[s][:N5] for s in SEEDS], dtype=float)
    main = np.array([E5_MAIN[s][:N5] for s in SEEDS], dtype=float)
    averaged_first = math.exp(np.mean(np.log(arm.mean(axis=0) / main.mean(axis=0))))
    paired_first = math.exp(np.mean(np.log(arm / main)))
    assert averaged_first == pytest.approx((4 / 3) ** (1 / 6))
    assert abs(averaged_first - paired_first) > 0.01  # on these outcomes the order matters
    entry = num[f"{q}.cost.gm_ratio"]
    assert entry["value"] == pytest.approx(averaged_first) and entry["n"] == N5
    assert entry["ci"][0] <= entry["ci"][1]
    assert num[f"{q}.n"]["value"] == N5 and num[f"{q}.cost.excluded"]["value"] == 0
    for k in BUDGETS[:-1]:
        diff = np.mean((arm <= k).mean(axis=0) - (main <= k).mean(axis=0))
        assert num[f"{q}.fullcov.K{k}.diff"]["value"] == pytest.approx(100 * diff)
    assert num[f"{q}.fullcov.K1.diff"]["value"] == pytest.approx(-100 / 12)
    assert f"{q}.fullcov.K{P}.diff" not in num  # every K but P
    described = f"E5.arxiv.planted.R0.hybrid.arm.{rr.segment(X_ARM)}"
    assert num[f"{described}.fullcov.K2"]["value"] == pytest.approx(100 * (arm <= 2).mean())
    assert num[f"{described}.seeds"]["value"] == 2
    assert f"{described}.cost.gm_ratio_p1_vs_index" not in num


def test_e5_holm_family_is_the_table_without_duplicate_rows(tmp_path):
    _, doc, _ = reduce_to(tmp_path, e5=[write_e5(tmp_path, dup="weights")])
    num, tables = doc["numbers"], doc["tables"]
    prefix = "E5.arxiv.planted.R0."
    pvalues = {k[: -len(".p")]: e["value"] for k, e in num.items() if k.startswith(prefix) and k.endswith(".cost.p")}
    assert sorted(pvalues) == sorted(f"{prefix}{policy}.vs_main.{rr.segment(a)}.cost"
                                     for policy in ("hybrid", "topk") for a in (X_ARM, CONTROL))
    for key, adjusted in holm_by_hand(pvalues).items():
        assert num[f"{key}.p_holm"]["value"] == pytest.approx(adjusted)
    rows = tables["E5.arxiv.planted.cost"]["rows"]
    assert [row[:3] for row in rows] == [["R0", policy, label] for policy in ("hybrid", "topk")
                                         for label in (X_ARM, CONTROL, f"{DUP} (duplicate of main)")]
    tested = [row for row in rows if not row[2].startswith(DUP)]
    assert all(row[7] == {"key": f"E5.arxiv.planted.R0.{row[1]}.vs_main.{rr.segment(row[2])}.cost.p"}
               for row in tested)
    fullcov_rows = tables["E5.arxiv.planted.fullcov"]["rows"]
    assert len(fullcov_rows) == 2 * 3 * (len(BUDGETS) - 1)


def test_duplicate_row_reports_the_model_it_duplicates(tmp_path):
    _, doc, out = reduce_to(tmp_path, e5=[write_e5(tmp_path, dup="weights")])
    num, tables = doc["numbers"], doc["tables"]
    base = "E5.arxiv.planted.R0.hybrid"
    seg = rr.segment(DUP)
    assert seg == "grid-cvar-barrier0_35-live64"
    assert not [k for k in num if f".arm.{seg}." in k or f".vs_main.{seg}." in k]
    label = f"{DUP} (duplicate of main)"
    arms_table = tables[f"{base}.arms"]
    row = next(r for r in arms_table["rows"] if r[0] == label)
    main_row = next(r for r in arms_table["rows"] if r[0] == "main")
    assert row[1:] == main_row[1:]
    assert all(cell["key"].startswith(f"{base}.arm.main.") for cell in row[1:] if isinstance(cell, dict))
    assert f"{DUP} has the settings of main and was evaluated on its weights" in arms_table["notes"]
    cost_row = next(r for r in tables["E5.arxiv.planted.cost"]["rows"] if r[1] == "hybrid" and r[2] == label)
    assert cost_row[6:] == ["same model", "-", "-"]
    assert label not in [r[0] for r in tables[f"{base}.seeds"]["rows"]]
    series = {r["series"] for r in read_csv(out / "figures" / f"{base}.fullcov_curve.csv")}
    assert series == {"main", X_ARM, CONTROL}


def test_duplicate_on_its_own_export_keeps_its_own_numbers_without_a_test(tmp_path):
    _, doc, _ = reduce_to(tmp_path, e5=[write_e5(tmp_path, dup="own")])
    num = doc["numbers"]
    base = "E5.arxiv.planted.R0.hybrid"
    seg = rr.segment(DUP)
    own = np.array([E5_OWN[s][:N5] for s in SEEDS], dtype=float)
    assert num[f"{base}.arm.{seg}.n"]["value"] == N5
    assert num[f"{base}.arm.{seg}.fullcov.K2"]["value"] == pytest.approx(100 * (own <= 2).mean())
    assert not [k for k in num if f".vs_main.{seg}." in k]
    arms_table = doc["tables"][f"{base}.arms"]
    row = next(r for r in arms_table["rows"] if r[0] == f"{DUP} (duplicate of main)")
    assert row[1] == {"key": f"{base}.arm.main.n"}
    assert "was evaluated on its own export" in arms_table["notes"]


def test_pe6_compares_main_with_the_mean_aggregation_control(tmp_path):
    assert rr.pe6_control("arxiv") == CONTROL
    assert rr.pe6_control("mag") == "no_cvar"
    assert list(rr.factorial_arms("arxiv")) == [
        "grid-mean-barrier0-live0", "grid-mean-barrier0-live64", "grid-mean-barrier0.35-live0",
        "grid-mean-barrier0.35-live64", "grid-cvar-barrier0-live0", "grid-cvar-barrier0-live64",
        "grid-cvar-barrier0.35-live0", "grid-cvar-barrier0.35-live64", "infonce_only"]
    _, doc, _ = reduce_to(tmp_path, e5=[write_e5(tmp_path, dup="weights")])
    num, tables = doc["numbers"], doc["tables"]
    assert num["PE6.arxiv.control_arm"]["value"] == CONTROL and num["PE6.arxiv.control_arm"]["unit"] == "text"
    main = np.array([E5_MAIN[s][:N5] for s in SEEDS], dtype=float)
    control = np.array([E5_CONTROL[s][:N5] for s in SEEDS], dtype=float)
    q = "PE6.arxiv.planted.R0.hybrid"
    assert num[f"{q}.cost.gm_ratio"]["value"] == pytest.approx(
        math.exp(np.mean(np.log(main.mean(axis=0) / control.mean(axis=0)))))
    for policy in ("hybrid", "topk"):
        q = f"PE6.arxiv.planted.R0.{policy}"
        e5 = f"E5.arxiv.planted.R0.{policy}.vs_main.{rr.segment(CONTROL)}"
        assert num[f"{q}.cost.gm_ratio"]["value"] * num[f"{e5}.cost.gm_ratio"]["value"] == pytest.approx(1.0)
        for k in BUDGETS[:-1]:
            assert num[f"{q}.fullcov.K{k}.diff"]["value"] == pytest.approx(-num[f"{e5}.fullcov.K{k}.diff"]["value"])
            assert f"{q}.fullcov.K{k}.p_holm" in num
        assert f"{q}.fullcov.K{P}.diff" not in num
    assert [row[:2] for row in tables["PE6.arxiv.planted.cost"]["rows"]] == [["R0", "hybrid"], ["R0", "topk"]]
    assert len(tables["PE6.arxiv.planted.fullcov"]["rows"]) == 2 * (len(BUDGETS) - 1)
    assert "PE6.arxiv.sm.R0.hybrid.cost.gm_ratio" in num
    assert tables["PE6.arxiv.planted.cost"]["notes"].startswith("PE6:")


def test_identical_arm_gives_the_identity_ratio(tmp_path):
    _, doc, _ = reduce_to(tmp_path, e6=[write_e6(tmp_path)])
    num, tables = doc["numbers"], doc["tables"]
    q = f"E6.arxiv.planted.R0.hybrid.vs_main.{GIN}"
    ratio = num[f"{q}.cost.gm_ratio"]
    assert ratio["value"] == pytest.approx(1.0) and ratio["ci"] == pytest.approx([1.0, 1.0])
    assert num[f"{q}.cost.p"]["value"] == 1.0
    for k in BUDGETS[:-1]:
        diff = num[f"{q}.fullcov.K{k}.diff"]
        assert diff["value"] == 0 and diff["ci"] == [0, 0] and num[f"{q}.fullcov.K{k}.p"]["value"] == 1.0
    nores = np.array([E6_NORES[s][:N5] for s in SEEDS], dtype=float)
    main = np.array([E5_MAIN[s][:N5] for s in SEEDS], dtype=float)
    assert num[f"E6.arxiv.planted.R0.hybrid.vs_main.{NORES}.cost.gm_ratio"]["value"] == pytest.approx(
        math.exp(np.mean(np.log(nores.mean(axis=0) / main.mean(axis=0)))))
    assert [row[0] for row in tables["E6.arxiv.planted.R0.hybrid.arms"]["rows"]] == ["main", GIN, NORES]
    assert "architecture, depth or residual variant" in tables["E6.arxiv.planted.cost"]["notes"]
    assert not [k for k in num if k.startswith("PE6.")]  # PE6 reads E5 only


# ----------------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------------
def refused(tmp_path: Path, match: str, **inputs) -> None:
    out = tmp_path / "refused"
    with pytest.raises(SystemExit, match=match):
        rr.reduce(out=str(out), **inputs)
    assert not out.exists()  # refused before anything is written


def test_a_missing_seed_breaks_the_pairing(tmp_path):
    refused(tmp_path, "missing seed breaks", e5=[write_e5(tmp_path, x_seeds=(7301,))])
    refused(tmp_path, "missing seed breaks", e4=[write_e4(tmp_path, arms=e4_arms(fused_seeds=(7301,)))])


def test_mixed_splits_are_refused(tmp_path):
    dev = write_e4(tmp_path, name="e4-dev", split="dev")
    validation = write_e5(tmp_path, name="e5-validation", split="validation")
    refused(tmp_path, "different splits", e4=[dev], e5=[validation], allow_dev=True)


def test_a_dev_split_needs_allow_dev(tmp_path):
    dev = write_e4(tmp_path, split="dev")
    refused(tmp_path, "--allow-dev", e4=[dev])
    summary, doc, _ = reduce_to(tmp_path, e4=[dev], allow_dev=True)
    assert summary["split"] == doc["split"] == "dev"


def test_an_incomplete_run_is_refused(tmp_path):
    refused(tmp_path, "status 'running'", e4=[write_e4(tmp_path, complete=False)])


def test_a_record_of_another_schema_is_refused(tmp_path):
    run = write_e4(tmp_path, mutate=lambda rec: {**rec, "schema": "jigsaw-retrieval-record-v0"})
    refused(tmp_path, "jigsaw-retrieval-record-v0", e4=[run])


def test_mixed_protocol_hashes_are_refused(tmp_path):
    e4, e5 = write_e4(tmp_path), write_e5(tmp_path)
    path = Path(e5) / manifest.MANIFEST_NAME
    info = json.loads(path.read_text(encoding="utf-8"))
    info["protocol_hash"] = "0" * 64
    path.write_text(json.dumps(info), encoding="utf-8")
    refused(tmp_path, "different protocol hashes", e4=[e4], e5=[e5])


def test_a_protocol_hash_other_than_the_codes_is_refused(tmp_path):
    e4 = write_e4(tmp_path)
    path = Path(e4) / manifest.MANIFEST_NAME
    info = json.loads(path.read_text(encoding="utf-8"))
    info["protocol_hash"] = "0" * 64
    path.write_text(json.dumps(info), encoding="utf-8")
    refused(tmp_path, "is not the code's", e4=[e4])


def test_a_duplicate_on_its_targets_weights_must_have_its_records(tmp_path):
    refused(tmp_path, "its records differ", e5=[write_e5(tmp_path, dup="differ")])


def test_other_refusals(tmp_path):
    e4 = write_e4(tmp_path)
    refused(tmp_path, "given as E5", e5=[e4])
    refused(tmp_path, "given twice", e4=[e4, e4])
    with pytest.raises(SystemExit, match="lies inside"):
        rr.reduce(e4=[e4], out=str(Path(e4) / "numbers"))
    free = write_run(tmp_path / "free", "retrieval", "test", e4_arms(), query_list("test", E4_FAMILIES))
    refused(tmp_path, "experiment 'retrieval'", e4=[free])
    unplanned = write_e4(tmp_path, name="e4-plan", planned_extra=("router+hybrid",))
    refused(tmp_path, "planned arms", e4=[unplanned])

    def wrong_fullcov(rec: dict) -> dict:
        return {**rec, "fullcov_at": {k: True for k in rec["fullcov_at"]}} if (rec["cost"] or 0) > 1 else rec

    refused(tmp_path, "disagrees with cost", e4=[write_e4(tmp_path, name="e4-fullcov", mutate=wrong_fullcov)])
    changed = write_e4(tmp_path, name="e4-changed")
    with open(Path(changed) / "records" / REGIME / "index+hybrid.jsonl", "a", encoding="utf-8") as handle:
        handle.write("\n")
    refused(tmp_path, "does not verify", e4=[changed])
    refused(tmp_path, "no inputs")


def test_statistics_other_than_the_implemented_rules_are_refused(monkeypatch):
    rr.check_statistics()
    monkeypatch.setitem(protocol.STATISTICS, "alpha", 0.01)
    with pytest.raises(SystemExit, match="alpha"):
        rr.check_statistics()


# ----------------------------------------------------------------------------
# E4 split by the pattern audit (E20)
# ----------------------------------------------------------------------------
TRAINING_SEEDS = tuple(protocol.SEEDS["training"])
SEEN_AT = (0, 4, 9, 11)  # three planted positives, and a standard-SM query (in neither seen nor novel)


def query_file(tmp_path: Path, name: str = "R0.json", content: str = "the query set") -> Path:
    path = tmp_path / "queries" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_audit(tmp_path: Path, qfile: Path, *, name: str = "audit", split: str = "test", seen=SEEN_AT,
                seeds=TRAINING_SEEDS, smoke: bool = False, mutate=None) -> str:
    """A pattern_audit.py output of the main encoders of seeds; seen: the query indices the first one drew."""
    queries = query_list(split, E4_FAMILIES)
    encoders = {rev.main_model(DATASET, REGIME, s).model_id: s for s in seeds}
    first = next(iter(encoders))
    entry = {"query_set": {"path": str(qfile), "sha256": manifest.sha256_file(qfile)["sha256"]},
             "uids": [q[0] for q in queries], "encoders": list(encoders),
             "seen": {queries[i][0]: {first: {"first": [1, 0, i], "hits": 1}} for i in seen}}
    doc = {"schema": rr.AUDIT_SCHEMA, "dataset": DATASET, "split": split, "protocol_hash": protocol.protocol_hash(),
           "complete": True, "regimes": {REGIME: entry},
           "encoders": {m: {"regime": REGIME, "serves": [REGIME], "seed": s, "smoke": smoke}
                        for m, s in encoders.items()}}
    path = tmp_path / name
    with manifest.RunManifest(path, experiment=rr.AUDIT_EXPERIMENT, config={}, split=split,
                              inputs={f"queries/{REGIME}": qfile}):
        (path / rr.AUDIT_FILE).write_text(json.dumps(mutate(doc) if mutate else doc), encoding="utf-8")
    return str(path)


def audited_e4(tmp_path: Path, **audit) -> tuple[str, str]:
    qfile = query_file(tmp_path)
    return write_e4(tmp_path, inputs={f"queries/{REGIME}": qfile}), write_audit(tmp_path, qfile, **audit)


def test_e4_splits_the_planted_positives_by_the_pattern_audit(tmp_path):
    e4, audit = audited_e4(tmp_path)
    summary, doc, out = reduce_to(tmp_path, e4=[e4], pattern_audit=[audit])
    num, tables = doc["numbers"], doc["tables"]
    for sub, idx in (("seen", [0, 4, 9]), ("novel", [1, 2, 3, 5, 6, 7, 8])):
        base = f"E4.arxiv.{sub}.R0.hybrid"
        index = [E4_INDEX[i] for i in idx]
        s1, s2 = [E4_LEARNED[7301][i] for i in idx], [E4_LEARNED[7302][i] for i in idx]
        assert num[f"{base}.index.n"]["value"] == len(idx)
        for k in BUDGETS:
            hits = int(covered(index, k).sum())
            entry = num[f"{base}.index.fullcov.K{k}"]
            assert entry["value"] == pytest.approx(100 * hits / len(idx)) and entry["n"] == len(idx)
            assert entry["ci"] == pytest.approx(list(wilson(hits, len(idx))))
            learned = num[f"{base}.learned.fullcov.K{k}"]
            assert learned["value"] == pytest.approx(100 * np.mean((covered(s1, k) + covered(s2, k)) / 2))
        # every number of the subset reads the audit as well as the run
        assert num[f"{base}.index.n"]["source"] == num[f"{base}.learned.fullcov.K1"]["source"] == \
            ["E20.arxiv", "E4.arxiv"]
        assert "pattern audit" in tables[f"{base}.fullcov"]["notes"]
        assert (out / "figures" / f"{base}.fullcov_curve.csv").is_file()
    # seen and novel partition the planted positives: 3 x 50 + 7 x 400/7 = 10 x 55
    seen, novel = (num[f"E4.arxiv.{s}.R0.hybrid.learned.fullcov.K1"]["value"] for s in ("seen", "novel"))
    assert (seen, novel) == (pytest.approx(50.0), pytest.approx(400 / 7))
    assert 3 * seen + 7 * novel == pytest.approx(10 * num["E4.arxiv.planted.R0.hybrid.learned.fullcov.K1"]["value"])
    # the pre-registered numbers neither change nor read the audit, and seen and novel get no PE5 test
    alone = rr.reduce(e4=[e4], out=str(tmp_path / "alone"))
    plain = nj.load(tmp_path / "alone" / "numbers.json")["numbers"]
    assert {k: e["value"] for k, e in plain.items()} == {k: num[k]["value"] for k in plain}
    assert set(num) - set(plain) == {k for k in num if k.startswith(("E4.arxiv.seen.", "E4.arxiv.novel."))}
    assert num["E4.arxiv.planted.R0.hybrid.index.n"]["source"] == ["E4.arxiv"]
    assert alone["sources"] == ["E4.arxiv"] and summary["sources"] == ["E20.arxiv", "E4.arxiv"]
    assert doc["sources"]["E20.arxiv"] == {"path": Path(audit).as_posix(), "experiment": "E20", "dataset": "arxiv",
                                           "manifest_sha256": nj.file_sha256(Path(audit) / manifest.MANIFEST_NAME)}


def test_an_empty_subset_is_left_out_and_the_cli_takes_the_audit(tmp_path, capsys):
    e4, audit = audited_e4(tmp_path, seen=(11,))  # the encoders drew only a standard-SM query
    out = tmp_path / "numbers"
    assert rr.main(["--e4", e4, "--pattern-audit", audit, "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["sources"] == ["E20.arxiv", "E4.arxiv"]
    num = nj.load(out / "numbers.json")["numbers"]
    assert num["E4.arxiv.novel.R0.hybrid.index.n"]["value"] == N4
    assert not [k for k in num if ".seen." in k]
    args = rr.build_parser().parse_args(["--e4", "a", "--pattern-audit", "b", "--pattern-audit", "c", "--out", "o"])
    assert args.pattern_audit == ["b", "c"]


def test_an_audit_without_an_e4_run_is_named_in_a_warning(tmp_path):
    audit = write_audit(tmp_path, query_file(tmp_path))
    summary, doc, _ = reduce_to(tmp_path, e5=[write_e5(tmp_path)], pattern_audit=[audit])
    assert [w for w in summary["warnings"] if w == f"{audit} audits arxiv, which no --e4 run evaluates"]
    assert not [k for k in doc["numbers"] if ".seen." in k or ".novel." in k]


def test_pattern_audit_refusals(tmp_path):
    qfile = query_file(tmp_path)
    e4 = write_e4(tmp_path, inputs={f"queries/{REGIME}": qfile})
    uids = [q[0] for q in query_list("test", E4_FAMILIES)]

    def audit(name: str, **kwargs) -> str:
        return write_audit(tmp_path, kwargs.pop("qfile", qfile), name=name, **kwargs)

    def entry(**changes):
        return lambda doc: {**doc, "regimes": {REGIME: {**doc["regimes"][REGIME], **changes}}}

    def refused_with(match: str, *audits: str, runs=(e4,), **flags) -> None:
        refused(tmp_path, match, e4=list(runs), pattern_audit=list(audits), **flags)

    # the audit and its manifest
    refused_with("given as pattern_audit", write_e4(tmp_path, name="not-an-audit"))
    refused_with("not a complete", audit("incomplete", mutate=lambda d: {**d, "complete": False}))
    refused_with("not a complete", audit("schema", mutate=lambda d: {**d, "schema": "jigsaw-pattern-audit-v0"}))
    refused_with("different splits or protocol hashes", audit("doc-split", mutate=lambda d: {**d, "split": "dev"}))
    refused_with("unknown dataset 'cora_x'", audit("dataset", mutate=lambda d: {**d, "dataset": "cora_x"}))
    hashed = audit("protocol", mutate=lambda d: {**d, "protocol_hash": "0" * 64})
    path = Path(hashed) / manifest.MANIFEST_NAME
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "protocol_hash": "0" * 64}),
                    encoding="utf-8")
    refused_with("different protocol hashes", hashed)
    refused_with("an audit of split 'dev'", audit("dev", split="dev"))
    refused_with("is also audited by", audit("first"), audit("second"))
    # its encoders and seen queries
    refused_with("a test split needs the main encoders of every training seed", audit("seeds", seeds=SEEDS))
    refused_with("a test split needs the main encoders of every training seed", audit("smoke", smoke=True))
    refused_with("no entry in encoders", audit("undescribed", mutate=lambda d: {**d, "encoders": {}}))
    other = rev.main_model(DATASET, REGIME, 7399).model_id
    refused_with("names another encoder",
                 audit("seen-other", mutate=entry(seen={uids[0]: {other: {"first": [1, 0, 0], "hits": 1}}})))
    refused_with("is not in the set or names", audit("seen-outside", mutate=entry(uids=uids[1:])))
    # the E4 runs it splits
    refused_with("ran query set", audit("other-set", qfile=query_file(tmp_path, "R0-other.json", "another set")))
    refused_with("records no sha256", audit("unhashed"), runs=[write_e4(tmp_path, name="e4-unhashed")])
    refused_with("1 queries are not in the set", audit("fewer", seen=(4, 9), mutate=entry(uids=uids[1:])))
    dev_file = query_file(tmp_path, "R0-dev.json", "the dev query set")
    dev = write_e4(tmp_path, name="e4-dev", split="dev", inputs={f"queries/{REGIME}": dev_file})
    partial = write_audit(tmp_path, dev_file, name="dev-7301", split="dev", seeds=(7301,))
    refused_with("seed7302-main, which .* did not audit", partial, runs=[dev], allow_dev=True)


# ----------------------------------------------------------------------------
# Merging, layerwise, output format
# ----------------------------------------------------------------------------
def test_runs_of_one_scope_merge_and_an_arm_in_two_runs_is_refused(tmp_path):
    arms = e4_arms()
    queries = query_list("test", E4_FAMILIES)
    a = write_run(tmp_path / "e4-a", "E4", "test", {k: v for k, v in arms.items() if k.policy == "hybrid"}, queries)
    b = write_run(tmp_path / "e4-b", "E4", "test", {k: v for k, v in arms.items() if k.policy == "topk"}, queries)
    figures = tmp_path / "numbers" / "figures"
    figures.mkdir(parents=True)
    (figures / "old-figure.csv").write_text("series,x,y,lo,hi\n", encoding="utf-8")
    summary, doc, _ = reduce_to(tmp_path, e4=[b, a])
    assert summary["sources"] == ["E4.arxiv.run1", "E4.arxiv.run2"]
    assert doc["sources"]["E4.arxiv.run1"]["path"] == Path(a).as_posix()
    assert doc["sources"]["E4.arxiv.run1"]["manifest_sha256"] == nj.file_sha256(Path(a) / manifest.MANIFEST_NAME)
    assert doc["numbers"]["E4.arxiv.planted.R0.hybrid.index.n"]["source"] == ["E4.arxiv.run1"]
    assert doc["numbers"]["E4.arxiv.planted.R0.topk.index.n"]["source"] == ["E4.arxiv.run2"]
    assert any("old-figure" in w for w in summary["warnings"])
    index_only = {k: v for k, v in arms.items() if k.policy == "hybrid" and k.ranker == "index"}
    c = write_run(tmp_path / "e4-c", "E4", "test", index_only, queries)
    refused(tmp_path, "is in both", e4=[a, b, c])


def test_layerwise_tables_and_curves(tmp_path):
    summary, doc, out = reduce_to(tmp_path, layerwise=[write_layerwise(tmp_path)])
    assert summary["split"] == doc["split"] == oversmoothing.SPLIT
    num, tables = doc["numbers"], doc["tables"]
    base = "E6_layerwise.arxiv.R0"
    assert num[f"{base}.main.seeds"]["value"] == 2
    key = f"{base}.main.layer-1.mean_average_distance"  # layer "1" is the third entry: 0.2, 0.3 over the seeds
    assert num[f"{key}.mean"]["value"] == pytest.approx(statistics.mean([0.2, 0.3]))
    assert num[f"{key}.sd"]["value"] == pytest.approx(statistics.stdev([0.2, 0.3]))
    assert num[f"{key}.mean"]["n"] == num[f"{key}.sd"]["n"] == 2
    assert num[f"{base}.main_knockout.layer-1.mean_average_distance.mean"]["value"] == pytest.approx(0.45)
    assert f"{base}.main.layer-readout.mean_average_distance.mean" not in num
    assert f"{base}.arch-gin-L2-res.layer-1.mean_average_distance.sd" not in num  # one seed: no SD
    metrics = oversmoothing.metric_names()
    table = tables[f"{base}.main"]
    assert table["columns"] == ["Layer", "Width", *[f"{m} {s}" for m in metrics for s in ("mean", "SD")]]
    assert [row[0] for row in table["rows"]] == oversmoothing.layer_names(2)
    readout = table["rows"][-1]
    assert readout[table["columns"].index("mean_average_distance mean")] == "n/a"
    assert readout[table["columns"].index("block_separation SD")] == {
        "key": f"{base}.main.layer-readout.block_separation.sd"}
    gin = tables[f"{base}.arch-gin-L2-res"]
    assert gin["rows"][2][gin["columns"].index("mean_average_distance SD")] == "n/a"
    assert "not a model trained without it" in tables[f"{base}.main_knockout"]["notes"]
    assert list(tables).index(f"{base}.main") < list(tables).index(f"{base}.main_knockout")
    rows = read_csv(out / "figures" / f"{base}.block_separation.csv")
    assert {r["series"] for r in rows} == {"main", "main+knockout", "arch-graphsage-L2-nores", "arch-gin-L2-res"}
    main = [r for r in rows if r["series"] == "main"]
    assert [float(r["x"]) for r in main] == [-1, 0, 1, 2, 3]
    for i, r in enumerate(main):
        assert float(r["y"]) == pytest.approx(statistics.mean([0.5 + 0.01 * i, 0.6 + 0.01 * i]))
        assert r["lo"] == r["hi"] == ""
    nodes = [r for r in read_csv(out / "figures" / f"{base}.mean_average_distance.csv") if r["series"] == "main"]
    assert [float(r["x"]) for r in nodes] == [-1, 0, 1, 2]  # no node metrics at readout
    assert "log_x" not in doc["figures"][f"{base}.block_separation"]
    build_paper.load_input(out / "numbers.json")


def test_written_numbers_load_in_the_paper_builder(tmp_path):
    runs = {"E4.arxiv": write_e4(tmp_path), "E5.arxiv": write_e5(tmp_path, dup="weights"),
            "E6.arxiv": write_e6(tmp_path), "E6_layerwise.arxiv": write_layerwise(tmp_path)}
    summary, doc, out = reduce_to(tmp_path, e4=[runs["E4.arxiv"]], e5=[runs["E5.arxiv"]], e6=[runs["E6.arxiv"]],
                                  layerwise=[runs["E6_layerwise.arxiv"]])
    loaded = build_paper.load_input(out / "numbers.json")
    assert loaded.data["split"] == "test" and doc["protocol_hash"] == protocol.protocol_hash()
    assert doc["frozen"] is (protocol.FROZEN_PROTOCOL_HASH == protocol.protocol_hash())
    assert set(doc["sources"]) == set(runs) == set(summary["sources"])
    for run_id, path in runs.items():
        assert doc["sources"][run_id]["manifest_sha256"] == nj.file_sha256(Path(path) / manifest.MANIFEST_NAME)
    assert set(doc["figures"]) == {p.stem for p in (out / "figures").glob("*.csv")} == set(loaded.rows)
    # every number but the text key carries n, and names its sources
    assert [k for k, e in doc["numbers"].items() if e["n"] is None] == ["PE6.arxiv.control_arm"]
    assert all(e["source"] for e in doc["numbers"].values())
    # every key, table id and figure id is one of the patterns the module docstring defines
    scope, sub, reg, pol, arm = r"arxiv", r"(planted|sm)", r"R0", r"(hybrid|topk)", r"[A-Za-z0-9_-]+"
    desc = r"(n|errors|seeds|fullcov\.K\d+|cost\.(mean|median)|seed\d+\.(fullcov\.K\d+|cost\.mean))"
    test = r"(n|cost\.(gm_ratio|excluded|p|p_holm)|fullcov\.K\d+\.(diff|p|p_holm))"
    key_patterns = [
        rf"E4\.{scope}\.{sub}\.{reg}\.{pol}\.[a-z_]+\.({desc}|cost\.gm_ratio_p1_vs_index)",
        rf"PE5\.{scope}\.{sub}\.{reg}\.[a-z_]+_vs_(index|fused_learned_mean)\.{test}",
        rf"PE6\.{scope}\.control_arm", rf"PE6\.{scope}\.{sub}\.{reg}\.{pol}\.{test}",
        rf"E[56]\.{scope}\.{sub}\.{reg}\.{pol}\.arm\.{arm}\.{desc}",
        rf"E[56]\.{scope}\.{sub}\.{reg}\.{pol}\.vs_main\.{arm}\.{test}",
        rf"E6_layerwise\.{scope}\.{reg}\.{arm}\.(seeds|layer-(input|proj|\d+|readout)\.[a-z_]+\.(mean|sd))"]
    table_patterns = [rf"E4\.{scope}\.{sub}\.{reg}\.{pol}\.(fullcov|cost|seeds)", rf"PE[56]\.{scope}\.{sub}\.(cost|fullcov)",
                      rf"E[56]\.{scope}\.{sub}\.{reg}\.{pol}\.(arms|seeds)", rf"E[56]\.{scope}\.{sub}\.(cost|fullcov)",
                      rf"E6_layerwise\.{scope}\.{reg}\.{arm}"]
    figure_patterns = [rf"E[456]\.{scope}\.{sub}\.{reg}\.{pol}\.fullcov_curve",
                       rf"E6_layerwise\.{scope}\.{reg}\.[a-z_]+"]
    for ids, patterns in ((doc["numbers"], key_patterns), (doc["tables"], table_patterns),
                          (doc["figures"], figure_patterns)):
        assert [i for i in ids if not any(re.fullmatch(p, i) for p in patterns)] == []
    assert {f"E6_layerwise.arxiv.R0.{m}" for m in oversmoothing.metric_names()} <= set(doc["figures"])


def test_frozen_needs_the_frozen_hash_and_every_input_run_under_it(tmp_path, monkeypatch):
    before = write_e4(tmp_path, name="e4-before")  # run while FROZEN_PROTOCOL_HASH was unset
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", protocol.protocol_hash())
    assert protocol.is_frozen()
    e4, e5 = write_e4(tmp_path), write_e5(tmp_path)  # their manifests record protocol_frozen true
    summary, doc, out = reduce_to(tmp_path, e4=[e4], e5=[e5])
    assert summary["frozen"] is doc["frozen"] is True
    assert not [w for w in summary["warnings"] if "frozen" in w]
    assert build_paper.dev_reasons([build_paper.load_input(out / "numbers.json")]) == []
    mixed = rr.reduce(e4=[before], e5=[e5], out=str(tmp_path / "numbers-mixed"))
    assert mixed["frozen"] is nj.load(tmp_path / "numbers-mixed" / "numbers.json")["frozen"] is False
    assert [w for w in mixed["warnings"] if "before the protocol was frozen" in w and before in w]
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", "f" * 64)  # the runs claim frozen off the frozen hash
    other = rr.reduce(e4=[e4], e5=[e5], out=str(tmp_path / "numbers-other"))
    assert other["frozen"] is False
    assert not [w for w in other["warnings"] if "before the protocol was frozen" in w]


def test_cli_prints_a_summary_and_reports_families(tmp_path, capsys):
    e4 = write_e4(tmp_path)
    out = tmp_path / "numbers"
    assert rr.main(["--e4", e4, "--out", str(out), "--by-family"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["split"] == "test" and summary["sources"] == ["E4.arxiv"] and summary["numbers"] > 0
    num = nj.load(out / "numbers.json")["numbers"]
    assert num["E4.arxiv.family-single.R0.hybrid.index.n"]["value"] == 4
    assert num["E4.arxiv.family-sm_any.R0.hybrid.index.n"]["value"] == 2
    assert not [k for k in num if k.startswith("PE5.arxiv.family-")]
    args = rr.build_parser().parse_args(["--e4", "a", "b", "--e4", "c", "--out", "o"])
    assert args.e4 == ["a", "b", "c"] and args.e5 == [] and not args.allow_dev


def test_key_segments_and_layer_positions():
    assert rr.segment(CONTROL) == "grid-mean-barrier0_35-live64"
    assert rr.segment("main+knockout") == "main_knockout"
    assert rr.scope_name("arxiv", PRIMARY) == "arxiv"
    assert rr.scope_name("arxiv", "fine.2x") == "arxiv-part-fine_2x"
    assert [rr.layer_x(name, 3) for name in oversmoothing.layer_names(3)] == [-1, 0, 1, 2, 3, 4]
    with pytest.raises(SystemExit):
        rr.layer_x("4", 3)


# ----------------------------------------------------------------------------
# Imports and the trainer's naming
# ----------------------------------------------------------------------------
def test_the_reducer_imports_no_torch():
    code = "import sys, scripts.canonical.reduce_retrieval; print('torch' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_the_audit_names_follow_pattern_audit():
    """pattern_audit.py imports torch through train_encoder, so its names are read from its source."""
    tree = ast.parse((REPO_ROOT / "scripts" / "canonical" / "pattern_audit.py").read_text(encoding="utf-8"))
    names = {t.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Constant) for t in node.targets if isinstance(t, ast.Name)}
    assert (rr.AUDIT_SCHEMA, rr.AUDIT_EXPERIMENT, rr.AUDIT_FILE) == \
        (names["SCHEMA"], names["EXPERIMENT"], names["OUTPUT"])


def _functions(path: Path, names: set[str]) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}


def _body(node: ast.FunctionDef) -> list[str]:
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # the docstring
    return [ast.dump(stmt) for stmt in body]


def trainer_namespace() -> dict[str, Any]:
    """train_encoder's naming, arm enumeration and loss resolution, executed from its source without torch."""
    tree = ast.parse((REPO_ROOT / "scripts" / "canonical" / "train_encoder.py").read_text(encoding="utf-8"))
    functions = {"_format_value", "grid_arm_name", "arch_arm_name", "arms", "_per_dataset", "loss_config"}
    constants = {"FACTOR_LABELS", "LOSS_DEFAULTS", "MAIN_ARM"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in constants for t in targets):
                body.append(node)
    code = compile(ast.Module(body=body, type_ignores=[]), "train_encoder.py", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    namespace: dict[str, Any] = {"protocol": protocol, "itertools": itertools, "Mapping": typing.Mapping,
                                 "Any": typing.Any}
    exec(code, namespace)
    assert functions | constants <= set(namespace)
    return namespace


def test_arm_names_and_the_pe6_control_follow_the_trainer():
    names = {"_format_value", "grid_arm_name", "arch_arm_name"}
    trainer = _functions(REPO_ROOT / "scripts" / "canonical" / "train_encoder.py", names)
    reducer = _functions(Path(rr.__file__), names)
    assert set(trainer) == set(reducer) == names
    for name in names:
        assert _body(reducer[name]) == _body(trainer[name]), name
    ns = trainer_namespace()
    assert rr.FACTOR_LABELS == ns["FACTOR_LABELS"] and rev.MAIN_ARM == ns["MAIN_ARM"]
    for dataset in ("arxiv", "mag"):
        trainer_arms = ns["arms"](dataset)
        factorial = {name: e["loss"] for name, e in trainer_arms.items() if e["kind"] == "factorial"}
        assert list(rr.factorial_arms(dataset).items()) == list(factorial.items())
        assert rr.architecture_arms(dataset) == [n for n, e in trainer_arms.items() if e["kind"] == "architecture"]
        main = ns["loss_config"](dataset)
        controls = []
        for name, overrides in factorial.items():
            loss = ns["loss_config"](dataset, overrides)
            if {k for k in main if loss[k] != main[k]} == {"coverage_aggregation"} and loss["coverage_aggregation"] == "mean":
                controls.append(name)
        assert controls == [rr.pe6_control(dataset)]
