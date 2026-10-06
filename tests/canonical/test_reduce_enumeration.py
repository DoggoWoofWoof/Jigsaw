"""Tests for the E15 reducer (scripts/canonical/reduce_enumeration.py) on synthetic enumeration runs."""

from __future__ import annotations

import json

import pytest

from scripts.canonical import enumeration, manifest, numbers_json, protocol
from scripts.canonical import reduce_enumeration as re15
from scripts.canonical import reducer as rd

JIG = f"jigsaw@{protocol.SEEDS['training'][0]}"
IDX = "jigsaw_index"
BUDGETS = (1, 5, 10)


@pytest.fixture(autouse=True)
def _few_draws(monkeypatch):
    monkeypatch.setattr(re15, "DRAWS", {"boot": 400, "quantile": 200})


def rec(uid, arm, status, embeddings, recalls, *, planted=("kept", "kept", "kept"), planted_found=True,
        seconds=1.0, unsound=0):
    steps = [{"step": i, "budget": b, "final": i == len(BUDGETS) - 1, "closed": i == len(BUDGETS) - 1,
              "candidate_nodes": 10 * (i + 1), "targets": 1, "build_seconds": 0.01 * (i + 1),
              "located": {"kept": round(r * embeddings), "outside": embeddings - round(r * embeddings), "capped": 0,
                          "unsound": unsound if i == 0 else 0},
              "recall": r, "planted": p}
             for i, (b, r, p) in enumerate(zip(BUDGETS, recalls, planted))]
    out = {"schema": re15.SCHEMA, "uid": uid, "family": "k_hop", "size": 5, "kind": "positive", "regime": "R0",
           "method": arm.split("@")[0], "dataset": "corafull", "certified": None, "limit": 5, "T": 30.0,
           "status": status, "sound": unsound == 0 and not (status == "complete" and planted_found is False),
           "seconds": seconds + 0.5,
           "global": {"status": status, "seconds": seconds, "targets": 1, "embeddings": embeddings,
                      "candidate_nodes": 40},
           "steps": steps, "arm": arm, "backend": "networkx"}
    if planted_found is not None:
        out["planted_found"] = planted_found
    return out


def queries_for(arm, **overrides):
    recs = [rec("u0", arm, "complete", 3, (1 / 3, 2 / 3, 1.0), planted=("outside", "kept", "kept")),
            rec("u1", arm, "complete", 2, (1.0, 1.0, 1.0), seconds=3.0),
            rec("u2", arm, "partial", 5, (0.2, 0.6, 1.0), seconds=30.0),
            rec("u3", arm, "partial", 1, (0.0, 1.0, 1.0), seconds=30.0)]
    for uid, change in overrides.items():
        recs = [change(r) if r["uid"] == uid else r for r in recs]
    return recs


def make_run(root, name, recs_by_arm, *, split="dev", shard=None, config=None, qsha="a" * 64):
    run = root / name
    for arm, recs in recs_by_arm.items():
        path = run / "records" / "R0" / rd.record_file(arm, shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    cfg = {"schema": re15.SCHEMA, "experiment": "E15", "dataset": "corafull", "partition": "primary",
           "regimes": ["R0"], "arms": list(recs_by_arm), "limit": 5, "T": 30.0, "positives": 4,
           "solve_steps": False, "budgets": None, "router_tau": None, "families": None, "sizes": None,
           "max_queries": None, "shard": list(shard) if shard else None,
           "query_sets": {"R0": {"path": "q/R0.npz", "split": split, "master": "20260607",
                                 "selected": len(next(iter(recs_by_arm.values())))}}}
    cfg.update(config or {})
    doc = {"schema": "jigsaw-canonical-manifest-v1", "experiment": "E15", "split": split, "status": "complete",
           "protocol_hash": protocol.protocol_hash(), "protocol_frozen": False, "config": cfg,
           "inputs": {"queries/R0": {"sha256": qsha}}, "outputs": manifest.output_hashes(run)}
    (run / manifest.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")
    return run


DEV = rd.Flags(allow_dev=True)


def test_constants_match_the_enumeration_module():
    assert re15.SCHEMA == enumeration.SCHEMA
    assert re15.KEPT == enumeration.KEPT
    assert re15.plan_arms() == enumeration.plan_arms()


def test_recall_per_step_counts_and_outputs(tmp_path):
    run = make_run(tmp_path, "E15", {JIG: queries_for(JIG), IDX: queries_for(IDX)})
    out = tmp_path / "out"
    assert re15.execute([str(run)], out, DEV, argv=["t"]) == 0
    data = numbers_json.load(out / "numbers.json")
    n = data["numbers"]
    k = "E15.corafull.R0.jigsaw-at-7301"
    assert data["split"] == "dev" and data["frozen"] is False
    assert {m: n[f"{k}.{m}"]["value"] for m in ("queries", "complete", "partial_limit", "partial_time", "errors")} \
        == {"queries": 4, "complete": 2, "partial_limit": 1, "partial_time": 1, "errors": 0}
    assert n[f"{k}.complete_rate"]["value"] == pytest.approx(50.0)
    assert n[f"{k}.k1.recall"]["value"] == pytest.approx(100 * (1 / 3 + 1) / 2)
    assert n[f"{k}.k1.recall"]["n"] == 2
    lo, hi = n[f"{k}.k1.recall"]["ci"]
    assert lo <= n[f"{k}.k1.recall"]["value"] <= hi
    assert n[f"{k}.k5.recall"]["value"] == pytest.approx(100 * (2 / 3 + 1) / 2)
    assert n[f"{k}.k1.full_recall"]["value"] == pytest.approx(50.0)
    assert n[f"{k}.k1.planted_kept"]["value"] == pytest.approx(75.0) and n[f"{k}.k1.planted_kept"]["n"] == 4
    assert n[f"{k}.k1.partial_recall"]["value"] == pytest.approx(10.0)
    assert n[f"{k}.final.recall"]["value"] == pytest.approx(100.0)
    assert n[f"{k}.final.recall"]["ci"] == pytest.approx([100.0, 100.0])
    assert n[f"{k}.embeddings_median"]["value"] == pytest.approx(2.5)
    assert n[f"{k}.seconds.median"]["value"] == pytest.approx(16.5)
    assert n[f"{k}.k10.build_seconds_median"]["value"] == pytest.approx(0.03)
    assert all(n[f"{k}.violations.{v}"]["value"] == 0 for v in re15.VIOLATIONS)
    assert "E15.corafull.R0.jigsaw_index.k5.recall" in n
    assert n["E15.holds"]["value"] == "yes"
    assert set(data["tables"]) == {"E15.corafull.R0", "E15.summary"}
    assert len(data["tables"]["E15.corafull.R0"]["rows"]) == 2 * len(BUDGETS)
    assert data["figures"]["E15.corafull.R0.recall"]["log_x"] is True
    assert (out / "tables" / "E15.summary.csv").read_text(encoding="utf-8").startswith("Dataset,Regime,System")
    assert manifest.verify_manifest(out) == []
    assert json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))["experiment"] == "reduce-E15"


def test_violations_fail_the_reduction_unless_allowed(tmp_path):
    def lose_planted(r):
        return {**r, "planted_found": False, "sound": False}

    def lose_by_pruning(r):
        r = json.loads(json.dumps(r))
        r["steps"][0]["located"]["unsound"] = 1
        r["sound"] = False
        return r

    def crash(r):
        return {k: v for k, v in r.items() if k not in ("global", "limit", "T")} | {
            "status": "error", "sound": False, "error": "KeyError: x", "traceback": "...", "steps": []}

    recs = {JIG: queries_for(JIG, u0=lose_planted, u2=lose_by_pruning, u3=crash), IDX: queries_for(IDX)}
    run = make_run(tmp_path, "E15", recs)
    assert re15.execute([str(run)], tmp_path / "out", DEV, argv=["t"]) == 3
    report = json.loads((tmp_path / "out" / "violations.json").read_text(encoding="utf-8"))
    assert report["holds"] is False
    assert sorted((v["uid"], v["kind"]) for v in report["violations"]) == [
        ("u0", "planted_missing"), ("u2", "lost_by_pruning"), ("u3", "crash")]
    n = numbers_json.load(tmp_path / "out" / "numbers.json")["numbers"]
    assert n["E15.holds"]["value"] == "no" and n["E15.corafull.R0.jigsaw-at-7301.errors"]["value"] == 1
    allowed = rd.Flags(allow_dev=True, allow_pe1_failure=True)
    assert re15.execute([str(run)], tmp_path / "out2", allowed, argv=["t"]) == 0


def test_shards_merge_and_must_be_complete(tmp_path):
    a = {arm: queries_for(arm)[:2] for arm in (JIG, IDX)}
    b = {arm: queries_for(arm)[2:] for arm in (JIG, IDX)}
    s0 = make_run(tmp_path, "s0", a, shard=(0, 2))
    s1 = make_run(tmp_path, "s1", b, shard=(1, 2))
    assert re15.execute([str(tmp_path / "s*")], tmp_path / "out", DEV, argv=["t"]) == 0
    n = numbers_json.load(tmp_path / "out" / "numbers.json")["numbers"]
    assert n["E15.corafull.R0.jigsaw-at-7301.queries"]["value"] == 4
    with pytest.raises(rd.ReduceError, match="every shard 0..n-1"):
        re15.execute([str(s0)], tmp_path / "o1", DEV)
    make_run(tmp_path / "other", "s1", b, shard=(1, 2), qsha="b" * 64)
    with pytest.raises(rd.ReduceError, match="different query sets"):
        re15.execute([str(s0), str(tmp_path / "other" / "s1")], tmp_path / "o2", DEV)
    assert s1.exists()


def test_refusals(tmp_path):
    good = make_run(tmp_path, "good", {JIG: queries_for(JIG), IDX: queries_for(IDX)})
    with pytest.raises(rd.ReduceError, match="--allow-dev"):
        re15.execute([str(good)], tmp_path / "o0", rd.Flags())
    with pytest.raises(rd.ReduceError, match="inside the run directory"):
        re15.execute([str(good)], good / "out", DEV)
    sealed = make_run(tmp_path, "sealed", {JIG: queries_for(JIG)}, split="test", config={"max_queries": 4})
    with pytest.raises(rd.ReduceError, match="pre-registered run; not allowed: max_queries; limit 5"):
        re15.execute([str(sealed)], tmp_path / "o1", rd.Flags())
    short = make_run(tmp_path, "short", {JIG: queries_for(JIG)[:3], IDX: queries_for(IDX)[:3]},
                     config={"query_sets": {"R0": {"split": "dev", "selected": 4}}})
    with pytest.raises(rd.ReduceError, match="incomplete run is resumed"):
        re15.execute([str(short)], tmp_path / "o2", DEV)
    uneven = make_run(tmp_path, "uneven", {JIG: queries_for(JIG), IDX: queries_for(IDX)[:3] + [
        {**queries_for(IDX)[3], "uid": "u9"}]})
    with pytest.raises(rd.ReduceError, match="arms ran different queries"):
        re15.execute([str(uneven)], tmp_path / "o3", DEV)
    wrong = make_run(tmp_path, "wrong", {JIG: [{**r, "limit": 7} for r in queries_for(JIG)]})
    with pytest.raises(rd.ReduceError, match="ran with limit 7"):
        re15.execute([str(wrong)], tmp_path / "o4", DEV)
    ladders = make_run(tmp_path, "ladders", {JIG: queries_for(JIG, u1=lambda r: {**r, "steps": r["steps"][:2]})})
    with pytest.raises(rd.ReduceError, match="different ladders"):
        re15.execute([str(ladders)], tmp_path / "o5", DEV)
    (good / "records" / "R0" / f"{IDX}.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(rd.ReduceError, match="differs from its manifest"):
        re15.execute([str(good)], tmp_path / "o6", DEV)
    with pytest.raises(rd.ReduceError, match="0 records for 4 selected"):
        re15.execute([str(good)], tmp_path / "o7", rd.Flags(allow_dev=True, no_verify=True))


def test_cli_exit_codes(tmp_path, capsys):
    run = make_run(tmp_path, "E15", {JIG: queries_for(JIG), IDX: queries_for(IDX)})
    assert re15.main(["--runs", str(run), "--out", str(tmp_path / "out")]) == 2
    assert "--allow-dev" in capsys.readouterr().err
    assert re15.main(["--runs", str(run), "--out", str(tmp_path / "out"), "--allow-dev"]) == 0
    assert json.loads(capsys.readouterr().out)["exit"] == 0
