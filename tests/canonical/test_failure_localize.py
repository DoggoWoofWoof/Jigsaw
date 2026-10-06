"""failure_localize (E18): every category from hand-built trails, then on real cascade records."""

from __future__ import annotations

import math
import shutil

import pytest

from scripts.canonical import backends, cascade, protocol, queries, rankers, stats
from scripts.canonical import failure_localize as fl
from scripts.canonical import store as store_mod

N = 8  # nodes of the hand-built queries
REAL_T = 2.0


def where(in_candidate: int = N, alive: int | None = N, component: str | None = "kept", target: int | None = 0):
    """A prune.locate result for the planted witness."""
    return {"nodes": N, "in_candidate": in_candidate, "alive": alive, "component": component, "target": target}


def outside(in_candidate: int = 3):
    """A witness only partly inside the candidate: pruning is never asked about it."""
    return where(in_candidate=in_candidate, alive=None, component=None, target=None)


def call(target: int, t: float, seconds: float, status: str, **extra):
    return {"target": target, "t": t, "seconds": seconds, "status": status, **extra}


def step(i: int, t_start: float, *, witness=None, calls=None, closed: bool = False, final: bool = False,
         status: str = "TIMEOUT", **extra):
    s = {"step": i, "t_start": t_start, "t_end": t_start + 1.0, "closed": closed, "final": final, "status": status,
         **extra}
    if witness is not None:
        s["witness"] = witness
    if calls is not None:
        s["solver_calls"] = calls
    return s


def record(kind: str = "positive", status: str = "TIMEOUT", *, steps=(), T: float = 30.0, elapsed: float | None = None,
           method: str = "jigsaw_index", **extra):
    rec = {"uid": "q", "kind": kind, "status": status, "T": T, "elapsed": T if elapsed is None else elapsed,
           "method": method, "arm": method, "dataset": "synth", "regime": "R0", "steps": list(steps),
           "time_found": None, "time_decided": None, "decided_step": None,
           "certified": "planted" if kind == "positive" else "none"}
    rec.update(extra)
    return rec


def found(t: float) -> dict:
    """The fields of a FOUND verified at t."""
    return {"status": "FOUND", "witness_check": "ok", "time_found": t, "time_decided": t, "elapsed": t}


# ----------------------------------------------------------------------------
# Positives
# ----------------------------------------------------------------------------
def test_a_positive_is_solved_from_time_found_on():
    s0 = step(0, 0.0, witness=where(), calls=[call(0, 1.0, 1.9, "FOUND")], status="FOUND")
    rec = record(**found(3.0), steps=[s0], decided_step=0)
    assert fl.classify(rec, 30.0) == ("solved", {"time_found": 3.0, "overlap_rescued": False})
    assert fl.classify(rec, 3.0)[0] == "solved"
    assert fl.classify(rec, 2.95) == ("solver_budget", {"step": 0, "target": 0, "found_after_call": True,
                                                        "overlap_rescued": False})
    assert fl.classify(rec, 2.0) == ("solver_budget", {"step": 0, "target": 0, "running": True,
                                                       "overlap_rescued": False})
    assert fl.classify(rec, 0.5) == ("solver_queue", {"step": 0, "target": 0, "overlap_rescued": False})
    # before any step started: no candidate has held the witness yet, and a later one will
    assert fl.classify(rec, 0.0) == ("retrieval", {"max_covered": 0.0, "steps_started": 0, "coverage": "late",
                                                   "overlap_rescued": False})


def test_soundness_failures_hold_at_every_horizon():
    rejected = record(status="ERROR", error="witness rejected: edge (1, 2) missing", elapsed=2.0)
    crash = record(status="ERROR", error="RuntimeError: boom", elapsed=2.0)
    unverified = record(status="FOUND", witness_check="label mismatch", elapsed=2.0)
    for t in (0.0, 1.0, 30.0):
        assert fl.classify(rejected, t)[0] == "unsound_witness"
        assert fl.classify(crash, t)[0] == "error"
        assert fl.classify(unverified, t)[0] == "unsound_witness"
    steps = [step(0, 0.0), step(1, 2.0, closed=True, status="NONE")]
    proof = record(status="NONE", time_decided=4.0, elapsed=4.0, decided_step=1, steps=steps)
    assert fl.classify(proof, 4.0) == ("unsound_none", {"certificate": None, "decided_step": 1,
                                                        "overlap_rescued": False})
    assert fl.classify(proof, 3.9)[0] == "retrieval"  # not proved yet, and no step held the witness


def test_pruning_and_component_losses_are_unsound():
    assert fl.classify(record(steps=[step(0, 0.0, witness=where(alive=7, component=None, target=None))]), 30.0) == \
        ("unsound_prune", {"step": 0, "overlap_rescued": False})
    # alive None with the whole witness inside: pruning never ran on a candidate that held it
    never_pruned = record(steps=[step(0, 0.0, witness=where(alive=None, component=None, target=None))])
    assert fl.classify(never_pruned, 30.0)[0] == "unsound_prune"
    for component in ("filtered", "split"):
        rec = record(steps=[step(0, 0.0, witness=where(component=component, target=None))])
        assert fl.classify(rec, 30.0) == ("unsound_component", {"step": 0, "overlap_rescued": False})
    # a loss anywhere in S_t outranks a later step that reached the solver
    later = step(1, 2.0, witness=where(), calls=[call(0, 2.5, 1.0, "TIMEOUT", killed_at_cap=True)])
    both = record(steps=[step(0, 0.0, witness=where(alive=7, component=None, target=None)), later])
    assert fl.classify(both, 30.0)[0] == "unsound_prune"
    assert fl.classify(both, 1.0)[0] == "unsound_prune"


def test_solver_categories_read_the_call_on_the_witness_target():
    kept1 = where(target=1)
    calls = [call(0, 0.2, 0.3, "NONE"), call(1, 0.5, 2.0, "TIMEOUT", killed_at_cap=True)]
    open_step = record(steps=[step(0, 0.0, witness=kept1, calls=calls)])
    assert fl.classify(open_step, 30.0) == ("solver_step_cap", {"step": 0, "target": 1, "overlap_rescued": False})
    assert fl.classify(open_step, 1.0) == ("solver_budget", {"step": 0, "target": 1, "running": True,
                                                             "overlap_rescued": False})
    assert fl.classify(open_step, 0.4)[0] == "solver_queue"  # target 0 still ran: the witness's call had not begun
    assert fl.classify(record(steps=[step(0, 0.0, witness=kept1, calls=calls, closed=True)]), 30.0)[0] == \
        "solver_budget"
    final = record(steps=[step(0, 0.0, witness=kept1, calls=[call(1, 0.5, 29.0, "TIMEOUT")], final=True)])
    assert fl.classify(final, 30.0)[0] == "solver_budget"
    wrong = record(steps=[step(0, 0.0, witness=kept1, calls=[call(0, 0.1, 0.1, "NONE"), call(1, 0.2, 0.1, "NONE")],
                               status="NONE")])
    assert fl.classify(wrong, 30.0) == ("unsound_solver", {"step": 0, "target": 1, "overlap_rescued": False})
    # "unfiltered" (no component step) answers through target 0
    unfiltered = record(steps=[step(0, 0.0, witness=where(component="unfiltered", target=None),
                                    calls=[call(0, 0.1, 5.0, "TIMEOUT")], closed=True)])
    assert fl.classify(unfiltered, 30.0) == ("solver_budget", {"step": 0, "target": 0, "overlap_rescued": False})
    odd = record(steps=[step(0, 0.0, witness=where(), calls=[call(0, 0.1, 0.1, "ERROR")])])
    assert fl.classify(odd, 30.0) == ("error", {"step": 0, "target": 0, "call_status": "ERROR",
                                                "overlap_rescued": False})


def test_solver_queue_covers_calls_that_never_ran():
    not_started = record(steps=[step(0, 0.0, witness=where(),
                                     calls=[{"target": 0, "status": fl.NOT_STARTED, "t": 1.0}])])
    missing = record(steps=[step(0, 0.0, witness=where(target=2),
                                 calls=[call(0, 0.1, 1.0, "NONE"), call(1, 1.1, 0.9, "TIMEOUT", killed_at_cap=True)])])
    late = record(steps=[step(0, 0.0, witness=where(), calls=[call(0, 4.0, 1.0, "TIMEOUT")])])
    for rec in (not_started, missing):
        assert fl.classify(rec, 30.0)[0] == "solver_queue"
    assert fl.classify(late, 4.0)[0] == "solver_queue"  # a call starting at t had not started by t
    assert fl.classify(late, 4.5)[0] == "solver_budget"


def test_skipped_steps_inherit_the_trail_and_calls_they_repeat():
    s0 = step(0, 0.0, witness=where(), calls=[call(0, 0.1, 2.0, "TIMEOUT", killed_at_cap=True)])
    s1 = step(1, 2.2, skipped=fl.IDENTICAL_CANDIDATE)
    assert fl.witness_trail([s0, s1])[1][1] is s0["witness"]
    assert fl.classify(record(steps=[s0, s1]), 30.0) == ("solver_step_cap", {"step": 1, "target": 0,
                                                                              "overlap_rescued": False})
    # identical targets: the step keeps its own witness and answers with the calls it repeats
    s1t = step(1, 2.2, witness=where(), skipped="identical targets")
    assert fl.classify(record(steps=[s0, s1t]), 30.0) == ("solver_step_cap", {"step": 1, "target": 0,
                                                                               "overlap_rescued": False})
    # a step that is not a skip answers for itself: no call on its target means the call never ran
    s1n = step(1, 2.2, witness=where(), calls=[])
    assert fl.classify(record(steps=[s0, s1n]), 30.0)[0] == "solver_queue"
    # steps that never started are outside S_t
    rec = record(steps=[step(0, 0.0, witness=outside()), step(1, 5.0, status=fl.NOT_STARTED)])
    assert len(fl.started_steps(rec, 30.0)) == 1 and fl.started_steps(rec, 0.0) == []


def test_component_cap_and_incomplete_positives():
    capped = record("positive", "INCOMPLETE", elapsed=0.5,
                    steps=[step(0, 0.0, witness=where(component="capped", target=None),
                                calls=[call(0, 0.1, 0.1, "NONE")], status="INCOMPLETE")])
    assert fl.classify(capped, 30.0) == ("component_cap", {"step": 0, "overlap_rescued": False})
    incomplete = record("positive", "INCOMPLETE", elapsed=4.0, steps=[step(0, 0.0, witness=outside())])
    assert fl.classify(incomplete, 5.0) == ("component_cap", {"step": None, "overlap_rescued": False})
    assert fl.classify(incomplete, 3.0)[0] == "retrieval"  # it had not returned yet


def test_retrieval_reports_coverage_and_whether_a_later_step_holds_the_witness():
    s0 = step(0, 0.0, witness=outside(3), fullcov=False, homecov=False, covered=3)
    s1 = step(1, 6.0, witness=where(), fullcov=True, homecov=True, covered=N,
              calls=[call(0, 6.5, 1.0, "TIMEOUT", killed_at_cap=True)])
    assert fl.classify(record(steps=[s0, s1]), 5.0) == ("retrieval", {"max_covered": 0.375, "steps_started": 1,
                                                                      "coverage": "late", "overlap_rescued": False})
    assert fl.classify(record(steps=[s0]), 30.0)[1]["coverage"] == "never"
    assert fl.classify(record(steps=[s0, s1]), 30.0)[0] == "solver_step_cap"


def test_overlap_rescued_is_the_first_holding_step_through_the_halo():
    s0 = step(0, 0.0, fullcov=False, homecov=False, covered=5)
    s1 = step(1, 2.0, witness=where(), fullcov=True, homecov=False, covered=N,
              calls=[call(0, 2.1, 0.5, "FOUND")], status="FOUND")
    assert fl.overlap_rescued(record(steps=[s0, s1]))
    assert not fl.overlap_rescued(record(steps=[s0, dict(s1, homecov=True)]))
    assert not fl.overlap_rescued(record(steps=[step(0, 0.0, status=fl.NOT_STARTED), s1]))
    assert fl.classify(record(**found(3.0), steps=[s0, s1]), 30.0)[1]["overlap_rescued"] is True
    assert fl.classify(record(steps=[s0, s1]), 1.0)[1]["overlap_rescued"] is True  # a property of the record


def test_direct_and_external_records():
    direct = record(method="direct", steps=[step(0, 0.0, calls=[call(0, 0.1, 29.9, "TIMEOUT")], closed=True,
                                                 final=True)])
    assert fl.is_direct(direct) and not fl.is_direct(record())
    assert fl.classify(direct, 30.0) == ("solver_budget", {"direct": True, "overlap_rescued": False})
    ext = record("positive", "NONE", method="classical_gql", external={"none_certified": False}, elapsed=4.0)
    assert fl.is_external(ext) and fl.is_external(record(method="gnnpe")) and not fl.is_external(record())
    assert fl.classify(ext, 3.0) == ("solver_budget", {"external": True, "overlap_rescued": False})
    assert fl.classify(ext, 5.0) == ("none_unverified", {"overlap_rescued": False})
    neg = dict(ext, kind="negative", certified="none")
    assert fl.classify(neg, 5.0) == ("incomplete", {"none_unverified": True})
    assert fl.classify(neg, 3.0) == ("closed_step_timeout", {"external": True})
    proved = dict(neg, time_decided=4.0, external={"none_certified": True})
    assert fl.classify(proved, 5.0) == ("closed_none", {"decided_step": None})
    assert fl.classify(dict(proved, kind="positive", certified="planted"), 5.0)[0] == "unsound_none"


# ----------------------------------------------------------------------------
# Negatives
# ----------------------------------------------------------------------------
def test_negative_categories():
    cert = record("negative", "NONE", time_decided=0.01, elapsed=0.01, certificate="global label multiset")
    assert fl.classify(cert, 1.0) == ("certificate", {"certificate": "global label multiset"})
    assert fl.classify(cert, 0.005) == ("no_closed_step", {"steps_started": 0})
    steps = [step(0, 0.0), step(1, 3.0, closed=True, status="NONE")]
    closed = record("negative", "NONE", time_decided=4.0, elapsed=4.0, decided_step=1, steps=steps)
    assert fl.classify(closed, 4.0) == ("closed_none", {"decided_step": 1})
    assert fl.classify(closed, 3.5) == ("closed_step_timeout", {"step": 1, "steps_started": 2})
    assert fl.classify(closed, 2.0) == ("no_closed_step", {"steps_started": 1})
    incomplete = record("negative", "INCOMPLETE", elapsed=4.0, steps=steps)
    assert fl.classify(incomplete, 5.0) == ("incomplete", {"decided_step": None})
    assert fl.classify(incomplete, 3.5)[0] == "closed_step_timeout"
    for t in (0.0, 30.0):  # a verified embedding of a certified-none negative, whenever it came
        assert fl.classify(record("negative", **found(2.0)), t)[0] == "contradiction"
    for label in ("found", "undecided", "uncertified"):
        rec = record("negative", **found(2.0), certified=label, steps=steps[:1])
        assert fl.classify(rec, 2.0) == ("found", {"time_found": 2.0, "certified": label})
        assert fl.classify(rec, 1.0) == ("no_closed_step", {"steps_started": 1})
    rejected = record("negative", "ERROR", error="witness rejected: x")
    crash = record("negative", "ERROR", error="boom")
    unchecked = record("negative", "FOUND", witness_check="missing edge")
    assert [fl.classify(r, 1.0)[0] for r in (rejected, crash, unchecked)] == ["unsound_witness", "error",
                                                                             "unsound_witness"]
    assert fl.UNSOUND >= {"unsound_witness", "contradiction"} and "found" not in fl.UNSOUND


def test_horizons_are_checked():
    rec = record(T=5.0)
    with pytest.raises(ValueError, match="beyond the record's T"):
        fl.classify(rec, 6.0)
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="bad horizon"):
            fl.classify(rec, bad)
    assert fl.check_horizon(rec, 5.0 + 1e-12) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="on a negative"):
        fl.classify_positive(record("negative"), 1.0)
    with pytest.raises(ValueError, match="on a positive"):
        fl.classify_negative(record(), 1.0)


def test_summarize_counts_shares_and_skips_records_shorter_than_the_horizon():
    recs = [record(**found(1.0), arm="a"), record(**found(10.0), arm="a"), record(arm="a"),
            record(**found(2.0), arm="a", T=5.0, uid="short"),
            record("negative", "NONE", time_decided=1.0, elapsed=1.0, arm="b")]
    out = fl.summarize(recs, [5.0, 30.0])
    a5, a30 = out["a"]["synth"]["R0"][5.0], out["a"]["synth"]["R0"][30.0]
    assert (a5["n"], a5["counts"]) == (4, {"retrieval": 2, "solved": 2})
    assert (a30["n"], a30["counts"]) == (3, {"retrieval": 1, "solved": 2})
    lo, hi = stats.wilson_interval(2, 3)
    assert a30["shares"]["solved"] == {"share": 2 / 3, "lo": lo, "hi": hi} and a30["overlap_rescued"] == 0
    assert out["b"]["synth"]["R0"][30.0]["counts"] == {"closed_none": 1}
    flat = fl.summarize(recs, [30.0], key=lambda r: r["arm"])
    assert flat["a"][30.0]["n"] == 3 and set(flat) == {"a", "b"}


# ----------------------------------------------------------------------------
# Real cascade records
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_runs(clustered_store, tmp_path_factory):
    """Records of filterall, jigsaw_index, direct and a component cap of 1 on a small dev query set."""
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("localize")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    rankers.build_artifacts(gstore, ["R0"])
    cascade.build_label_index(gstore, "R0")
    queries.SplitBuilder(gstore, "synth", "dev", "20260607", sizes=[8], max_per_cell=1).build(
        ["R0"], copy / "queries", master_tag="20260607")
    qs = queries.QuerySet.load(copy / "queries" / "R0.npz").queries
    bank = rankers.RankerBank(gstore, "R0", router_tau=1.0)
    arms = {"filterall": ("filterall", {}, REAL_T), "jigsaw_index": ("jigsaw_index", {"budgets": [1, 2]}, REAL_T),
            "direct": ("direct", {}, REAL_T),
            "capped": ("jigsaw_index", {"budgets": [1, 2], "component_cap": 1}, REAL_T / 2)}
    runs = {}
    for name, (method, kw, T) in arms.items():
        c = cascade.Cascade(gstore, "R0", method, backends.NetworkXBackend(),
                            bank=bank if protocol.METHODS[method].get("ranker") else None, **kw)
        runs[name] = [c.run(q, time_limit=T) for q in qs]
    return runs


def test_real_records_classify_soundly_at_every_horizon(real_runs):
    last: dict[str, set[str]] = {"positive": set(), "negative": set()}
    for name, recs in real_runs.items():
        assert {r["kind"] for r in recs} == {"positive", "negative"}
        for rec in recs:
            assert rec["status"] != backends.ERROR, rec.get("error")
            T = float(rec["T"])
            grid = [0.0, 0.25 * T, 0.5 * T, T]
            categories = []
            for t in grid:
                category, detail = fl.classify(rec, t)
                assert category not in fl.UNSOUND, (name, rec["uid"], t, category, detail)
                assert category in (fl.POSITIVE_CATEGORIES if rec["kind"] == "positive" else fl.NEGATIVE_CATEGORIES)
                categories.append(category)
            with pytest.raises(ValueError):
                fl.classify(rec, T + 1.0)
            # nothing had started before 0
            if rec["kind"] == "negative":
                assert categories[0] == "no_closed_step"
            else:
                assert categories[0] == ("solver_budget" if name == "direct" else "retrieval")
            solved = [fl.solved_by(rec, t) for t in grid]
            assert solved == sorted(solved)  # once solved, solved at every later t
            if rec["kind"] == "positive" and solved[-1]:
                assert categories[-1] == "solved"
            if rec["kind"] == "negative" and fl.none_proved_by(rec, T):
                assert categories[-1] == ("certificate" if rec.get("certificate") else "closed_none")
            if rec["kind"] == "negative" and rec["status"] == cascade.INCOMPLETE and rec["elapsed"] <= T:
                assert name == "capped" and categories[-1] == "incomplete"
            last[rec["kind"]].add(categories[-1])
        summary = fl.summarize(recs, [0.0, REAL_T / 2], key=lambda r: r["kind"])
        assert sum(summary[k][0.0]["n"] for k in summary) == len(recs)
    assert "solved" in last["positive"] and last["negative"] & {"certificate", "closed_none"}
