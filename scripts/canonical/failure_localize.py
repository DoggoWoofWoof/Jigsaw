"""Failure localization (E18): where a query stands at time t, read from its record alone.

The cascade writes a trail into every record (cascade.py, "Failure localization"): each step's
start and end on the query clock, whether its candidate was closed, the coverage of the planted
witness by the step's blocks (homecov) and by blocks plus halo (fullcov, covered), where
prune.locate saw the witness go (in_candidate, alive, component, target), and every solver call's
target, start t and length. Because non-final steps have caps that do not depend on T, a record
run at T_max shows the state of the query at any t <= T_max, so every function here takes a
horizon t and refuses one above the record's T.

classify_positive(rec, t) gives one category per positive, in this order of precedence:

  solved              a verified FOUND with time_found <= t
  unsound_witness     an ERROR from a rejected witness (a false positive; at every t)
  error               any other ERROR (a crash; at every t)
  unsound_none        a NONE proof by t on a query that has an embedding
  -- undecided at t: walk S_t, the steps started before t (not_started ones excluded); a step
     skipped as "identical candidate" has no trail of its own and inherits the previous one --
  unsound_prune       the whole witness was in a candidate and LDF/NLF removed part of it
  unsound_component   the component filter dropped it ("filtered") or it spans components ("split")
  (the witness reached the solver: component "kept" or "unfiltered" with every node alive; the
   latest such step s decides, through its call on the witness's target: the "kept" target, or
   target 0 for "unfiltered"; a skipped step answers with the calls of the step it repeats)
  solver_step_cap     s is not closed and the call ended TIMEOUT or killed at the step cap
  solver_budget       s is closed or final and the call timed out, or the call was running at t
  solver_queue        the call had not started by t: not_started, missing (an earlier target used
                      the budget) or starting at or after t
  unsound_solver      the call answered NONE on a target that holds the witness
  component_cap       the component cap dropped the witness ("capped"), or the query ended
                      INCOMPLETE by t
  retrieval           no step so far held the whole witness; the detail gives the best covered
                      fraction and whether a later step of the record covers it ("late") or none
                      does ("never")

  Direct runs (one whole-graph step, no witness trail): an undecided positive is solver_budget.
  External runs (no steps): an undecided one is solver_budget, and a NONE that proves nothing
  (time_decided null, e.g. GNN-PE) is none_unverified once it has returned.

Every positive's detail also carries overlap_rescued: the first step whose candidate held the whole
witness had it through the halo only (homecov false, fullcov true).

classify_negative(rec, t) gives certificate or closed_none (a NONE proof by t: a global certificate
or a closed step), incomplete (INCOMPLETE, or an external NONE that proves nothing, returned by t),
closed_step_timeout (a closed or final step had started, or an external search was running, but no
proof by t), no_closed_step (the ladder had not reached a closed step), and the soundness failures
contradiction (a verified FOUND on a negative certified none), unsound_witness and error, which hold
at every t. rec["certified"] is the query's truth label (the reducer puts the certification file's
there); on a negative not certified none a verified FOUND by t is "found", since an embedding exists.

summarize() counts categories per group of records with Wilson 95% intervals. Everything here is a
pure function of the records: no store, no solver, no torch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from . import protocol, stats

FOUND, NONE, TIMEOUT, ERROR, INCOMPLETE = "FOUND", "NONE", "TIMEOUT", "ERROR", "INCOMPLETE"
NOT_STARTED = "not_started"
REJECTED = "witness rejected"
IDENTICAL_CANDIDATE = "identical candidate"

POSITIVE_CATEGORIES = ("solved", "unsound_witness", "error", "unsound_none", "unsound_prune", "unsound_component",
                       "solver_step_cap", "solver_budget", "solver_queue", "unsound_solver", "component_cap",
                       "retrieval", "none_unverified")
NEGATIVE_CATEGORIES = ("certificate", "closed_none", "no_closed_step", "closed_step_timeout", "incomplete",
                       "found", "contradiction", "unsound_witness", "error")
UNSOUND = frozenset({"unsound_witness", "unsound_none", "unsound_prune", "unsound_component", "unsound_solver",
                     "contradiction"})
EXTERNAL_METHODS = frozenset(m for m, spec in protocol.METHODS.items() if "external" in spec)
_TOLERANCE = 1e-9


# ----------------------------------------------------------------------------
# Record state at t
# ----------------------------------------------------------------------------
def check_horizon(rec: Mapping[str, Any], t: float) -> float:
    """t as a float; a horizon beyond the record's T would read time the query never had."""
    t = float(t)
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"bad horizon {t!r}")
    if t > float(rec["T"]) + _TOLERANCE:
        raise ValueError(f"horizon {t:g} s is beyond the record's T = {float(rec['T']):g} s ({rec.get('uid')})")
    return t


def solved_by(rec: Mapping[str, Any], t: float) -> bool:
    return (rec.get("status") == FOUND and rec.get("witness_check") == "ok" and rec.get("time_found") is not None
            and float(rec["time_found"]) <= t)


def none_proved_by(rec: Mapping[str, Any], t: float) -> bool:
    """A NONE that proves absence (time_decided set) by t; INCOMPLETE and unproven NONEs never are."""
    return rec.get("status") == NONE and rec.get("time_decided") is not None and float(rec["time_decided"]) <= t


def witness_rejected(rec: Mapping[str, Any]) -> bool:
    return rec.get("status") == ERROR and str(rec.get("error") or "").startswith(REJECTED)


def is_external(rec: Mapping[str, Any]) -> bool:
    return "external" in rec or rec.get("method") in EXTERNAL_METHODS


def is_direct(rec: Mapping[str, Any]) -> bool:
    return protocol.METHODS.get(str(rec.get("method")), {}).get("pruning") == "solver only"


def returned_by(rec: Mapping[str, Any], t: float) -> bool:
    elapsed = rec.get("elapsed")
    return elapsed is not None and float(elapsed) <= t


def started_steps(rec: Mapping[str, Any], t: float) -> list[dict[str, Any]]:
    """S_t: the steps that had started before t."""
    return [s for s in rec.get("steps") or [] if s.get("status") != NOT_STARTED and float(s["t_start"]) < t]


def witness_trail(steps: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    """(step, witness location) pairs; an identical-candidate skip repeats the previous location."""
    out: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    previous: Mapping[str, Any] | None = None
    for step in steps:
        where = step.get("witness")
        if where is None and step.get("skipped") == IDENTICAL_CANDIDATE:
            where = previous
        out.append((step, where))
        previous = where
    return out


def _whole(where: Mapping[str, Any] | None) -> bool:
    return where is not None and int(where["in_candidate"]) == int(where["nodes"])


def _pruned_away(where: Mapping[str, Any] | None) -> bool:
    # alive is None only when pruning never ran, which a candidate holding the witness rules out
    return _whole(where) and (where.get("alive") is None or int(where["alive"]) < int(where["nodes"]))


def _reached_solver(where: Mapping[str, Any] | None) -> bool:
    return (_whole(where) and where.get("alive") is not None and int(where["alive"]) == int(where["nodes"])
            and where.get("component") in ("kept", "unfiltered"))


def _holds_witness(step: Mapping[str, Any]) -> bool:
    return step.get("fullcov") is True or _whole(step.get("witness"))


def overlap_rescued(rec: Mapping[str, Any]) -> bool:
    """The first step whose candidate held the whole witness held it through the halo only."""
    for step in rec.get("steps") or []:
        if step.get("status") == NOT_STARTED:
            break
        if _holds_witness(step):
            return step.get("homecov") is False and step.get("fullcov") is True
    return False


# ----------------------------------------------------------------------------
# Positives
# ----------------------------------------------------------------------------
def _solver_step(trail: Sequence[tuple[Mapping[str, Any], Any]], k: int) -> Mapping[str, Any] | None:
    """The step whose solver calls answer for step k: k itself, or the step a skip repeats."""
    for j in range(k, -1, -1):
        step = trail[j][0]
        if "solver_calls" in step:
            return step
        if not step.get("skipped"):
            return None
    return None


def _solver_category(reached: Mapping[str, Any], calls_step: Mapping[str, Any] | None, where: Mapping[str, Any],
                     t: float) -> tuple[str, dict]:
    """The solver state of the witness target of step `reached`, read from `calls_step`'s calls."""
    target = int(where["target"]) if where.get("component") == "kept" and where.get("target") is not None else 0
    calls = [c for c in (calls_step or {}).get("solver_calls") or [] if int(c.get("target", -1)) == target]
    detail: dict[str, Any] = {"step": reached.get("step"), "target": target}
    call = calls[0] if calls else None
    if call is None or call.get("status") == NOT_STARTED or float(call["t"]) >= t:
        return "solver_queue", detail
    end = float(call["t"]) + float(call.get("seconds") or 0.0)
    if t < end:
        return "solver_budget", {**detail, "running": True}
    closed = bool(reached.get("closed") or reached.get("final"))
    if call.get("status") == TIMEOUT or call.get("killed_at_cap"):
        return ("solver_budget" if closed else "solver_step_cap"), detail
    if call.get("status") == NONE:
        return "unsound_solver", detail
    if call.get("status") == FOUND:
        # the call returned the witness before t; the record stamps time_found a moment later
        return "solver_budget", {**detail, "found_after_call": True}
    return "error", {**detail, "call_status": call.get("status")}


def _coverage_detail(rec: Mapping[str, Any], trail: Sequence[tuple[Mapping[str, Any], Any]], t: float) -> dict:
    nodes = next((int(w["nodes"]) for _, w in trail if w is not None), None)
    if nodes is None:
        nodes = next((int(s["witness"]["nodes"]) for s in rec.get("steps") or [] if s.get("witness")), None)
    if nodes is None and rec.get("query_nodes") is not None:
        nodes = int(rec["query_nodes"])
    fractions = []
    for step, where in trail:
        if step.get("covered") is not None and nodes:
            fractions.append(int(step["covered"]) / nodes)
        elif where is not None and int(where["nodes"]):
            fractions.append(int(where["in_candidate"]) / int(where["nodes"]))
    later = [s for s in rec.get("steps") or [] if s.get("status") != NOT_STARTED and float(s["t_start"]) >= t]
    return {"max_covered": round(max(fractions, default=0.0), 6), "steps_started": len(trail),
            "coverage": "late" if any(_holds_witness(s) for s in later) else "never"}


def _undecided_positive(rec: Mapping[str, Any], t: float) -> tuple[str, dict]:
    trail = witness_trail(started_steps(rec, t))
    for name, test in (("unsound_prune", _pruned_away),
                       ("unsound_component", lambda w: _whole(w) and w.get("component") in ("filtered", "split"))):
        hit = next((step for step, where in trail if test(where)), None)
        if hit is not None:
            return name, {"step": hit.get("step")}
    reached = [k for k, (_, where) in enumerate(trail) if _reached_solver(where)]
    if reached:
        k = reached[-1]
        return _solver_category(trail[k][0], _solver_step(trail, k), trail[k][1], t)
    capped = next((step for step, where in trail if _whole(where) and where.get("component") == "capped"), None)
    if capped is not None:
        return "component_cap", {"step": capped.get("step")}
    if rec.get("status") == INCOMPLETE and returned_by(rec, t):
        return "component_cap", {"step": rec.get("decided_step")}
    return "retrieval", _coverage_detail(rec, trail, t)


def classify_positive(rec: Mapping[str, Any], t: float) -> tuple[str, dict[str, Any]]:
    """The category of a positive at horizon t (module docstring) and its detail."""
    t = check_horizon(rec, t)
    if rec.get("kind") != "positive":
        raise ValueError(f"{rec.get('uid')}: classify_positive on a {rec.get('kind')} record")
    category, detail = _classify_positive(rec, t)
    return category, {**detail, "overlap_rescued": overlap_rescued(rec)}


def _classify_positive(rec: Mapping[str, Any], t: float) -> tuple[str, dict[str, Any]]:
    if solved_by(rec, t):
        return "solved", {"time_found": float(rec["time_found"])}
    if witness_rejected(rec):
        return "unsound_witness", {"error": str(rec.get("error"))}
    if rec.get("status") == ERROR:
        return "error", {"error": str(rec.get("error") or "")[:200]}
    if rec.get("status") == FOUND and rec.get("witness_check") != "ok":
        return "unsound_witness", {"witness_check": rec.get("witness_check")}  # a FOUND nobody verified
    if none_proved_by(rec, t):
        return "unsound_none", {"certificate": rec.get("certificate"), "decided_step": rec.get("decided_step")}
    if is_external(rec) and not rec.get("steps"):
        if rec.get("status") == NONE and rec.get("time_decided") is None and returned_by(rec, t):
            return "none_unverified", {}
        return "solver_budget", {"external": True}
    if is_direct(rec):
        return "solver_budget", {"direct": True}
    return _undecided_positive(rec, t)


# ----------------------------------------------------------------------------
# Negatives
# ----------------------------------------------------------------------------
def classify_negative(rec: Mapping[str, Any], t: float) -> tuple[str, dict[str, Any]]:
    """The category of a negative at horizon t (module docstring) and its detail."""
    t = check_horizon(rec, t)
    if rec.get("kind") != "negative":
        raise ValueError(f"{rec.get('uid')}: classify_negative on a {rec.get('kind')} record")
    if witness_rejected(rec):
        return "unsound_witness", {"error": str(rec.get("error"))}
    if rec.get("status") == ERROR:
        return "error", {"error": str(rec.get("error") or "")[:200]}
    if rec.get("status") == FOUND:
        if rec.get("witness_check") != "ok":
            return "unsound_witness", {"witness_check": rec.get("witness_check")}
        if rec.get("certified") == "none":
            return "contradiction", {"time_found": rec.get("time_found")}
        if solved_by(rec, t):  # an embedding exists: the negative was never certified none
            return "found", {"time_found": float(rec["time_found"]), "certified": rec.get("certified")}
    if none_proved_by(rec, t):
        if rec.get("certificate"):
            return "certificate", {"certificate": rec.get("certificate")}
        return "closed_none", {"decided_step": rec.get("decided_step")}
    if is_external(rec) and not rec.get("steps"):
        if rec.get("status") == NONE and rec.get("time_decided") is None and returned_by(rec, t):
            return "incomplete", {"none_unverified": True}
        return "closed_step_timeout", {"external": True}
    if rec.get("status") == INCOMPLETE and returned_by(rec, t):
        return "incomplete", {"decided_step": rec.get("decided_step")}
    started = started_steps(rec, t)
    closed = [s for s in started if s.get("closed") or s.get("final")]
    if closed:
        return "closed_step_timeout", {"step": closed[0].get("step"), "steps_started": len(started)}
    return "no_closed_step", {"steps_started": len(started)}


def classify(rec: Mapping[str, Any], t: float) -> tuple[str, dict[str, Any]]:
    """classify_positive or classify_negative by the record's kind."""
    return classify_positive(rec, t) if rec.get("kind") == "positive" else classify_negative(rec, t)


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def _default_key(rec: Mapping[str, Any]) -> tuple:
    return (rec.get("arm") or rec.get("method"), rec.get("dataset"), rec.get("regime"))


def summarize(records: Iterable[Mapping[str, Any]], horizons: Sequence[float], *,
              key: Callable[[Mapping[str, Any]], Any] = _default_key,
              classify_fn: Callable[[Mapping[str, Any], float], tuple[str, Mapping[str, Any]]] = classify,
              ) -> dict[Any, Any]:
    """Category counts and shares (Wilson 95%) per group and horizon.

    Returns nested dicts, one level per element of key(rec) (a tuple, or one value), then the
    horizon (float): {"n", "counts": {category: int}, "shares": {category: {"share", "lo", "hi"}},
    "overlap_rescued": int}. A horizon beyond a record's T skips that record, so n may fall with t.
    """
    buckets: dict[tuple, dict[float, dict[str, Any]]] = {}
    for rec in records:
        group = key(rec)
        group = group if isinstance(group, tuple) else (group,)
        for h in horizons:
            h = float(h)
            if h > float(rec["T"]) + _TOLERANCE:
                continue
            category, detail = classify_fn(rec, h)
            cell = buckets.setdefault(group, {}).setdefault(h, {"n": 0, "counts": {}, "overlap_rescued": 0})
            cell["n"] += 1
            cell["counts"][category] = cell["counts"].get(category, 0) + 1
            cell["overlap_rescued"] += int(bool((detail or {}).get("overlap_rescued")))
    out: dict[Any, Any] = {}
    for group, per_h in buckets.items():
        node = out
        for part in group[:-1]:
            node = node.setdefault(part, {})
        leaf = node.setdefault(group[-1], {})
        for h, cell in sorted(per_h.items()):
            shares = {}
            for category, count in sorted(cell["counts"].items()):
                lo, hi = stats.wilson_interval(count, cell["n"])
                shares[category] = {"share": count / cell["n"], "lo": lo, "hi": hi}
            leaf[h] = {"n": cell["n"], "counts": dict(sorted(cell["counts"].items())), "shares": shares,
                       "overlap_rescued": cell["overlap_rescued"]}
    return out
