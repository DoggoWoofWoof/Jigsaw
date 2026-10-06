"""E15 reducer: enumeration recall of the ladder, from enumeration.py's run directories.

    python -m scripts.canonical.reduce_enumeration --runs RUNS/E15/arxiv RUNS/E15/mag --out OUT
        [--allow-dev] [--no-verify] [--allow-violations]

Each --runs directory (a glob may name several) is one enumeration.py run: one dataset, its regimes and
arms, possibly one shard. OUT gets numbers.json and figures/*.csv (numbers_json), tables/*.csv,
violations.json and a manifest (experiment "reduce-E15"); build_paper takes OUT/numbers.json beside the
reducer's. The module reuses the reducer's manifest checks, record-file names and key grammar, and never
imports torch or seal.

Refusals. Every check raises reducer.ReduceError; the CLI prints it and exits 2 before writing anything.
  * Manifests: status complete, the current protocol hash, outputs that re-hash to the manifest
    (--no-verify skips only the re-hashing), experiment "E15", record schema jigsaw-enumeration-v1 and the
    primary partition.
  * One split over every run, in queries.TEST_SPLITS unless --allow-dev. On a test split every run has
    the protocol's limit, T, positives, regimes and arms (protocol.ENUMERATION) and no smoke setting.
  * One dataset is one configuration: its runs share regimes, arms, limit, T, positives and solve_steps,
    and are either one unsharded run or the shards 0..n-1 of one n. One query set (the manifest's sha256
    of input "queries/<regime>") per dataset and regime.
  * Records: one file per arm and regime with config.query_sets[regime].selected records; schema,
    dataset, regime, arm, kind "positive", limit and T as configured; no uid twice; every arm of a dataset
    and regime on the same uids; every non-error record of an arm on the same ladder budgets.

Measures (per dataset, regime and system; a system is an arm without a backend every arm shares):
  * Status counts: complete (the whole-graph enumeration finished below L + 1 embeddings), partial at the
    limit (L embeddings found) or in time (the limit T ran out first), and errors.
  * Recall at each ladder step K: the mean over complete queries of the share of the enumerated
    embeddings inside a kept target of step K, with a percentile bootstrap interval over queries; the
    share of complete queries with full recall and of queries whose planted embedding lies in a kept
    target (Wilson). Partial queries hold the solver's first L embeddings, a sample in solver order, so
    their recall is reported apart.
  * Time: the whole-graph enumeration seconds (median and p95 with bootstrap intervals) and each step's
    candidate build time (median).
  * Violations (the E15 counterpart of PE1): a rejected embedding, a complete enumeration without the
    planted embedding, an embedding of a candidate lost by pruning, a step whose solved embeddings differ
    from the located ones (--solve-steps), a solver error and a crash. E15.holds is "yes" when all are 0;
    otherwise the CLI writes everything and exits 3, unless --allow-violations.

Keys (S a system, K a ladder budget):
  E15.<ds>.<r>.<S>.{queries,complete,partial_limit,partial_time,errors} ; E15.<ds>.<r>.<S>.complete_rate
  E15.<ds>.<r>.<S>.k<K>.{recall,full_recall,planted_kept,partial_recall,build_seconds_median}
  E15.<ds>.<r>.<S>.final.{recall,full_recall,planted_kept} ; E15.<ds>.<r>.<S>.embeddings_median
  E15.<ds>.<r>.<S>.seconds.{median,p95} ; E15.<ds>.<r>.<S>.solve.{consistent,inconsistent,unchecked}
  E15.<ds>.<r>.<S>.violations.<kind> ; E15.holds
  Example: E15.arxiv.R0.jigsaw-at-7301.k10.recall.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import manifest, numbers_json, protocol, queries, stats
from . import reducer as rd

SCHEMA = "jigsaw-enumeration-v1"  # enumeration.SCHEMA; enumeration itself imports seal
KEPT = ("kept", "unfiltered")  # enumeration.KEPT
STATUSES = ("complete", "partial", "error")
VIOLATIONS = ("witness_rejected", "planted_missing", "lost_by_pruning", "inconsistent_solve", "solver_error",
              "crash", "unexplained")
SMOKE_SETTINGS = ("budgets", "router_tau", "families", "sizes", "max_queries")
PROTOCOL_DRAWS = {"boot": 20_000, "quantile": 5_000}
DRAWS = dict(PROTOCOL_DRAWS)
EXPERIMENT = "reduce-E15"
_TOL = 1e-9


def plan_arms() -> list[str]:
    """enumeration.plan_arms: the protocol's arms with the first training seed filled in."""
    return [a.replace("@first_training_seed", f"@{rd.first_seed()}") for a in protocol.ENUMERATION["arms"]]


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
@dataclass
class EnumRun:
    run_id: str
    path: Path
    dataset: str
    split: str
    frozen: bool
    shard: tuple[int, int] | None
    regimes: list[str]
    arms: list[str]
    limit: int
    T: float
    positives: int
    solve_steps: bool
    manifest_sha256: str
    query_sha: dict[str, str]
    selected: dict[str, int]


def load_run(path: Path, run_id: str, *, H: str, flags: rd.Flags) -> EnumRun:
    doc, sha = rd.read_manifest(path, H=H, flags=flags)
    if doc.get("experiment") != "E15":
        raise rd.ReduceError(f"{path}: manifest experiment {doc.get('experiment')!r}, not 'E15'")
    config = doc.get("config") or {}
    where = f"{path}: config"
    if config.get("schema") != SCHEMA:
        raise rd.ReduceError(f"{where} schema {config.get('schema')!r}, not {SCHEMA}")
    dataset = config.get("dataset")
    if dataset not in protocol.DATASETS:
        raise rd.ReduceError(f"{where} dataset {dataset!r} is not a protocol dataset")
    if (config.get("partition") or "primary") != "primary":
        raise rd.ReduceError(f"{where} partition {config.get('partition')!r}: E15 runs on the primary partition")
    split = doc.get("split")
    if not isinstance(split, str) or not split:
        raise rd.ReduceError(f"{path}: manifest has no split")
    regimes = config.get("regimes")
    if not isinstance(regimes, list) or not regimes or len(set(regimes)) != len(regimes):
        raise rd.ReduceError(f"{where} regimes {regimes!r}")
    for regime in regimes:
        rd.seg_regime(regime)
    arms = config.get("arms")
    if not isinstance(arms, list) or not arms or len(set(arms)) != len(arms):
        raise rd.ReduceError(f"{where} arms {arms!r}")
    for arm in arms:
        if rd.parse_arm(arm)[0] not in protocol.METHODS:
            raise rd.ReduceError(f"{where} arm {arm!r} is not a protocol method")
    try:
        limit, T, positives = int(config["limit"]), float(config["T"]), int(config["positives"])
    except (KeyError, TypeError, ValueError) as exc:
        raise rd.ReduceError(f"{where} lacks limit, T or positives") from exc
    if limit < 1 or not T > 0:
        raise rd.ReduceError(f"{where} limit {limit} and T {T}")
    shard = config.get("shard")
    if shard is not None:
        if (not isinstance(shard, list) or len(shard) != 2 or not all(isinstance(x, int) for x in shard)
                or not 0 <= shard[0] < shard[1]):
            raise rd.ReduceError(f"{where} shard {shard!r}")
        shard = (int(shard[0]), int(shard[1]))
    qsets, inputs = config.get("query_sets") or {}, doc.get("inputs") or {}
    query_sha: dict[str, str] = {}
    selected: dict[str, int] = {}
    for regime in regimes:
        q = qsets.get(regime) or {}
        if q.get("split") is not None and q.get("split") != split:
            raise rd.ReduceError(f"{where} query set of {regime} is split {q.get('split')!r}, the manifest's {split!r}")
        if not isinstance(q.get("selected"), int) or q["selected"] < 0:
            raise rd.ReduceError(f"{where} query_sets.{regime}.selected missing")
        hashed = inputs.get(f"queries/{regime}")
        digest = hashed.get("sha256") if isinstance(hashed, dict) else None
        if not digest:
            raise rd.ReduceError(f"{path}: the manifest holds no sha256 of input queries/{regime}")
        query_sha[regime], selected[regime] = str(digest), int(q["selected"])
    if split in queries.TEST_SPLITS:
        spec = protocol.ENUMERATION
        problems = [name for name in SMOKE_SETTINGS if config.get(name) is not None]
        outside = sorted(set(arms) - set(plan_arms()))
        if outside:
            problems.append(f"arms {outside}")
        for name, value, want in (("limit", limit, spec["limit"]), ("T", T, spec["T_seconds"]),
                                  ("positives", positives, spec["positives_per_dataset"])):
            if abs(float(value) - float(want)) > _TOL:
                problems.append(f"{name} {value:g} (the protocol's is {float(want):g})")
        extra = sorted(set(regimes) - set(spec["regimes"]))
        if extra:
            problems.append(f"regimes {extra}")
        if problems:
            raise rd.ReduceError(f"{path}: E15 on the {split} split is the pre-registered run; "
                                 f"not allowed: {'; '.join(problems)}")
    return EnumRun(run_id, path, dataset, split, bool(doc.get("protocol_frozen")), shard, list(regimes), list(arms),
                   limit, T, positives, bool(config.get("solve_steps")), sha, query_sha, selected)


def check_record(rec: Mapping[str, Any], *, run: EnumRun, regime: str, arm: str, path: Path) -> None:
    uid = rec.get("uid")
    if not isinstance(uid, str) or not uid:
        raise rd.ReduceError(f"{path}: a record without a uid")
    for name, want in (("schema", SCHEMA), ("dataset", run.dataset), ("regime", regime), ("arm", arm),
                       ("kind", "positive")):
        if rec.get(name) != want:
            raise rd.ReduceError(f"{path}: record {uid} has {name} {rec.get(name)!r}, not {want!r}")
    if rec.get("status") not in STATUSES:
        raise rd.ReduceError(f"{path}: record {uid} has status {rec.get('status')!r}")
    if "traceback" not in rec:  # a crash record stops before the settings are written
        if rec.get("limit") != run.limit or abs(float(rec.get("T", -1.0)) - run.T) > _TOL:
            raise rd.ReduceError(f"{path}: record {uid} ran with limit {rec.get('limit')} and T {rec.get('T')}, "
                                 f"the run's are {run.limit} and {run.T:g}")
    if rec["status"] != "error" and not isinstance(rec.get("global"), dict):
        raise rd.ReduceError(f"{path}: record {uid} has no whole-graph enumeration")


@dataclass
class Group:
    """One dataset's E15 runs (shards of one configuration) and their records."""
    dataset: str
    runs: list[EnumRun]
    regimes: list[str]
    arms: list[str]
    systems: dict[str, str]
    limit: int
    T: float
    solve_steps: bool
    records: dict[tuple[str, str], dict[str, dict[str, Any]]]


def system_names(arms: Sequence[str]) -> dict[str, str]:
    """Arm -> system: a +backend=X every arm shares (a dev --backend run) is dropped."""
    backends = {rd.parse_arm(a)[2].get("backend") for a in arms}
    shared = next(iter(backends)) if len(backends) == 1 else None
    names = {a: rd.strip_backend(a, shared) for a in arms}
    if len(set(names.values())) != len(names):
        raise rd.ReduceError(f"arms {list(arms)} name the same system twice")
    return names


def assemble(runs: Sequence[EnumRun]) -> dict[str, Group]:
    by_dataset: dict[str, list[EnumRun]] = {}
    for run in runs:
        by_dataset.setdefault(run.dataset, []).append(run)
    groups: dict[str, Group] = {}
    for ds, members in sorted(by_dataset.items()):
        base = members[0]
        for run in members[1:]:
            for attr in ("regimes", "arms", "limit", "T", "positives", "solve_steps"):
                if getattr(run, attr) != getattr(base, attr):
                    raise rd.ReduceError(f"{ds}: {run.path} and {base.path} differ in {attr}; "
                                         "one dataset is one configuration")
        shards = [run.shard for run in members]
        if len(members) > 1 or shards[0] is not None:
            if any(s is None for s in shards):
                raise rd.ReduceError(f"{ds}: an unsharded run beside other runs")
            counts = {s[1] for s in shards}
            indices = sorted(s[0] for s in shards)
            if len(counts) != 1 or indices != list(range(next(iter(counts)))):
                raise rd.ReduceError(f"{ds}: shards {indices} of {sorted(counts)}; every shard 0..n-1 of one n once")
        records: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for regime in base.regimes:
            shas = {run.query_sha[regime] for run in members}
            if len(shas) != 1:
                raise rd.ReduceError(f"{ds} {regime}: the runs used different query sets")
            for arm in base.arms:
                merged: dict[str, dict[str, Any]] = {}
                for run in members:
                    path = run.path / "records" / regime / rd.record_file(arm, run.shard)
                    if not path.is_file():
                        raise rd.ReduceError(f"{run.path}: no records for {arm} in {regime} ({path.name})")
                    recs = rd.read_jsonl(path)
                    if len(recs) != run.selected[regime]:
                        raise rd.ReduceError(f"{path}: {len(recs)} records for {run.selected[regime]} selected "
                                             "queries; an incomplete run is resumed, not reduced")
                    for rec in recs:
                        check_record(rec, run=run, regime=regime, arm=arm, path=path)
                        if rec["uid"] in merged:
                            raise rd.ReduceError(f"{path}: query {rec['uid']} appears twice for {arm}")
                        merged[rec["uid"]] = rec
                records[(regime, arm)] = merged
            if len({frozenset(records[(regime, arm)]) for arm in base.arms}) != 1:
                raise rd.ReduceError(f"{ds} {regime}: the arms ran different queries")
        groups[ds] = Group(ds, members, list(base.regimes), list(base.arms), system_names(base.arms), base.limit,
                           base.T, base.solve_steps, records)
    return groups


# ----------------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------------
def violations(rec: Mapping[str, Any]) -> list[str]:
    """The E15 violations a record shows (empty when sound)."""
    if rec["status"] == "error":
        if "traceback" in rec:
            return ["crash"]
        if str(rec.get("error") or "").startswith("witness rejected"):
            return ["witness_rejected"]
        return ["solver_error"]
    found = []
    if rec["status"] == "complete" and not rec.get("planted_found"):
        found.append("planted_missing")  # every E15 query is a positive with a planted embedding
    steps = rec.get("steps") or []
    if any((s.get("located") or {}).get("unsound", 0) for s in steps):
        found.append("lost_by_pruning")
    if any((s.get("solve") or {}).get("consistent") is False or (s.get("solve") or {}).get("status") == "error"
           for s in steps):
        found.append("inconsistent_solve")
    if not found and rec.get("sound") is False:
        found.append("unexplained")
    return found


def ladder(recs: Sequence[Mapping[str, Any]], where: str) -> list[tuple[int, bool]]:
    """(budget, final) of each step, the same for every record."""
    seen = {tuple((int(s["budget"]), bool(s["final"])) for s in rec["steps"]) for rec in recs}
    if len(seen) > 1:
        raise rd.ReduceError(f"{where}: records walked different ladders {sorted(seen)[:3]}")
    return list(next(iter(seen))) if seen else []


def mean_ci(values: Sequence[float]) -> tuple[float, tuple[float, float]]:
    x = np.asarray(values, dtype=float)
    lo, hi = stats.paired_bootstrap_ci(x, n_boot=DRAWS["boot"], seed=rd.SEED)
    return float(x.mean()), (lo, hi)


@dataclass
class Summary:
    counts: dict[str, int]
    steps: list[dict[str, Any]]
    violations: dict[str, list[str]]


def summarize(recs: Sequence[Mapping[str, Any]], *, limit: int, where: str) -> Summary:
    ok = [r for r in recs if r["status"] != "error"]
    complete = [r for r in ok if r["status"] == "complete"]
    partial = [r for r in ok if r["status"] == "partial"]
    at_limit = [r for r in partial if int(r["global"].get("embeddings", 0)) >= limit]
    counts = {"queries": len(recs), "complete": len(complete), "partial_limit": len(at_limit),
              "partial_time": len(partial) - len(at_limit), "errors": len(recs) - len(ok)}
    found: dict[str, list[str]] = {kind: [] for kind in VIOLATIONS}
    for rec in recs:
        for kind in violations(rec):
            found[kind].append(rec["uid"])
    steps = []
    for i, (budget, final) in enumerate(ladder(ok, where)):
        step = {"budget": budget, "final": final,
                "recall": [r["steps"][i]["recall"] for r in complete if r["steps"][i].get("recall") is not None],
                "partial_recall": [r["steps"][i]["recall"] for r in partial
                                   if r["steps"][i].get("recall") is not None],
                "planted": [r["steps"][i].get("planted") for r in ok if r["steps"][i].get("planted") is not None],
                "build_seconds": [float(r["steps"][i]["build_seconds"]) for r in ok],
                "solve": [r["steps"][i]["solve"].get("consistent") for r in ok if "solve" in r["steps"][i]]}
        steps.append(step)
    return Summary(counts, steps, found)


# ----------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------
def write_group(group: Group, w: rd.Writer, report: dict[str, Any]) -> None:
    ds_seg = rd.escape(group.dataset)
    sources = sorted(run.run_id for run in group.runs)
    summary_rows: list[list[Any]] = []
    for regime in group.regimes:
        base = f"E15.{ds_seg}.{rd.seg_regime(regime)}"
        figure_rows: list[dict[str, Any]] = []
        table_rows: list[list[Any]] = []
        for arm in group.arms:
            system = group.systems[arm]
            recs = [group.records[(regime, arm)][u] for u in sorted(group.records[(regime, arm)])]
            s = summarize(recs, limit=group.limit, where=f"{group.dataset} {regime} {arm}")
            k = f"{base}.{rd.seg_system(system)}"
            for name, value in s.counts.items():
                w.count(f"{k}.{name}", value, sources)
            if recs:
                w.nums.rate(f"{k}.complete_rate", s.counts["complete"], len(recs), source=sources)
            for kind in VIOLATIONS:
                w.count(f"{k}.violations.{kind}", len(s.violations[kind]), sources)
                report["violations"].extend({"dataset": group.dataset, "regime": regime, "arm": arm, "kind": kind,
                                             "uid": uid} for uid in s.violations[kind])
            ok = [r for r in recs if r["status"] != "error"]
            complete = [r for r in ok if r["status"] == "complete"]
            if complete:
                w.value(f"{k}.embeddings_median", float(np.median([r["global"]["embeddings"] for r in complete])),
                        "number", sources, n=len(complete), digits=0,
                        note="embeddings of the whole-graph enumeration, complete queries")
            if ok:
                seconds = [float(r["global"]["seconds"]) for r in ok]
                for name, q in (("median", 0.5), ("p95", 0.95)):
                    point, lo, hi = stats.quantile_with_ci(seconds, q, n_boot=DRAWS["quantile"], seed=rd.SEED)
                    w.value(f"{k}.seconds.{name}", point, "seconds", sources, ci=(lo, hi), n=len(seconds), digits=2,
                            note="whole-graph enumeration time, censored at the run's T")
            solve = {"consistent": 0, "inconsistent": 0, "unchecked": 0}
            for step in s.steps:
                kk = f"{k}.k{step['budget']}"
                cells = {}
                for name, values, note in (("recall", step["recall"], "complete queries"),
                                           ("partial_recall", step["partial_recall"],
                                            "partial queries: the solver's first L embeddings")):
                    if values:
                        mean, (lo, hi) = mean_ci(values)
                        cells[name] = w.value(f"{kk}.{name}", 100.0 * mean, "pct", sources,
                                              ci=(100.0 * lo, 100.0 * hi), n=len(values), digits=1,
                                              note=f"mean share of the enumerated embeddings in a kept target; {note}")
                if step["recall"]:
                    full = sum(1 for v in step["recall"] if v >= 1.0 - _TOL)
                    cells["full_recall"] = w.nums.rate(f"{kk}.full_recall", full, len(step["recall"]),
                                                       source=sources, note="complete queries with recall 1")
                if step["planted"]:
                    kept = sum(1 for v in step["planted"] if v in KEPT)
                    cells["planted_kept"] = w.nums.rate(f"{kk}.planted_kept", kept, len(step["planted"]),
                                                        source=sources, note="planted embedding in a kept target")
                if step["build_seconds"]:
                    w.value(f"{kk}.build_seconds_median", float(np.median(step["build_seconds"])), "seconds",
                            sources, n=len(step["build_seconds"]), digits=3, note="candidate build and pruning")
                if step["final"]:
                    for name in ("recall", "full_recall", "planted_kept"):
                        if name in cells:
                            entry = w.nums.numbers[cells[name]]
                            w.value(f"{k}.final.{name}", entry["value"], "pct", sources, ci=entry["ci"], n=entry["n"],
                                    digits=1, note=f"the final step (K = {step['budget']} blocks)")
                for verdict in step["solve"]:
                    solve["consistent" if verdict is True else "inconsistent" if verdict is False else "unchecked"] += 1
                if "recall" in cells:
                    entry = w.nums.numbers[cells["recall"]]
                    figure_rows.append({"series": system, "x": float(step["budget"]), "y": entry["value"],
                                        "lo": entry["ci"][0], "hi": entry["ci"][1]})
                table_rows.append([system, str(step["budget"]) + (" (final)" if step["final"] else ""),
                                   w.cell(f"{kk}.recall", ci=True), w.cell(f"{kk}.full_recall", ci=True),
                                   w.cell(f"{kk}.planted_kept", ci=True), w.cell(f"{kk}.partial_recall")])
            if group.solve_steps:
                for name, value in solve.items():
                    w.count(f"{k}.solve.{name}", value, sources)
            summary_rows.append([group.dataset, regime, system, w.cell(f"{k}.queries"), w.cell(f"{k}.complete"),
                                 w.cell(f"{k}.partial_limit"), w.cell(f"{k}.partial_time"), w.cell(f"{k}.errors"),
                                 w.cell(f"{k}.seconds.median"), w.cell(f"{k}.final.recall", ci=True)])
        if table_rows:
            w.nums.table(base, ["System", "K", "Recall (complete)", "Full recall", "Planted kept", "Recall (partial)"],
                         table_rows, notes=f"L = {group.limit}, T = {group.T:g} s; recall is the mean share of the "
                                           "enumerated embeddings inside a kept target of step K")
        if figure_rows:
            w.nums.figure(f"{base}.recall", "curve", figure_rows, x="K (blocks)", y="Recall (%)", log_x=True,
                          notes="complete queries; bootstrap 95% bands")
    report["summary_rows"].extend(summary_rows)


def reduce_groups(groups: Mapping[str, Group], *, split: str, frozen: bool, H: str) -> tuple[numbers_json.Numbers,
                                                                                               dict[str, Any]]:
    nums = numbers_json.Numbers(split=split, frozen=frozen, protocol_hash=H)
    for group in groups.values():
        for run in group.runs:
            nums.source(run.run_id, path=run.path, manifest_sha256=run.manifest_sha256, experiment="E15",
                        dataset=group.dataset)
    w = rd.Writer(nums)
    report: dict[str, Any] = {"violations": [], "summary_rows": []}
    for ds in sorted(groups):
        write_group(groups[ds], w, report)
    all_sources = sorted(nums.sources)
    holds = not report["violations"]
    w.text("E15.holds", "yes" if holds else "no", all_sources)
    nums.table("E15.summary", ["Dataset", "Regime", "System", "Queries", "Complete", "Partial (limit)",
                               "Partial (time)", "Errors", "Enumeration s (median)", "Final-step recall"],
               report.pop("summary_rows"))
    report["holds"] = holds
    return nums, report


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def expand(patterns: Sequence[str]) -> list[Path]:
    dirs: list[Path] = []
    for pattern in patterns:
        hits = sorted(Path(p) for p in glob.glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        if not hits:
            raise rd.ReduceError(f"--runs {pattern} matches nothing")
        for hit in hits:
            if not hit.is_dir():
                raise rd.ReduceError(f"--runs {hit} is not a directory")
            real = hit.resolve()
            if real in dirs:
                raise rd.ReduceError(f"--runs names {real} twice")
            dirs.append(real)
    return dirs


def load_runs(dirs: Sequence[Path], flags: rd.Flags, H: str) -> list[EnumRun]:
    runs: list[EnumRun] = []
    per_dataset: dict[str, int] = {}
    for path in dirs:
        run = load_run(path, "pending", H=H, flags=flags)
        n = per_dataset.get(run.dataset, 0)
        per_dataset[run.dataset] = n + 1
        run.run_id = f"E15.{rd.escape(run.dataset)}.{n}"
        runs.append(run)
    splits = {run.split for run in runs}
    if len(splits) != 1:
        raise rd.ReduceError(f"the runs span splits {sorted(splits)}")
    split = next(iter(splits))
    if split not in queries.TEST_SPLITS and not flags.allow_dev:
        raise rd.ReduceError(f"the runs are on the {split} split; reduce a test split, or pass --allow-dev")
    return runs


def check_out(out: Path, runs: Sequence[EnumRun]) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise rd.ReduceError(f"{out} exists and is not an empty directory")
    real = out.resolve()
    for run in runs:
        if real == run.path or run.path in real.parents:
            raise rd.ReduceError(f"{out} lies inside the run directory {run.path}")


def execute(run_patterns: Sequence[str], out: str | Path, flags: rd.Flags = rd.Flags(),
            argv: Sequence[str] | None = None) -> int:
    """Reduce E15 runs into OUT; 0, or 3 when a violation is found (without allow_pe1_failure)."""
    H = protocol.protocol_hash()
    out = Path(out)
    runs = load_runs(expand(run_patterns), flags, H)
    check_out(out, runs)
    groups = assemble(runs)
    frozen = all(run.frozen for run in runs)
    nums, report = reduce_groups(groups, split=runs[0].split, frozen=frozen, H=H)
    config = {"runs": [str(run.path) for run in runs], "draws": dict(DRAWS),
              "flags": {"allow_dev": flags.allow_dev, "no_verify": flags.no_verify,
                        "allow_violations": flags.allow_pe1_failure}}
    inputs = {f"manifest/{run.run_id}": run.path / manifest.MANIFEST_NAME for run in runs}
    with manifest.RunManifest(out, experiment=EXPERIMENT, config=config, split=runs[0].split, inputs=inputs,
                              argv=argv) as run:
        nums.write(out)
        (out / "tables").mkdir(exist_ok=True)
        for table_id in sorted(nums.tables):
            (out / "tables" / f"{table_id}.csv").write_text(rd.table_csv(nums, table_id), encoding="utf-8")
        (out / "violations.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        run.note(holds=report["holds"], numbers=len(nums.numbers), tables=len(nums.tables), figures=len(nums.figures),
                 violations=len(report["violations"]))
    if not report["holds"] and not flags.allow_pe1_failure:
        print(f"E15 violations: {len(report['violations'])} (see {out / 'violations.json'})", file=sys.stderr)
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="enumeration.py run directories (globs allowed)")
    p.add_argument("--out", required=True, help="a new or empty directory")
    p.add_argument("--allow-dev", action="store_true", help="reduce a dev or validation split")
    p.add_argument("--no-verify", action="store_true", help="skip re-hashing run outputs")
    p.add_argument("--allow-violations", action="store_true", help="exit 0 even when a violation is found")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    DRAWS.clear()
    DRAWS.update(PROTOCOL_DRAWS)  # the CLI always resamples with the protocol's counts
    flags = rd.Flags(allow_dev=args.allow_dev, no_verify=args.no_verify, allow_pe1_failure=args.allow_violations)
    try:
        code = execute(args.runs, args.out, flags,
                       argv=sys.argv if argv is None else ["reduce_enumeration", *argv])
    except rd.ReduceError as exc:
        print(f"reduce_enumeration refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "exit": code}, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
