"""Enumeration recall of the ladder (E15).

    python -m scripts.canonical.enumeration --store-root STORES --dataset arxiv \\
        --queries QUERIES/arxiv/dev/20260607 --models-root MODELS --out RUNS/E15/arxiv-dev-20260607
    python -m scripts.canonical.enumeration ... --arms jigsaw_index --limit 50 --T 60 --solve-steps

The find-first runs answer each query with one embedding. E15 measures how much
of the embedding set the ladder's candidates hold. For each positive query and
arm (protocol.ENUMERATION):

  1. Up to L embeddings are enumerated on the label-filtered graph, pruned and
     split into components as in the cascade. That graph holds every embedding,
     so the enumeration is complete when the solver finishes every target below
     L + 1 embeddings within the time limit. Each embedding is checked against
     the raw store (verify.check_embedding), and a complete enumeration must
     contain the planted one.
  2. The arm's ladder is walked without solving. Step i selects its blocks with
     the arm's ranker and policy and builds its candidate G[C(S_i)] and targets.
     An embedding whose nodes all lie in the candidate is an embedding of it (the
     candidate is induced), and sound pruning keeps it whole in one kept target;
     prune.locate checks this for every embedding. recall(i) is the share of the
     enumerated embeddings that reach a kept target.
  3. With --solve-steps every non-final step's targets are enumerated up to L as
     well. When the whole-graph enumeration is complete, the embeddings found at
     a step must be exactly the ones located in its kept targets; the record says
     whether they are.

A partial enumeration (L reached or time out) holds the solver's first L
embeddings, so its recall is that of a sample in solver order; the reducer
reports complete and partial queries apart. Records go to
<out>/records/<regime>/<arm>.jsonl (schema jigsaw-enumeration-v1) under a run
manifest, and running the same command again resumes. A sealed query set runs
only with the protocol's settings, --unseal, a frozen protocol and committed
code (seal.authorize_sealed_run).
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")  # before numpy loads, as in run_experiment

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterator, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from . import backends, cascade, fastcand, manifest, protocol, prune, queries, rankers, seal, verify  # noqa: E402
from . import run_experiment as rx  # noqa: E402
from .store import GraphStore, LocalGraph  # noqa: E402

SCHEMA = "jigsaw-enumeration-v1"
KEPT = ("kept", "unfiltered")  # locate() outcomes that put an embedding inside a solver target


def plan_arms() -> list[str]:
    first = int(protocol.SEEDS["training"][0])
    return [a.replace("@first_training_seed", f"@{first}") for a in protocol.ENUMERATION["arms"]]


# ----------------------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------------------
def enumerate_targets(backend: Any, query: LocalGraph, targets: Sequence[LocalGraph], *, limit: int,
                      time_limit: float) -> dict[str, Any]:
    """Up to `limit` embeddings over the targets in order, as arrays of global node ids.

    The solver is asked for one embedding more than it may return, so a target
    finished below that count proves its list complete."""
    t0 = time.perf_counter()
    found: list[np.ndarray] = []
    status = "complete"
    error = None
    for target in targets:
        left = time_limit - (time.perf_counter() - t0)
        if len(found) > limit or left <= 0:
            status = "partial"
            break
        out = backend.solve(query, target, time_limit=left, max_solutions=limit + 1 - len(found))
        if out.status == backends.FOUND:
            gids = np.asarray(target.global_ids, dtype=np.int64)
            found.extend(gids[np.asarray(phi, dtype=np.int64)] for phi in out.embeddings)
            if out.detail.get("killed_at_cap"):
                status = "partial"
                break
        elif out.status == backends.TIMEOUT:
            status = "partial"
            break
        elif out.status != backends.NONE:
            status, error = "error", str(out.detail.get("error") or out.status)
            break
    if len(found) > limit:
        status = "partial"
        found = found[:limit]
    result = {"status": status, "found": found, "seconds": time.perf_counter() - t0, "targets": len(targets)}
    if error is not None:
        result["error"] = error
    return result


def ladder_steps(c: cascade.Cascade, q: queries.Query, codes: np.ndarray, query: LocalGraph,
                 req: prune.QueryRequirements) -> Iterator[tuple[dict[str, Any], LocalGraph, prune.PruneResult]]:
    """The arm's ladder as Cascade.run builds it, without solving: (step, candidate, pruned) per step."""
    global_nodes = int(np.asarray(c.table.label_frequency(req.labels)).sum())
    selector = None
    if c.spec.get("ranker"):
        ranking, _ = c.bank.rank(c.spec["ranker"], c.query_inputs(q, codes))
        selector = rankers.make_selector(c.selection_policy, ranking, c.store.part, **c.policy_params)
    for i, budget in enumerate(c.budgets):
        final = budget >= c.num_blocks
        t0 = time.perf_counter()
        if final or selector is None:
            cand = c.global_candidate(codes, {})
        else:
            cand = fastcand.candidate(c.store, selector.prefix(budget), codes, c.regime, halo=c.halo, label_filter=True)
        result = prune.prune(cand, query, mode=c.component_mode, component_cap=c.component_cap,
                             requirements=req, max_iterations=c.prune_passes)
        yield ({"step": i, "budget": int(budget), "final": final, "closed": bool(final or cand.n == global_nodes),
                "candidate_nodes": int(cand.n), "targets": len(result.targets),
                "build_seconds": time.perf_counter() - t0}, cand, result)


def enumerate_query(c: cascade.Cascade, q: queries.Query, *, limit: int, time_limit: float,
                    solve_steps: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind, "regime": c.regime,
                           "method": c.method, "dataset": c.store.dataset, "certified": q.certified,
                           "limit": int(limit), "T": float(time_limit), "options": c.options(), "steps": []}
    codes = c.table.query_codes(q.payload, c.store)
    query = q.local_graph(codes)
    req = prune.query_requirements(query)
    # the ground truth: the label-filtered graph with the canonical pruning and no component cap
    whole = c.global_candidate(codes, {})
    pruned = prune.prune(whole, query, mode="separate", requirements=req)
    glob = enumerate_targets(c.backend, query, pruned.targets, limit=limit, time_limit=time_limit)
    embs = glob.pop("found")
    rec["global"] = {**glob, "embeddings": len(embs), "candidate_nodes": int(whole.n)}
    t_check = time.perf_counter()
    for emb in embs:
        check = verify.check_embedding(c.store, q.payload, q.edges, emb, c.regime)
        if not check.ok:
            rec.update({"status": "error", "error": f"witness rejected: {check.reason}", "sound": False,
                        "seconds": time.perf_counter() - t0})
            return rec
    rec["global"]["check_seconds"] = time.perf_counter() - t_check
    planted = np.asarray(q.planted, dtype=np.int64) if q.planted is not None else None
    if planted is not None and embs:
        rec["planted_found"] = any(np.array_equal(e, planted) for e in embs)
    if glob["status"] == "error":
        rec.update({"status": "error", "error": glob.get("error"), "sound": True, "seconds": time.perf_counter() - t0})
        return rec
    sound = not (glob["status"] == "complete" and planted is not None and not rec.get("planted_found"))
    for step, cand, result in ladder_steps(c, q, codes, query, req):
        located = {"kept": 0, "outside": 0, "capped": 0, "unsound": 0}
        kept: list[tuple[int, ...]] = []
        for emb in embs:
            where = prune.locate(result, cand, emb)
            if where["in_candidate"] < where["nodes"]:
                located["outside"] += 1
            elif where["component"] in KEPT:
                located["kept"] += 1
                kept.append(tuple(int(v) for v in emb))
            elif where["component"] == "capped":
                located["capped"] += 1
            else:
                located["unsound"] += 1  # an embedding of the candidate lost by pruning
        sound = sound and located["unsound"] == 0
        step.update({"located": located, "recall": located["kept"] / len(embs) if embs else None})
        if planted is not None:
            step["planted"] = prune.locate(result, cand, planted)["component"]
        if solve_steps and not step["final"]:
            if glob["status"] == "complete":
                sol = enumerate_targets(c.backend, query, result.targets, limit=limit, time_limit=time_limit)
                got = {tuple(int(v) for v in e) for e in sol.pop("found")}
                consistent = got == set(kept) if sol["status"] == "complete" else None
                step["solve"] = {**sol, "embeddings": len(got), "consistent": consistent}
                sound = sound and consistent is not False and sol["status"] != "error"
            else:
                step["solve"] = {"status": "skipped", "reason": "the whole-graph enumeration is partial"}
        rec["steps"].append(step)
    rec.update({"status": glob["status"], "sound": bool(sound), "seconds": time.perf_counter() - t0})
    return rec


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def check_sealed(args: argparse.Namespace, arms: Sequence[rx.Arm]) -> None:
    """A sealed E15 run uses the protocol's arms (or a subset), limit, time, selection and ladder."""
    spec = protocol.ENUMERATION
    problems = [flag for flag, value in (("--budgets", args.budgets), ("--router-tau", args.router_tau),
                                         ("--families", args.families), ("--sizes", args.sizes),
                                         ("--max-queries", args.max_queries), ("--backend", args.backend))
                if value is not None]
    registered = {rx.Arm.parse(a).name for a in plan_arms()}
    outside = [a.name for a in arms if a.name not in registered]
    if outside:
        problems.append(f"arms outside E15: {', '.join(outside)}")
    for flag, value, want in (("--limit", args.limit, spec["limit"]), ("--T", args.T, spec["T_seconds"]),
                              ("--positives", args.positives, spec["positives_per_dataset"])):
        if value is not None and float(value) != float(want):
            problems.append(f"{flag} {value:g} (the protocol's is {want:g})")
    if args.regimes and not set(args.regimes) <= set(spec["regimes"]):
        problems.append(f"regimes outside E15: {sorted(set(args.regimes) - set(spec['regimes']))}")
    if args.partition != "primary":
        problems.append(f"--partition {args.partition}")
    if problems:
        raise SystemExit(f"a sealed run is a pre-registered experiment; not allowed: {'; '.join(problems)}")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    spec = protocol.ENUMERATION
    limit = int(spec["limit"] if args.limit is None else args.limit)
    T = float(spec["T_seconds"] if args.T is None else args.T)
    positives = int(spec["positives_per_dataset"] if args.positives is None else args.positives)
    regimes = list(args.regimes or spec["regimes"])
    arms = [rx.Arm.parse(a) for a in (args.arms or plan_arms())]
    if args.backend:
        arms = [a.with_setting("backend", args.backend) for a in arms]
    if len({a.name for a in arms}) != len(arms):
        raise SystemExit("an arm is listed twice")
    if any(a.get("label_filter") is False for a in arms):
        raise SystemExit("E15 measures the label-filtered candidates: drop +label_filter=0")
    if limit < 1 or T <= 0:
        raise SystemExit("--limit needs at least 1 and --T a positive time")
    shard = None
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit("--shard i/n needs 0 <= i < n")
        shard = (index, count)
    plan = rx.Plan("E15", arms, T, regimes, {"kinds": ["positive"], "totals": {"positive": positives}})
    sets = rx.load_sets(args, plan)
    split = str(next(iter(sets.values())).meta.get("split", "dev"))
    if split in queries.TEST_SPLITS:
        check_sealed(args, arms)  # before the gate, so a refused run leaves no unseal entry
        seal.authorize_sealed_run(unseal=args.unseal, reason=args.reason or "run E15")
    selected = {r: rx.select_queries(sets[r].queries, kinds=["positive"], families=args.families, sizes=args.sizes,
                                     totals={"positive": positives}, limit=args.max_queries, shard=shard)
                for r in regimes}
    threads, limiter = rx.pin_threads(int(protocol.SERVING["threads"]))
    try:
        store = GraphStore(args.store_root, args.dataset, mmap=True, partition=args.partition)
        solvers: dict[str, Any] = {}
        cascades = {r: rx.build_cascades(store, r, arms, args, solvers) for r in regimes}
        out = Path(args.out)
        config = {
            "schema": SCHEMA, "experiment": "E15", "dataset": args.dataset, "partition": args.partition,
            "regimes": regimes, "arms": [a.name for a in arms], "limit": limit, "T": T, "positives": positives,
            "solve_steps": bool(args.solve_steps), "budgets": args.budgets, "router_tau": args.router_tau,
            "solver_bin": args.solver_bin, "families": args.families, "sizes": args.sizes,
            "max_queries": args.max_queries, "shard": list(shard) if shard else None,
            "query_sets": {r: {"path": str(Path(sets[r].path).resolve()), "split": sets[r].meta.get("split"),
                               "master": sets[r].meta.get("master"), "selected": len(selected[r])} for r in regimes},
            "cascade_options": {r: {a.name: {**c.options(), "budgets": list(c.budgets)}
                                    for a, c in zip(arms, cascades[r])} for r in regimes},
            "threads": int(protocol.SERVING["threads"]),
        }
        attempt = rx.check_resume(out, config)
        glasgow = args.solver_bin or os.environ.get("GLASGOW_SOLVER_BIN")
        summary: dict[str, Any] = {"attempt": attempt, "regimes": {}}
        with manifest.RunManifest(out, experiment="E15", config=config, split=split,
                                  inputs=rx.run_inputs(args, plan, store, sets),
                                  solvers={"glasgow": glasgow} if "glasgow" in solvers else {},
                                  fast_input_hashes=args.fast_hashes) as run:
            run.note(threads=threads, attempt=attempt)
            for regime in regimes:
                summary["regimes"][regime] = run_regime(regime, arms, cascades[regime], selected[regime], out,
                                                        shard, attempt, limit=limit, T=T,
                                                        solve_steps=bool(args.solve_steps))
            summary["torch_imported"] = "torch" in sys.modules
            run.note(**summary)
        return summary
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def run_regime(regime: str, arms: Sequence[rx.Arm], cascades: Sequence[cascade.Cascade], qs: Sequence[queries.Query],
               out: Path, shard: tuple[int, int] | None, attempt: int, *, limit: int, T: float,
               solve_steps: bool) -> dict[str, Any]:
    files = [rx.RecordFile(out / "records" / regime / rx.record_name(arm, shard)) for arm in arms]
    counts = {arm.name: {"run": 0, "recorded_before": len(f.done)} for arm, f in zip(arms, files)}
    try:
        for position, q in enumerate(qs):
            for slot, j in enumerate(rx.arm_order(len(arms), position)):
                arm, c, f = arms[j], cascades[j], files[j]
                if q.uid in f.done:
                    continue
                try:
                    rec = enumerate_query(c, q, limit=limit, time_limit=T, solve_steps=solve_steps)
                except Exception as exc:  # a bug: recorded, the run goes on
                    rec = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind, "regime": regime,
                           "method": c.method, "dataset": c.store.dataset, "status": "error", "sound": False,
                           "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8),
                           "steps": []}
                rec.update({"schema": SCHEMA, "arm": arm.name, "arm_settings": dict(arm.settings),
                            "train_seed": arm.seed, "query_index": position, "arm_slot": slot, "attempt": attempt,
                            "backend": c.backend.name, "model_id": arm.model_id(c.store.dataset, regime)})
                f.write(rec)
                counts[arm.name]["run"] += 1
    finally:
        for f in files:
            f.close()
    return {"queries": len(qs), "arms": counts}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="*", help="arms (default protocol.ENUMERATION)")
    p.add_argument("--store-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default="primary")
    p.add_argument("--queries", required=True, help="directory of <regime>.npz query sets of one split master")
    p.add_argument("--regimes", nargs="*")
    p.add_argument("--models-root", default="models/canonical")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="embeddings per query (default protocol)")
    p.add_argument("--T", type=float, default=None, help="seconds per enumeration (default protocol)")
    p.add_argument("--positives", type=int, default=None, help="positive queries per regime (default protocol)")
    p.add_argument("--solve-steps", action="store_true", help="also enumerate every non-final step's targets")
    p.add_argument("--budgets", type=int, nargs="*", help="smoke runs only: replaces the protocol ladder")
    p.add_argument("--router-tau", type=float, default=None, help="before the freeze: the router threshold")
    p.add_argument("--backend", default=None, help="enumerating solver of every arm without +backend")
    p.add_argument("--solver-bin", default=None, help="Glasgow binary (default $GLASGOW_SOLVER_BIN)")
    p.add_argument("--families", nargs="*")
    p.add_argument("--sizes", type=int, nargs="*")
    p.add_argument("--max-queries", type=int, default=None, help="smoke runs only: the first N selected queries")
    p.add_argument("--shard", default=None, help="i/n: every n-th selected query from the i-th")
    p.add_argument("--unseal", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--fast-hashes", action="store_true", help="fingerprint large inputs instead of hashing them")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    summary = execute(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, default=rx._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
