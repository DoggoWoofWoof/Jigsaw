"""Run one external exact matcher over canonical query sets, writing run_experiment.py's records.

    python -m scripts.canonical.run_external --experiment E7 --method classical_gql --store-root STORES \\
        --dataset arxiv --queries QUERIES/arxiv/dev/20260607 --out RUNS/E7/arxiv-dev-20260607-classical_gql
    python -m scripts.canonical.run_external --method gnnpe --regimes R1 --T 120 ...

Methods are the protocol.METHODS entries with "external": classical_gql, classical_cfl and
classical_dpiso (SubgraphMatching presets, classical.FullGraphClassicalRunner) and gnnpe
(gnnpe_adapter.GNNPERunner). Each runs on the whole data graph with one solver process per
query, and one run process runs one method. experiment_plan() fixes the methods, T, regimes,
caches and query selection of E2, E7, E10 and E12.

Setup. Each regime's runner writes (or reuses) its data graph file in --cache-dir (default
<store-root>/_external_cache), and for GNN-PE its offline index and one warm-up query
(prepare(), never charged to a query), before anything is written under --out.
--prepare-only stops there, so the shards of a GNN-PE run can share one prepared index.

Records. <out>/records/<regime>/<method>[.shard<i>of<n>].jsonl and <out>/manifest.json in the
format of run_experiment.py (schema jigsaw-run-record-v1, arm = method, mode resident: every
solver process loads its whole data graph file). Running the same command again resumes by
run_experiment's rules. elapsed is the method's wall time, capped at T: the sigma codes of the
query (stage query_labels), writing the query file (write_query) and the whole solver process,
which re-reads its data graph on every run. stages adds what the binary reports: SubgraphMatching's
load and total (solve) times; GNN-PE's Query Time (solve) and the rest of its run (load). The
witness (global node ids in query order) is checked by verify.check_embedding outside the timed
part, exactly as cascade._finish does: a rejected witness is an ERROR. time_found = time_decided =
elapsed for a verified FOUND. A NONE decides only when it is a proof: a sound whole-graph reason
(classical.whole_graph_none) or a classical search that finished inside the cap. A GNN-PE search
NONE is not a certificate (gnnpe_adapter.SEMANTIC_DIFFERENCES), so its time_decided is null and
external.none_certified is false. An exception is recorded as ERROR with its traceback and the
run goes on. peak_rss_bytes is the sampled peak RSS of the solver process tree (psutil, 20 ms).

Caches. warm runs one untimed pass first (capped at --warm-T seconds per query); cold (Linux)
evicts the solver's input files from the page cache before every query.

Sealed query sets (meta.split test or reserve_*) run only as a named experiment (its method, T,
regimes, cache and query selection) and only through seal.authorize_sealed_run.
"""

from __future__ import annotations

from . import run_experiment as rx  # first: it pins the BLAS thread variables before numpy loads

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from . import classical, gnnpe_adapter, manifest, protocol, queries, seal, verify  # noqa: E402
from .backends import ERROR, FOUND, NONE, SolveOutcome  # noqa: E402
from .store import GraphStore  # noqa: E402

SCHEMA = rx.SCHEMA
CLASSICAL = {"classical_gql": "gql", "classical_cfl": "cfl", "classical_dpiso": "dpiso"}
METHODS = (*CLASSICAL, "gnnpe")
EXPERIMENTS = ("E2", "E7", "E10", "E12")
MODE = "resident"
CACHE_DIRNAME = "_external_cache"
DATA_FORMAT = "text"  # the input format upstream documents; -csr only skips the text parse
GNNPE_PARTITIONS = 5  # upstream default -p (src/main.cpp line 50)
BINARY_VARIABLES = {"classical": "SUBGRAPHMATCHING_BIN", "gnnpe": "GNNPE_BIN"}


# ----------------------------------------------------------------------------
# Named experiments
# ----------------------------------------------------------------------------
def experiment_plan(name: str, dataset: str) -> dict[str, Any]:
    """Methods, T, regimes, caches and query selection of the external side of an experiment."""
    t_max = protocol.TIME["run_T_max_seconds"]
    regimes = protocol.regimes_for(dataset)
    if name == "E2":
        # memory, I/O and latency, on the query selection of run_experiment's E2
        return {"methods": list(CLASSICAL), "T": protocol.TIME["primary_T_seconds"], "regimes": ["R0"],
                "select": {"per_cell": 10}, "mode": MODE, "caches": ["cold", "warm"]}
    if name == "E7":
        # the classical side of the selectivity sweep, on run_experiment's E7 selection
        return {"methods": list(CLASSICAL), "T": t_max["E7"], "regimes": regimes, "select": {"per_cell": 50},
                "mode": MODE}
    if name == "E10":
        # scale-out: the classical matchers in memory on the whole graph, on run_experiment's E10 queries,
        # uncapped (the large host) and under the scale-out cap (streaming.py --experiment E10 --external)
        ours = rx.experiment_plan("E10", dataset)
        return {"methods": list(CLASSICAL), "T": ours["T"], "regimes": list(ours["regimes"]),
                "select": dict(ours["select"]), "mode": MODE, "caches": list(ours["caches"])}
    if name == "E12":
        # GNN-PE head to head on class labels; its records join E1's on uid, so it takes E1's T_max
        return {"methods": ["gnnpe"], "T": t_max.get("E12", t_max["E1"]),
                "regimes": [r for r in ("R1",) if r in regimes], "select": {"per_cell": 50}, "mode": MODE}
    raise ValueError(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS)}")


@dataclass
class Plan:
    experiment: str | None
    method: str
    T: float
    regimes: list[str]
    select: dict[str, Any]


def resolve_plan(args: argparse.Namespace) -> Plan:
    if args.method in CLASSICAL and any(v is not None for v in (args.gnnpe_partitions, args.gnnpe_membership,
                                                                args.gnnpe_max_paths)):
        raise SystemExit("--gnnpe-* options apply to --method gnnpe only")
    if args.method == "gnnpe" and args.data_format is not None:
        raise SystemExit("--data-format applies to the classical methods only")
    if args.experiment:
        try:
            spec = experiment_plan(args.experiment, args.dataset)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        select = dict(spec["select"])
        T = float(args.T if args.T is not None else spec["T"])
        regimes = list(args.regimes or spec["regimes"])
    else:
        if args.T is None or not args.regimes:
            raise SystemExit("without --experiment give --T and --regimes")
        select, T, regimes = {}, float(args.T), list(args.regimes)
    for key in ("kinds", "families", "sizes", "per_cell", "limit"):
        value = getattr(args, key)
        if value is not None:
            select[key] = value
    if not regimes:
        raise SystemExit(f"{args.experiment} has no regime on {args.dataset}")
    unknown = [r for r in regimes if r not in protocol.LABEL_REGIMES]
    if unknown:
        raise SystemExit(f"unknown regimes {unknown}")
    if T <= 0:
        raise SystemExit("--T must be positive")
    return Plan(args.experiment, args.method, T, regimes, select)


def check_sealed_plan(args: argparse.Namespace, plan: Plan) -> None:
    """A sealed run is a pre-registered experiment: one of its methods, its T, regimes, cache and selection."""
    problems = [flag for flag, value in (("--limit", args.limit), ("--kinds", args.kinds),
                                         ("--families", args.families), ("--sizes", args.sizes),
                                         ("--per-cell", args.per_cell), ("--uids", args.uids),
                                         ("--data-format", args.data_format),
                                         ("--gnnpe-partitions", args.gnnpe_partitions)) if value is not None]
    if not args.experiment:
        problems.append("ad-hoc runs (name an --experiment)")
    else:
        spec = experiment_plan(args.experiment, args.dataset)
        if plan.method not in spec["methods"]:
            problems.append(f"--method {plan.method} is outside {args.experiment}")
        if plan.T != float(spec["T"]):
            problems.append(f"--T {plan.T:g} (the plan's is {float(spec['T']):g})")
        if not set(plan.regimes) <= set(spec["regimes"]):
            problems.append(f"regimes outside the plan: {sorted(set(plan.regimes) - set(spec['regimes']))}")
        if args.cache not in spec.get("caches", ["none"]):
            problems.append(f"--cache {args.cache}")
    if problems:
        raise SystemExit(f"a sealed run is a pre-registered experiment; not allowed: {'; '.join(problems)}")


# ----------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------
def cache_root(args: argparse.Namespace) -> Path:
    return Path(args.cache_dir) if args.cache_dir else Path(args.store_root) / CACHE_DIRNAME


def backend_name(method: str) -> str:
    return CLASSICAL.get(method, method)


def make_runner(method: str, store: GraphStore, regime: str, args: argparse.Namespace,
                membership: np.ndarray | None = None) -> Any:
    """FullGraphClassicalRunner or GNNPERunner for one regime; --binary may be a command prefix."""
    binary = None if not args.binary else (args.binary[0] if len(args.binary) == 1 else list(args.binary))
    if method in CLASSICAL:
        return classical.FullGraphClassicalRunner(store, regime, CLASSICAL[method], cache_dir=cache_root(args),
                                                  binary=binary, data_format=args.data_format or DATA_FORMAT)
    return gnnpe_adapter.GNNPERunner(store, regime, cache_dir=cache_root(args), binary=binary,
                                     partitions=args.gnnpe_partitions or GNNPE_PARTITIONS,
                                     threads=int(protocol.SERVING["threads"]), membership=membership,
                                     max_paths=args.gnnpe_max_paths)


def build_runners(store: GraphStore, plan: Plan, args: argparse.Namespace) -> dict[str, Any]:
    """One runner per regime; refuses a missing binary before anything is written."""
    membership = None
    if args.gnnpe_membership:
        membership = np.load(args.gnnpe_membership, allow_pickle=False)
        if membership.shape != (int(store.num_nodes),):
            raise SystemExit(f"{args.gnnpe_membership} holds {membership.shape}, not one part per data vertex")
    runners = {r: make_runner(plan.method, store, r, args, membership) for r in plan.regimes}
    first = runners[plan.regimes[0]]
    if first.binary_path is None:
        kind = "classical" if plan.method in CLASSICAL else "gnnpe"
        name = (first.backend if kind == "classical" else first).binary
        raise SystemExit(f"{plan.method}: solver binary {name!r} not found; build it with external/build_*.sh "
                         f"and set {BINARY_VARIABLES[kind]}, or pass --binary")
    return runners


def binary_digest(path: str | None) -> str | None:
    if not path:
        return None
    entry = manifest.sha256_file(path, fast=True)
    return str(entry.get("sha256") or entry.get("fingerprint"))


def external_settings(method: str, runner: Any, digest: str | None) -> dict[str, Any]:
    """The runner options that change results; a resumed run must match them."""
    spec = protocol.METHODS[method]
    out: dict[str, Any] = {"tool": spec["external"], "algorithm": spec.get("algorithm", spec["external"]),
                           "binary_sha256": digest, "max_solutions": 1}
    if method in CLASSICAL:
        out.update(preset=list(classical.PRESETS[CLASSICAL[method]]), data_format=runner.data_format)
    else:
        source = (gnnpe_adapter.METIS_RECURSIVE if runner.membership is None
                  else gnnpe_adapter._membership_sha256(runner.membership))
        out.update(partitions=runner.partitions, embedding_dim=runner.embedding_dim,
                   path_length=gnnpe_adapter.PATH_LENGTH, startup_allowance=runner.startup_allowance,
                   threads=runner.threads, membership_source=source)
    return out


def prepared_file(runner: Any) -> Path:
    """The record of a runner's setup: the data graph sidecar, or GNN-PE's prepared.json."""
    if isinstance(runner, gnnpe_adapter.GNNPERunner):
        return runner.root / "prepared.json"
    return classical.sidecar_path(runner.data_path)


def solver_files(runner: Any) -> list[Path]:
    """What a solver process reads besides the query: the files a cold run evicts."""
    if isinstance(runner, gnnpe_adapter.GNNPERunner):
        return [runner.root]
    if runner.data_format == "csr":
        return list(classical.csr_files(runner.data_path))
    return [Path(runner.data_path)]


# ----------------------------------------------------------------------------
# One query
# ----------------------------------------------------------------------------
def none_certified(out: SolveOutcome) -> bool:
    """A NONE that proves absence: a sound whole-graph reason, or a finished search that claims completeness."""
    if out.status != NONE:
        return False
    if out.detail.get("trivial"):
        return True
    if out.detail.get("none_unverified"):
        return False
    return out.detail.get("returncode") == 0 and not out.detail.get("killed_at_cap")


def reported_stages(method: str, out: SolveOutcome) -> dict[str, float]:
    """Stage times the binary reports: load and solve (SubgraphMatching); solve and the rest (GNN-PE)."""
    detail = out.detail
    stages: dict[str, float] = {}
    if method in CLASSICAL:
        for key, name in (("load_seconds", "load"), ("total_seconds", "solve")):
            if isinstance(detail.get(key), (int, float)):
                stages[name] = float(detail[key])
    elif isinstance(detail.get("query_time_ms"), (int, float)):
        stages["solve"] = float(detail["query_time_ms"]) / 1000.0
        # everything outside Query Time: process start, graph and path loading, embedding recomputation
        stages["load"] = max(0.0, float(out.elapsed) - float(detail.get("io_seconds") or 0.0) - stages["solve"])
    return stages


def _start_sampler() -> Any:
    """Samples this process's children (the solver tree); None without psutil."""
    try:
        from .memprobe import TreeSampler

        return TreeSampler(os.getpid(), include_root=False).start()
    except Exception:
        return None


def _stop_sampler(sampler: Any) -> tuple[int | None, int]:
    if sampler is None:
        return None, 0
    stats = sampler.stop()
    return (int(stats.peak_rss_bytes) or None), int(stats.samples)


def _finish_found(rec: dict[str, Any], store: GraphStore, q: queries.Query, regime: str, out: SolveOutcome,
                  elapsed: float) -> None:
    """cascade._finish for a FOUND: the independent witness check, outside the timed part."""
    if not out.embeddings:
        rec.update({"status": ERROR, "error": "FOUND without a witness"})
        return
    witness = np.asarray(out.embeddings[0], dtype=np.int64)
    t0 = time.perf_counter()
    check = verify.check_embedding(store, q.payload, q.edges, witness, regime)
    rec["witness_check_seconds"] = time.perf_counter() - t0
    rec["witness"] = witness.astype(int).tolist()
    rec["witness_check"] = "ok" if check.ok else check.reason
    if check.ok:
        rec["time_found"] = rec["time_decided"] = elapsed
    else:
        rec["status"] = ERROR  # a rejected witness is a soundness failure, never a solve
        rec["error"] = f"witness rejected: {check.reason}"


def run_query(runner: Any, method: str, store: GraphStore, regime: str, q: queries.Query, T: float, *,
              binary: dict[str, Any]) -> dict[str, Any]:
    """One timed query: cascade.run()'s record fields plus the external block, io and wall time."""
    spec = protocol.METHODS[method]
    io_before = rx.io_snapshot()
    t_wall = time.perf_counter()
    rec: dict[str, Any] = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind, "regime": regime,
                           "method": method, "dataset": store.dataset, "T": T, "certified": q.certified,
                           "query_nodes": q.n, "query_edges": q.m, "decided_step": None, "steps": [],
                           "time_found": None, "time_decided": None, "stages": {}}
    external: dict[str, Any] = {"tool": spec["external"], "algorithm": spec.get("algorithm", spec["external"]),
                                **binary, "none_certified": False, "load_seconds": None}
    rec["external"] = external
    t0 = time.perf_counter()
    try:
        codes = store.sigma(regime).query_codes(q.payload, store)
        edges = np.asarray(q.edges, dtype=np.int64).reshape(-1, 2)
        labels_seconds = time.perf_counter() - t0
        sampler = _start_sampler()
        try:
            out = runner.solve(codes, edges, time_limit=T - labels_seconds, max_solutions=1)
        finally:
            peak, samples = _stop_sampler(sampler)
        elapsed = min(T, labels_seconds + float(out.elapsed))
        stages = {"query_labels": labels_seconds}
        if out.detail.get("io_seconds") is not None:
            stages["write_query"] = float(out.detail["io_seconds"])
        stages.update(reported_stages(method, out))
        rec.update({"status": out.status, "elapsed": elapsed, "stages": stages})
        external.update({"load_seconds": stages.get("load"), "returncode": out.detail.get("returncode"),
                         "killed_at_cap": out.detail.get("killed_at_cap"),
                         "detail": json.loads(json.dumps(out.detail, default=rx._json_default))})
        if peak is not None:
            rec["peak_rss_bytes"] = peak
            external["rss_samples"] = samples
        if out.status == FOUND:
            _finish_found(rec, store, q, regime, out, elapsed)
        elif out.status == NONE:
            external["none_certified"] = none_certified(out)
            if external["none_certified"]:
                rec["time_decided"] = elapsed
        elif out.status == ERROR:
            failed = f"the solver failed (exit code {out.detail.get('returncode')})"
            rec["error"] = str(out.detail.get("error") or failed)
    except Exception as exc:  # a bug: recorded and counted by the reducer, the run goes on
        rec.update({"status": ERROR, "elapsed": min(T, time.perf_counter() - t0), "time_found": None,
                    "time_decided": None, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8)})
    rec["wall_seconds"] = time.perf_counter() - t_wall
    io_after = rx.io_snapshot()
    rec["io"] = {k: io_after[k] - io_before[k] for k in io_after if k in io_before}
    return rec


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
def record_name(method: str, shard: tuple[int, int] | None) -> str:
    return f"{method}.shard{shard[0]}of{shard[1]}.jsonl" if shard else f"{method}.jsonl"


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    index, count = (int(x) for x in text.split("/"))
    if not 0 <= index < count:
        raise SystemExit("--shard i/n needs 0 <= i < n")
    return index, count


def run_config(args: argparse.Namespace, plan: Plan, shard: tuple[int, int] | None,
               sets: dict[str, queries.QuerySet], selected: dict[str, list[queries.Query]],
               runners: dict[str, Any], digest: str | None) -> dict[str, Any]:
    """Every setting that changes the records; a resumed run must match it exactly."""
    return {
        "schema": SCHEMA, "experiment": plan.experiment, "dataset": args.dataset, "partition": args.partition,
        "regimes": plan.regimes, "arms": [plan.method], "method": plan.method, "T": plan.T, "mode": MODE,
        "cache": args.cache, "warm_T": args.warm_T if args.cache == "warm" else None, "select": plan.select,
        "uids_file": args.uids, "shard": list(shard) if shard else None,
        "query_sets": {r: {"path": str(Path(sets[r].path).resolve()), "split": sets[r].meta.get("split"),
                           "master": sets[r].meta.get("master"), "selected": len(selected[r])} for r in plan.regimes},
        "external": external_settings(plan.method, runners[plan.regimes[0]], digest),
        "threads": int(protocol.SERVING["threads"]),
    }


def run_inputs(args: argparse.Namespace, plan: Plan, store: GraphStore, sets: dict[str, queries.QuerySet],
               runners: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {"store_meta": store.dir / "meta.json",
                              "partition": Path(store.part.dir) / "partition.json"}
    for regime in plan.regimes:
        inputs[f"queries/{regime}"] = sets[regime].path
        inputs[f"data_graph/{regime}"] = prepared_file(runners[regime])
    if args.gnnpe_membership:
        inputs["gnnpe_membership"] = Path(args.gnnpe_membership)
    return {k: v for k, v in inputs.items() if Path(v).exists()}


def evict(runner: Any) -> dict[str, Any]:
    from . import memprobe

    report = memprobe.evict_files(solver_files(runner))
    return {"evicted_files": report["files"], "evicted_bytes": report["bytes"], "method": report["method"],
            "errors": len(report["errors"])}


def run_regime(store: GraphStore, regime: str, plan: Plan, runner: Any, qs: list[queries.Query],
               args: argparse.Namespace, out: Path, shard: tuple[int, int] | None, attempt: int,
               binary: dict[str, Any]) -> dict[str, Any]:
    record = rx.RecordFile(out / "records" / regime / record_name(plan.method, shard))
    counts = {"run": 0, "recorded_before": len(record.done)}
    t0 = time.perf_counter()
    try:
        if args.cache == "warm":
            for q in qs:
                run_query(runner, plan.method, store, regime, q, min(plan.T, float(args.warm_T)), binary=binary)
        warm_seconds = time.perf_counter() - t0
        for position, q in enumerate(qs):
            if q.uid in record.done:
                continue
            cold = evict(runner) if args.cache == "cold" else None
            rec = run_query(runner, plan.method, store, regime, q, plan.T, binary=binary)
            rec.update({"schema": SCHEMA, "arm": plan.method, "arm_settings": {}, "train_seed": None,
                        "model_id": None, "query_index": position, "arm_slot": 0, "mode": MODE,
                        "cache": args.cache, "attempt": attempt, "backend": backend_name(plan.method)})
            if cold is not None:
                rec["cold"] = cold
            record.write(rec)
            counts["run"] += 1
    finally:
        record.close()
    return {"queries": len(qs), "warm_seconds": warm_seconds, "arms": {plan.method: counts},
            "recovered_bytes": {plan.method: record.recovered_bytes} if record.recovered_bytes else {}}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = resolve_plan(args)
    shard = parse_shard(args.shard)
    if args.cache == "cold" and not hasattr(os, "posix_fadvise"):
        raise SystemExit("cold runs evict the solver's input files: they need posix_fadvise (Linux)")
    if args.prepare_only:
        # setup only: reads no query set, so no sealed gate and nothing under --out
        store = GraphStore(args.store_root, args.dataset, mmap=True, partition=args.partition)
        runners = build_runners(store, plan, args)
        return {"prepared": {r: runners[r].prepare() for r in plan.regimes}}
    sets = rx.load_sets(args, plan)
    split = str(next(iter(sets.values())).meta.get("split", "dev"))
    if split in queries.TEST_SPLITS:
        check_sealed_plan(args, plan)  # before the gate, so a refused run leaves no unseal entry
        seal.authorize_sealed_run(unseal=args.unseal,
                                  reason=args.reason or f"run {plan.experiment or 'external'} {plan.method}")
    uids = None
    if args.uids:
        uids = [line.strip() for line in Path(args.uids).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = {r: rx.select_queries(sets[r].queries, uids=uids, shard=shard, **plan.select) for r in plan.regimes}

    threads, limiter = rx.pin_threads(int(protocol.SERVING["threads"]))
    try:
        # the driver maps the store: it only reads query labels and witnesses; each solver loads its own file
        store = GraphStore(args.store_root, args.dataset, mmap=True, partition=args.partition)
        runners = build_runners(store, plan, args)
        path = runners[plan.regimes[0]].binary_path
        binary = {"binary": path, "binary_sha256": binary_digest(path)}
        out = Path(args.out)
        config = run_config(args, plan, shard, sets, selected, runners, binary["binary_sha256"])
        prepared = {r: runners[r].prepare() for r in plan.regimes}  # untimed setup, outside --out
        attempt = rx.check_resume(out, config)
        summary: dict[str, Any] = {"attempt": attempt, "regimes": {}}
        with manifest.RunManifest(out, experiment=plan.experiment or "external", config=config, split=split,
                                  inputs=run_inputs(args, plan, store, sets, runners), solvers={plan.method: path},
                                  fast_input_hashes=args.fast_hashes) as run:
            run.note(threads=threads, attempt=attempt, cache_dir=str(cache_root(args)), prepare=prepared)
            for regime in plan.regimes:
                summary["regimes"][regime] = run_regime(store, regime, plan, runners[regime], selected[regime],
                                                        args, out, shard, attempt, binary)
            summary["memory"] = rx.peak_memory()
            run.note(**summary)
        return summary
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=list(EXPERIMENTS))
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--store-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default="primary")
    p.add_argument("--queries", required=True, help="directory of <regime>.npz query sets of one split master")
    p.add_argument("--regimes", nargs="*")
    p.add_argument("--out", required=True)
    p.add_argument("--T", type=float, default=None, help="time limit per query in seconds")
    p.add_argument("--binary", nargs="+", default=None, help="solver binary, or a command prefix such as "
                   "PYTHON WRAPPER.py (default $SUBGRAPHMATCHING_BIN or $GNNPE_BIN)")
    p.add_argument("--cache-dir", default=None, help=f"data graph files and the GNN-PE index "
                   f"(default <store-root>/{CACHE_DIRNAME})")
    p.add_argument("--data-format", choices=["text", "csr"], default=None,
                   help=f"classical input format (default {DATA_FORMAT})")
    p.add_argument("--gnnpe-partitions", type=int, default=None, help=f"GNN-PE -p (default {GNNPE_PARTITIONS})")
    p.add_argument("--gnnpe-membership", default=None, help=".npy with one GNN-PE part per data vertex "
                   "(default: METIS in process, as gnnpe.py computes it)")
    p.add_argument("--gnnpe-max-paths", type=int, default=None,
                   help="refuse a data graph whose offline stage would store more 3-vertex paths")
    p.add_argument("--prepare-only", action="store_true", help="write the data graph files and the GNN-PE index "
                   "for the plan's regimes, then exit without writing under --out")
    p.add_argument("--cache", choices=["none", "warm", "cold"], default="none")
    p.add_argument("--warm-T", type=float, default=5.0)
    p.add_argument("--kinds", nargs="*")
    p.add_argument("--families", nargs="*")
    p.add_argument("--sizes", type=int, nargs="*")
    p.add_argument("--per-cell", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke runs only: the first N selected queries")
    p.add_argument("--uids", default=None, help="file with one query uid per line")
    p.add_argument("--shard", default=None, help="i/n: every n-th selected query from the i-th")
    p.add_argument("--unseal", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--fast-hashes", action="store_true", help="fingerprint large inputs instead of hashing them")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = execute(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "memory"}, indent=2, sort_keys=True,
                     default=rx._json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
