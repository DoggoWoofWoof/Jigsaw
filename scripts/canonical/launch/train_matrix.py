"""The canonical training matrix: every encoder run as a train_encoder command.

    python -m scripts.canonical.launch.train_matrix --store-root STORE --queries-root QUERIES \\
        --run-root RUNS --models-root MODELS [--filter 'arxiv/*'] [--kind factorial] [--out matrix.json]

Rows (protocol.SEEDS["training"] seeds each):
  main          every dataset x the regimes with their own encoder (regimes_for,
                with every R2-* mapped to R2-512), dataset encoder, arm "main"
  factorial     FACTORIAL arms (the arxiv factor grid and extra arms, the mag arms)
  architecture  ARCHITECTURE_ABLATION variants
Ablation rows train in --ablation-regime (default R0). A row whose resolved
settings equal an earlier row's (for example the arxiv grid cell cvar, barrier
0.35, live 64, which is the main recipe) names that row in duplicate_of; it keeps
its own model_id, seed stream and directories. Every row selects its checkpoint
on the checkpoint-selection master and holds out the node sets of all three
validation masters (--held-out), so no validation query is ever a training draw.

The output is a JSON list of {model_id, kind, dataset, regime, arch, arm, seed,
command, argv, run_dir, val_queries, held_out, model_dir, est_gpu_hours,
est_basis, duplicate_of}. est_gpu_hours is a rough estimate from the schedule and the
dataset size (est_basis says how), not a measurement. Nothing runs unless
--execute is given: then the rows run one after another on this machine,
skipping rows whose exports exist and whose run manifest is complete, and
resuming rows that have checkpoints. --jobs N runs up to N ablation rows at a
time on one GPU (one row keeps it about a quarter busy); main rows still run
alone, because E17 reports their training time.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import posixpath
import shlex
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import manifest, protocol
from .. import train_encoder as te

KINDS = ("main", "factorial", "architecture")
VAL_TEMPLATE = "{queries_root}/{dataset}/validation/{master}/{regime}.npz"
RUN_TEMPLATE = "{run_root}/train/{model_id}"
MODEL_TEMPLATE = "{models_root}/{model_id}"
# The thread variables a scheduler sets from a job's CPUs; rows that share a job split them.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS")
POLL_SECONDS = 5.0

# Rough cost model for est_gpu_hours (one current datacenter GPU; an estimate, not a measurement).
APPROX_NODES = {"corafull": 19_793, "arxiv": 169_343, "mag": 1_939_743, "products": 2_449_029,
                "papers100m": 111_059_956}
TRAIN_NODES_PER_SECOND = 2.0e5  # forward and backward, 6 layers, hidden 256
ENCODE_NODES_PER_SECOND = 1.0e6  # forward only (bank refresh, validation, export)
STEP_OVERHEAD_SECONDS = 0.05  # sampling, covers and input building per step
VALIDATION_SECONDS_PER_QUERY = 0.05  # hybrid selection and covers per query
MEAN_QUERY_SIZE = 57.0  # mean of TRAINING["sampler"]["sizes"] = 20, 50, 100 (before the minimum size)
ARCH_COST = {"graphsage": 1.0, "gin": 1.1, "gat": 1.6, "rgcn": 2.0}
EST_BASIS = ("rough estimate, not a measurement: steps x (query + live block + fine part nodes) at "
             f"{TRAIN_NODES_PER_SECOND:.0e} nodes/s, bank refreshes and evaluations at "
             f"{ENCODE_NODES_PER_SECOND:.0e} nodes/s, x architecture cost x layers/6")


def dataset_nodes(dataset: str, store_root: str | Path | None = None) -> tuple[int, str]:
    """(number of nodes, source): the store's meta.json when present, else APPROX_NODES."""
    if store_root is not None:
        meta = Path(store_root) / dataset / "meta.json"
        if meta.exists():
            return int(json.loads(meta.read_text(encoding="utf-8"))["num_nodes"]), "store meta.json"
    return int(APPROX_NODES.get(dataset, 0)), "approximate"


def estimate_gpu_hours(plan: Mapping[str, Any], num_nodes: int, num_blocks: int | None = None) -> float:
    """Rough GPU hours of one train_encoder run (see EST_BASIS)."""
    schedule, loss, model = plan["schedule"], plan["loss"], plan["model"]
    blocks = int(num_blocks or protocol.DATASETS[plan["dataset"]]["partitions"])
    block_nodes = num_nodes / max(1, blocks)
    fine_nodes = block_nodes / float(protocol.DATASETS[plan["dataset"]].get("fine_per_block", 5))
    batch = int(schedule["batch_size"])
    steps = int(schedule["epochs"]) * int(schedule["steps_per_epoch"])
    live = min(int(loss["live_positives"]), blocks)
    fine = batch * int(loss["fine_positives_per_query"]) if float(loss["fine_nce_weight_alpha"]) > 0 else 0
    step_nodes = batch * MEAN_QUERY_SIZE + live * block_nodes + fine * fine_nodes
    cost = ARCH_COST.get(plan["arch"], 1.0) * int(model["layers"]) / 6.0
    refresh = max(1, int(loss["bank_refresh_steps"]))
    evaluations = math.ceil(int(schedule["epochs"]) / int(schedule["validation_every_epochs"]))
    split = protocol.QUERY_DESIGN["split_counts"]["validation"]
    val_queries = int(split.get("positive", 0))
    seconds = (steps * (step_nodes * cost / TRAIN_NODES_PER_SECOND + STEP_OVERHEAD_SECONDS)
               + math.ceil(steps / refresh) * num_nodes * cost / ENCODE_NODES_PER_SECOND
               + evaluations * (num_nodes * cost / ENCODE_NODES_PER_SECOND + val_queries * VALIDATION_SECONDS_PER_QUERY)
               + 2.0 * num_nodes * cost / ENCODE_NODES_PER_SECOND)
    return round(seconds / 3600.0, 2)


def _settings_key(plan: Mapping[str, Any]) -> str:
    keys = ("dataset", "regime", "seed", "arch", "loss", "model", "schedule", "sampler", "smoke")
    return protocol.canonical_json({k: plan[k] for k in keys})


def matches(entry: Mapping[str, Any], patterns: Sequence[str]) -> bool:
    """True when any glob matches "dataset/arm", the dataset alone or the model_id."""
    if not patterns:
        return True
    names = (f"{entry['dataset']}/{entry['arm']}", entry["dataset"], entry["model_id"])
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns for name in names)


def planned_rows(ablation_regime: str = "R0") -> list[tuple[str, str, str, str | None, int]]:
    """(kind, dataset, regime, arm, seed) of every row, main rows first; arm None means main."""
    seeds = [int(s) for s in protocol.SEEDS["training"]]
    rows: list[tuple[str, str, str, str | None, int]] = []
    for dataset in protocol.DATASETS:
        for regime in te.trained_regimes_for(dataset):
            rows += [("main", dataset, regime, None, seed) for seed in seeds]
    for kind in ("factorial", "architecture"):
        for dataset in protocol.DATASETS:
            for arm, spec in te.arms(dataset).items():
                if spec["kind"] == kind:
                    rows += [(kind, dataset, ablation_regime, arm, seed) for seed in seeds]
    return rows


def enumerate_matrix(*, store_root: str | Path, queries_root: str | Path, run_root: str | Path,
                     models_root: str | Path, ablation_regime: str = "R0", python: str = "python",
                     device: str = "auto", val_template: str = VAL_TEMPLATE,
                     filters: Sequence[str] = (), kinds: Iterable[str] = KINDS) -> list[dict[str, Any]]:
    """Every row of the matrix that passes the filters, with its command and estimate."""
    kinds = set(kinds)
    unknown = kinds - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}")
    master = te.SELECTION_MASTER
    seen: dict[str, str] = {}
    nodes: dict[str, tuple[int, str]] = {}
    out = []
    for kind, dataset, regime, arm, seed in planned_rows(ablation_regime):
        plan = te.resolve_plan(dataset, regime, seed, arm=arm or te.MAIN_ARM)
        key = _settings_key(plan)
        duplicate_of = seen.setdefault(key, plan["model_id"])
        entry: dict[str, Any] = {"model_id": plan["model_id"], "kind": kind, "dataset": dataset,
                                 "regime": plan["regime"], "arch": plan["arch"], "arm": plan["arm"], "seed": seed}
        if kind not in kinds or not matches(entry, filters):
            continue
        if dataset not in nodes:
            nodes[dataset] = dataset_nodes(dataset, store_root)
        fields = {"queries_root": str(queries_root), "dataset": dataset, "master": master, "regime": plan["regime"],
                  "run_root": str(run_root), "models_root": str(models_root), "model_id": plan["model_id"]}
        run_dir = RUN_TEMPLATE.format(**fields)
        val_queries = val_template.format(**fields)
        # every validation master's node sets are held out of training (TRAINING["held_out_nodesets"])
        held_out = [posixpath.dirname(val_template.format(**{**fields, "master": m}))
                    for m in protocol.SEEDS["validation"]]
        argv = [python, "-m", "scripts.canonical.train_encoder", "--dataset", dataset, "--regime", plan["regime"],
                "--seed", str(seed), "--arch", plan["arch"], "--arm", plan["arm"], "--store-root", str(store_root),
                "--val-queries", val_queries, "--run-dir", run_dir, "--models-root", str(models_root),
                "--held-out", *held_out, "--device", device]
        num_nodes, source = nodes[dataset]
        entry.update({
            "command": shlex.join(argv),
            "argv": argv,
            "run_dir": run_dir,
            "val_queries": val_queries,
            "held_out": held_out,
            "model_dir": MODEL_TEMPLATE.format(**fields),
            "est_gpu_hours": estimate_gpu_hours(plan, num_nodes),
            "est_basis": f"{EST_BASIS}; {num_nodes} nodes ({source})",
            "duplicate_of": None if duplicate_of == plan["model_id"] else duplicate_of,
        })
        out.append(entry)
    return out


def summarize(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    hours: dict[str, float] = {}
    for entry in entries:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1
        hours[entry["dataset"]] = round(hours.get(entry["dataset"], 0.0) + float(entry["est_gpu_hours"]), 2)
    return {"rows": len(entries), "by_kind": by_kind,
            "duplicates": sum(1 for e in entries if e["duplicate_of"]),
            "est_gpu_hours_by_dataset": hours,
            "est_gpu_hours_total": round(sum(float(e["est_gpu_hours"]) for e in entries), 2),
            "note": "est_gpu_hours are rough estimates, not measurements"}


def row_complete(entry: Mapping[str, Any]) -> bool:
    """A row is done when its model and run exports exist and its run manifest is complete.

    A run stopped between writing the model export and closing its manifest is not done;
    --resume from its last checkpoint selects and exports again, then completes the manifest.
    """
    model_dir, run_dir = Path(entry["model_dir"]), Path(entry["run_dir"])
    if not (model_dir / te.EXPORT).is_file() or not (run_dir / te.EXPORT).is_file():
        return False
    try:
        record = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("status") == "complete"


def pending_rows(entries: Sequence[Mapping[str, Any]], *,
                 skip_duplicates: bool = False) -> list[tuple[Mapping[str, Any], list[str]]]:
    """(entry, argv) of every row still to run; a row with checkpoints resumes."""
    out = []
    for entry in entries:
        model_dir, run_dir = Path(entry["model_dir"]), Path(entry["run_dir"])
        if skip_duplicates and entry["duplicate_of"]:
            print(f"skip {entry['model_id']}: duplicate of {entry['duplicate_of']}", file=sys.stderr)
            continue
        if row_complete(entry):
            print(f"skip {entry['model_id']}: already exported", file=sys.stderr)
            continue
        if (model_dir / te.EXPORT).exists():
            print(f"finish {entry['model_id']}: exported, but its run manifest is not complete", file=sys.stderr)
        argv = list(entry["argv"])
        if te.list_checkpoints(run_dir):
            argv.append("--resume")
        out.append((entry, argv))
    return out


def shared_env(jobs: int, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a row that shares the job with others: an equal share of the job's threads."""
    env = dict(os.environ if environ is None else environ)
    total = int(env.get("OMP_NUM_THREADS") or os.cpu_count() or 1)
    env.update({name: str(max(1, total // jobs)) for name in THREAD_VARS})
    return env


def execute(entries: Sequence[Mapping[str, Any]], *, skip_duplicates: bool = False, keep_going: bool = False,
            jobs: int = 1, poll_seconds: float = POLL_SECONDS) -> int:
    """Run rows on this machine; returns the number of failed rows.

    With jobs 1 the rows run one after another and print to this process's output. With jobs > 1
    up to that many rows share the machine and its GPU, each with an equal share of the thread
    variables and its output in <run_dir>.log. A main row still runs alone with every thread,
    because E17 reports the main rows' own training time. A failed row stops new rows unless
    keep_going; rows already running finish.
    """
    rows = pending_rows(entries, skip_duplicates=skip_duplicates)
    failed = 0
    if jobs <= 1:
        for entry, argv in rows:
            print(f"run {entry['model_id']}", file=sys.stderr)
            code = subprocess.run(argv, check=False).returncode
            if code != 0:
                failed += 1
                print(f"failed {entry['model_id']}: exit {code}", file=sys.stderr)
                if not keep_going:
                    break
        return failed
    env = shared_env(jobs)
    queue = deque(rows)
    running: list[tuple[Mapping[str, Any], Any, Any]] = []
    stopped = False
    while running or (queue and not stopped):
        while queue and not stopped and len(running) < jobs:
            entry, argv = queue[0]
            alone = entry.get("kind") == "main"
            if running and (alone or any(e.get("kind") == "main" for e, _, _ in running)):
                break
            queue.popleft()
            log_path = Path(f"{entry['run_dir']}.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "ab")
            print(f"run {entry['model_id']}{' alone' if alone else ''}: output in {log_path}", file=sys.stderr)
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=None if alone else env)
            running.append((entry, proc, log))
            if alone:
                break
        time.sleep(poll_seconds)
        still = []
        for entry, proc, log in running:
            code = proc.poll()
            if code is None:
                still.append((entry, proc, log))
                continue
            log.close()
            if code == 0:
                print(f"done {entry['model_id']}", file=sys.stderr)
            else:
                failed += 1
                stopped = stopped or not keep_going
                print(f"failed {entry['model_id']}: exit {code} (see {entry['run_dir']}.log)", file=sys.stderr)
        running = still
    if queue:
        print(f"not started: {len(queue)} rows", file=sys.stderr)
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--queries-root", required=True, help="root of the validation query sets")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--val-template", default=VAL_TEMPLATE,
                        help="path of a row's validation set; fields: queries_root, dataset, master, regime")
    parser.add_argument("--ablation-regime", default="R0", choices=list(protocol.LABEL_REGIMES))
    parser.add_argument("--filter", action="append", default=[],
                        help="glob over dataset/arm, dataset or model_id (repeatable), e.g. 'arxiv/*', 'mag/no_*'")
    parser.add_argument("--kind", action="append", choices=list(KINDS), default=None)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None, help="write the JSON list here instead of stdout")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="list the rows only (default)")
    mode.add_argument("--execute", action="store_true", help="run the rows one after another on this machine")
    parser.add_argument("--skip-duplicates", action="store_true", help="with --execute: skip duplicate_of rows")
    parser.add_argument("--keep-going", action="store_true", help="with --execute: continue after a failed row")
    parser.add_argument("--jobs", type=int, default=1,
                        help="with --execute: rows at a time on one GPU (main rows still run alone)")
    args = parser.parse_args(argv)

    entries = enumerate_matrix(store_root=args.store_root, queries_root=args.queries_root, run_root=args.run_root,
                               models_root=args.models_root, ablation_regime=args.ablation_regime,
                               python=args.python, device=args.device, val_template=args.val_template,
                               filters=args.filter, kinds=args.kind or KINDS)
    text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(json.dumps(summarize(entries), sort_keys=True), file=sys.stderr)
    if args.execute:
        return 1 if execute(entries, skip_duplicates=args.skip_duplicates, keep_going=args.keep_going,
                            jobs=args.jobs) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
