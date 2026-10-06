"""E2 and E10 driver: memory, I/O and latency of the streaming path against the resident systems.

    python -m scripts.canonical.streaming --store-root STORES --dataset mag \\
        --queries QUERIES/mag/dev/20260607 --models-root MODELS --e2-root RUNS/E2 [--external] \\
        [--caps none 2 4 8] [--repeats 3]
    python -m scripts.canonical.streaming --experiment E10 --dataset papers100m ... --e2-root RUNS/E10

Every cell of E2 is one system in one mode, cache state and memory cap, run
--repeats times. The Jigsaw side is run_experiment.experiment_plan("E2"): its
arms in streaming mode (the lean path: memory-mapped store, numpy encoder, no
torch) and the resident single-step methods. --external adds the classical
matchers of run_external.experiment_plan("E2"), which load their whole data
graph in every solver process. E10 (--experiment E10, papers100M) measures the
same way with its own plan: the streaming arms under a cap below the store
size (protocol.SERVING["scale_out"]), cold, against the classical matchers
uncapped and capped. The plan fixes the caps and the repeats of each
experiment.

Each repeat is its own process, launched by memprobe.run_measured: the summed
RSS (and PSS on Linux) of the whole process tree, driver and solver
subprocesses alike, sampled every 20 ms, plus wait4 rusage; under a cap, a
cgroup v2 memory.max with swap disabled; when cold, the store (and the external
solver's cache directory) evicted from the page cache before the process starts,
and by the runner before every query. A system killed by the cap is a result
(PE4: success under each cap), recorded and never retried unless --retry-failed.

Layout (the reducer reads it):
    <e2-root>/<dataset>/<system>/<mode>-<cache>-cap<N|none>/rep<k>/
        measure.json   jigsaw-e2-measure-v1: the cell and memprobe's result
        run/           the runner's directory: manifest.json and records
        stdout.txt, stderr.txt
A cell with a measure.json is skipped, so an interrupted sweep resumes. The
repeats run in rounds (every cell once, then again), so drift of the machine
falls on every system alike.

A sealed query set runs only with the plan's systems, caches, caps and repeats;
every child passes the seal gate on its own (--unseal). measure.json names its
experiment, and a root that holds cells of another experiment is refused.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import manifest, memprobe, protocol, queries
from . import run_experiment as rx

SCHEMA = "jigsaw-e2-measure-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL_TIMEOUT = 6 * 3600.0
EXPERIMENTS = ("E2", "E10")


def plan_caps(plan: dict[str, Any]) -> list[float | None]:
    """The registered caps of a plan: uncapped first, then its memory_caps_gb."""
    return [None, *(float(c) for c in plan["memory_caps_gb"])]


def plan_repeats(plan: dict[str, Any]) -> int:
    return int(plan.get("repeats", protocol.SERVING["repeats"]))


@dataclass(frozen=True)
class Cell:
    system: str  # an arm name (run_experiment) or an external method (run_external)
    runner: str  # "run_experiment" or "run_external"
    mode: str
    cache: str
    cap_gb: float | None
    repeat: int

    @property
    def cap_name(self) -> str:
        return "none" if self.cap_gb is None else f"{self.cap_gb:g}"

    def directory(self, root: Path, dataset: str) -> Path:
        system = self.system.replace("/", "_")
        return root / dataset / system / f"{self.mode}-{self.cache}-cap{self.cap_name}" / f"rep{self.repeat}"


def parse_caps(values: Sequence[str] | None, default: Sequence[float | None] | None = None) -> list[float | None]:
    if not values:
        return list(default) if default is not None else [None, *(float(c) for c in protocol.SERVING["memory_caps_gb"])]
    caps: list[float | None] = []
    for value in values:
        cap = None if value == "none" else float(value)
        if cap is not None and cap <= 0:
            raise SystemExit(f"--caps: {value} is not a positive size in GB")
        caps.append(cap)
    if len(set(caps)) != len(caps):
        raise SystemExit("--caps lists a cap twice")
    return caps


def experiment_plan(experiment: str, dataset: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"streaming.py measures {', '.join(EXPERIMENTS)}, not {experiment}")
    try:
        return rx.experiment_plan(experiment, dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def plan_cells(dataset: str, *, caps: Sequence[float | None], caches: Sequence[str] | None, repeats: int,
               external: bool, systems: Sequence[str] | None = None, experiment: str = "E2") -> list[Cell]:
    """Every cell of the experiment in run order: rounds of repeats, each round every (cap, cache, mode, system)."""
    plan = experiment_plan(experiment, dataset)
    ext = None
    if external:
        from . import run_external  # the classical harness, only when its systems are measured

        ext = run_external.experiment_plan(experiment, dataset)
    chosen = list(caches or plan["caches"])
    unknown = [c for c in chosen if c not in plan["caches"]]
    if unknown:
        raise SystemExit(f"caches outside {experiment}: {unknown}")
    members: list[tuple[str, str, str]] = [(rx.Arm.parse(a).name, "run_experiment", mode)
                                           for mode, arms in plan["modes"].items() for a in arms]
    if ext is not None:
        members += [(m, "run_external", ext["mode"]) for m in ext["methods"]]
    if systems:
        wanted = set(systems)
        known = {m[0] for m in members}
        if wanted - known:
            raise SystemExit(f"systems outside {experiment}: {sorted(wanted - known)}; known: {sorted(known)}")
        members = [m for m in members if m[0] in wanted]
    return [Cell(system, runner, mode, cache, cap, rep)
            for rep in range(1, repeats + 1) for cap in caps for cache in chosen
            for system, runner, mode in members]


def child_command(cell: Cell, args: argparse.Namespace, run_dir: Path) -> list[str]:
    base = ["--store-root", str(args.store_root), "--dataset", args.dataset, "--queries", str(args.queries),
            "--out", str(run_dir), "--experiment", args.experiment, "--cache", cell.cache]
    if cell.runner == "run_experiment":
        cmd = [sys.executable, "-m", "scripts.canonical.run_experiment", *base, "--arms", cell.system,
               "--mode", cell.mode, "--models-root", str(args.models_root)]
        if args.solver_bin:
            cmd += ["--solver-bin", str(args.solver_bin)]
        if args.backend:
            cmd += ["--backend", args.backend]
        if args.router_tau is not None:
            cmd += ["--router-tau", str(args.router_tau)]
    else:
        cmd = [sys.executable, "-m", "scripts.canonical.run_external", *base, "--method", cell.system]
        if args.external_binary:
            cmd += ["--binary", *args.external_binary]
        if args.external_cache_dir:
            cmd += ["--cache-dir", str(args.external_cache_dir)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.per_cell is not None:
        cmd += ["--per-cell", str(args.per_cell)]
    if args.fast_hashes:
        cmd.append("--fast-hashes")
    if args.unseal:
        cmd += ["--unseal", "--reason", args.reason or f"{args.experiment} {cell.system} {cell.mode} {cell.cache} "
                                                         f"cap {cell.cap_name} rep {cell.repeat}"]
    return cmd


def evict_paths(cell: Cell, args: argparse.Namespace) -> list[Path]:
    if cell.cache != "cold":
        return []
    paths = [Path(args.store_root) / args.dataset]
    if cell.runner == "run_external":
        paths.append(Path(args.external_cache_dir) if args.external_cache_dir
                     else Path(args.store_root) / "_external_cache")
    return [p for p in paths if p.exists()]


def run_status(run_dir: Path) -> str | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return "unreadable"


def archive_attempt(cell_dir: Path) -> None:
    """Move a failed cell aside so the new measurement starts from an empty directory."""
    n = 1
    while (cell_dir.parent / f"{cell_dir.name}.attempt{n}").exists():
        n += 1
    cell_dir.rename(cell_dir.parent / f"{cell_dir.name}.attempt{n}")


def measure_cell(cell: Cell, args: argparse.Namespace, cell_dir: Path, mechanism: str) -> dict[str, Any]:
    run_dir = cell_dir / "run"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = child_command(cell, args, run_dir)
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    result = memprobe.run_measured(cmd, timeout=args.cell_timeout,
                                   memory_cap=f"{cell.cap_gb:g}G" if cell.cap_gb is not None else None,
                                   cap_mechanism=mechanism, evict=evict_paths(cell, args),
                                   env=dict(os.environ), cwd=REPO_ROOT, stdout_path=cell_dir / "stdout.txt",
                                   stderr_path=cell_dir / "stderr.txt")
    status = run_status(run_dir)
    measure = {
        "schema": SCHEMA, "experiment": args.experiment, "dataset": args.dataset, "system": cell.system,
        "arm": cell.system,
        "runner": cell.runner, "mode": cell.mode, "cache": cell.cache, "cap_gb": cell.cap_gb,
        "repeat": cell.repeat, "started_at": started, "protocol_hash": protocol.protocol_hash(),
        "queries": str(Path(args.queries).resolve()), "run_status": status,
        "completed": bool(result.get("exit_code") == 0 and not result.get("timed_out") and status == "complete"),
        "oom_killed": result.get("oom_killed"), "cell_timeout_seconds": args.cell_timeout,
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "cpu_count": os.cpu_count()},
        "memprobe": result,
    }
    tmp = cell_dir / "measure.json.tmp"
    tmp.write_text(json.dumps(measure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(cell_dir / "measure.json")
    return measure


def check_setup(args: argparse.Namespace, cells: Sequence[Cell], caps: Sequence[float | None]) -> str:
    """Refuse a sweep that cannot measure what it names; returns the cap mechanism."""
    if any(c.cache == "cold" for c in cells) and not hasattr(os, "posix_fadvise"):
        raise SystemExit("cold cells evict files from the page cache: they need posix_fadvise (Linux); "
                         "pass --caches warm elsewhere")
    mechanism = args.cap_mechanism
    if any(cap is not None for cap in caps):
        mechanism = memprobe.detect_cap_mechanism() if mechanism == "auto" else mechanism
        if mechanism not in ("systemd", "cgroup"):
            raise SystemExit(f"memory caps need a cgroup v2 mechanism (systemd-run or a delegated cgroup); "
                             f"found {mechanism!r}. Pass --caps none to measure without caps")
    qs_dir = Path(args.queries)
    meta_splits = set()
    plan = experiment_plan(args.experiment, args.dataset)
    for regime in plan["regimes"]:
        path = qs_dir / f"{regime}.npz"
        if not path.exists():
            raise SystemExit(f"{path} missing")
        meta_splits.add(str(queries.QuerySet.load(path, verify_hashes=False).meta.get("split", "dev")))
    sealed = bool(meta_splits & set(queries.TEST_SPLITS))
    if sealed:
        # the sweep may be split across machines by system, cache and cap; what a cell measures is fixed
        problems = [flag for flag, value in (("--limit", args.limit), ("--per-cell", args.per_cell),
                                             ("--router-tau", args.router_tau), ("--backend", args.backend))
                    if value is not None]
        if args.repeats != plan_repeats(plan):
            problems.append(f"--repeats {args.repeats} (the plan's is {plan_repeats(plan)})")
        extra = [c for c in caps if c not in plan_caps(plan)]
        if extra:
            problems.append(f"--caps {[c if c is not None else 'none' for c in extra]} outside the protocol")
        if not args.unseal:
            problems.append("sealed query sets need --unseal")
        if problems:
            raise SystemExit(f"a sealed {args.experiment} sweep is the pre-registered one; not allowed: "
                             f"{'; '.join(problems)}")
        # every child passes the gate (and logs it); a sweep that would fail it in every cell stops here
        if not protocol.is_frozen():
            raise SystemExit("protocol is not frozen: the children would refuse every sealed cell")
        if manifest.git_info().get("dirty_code"):
            raise SystemExit("tracked code differs from HEAD: the children would refuse every sealed cell")
    elif args.unseal:
        raise SystemExit("--unseal applies to sealed query sets only")
    return mechanism


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = experiment_plan(args.experiment, args.dataset)
    if args.repeats is None:
        args.repeats = plan_repeats(plan)
    if args.repeats < 1:
        raise SystemExit("--repeats needs at least 1")
    caps = parse_caps(args.caps, plan_caps(plan))
    cells = plan_cells(args.dataset, caps=caps, caches=args.caches, repeats=args.repeats,
                       external=args.external, systems=args.systems, experiment=args.experiment)
    mechanism = check_setup(args, cells, caps)
    root = Path(args.e2_root)
    summary: dict[str, Any] = {"experiment": args.experiment, "cells": len(cells), "measured": 0, "skipped": 0,
                               "failed": 0, "cap_mechanism": mechanism}
    if args.dry_run:
        summary["plan"] = [str(c.directory(root, args.dataset)) for c in cells]
        return summary
    for found in sorted((root / args.dataset).glob("*/*/rep*/measure.json")):
        other = json.loads(found.read_text(encoding="utf-8")).get("experiment", "E2")
        if other != args.experiment:
            raise SystemExit(f"{root / args.dataset} holds {other} cells ({found}); give {args.experiment} "
                             "its own --e2-root")
    for cell in cells:
        cell_dir = cell.directory(root, args.dataset)
        existing = cell_dir / "measure.json"
        if existing.exists():
            done = json.loads(existing.read_text(encoding="utf-8"))
            if done.get("completed") or not args.retry_failed:
                summary["skipped"] += 1
                continue
            archive_attempt(cell_dir)
        elif cell_dir.exists():
            archive_attempt(cell_dir)  # a measurement that died with the driver
        measure = measure_cell(cell, args, cell_dir, mechanism)
        summary["measured"] += 1
        summary["failed"] += 0 if measure["completed"] else 1
        peak, wall = measure["memprobe"].get("peak_rss_bytes"), measure["memprobe"].get("wall_seconds")
        print(f"{cell.system} {cell.mode}-{cell.cache}-cap{cell.cap_name} rep{cell.repeat}: "
              f"completed={measure['completed']} peak_rss={'-' if peak is None else f'{peak / 2**20:.1f} MiB'} "
              f"wall={'-' if wall is None else f'{wall:.1f}s'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--queries", required=True, help="directory of <regime>.npz query sets of one split master")
    p.add_argument("--models-root", default="models/canonical")
    p.add_argument("--experiment", choices=list(EXPERIMENTS), default="E2")
    p.add_argument("--e2-root", "--root", dest="e2_root", required=True,
                   help="measurement root of this experiment (for example RUNS/E2 or RUNS/E10)")
    p.add_argument("--external", action="store_true", help="also measure run_external's methods of the experiment")
    p.add_argument("--systems", nargs="*", default=None, help="dev only: a subset of the experiment's systems")
    p.add_argument("--caps", nargs="*", default=None, help="none and/or GB values (default none plus the plan's "
                   "memory_caps_gb)")
    p.add_argument("--caches", nargs="*", default=None, help="subset of the plan's caches (default all)")
    p.add_argument("--repeats", type=int, default=None, help="default: the plan's (protocol.SERVING)")
    p.add_argument("--cap-mechanism", default="auto", choices=["auto", "systemd", "cgroup"])
    p.add_argument("--cell-timeout", type=float, default=DEFAULT_CELL_TIMEOUT,
                   help="wall seconds before a cell's process tree is killed")
    p.add_argument("--solver-bin", default=None, help="Glasgow binary for the Jigsaw side")
    p.add_argument("--router-tau", type=float, default=None, help="before the freeze: the router threshold")
    p.add_argument("--external-binary", nargs="+", default=None, help="run_external --binary")
    p.add_argument("--external-cache-dir", default=None, help="run_external --cache-dir")
    p.add_argument("--backend", default=None, help="dev only: the Jigsaw side's verifier (run_experiment --backend)")
    p.add_argument("--limit", type=int, default=None, help="smoke runs only: the first N selected queries")
    p.add_argument("--per-cell", type=int, default=None, help="dev only: at most N queries of each (family, size) "
                                                              "cell instead of the plan's")
    p.add_argument("--retry-failed", action="store_true", help="re-measure cells that did not complete "
                   "(the old attempt is kept as <cell>.attemptN)")
    p.add_argument("--dry-run", action="store_true", help="list the cells and exit")
    p.add_argument("--unseal", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--fast-hashes", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    summary = execute(build_parser().parse_args(argv))
    print(json.dumps({k: v for k, v in summary.items() if k != "plan"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
