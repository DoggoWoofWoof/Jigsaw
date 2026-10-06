"""Run cascade methods over query sets: one JSON record per query and arm.

    python -m scripts.canonical.run_experiment --experiment E1 --store-root STORES --dataset arxiv \\
        --queries QUERIES/arxiv/dev/20260607 --models-root MODELS --out RUNS/E1/arxiv-dev-20260607
    python -m scripts.canonical.run_experiment --arms jigsaw_index filterall "jigsaw@7301+halo=0" \\
        --regimes R0 --T 30 ...

Arms. An arm is a cascade method (protocol.METHODS) with its settings, written
METHOD[@SEED][+key=value]... SEED selects the main-arm encoder of that training
seed for the learned rankers. The keys are backend, model (a model id that
replaces the seed's), step_cap, the operator ablations of E9 (halo,
label_filter, prune_passes, component_mode and component_cap) and the
sensitivity settings of E16 (halo_hops, seed_count and policy), escalate (seconds after which the
ladder skips to its whole-graph step) and skip_below (label nodes up to which a query skips retrieval and
the ladder). Both default to the protocol's value for the dataset and regime (none before the freeze) and
+escalate=off or +skip_below=off switches one off. A setting equal
to the protocol's is implied and dropped from the name (except those two, whose protocol value varies with the regime). experiment_plan() fixes
the arms, the time limit, the regimes and the query selection of each named
experiment, and for E16 the arms of every partition it runs on (--partition).

Order. Queries run in set order. For each query the arms run in an order
rotated by the query's position, so drift of the machine over a long run
(temperature, page cache, other tenants) falls on every arm alike.

Records. <out>/records/<regime>/<arm>.jsonl (.shard<i>of<n>.jsonl when sharded)
holds Cascade.run() plus the runner's fields for every (query, arm), among them
the query's I/O and page-fault counters ("io": storage bytes read, faults of the
driver and of its solver subprocesses; Linux). Running the
same command again resumes: it keeps complete lines, skips recorded pairs, and
refuses to continue when a setting that changes results differs from the first
attempt. An exception inside a run is recorded as ERROR with its traceback and
the run continues; the reducer counts every ERROR against PE1.

Sealed query sets (meta.split test or reserve_*) run only as a named experiment
(its arms or a subset of them, its T, regimes, mode, cache and query selection;
the protocol's ladder and router threshold), and only with --unseal, a frozen
protocol and committed code (seal.authorize_sealed_run).

Modes. resident loads the store arrays into RAM and, before the first query,
the inputs each arm reads; streaming memory-maps them (the lean path of E2) and
never imports torch. Cache warm runs one untimed pass first, capped at --warm-T
seconds per run; cache cold (Linux) evicts the store files from the page cache
before every run, releasing the mappings first when streaming (a resident run
keeps its loaded arrays, so for it cold means a start from disk, which E2's
driver, streaming.py, arranges by evicting the store before the process starts).
BLAS and the solvers run with protocol.SERVING["threads"] threads.
"""

from __future__ import annotations

import os

# Before numpy loads: BLAS reads these once (protocol.SERVING["threads"], enforced by pin_threads()).
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from . import backends, cascade, manifest, protocol, queries, rankers, seal, serving  # noqa: E402
from .store import GraphStore  # noqa: E402

SCHEMA = "jigsaw-run-record-v1"
LEARNED_METHODS = ("jigsaw", "jigsaw_learned", "jigsaw_fused", "mean_rrf")
FIXED_METHODS = ("jigsaw_index", "mean_feature", "topo_feature", "random", "filterall")
ABLATIONS = ("", "+halo=0", "+prune_passes=0", "+component_mode=none", "+component_cap=50", "+label_filter=0")
BACKENDS = {"glasgow": "glasgow", "networkx": "networkx", "networkx_vf2": "networkx", "vf2": "networkx",
            "gql": "gql", "cfl": "cfl", "dpiso": "dpiso"}
# The bank inputs each ranker reads (block embeddings load with the cascade).
RANKER_INPUTS = {"index": ("postings",), "index_count": ("postings",), "fused_learned_index": ("postings",),
                 "router": ("postings",), "mean_feature": ("mean_inputs",), "fused_learned_mean": ("mean_inputs",),
                 "topo_feature": ("topo",)}


def _flag(text: str) -> bool:
    if text in ("1", "true", "True", "yes"):
        return True
    if text in ("0", "false", "False", "no"):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def _backend(text: str) -> str:
    if text not in BACKENDS:
        raise ValueError(f"unknown backend {text!r}; known: {', '.join(sorted(set(BACKENDS.values())))}")
    return BACKENDS[text]


def _component_mode(text: str) -> str:
    if text not in ("separate", "merged", "none"):
        raise ValueError(f"unknown component mode {text!r}")
    return text


def _hops(text: str) -> int:
    hops = int(text)
    if hops < 2:
        raise ValueError(f"halo_hops={text}: the canonical halo has one hop and +halo=0 drops it; "
                         "halo_hops takes 2 or more")
    return hops


def _policy(text: str) -> str:
    if text not in ("hybrid", "topk"):
        raise ValueError(f"unknown selection policy {text!r}")
    return text


def _escalate(text: str) -> float | str:
    if text == cascade.LADDER_OFF:
        return text
    seconds = float(text)
    if seconds <= 0:
        raise ValueError(f"escalate={text}: the jump to the whole-graph step takes a positive number of seconds")
    return seconds


def _skip_below(text: str) -> int | str:
    if text == cascade.LADDER_OFF:
        return text
    nodes = int(text)
    if nodes < 1:
        raise ValueError(f"skip_below={text}: the label-node count below which the ladder is skipped is at least 1")
    return nodes


def _seed_count(text: str) -> int:
    count = int(text)
    if count < 1:
        raise ValueError(f"seed_count={text}: the hybrid policy needs at least one seed block")
    return count


ARM_KEYS = {"backend": _backend, "model": str, "step_cap": float, "halo": _flag, "label_filter": _flag,
            "prune_passes": int, "component_mode": _component_mode, "component_cap": int,
            "halo_hops": _hops, "policy": _policy, "seed_count": _seed_count, "escalate": _escalate,
            "skip_below": _skip_below}
CASCADE_KEYS = ("step_cap", "halo", "label_filter", "prune_passes", "component_mode", "component_cap", "policy",
                "escalate", "skip_below")


# ----------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    method: str
    seed: int | None = None
    settings: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Arm":
        head, *pairs = text.split("+")
        method, _, seed = head.partition("@")
        spec = protocol.METHODS.get(method)
        if spec is None or "external" in spec:
            raise ValueError(f"{method!r} is not a cascade method")
        settings: dict[str, Any] = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep or key not in ARM_KEYS:
                raise ValueError(f"bad arm setting {pair!r} in {text!r}; keys: {', '.join(ARM_KEYS)}")
            if key in settings:
                raise ValueError(f"{key} given twice in {text!r}")
            settings[key] = ARM_KEYS[key](value)
        if settings.get("backend") == protocol.VERIFIERS["default"]:
            del settings["backend"]  # one name per arm: the default verifier is implied
        if "halo_hops" in settings and "halo" in settings:
            raise ValueError(f"halo and halo_hops both set in {text!r}")
        if ("escalate" in settings or "skip_below" in settings) and not spec.get("ranker"):
            raise ValueError(f"{method} selects no blocks and has no ladder to skip: drop escalate and skip_below")
        if ("policy" in settings or "seed_count" in settings) and not spec.get("ranker"):
            raise ValueError(f"{method} selects no blocks: drop policy and seed_count")
        if "seed_count" in settings and settings.get("policy", spec["policy"]) != "hybrid":
            raise ValueError(f"seed_count sets the hybrid policy's seeds; {text!r} selects top-k")
        if settings.get("policy") == spec["policy"]:
            del settings["policy"]  # the method's own policy is implied
        if settings.get("seed_count") == protocol.SELECTION["hybrid"]["seed_count"]:
            del settings["seed_count"]  # the protocol's seed count is implied
        learned = spec.get("ranker") in rankers.NEEDS_EMBEDDING
        if learned and not seed and "model" not in settings:
            raise ValueError(f"{method} ranks with an encoder: write {method}@SEED or +model=ID")
        if not learned and (seed or "model" in settings):
            raise ValueError(f"{method} reads no encoder; drop the seed and model")
        return cls(method, int(seed) if seed else None, tuple(sorted(settings.items())))

    @property
    def name(self) -> str:
        text = self.method + (f"@{self.seed}" if self.seed is not None else "")
        for key, value in self.settings:
            text += f"+{key}={int(value) if isinstance(value, bool) else value}"
        return text

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self.settings).get(key, default)

    def with_setting(self, key: str, value: Any) -> "Arm":
        """This arm with `key` set, unless the arm already sets it."""
        return Arm.parse(f"{self.name}+{key}={value}") if self.get(key) is None else self

    @property
    def backend(self) -> str:
        return self.get("backend") or protocol.VERIFIERS["default"]

    @property
    def learned(self) -> bool:
        return protocol.METHODS[self.method].get("ranker") in rankers.NEEDS_EMBEDDING

    def model_id(self, dataset: str, regime: str) -> str | None:
        if not self.learned:
            return None
        return self.get("model") or serving.default_model_id(dataset, regime, int(self.seed))


# ----------------------------------------------------------------------------
# Named experiments
# ----------------------------------------------------------------------------
def experiment_plan(name: str, dataset: str) -> dict[str, Any]:
    """Arms, time limit, regimes and query selection of a named experiment (plan section 4)."""
    seeds = [int(s) for s in protocol.SEEDS["training"]]
    first = seeds[0]
    regimes = protocol.regimes_for(dataset)
    t_max = protocol.TIME["run_T_max_seconds"]
    if name == "E1":
        # one run at T_max gives solved@t for every t <= T_max (E3 reads the same records)
        return {"arms": [f"{m}@{s}" for m in LEARNED_METHODS for s in seeds] + list(FIXED_METHODS),
                "T": t_max["E1"], "regimes": [r for r in ("R0", "R1") if r in regimes], "select": {}}
    if name == "E2":
        # memory, I/O and latency: streaming.py runs each (arm, mode, cache, cap) in its own measured
        # process; the lean streaming path against the resident single-step methods
        streaming = [f"jigsaw@{first}", "jigsaw_index", "filterall"]
        resident = ["filterall", "direct"]
        return {"arms": list(dict.fromkeys(streaming + resident)), "T": protocol.TIME["primary_T_seconds"],
                "regimes": ["R0"], "select": {"per_cell": 10},
                "modes": {"streaming": streaming, "resident": resident}, "caches": ["cold", "warm"],
                "memory_caps_gb": list(protocol.SERVING["memory_caps_gb"])}
    if name == "E10":
        # scale-out beyond RAM: the streaming path under a cap below the store size, measured by
        # streaming.py as E2 is; the classical matchers load the whole graph (run_external's E10)
        scale = protocol.SERVING["scale_out"]
        if dataset not in scale["datasets"]:
            raise ValueError(f"E10 runs on {', '.join(scale['datasets'])}, not {dataset}")
        streaming = [f"jigsaw@{first}", "jigsaw_index", "filterall"]
        return {"arms": streaming, "T": protocol.TIME["primary_T_seconds"], "regimes": ["R0"],
                "select": {"totals": {"positive": 600, "negative": 200}}, "modes": {"streaming": streaming},
                "caches": ["cold"], "memory_caps_gb": list(scale["memory_caps_gb"]), "repeats": int(scale["repeats"])}
    if name == "E3x":
        # the 600 s extension, on the queries E1 left undecided (--uids)
        return {"arms": [f"jigsaw@{first}", "filterall"], "T": t_max["E3_extension"],
                "regimes": [r for r in ("R0", "R1") if r in regimes], "select": {"needs_uids": True}}
    if name == "E7":
        # the Jigsaw side of the selectivity sweep; the classical side runs through external/ on the same sets
        return {"arms": [f"jigsaw@{first}", f"jigsaw@{first}+backend=glasgow", "filterall", "filterall+backend=glasgow"],
                "T": t_max["E7"], "regimes": regimes, "select": {"per_cell": 50}}
    if name == "E8":
        # single-step methods: one run at 600 s also gives solved@30
        return {"arms": ["direct", "filterall"], "T": max(t_max["E8"]), "regimes": ["R0"],
                "select": {"kinds": ["positive"], "per_cell": 5}}
    if name == "E9":
        return {"arms": [f"jigsaw@{first}{a}" for a in ABLATIONS], "T": protocol.TIME["primary_T_seconds"],
                "regimes": ["R0"], "select": {"totals": {"positive": 300, "negative": 100}}}
    if name == "E16":
        # sensitivity: halo hops {0, 1, 2}, hybrid seed count {10, 20, 40} and top-k selection on the
        # primary partition; the partitioner and the block count on each sensitivity partition
        # (--partition NAME, the ladder scaled to its block count), with the two fixed references
        base = f"jigsaw@{first}"
        primary = [base, f"{base}+halo=0", f"{base}+halo_hops=2", f"{base}+seed_count=10",
                   f"{base}+seed_count=40", f"{base}+policy=topk", "jigsaw_index", "filterall"]
        partitions = {"primary": primary, **{p: [base, "jigsaw_index", "filterall"] for p in sensitivity_partitions()}}
        return {"arms": primary, "T": protocol.TIME["primary_T_seconds"], "regimes": ["R0"],
                "select": {"totals": {"positive": 300, "negative": 100}}, "partitions": partitions}
    raise ValueError(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS)}")


EXPERIMENTS = ("E1", "E2", "E3x", "E7", "E8", "E9", "E10", "E16")


def sensitivity_partitions() -> list[str]:
    """The partitions of E16 besides the primary one, as prepare_data names them."""
    scales = [float(s) for s in protocol.PARTITIONING["partition_scale_for_sensitivity"] if float(s) != 1.0]
    return list(protocol.PARTITIONING["alternatives_for_sensitivity"]) + [f"metis_x{s:g}" for s in scales]


def plan_partitions(spec: dict[str, Any]) -> dict[str, list[str]]:
    """The arms of a named experiment on each partition it runs on (the primary one unless it says)."""
    return spec.get("partitions", {"primary": spec["arms"]})


def check_sealed_plan(args: argparse.Namespace, plan: Plan) -> None:
    """A sealed run is a pre-registered experiment: its arms (a subset, for one process per arm), its T,
    its regimes and its query selection, with the protocol's ladder and router threshold."""
    problems = [flag for flag, value in (("--budgets", args.budgets), ("--limit", args.limit),
                                         ("--router-tau", args.router_tau), ("--kinds", args.kinds),
                                         ("--families", args.families), ("--sizes", args.sizes),
                                         ("--per-cell", args.per_cell)) if value is not None]
    if not args.experiment:
        problems.append("ad-hoc --arms (name an --experiment)")
    else:
        spec = experiment_plan(args.experiment, args.dataset)
        modes = spec.get("modes", {"resident": spec["arms"]})  # a plan without modes runs resident
        partitions = plan_partitions(spec)
        if args.mode not in modes:
            problems.append(f"--mode {args.mode}")
        if args.partition not in partitions:
            problems.append(f"--partition {args.partition}")
        registered = {Arm.parse(a).name for a in modes.get(args.mode, [])} & \
            {Arm.parse(a).name for a in partitions.get(args.partition, [])}
        outside = [a.name for a in plan.arms if a.name not in registered]
        if outside:
            problems.append(f"arms outside {args.experiment} in {args.mode} mode on partition {args.partition}: "
                            f"{', '.join(outside)}")
        if plan.T != float(spec["T"]):
            problems.append(f"--T {plan.T:g} (the plan's is {float(spec['T']):g})")
        if not set(plan.regimes) <= set(spec["regimes"]):
            problems.append(f"regimes outside the plan: {sorted(set(plan.regimes) - set(spec['regimes']))}")
        if args.cache not in spec.get("caches", ["none"]):
            problems.append(f"--cache {args.cache}")
    if problems:
        raise SystemExit(f"a sealed run is a pre-registered experiment; not allowed: {'; '.join(problems)}")


# ----------------------------------------------------------------------------
# Query selection
# ----------------------------------------------------------------------------
def _cells(qs: Sequence[queries.Query]) -> dict[tuple[str, int], list[queries.Query]]:
    cells: dict[tuple[str, int], list[queries.Query]] = {}
    for q in qs:
        cells.setdefault((q.family, q.size), []).append(q)
    return cells


def balanced(qs: Sequence[queries.Query], total: int) -> list[queries.Query]:
    """`total` queries taken round-robin over the (family, size) cells, each cell in set order."""
    cells = list(_cells(qs).values())
    taken: set[str] = set()
    depth = 0
    while len(taken) < total and any(depth < len(c) for c in cells):
        for cell in cells:
            if depth < len(cell) and len(taken) < total:
                taken.add(cell[depth].uid)
        depth += 1
    return [q for q in qs if q.uid in taken]


def select_queries(qs: Sequence[queries.Query], *, kinds: Iterable[str] | None = None,
                   families: Iterable[str] | None = None, sizes: Iterable[int] | None = None,
                   per_cell: int | None = None, totals: dict[str, int] | None = None,
                   uids: Iterable[str] | None = None, limit: int | None = None,
                   shard: tuple[int, int] | None = None) -> list[queries.Query]:
    """The queries of a run, in set order."""
    out = list(qs)
    if uids is not None:
        wanted = set(uids)
        out = [q for q in out if q.uid in wanted]
    if kinds:
        out = [q for q in out if q.kind in set(kinds)]
    if families:
        out = [q for q in out if q.family in set(families)]
    if sizes:
        out = [q for q in out if q.size in {int(s) for s in sizes}]
    if per_cell is not None:
        keep = {q.uid for cell in _cells(out).values() for q in cell[:int(per_cell)]}
        out = [q for q in out if q.uid in keep]
    if totals:
        keep = set()
        for kind, total in totals.items():
            keep |= {q.uid for q in balanced([q for q in out if q.kind == kind], int(total))}
        out = [q for q in out if q.uid in keep]
    if limit is not None:
        out = out[:int(limit)]
    if shard is not None:
        index, count = shard
        out = [q for i, q in enumerate(out) if i % count == index]
    return out


def arm_order(num_arms: int, position: int) -> list[int]:
    """Arm indices for the query at `position`: a rotation, so every arm takes every slot equally often."""
    return [(position + j) % num_arms for j in range(num_arms)]


# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


class RecordFile:
    """Append-only JSONL of one (regime, arm); a torn last line from a crash is cut on open."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        self.recovered_bytes = 0
        if path.exists():
            data = path.read_bytes()
            end = data.rfind(b"\n") + 1
            if end < len(data):
                self.recovered_bytes = len(data) - end
                with open(path, "r+b") as handle:
                    handle.truncate(end)
                data = data[:end]
            for line in data.splitlines():
                if line.strip():
                    self.done.add(json.loads(line)["uid"])
        self.handle = open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.done.add(record["uid"])

    def close(self) -> None:
        self.handle.close()


def read_records(path: str | os.PathLike) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def pin_threads(threads: int) -> tuple[dict[str, Any], Any]:
    """Limit the BLAS pools already loaded; the environment covers solver processes and late loads."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    info: dict[str, Any] = {"threads": threads, "environment": {k: os.environ.get(k) for k in THREAD_VARIABLES}}
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        info["threadpoolctl"] = "missing: only the environment variables apply"
        return info, None
    limiter = threadpool_limits(limits=threads)
    info["pools"] = [{k: p.get(k) for k in ("internal_api", "num_threads", "version")} for p in threadpool_info()]
    if any(p["num_threads"] != threads for p in info["pools"]):
        limiter.restore_original_limits()
        raise RuntimeError(f"a BLAS pool ignored the thread limit: {info['pools']}")
    return info, limiter


def peak_memory() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import resource

        scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KiB on Linux
        out["self_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        out["children_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    except ImportError:
        try:
            import psutil

            out["self_peak_wset_bytes"] = int(getattr(psutil.Process().memory_info(), "peak_wset", 0))
        except ImportError:
            out["note"] = "no resource or psutil"
    return out


def mapped_arrays(store: GraphStore, cascades: Iterable[cascade.Cascade]) -> list[np.ndarray]:
    arrays = list(store.open_arrays())
    for c in cascades:
        if c._label_index is not None:
            arrays += [c._label_index.ptr, c._label_index.nodes]
        if c.bank is not None and c.bank._postings is not None:
            arrays += [c.bank._postings.ptr, c.bank._postings.blocks]
    return [a for a in arrays if isinstance(a, np.memmap)]


def make_cold(store: GraphStore, cascades: Iterable[cascade.Cascade]) -> dict[str, Any]:
    """Drop the pages of every mapped store file: madvise(DONTNEED) on the mappings, then fadvise."""
    from . import memprobe

    released = sum(1 for a in mapped_arrays(store, cascades) if memprobe.release_mapping(a))
    report = memprobe.evict_files([store.dir])
    return {"released": released, "evicted_files": report["files"], "evicted_bytes": report["bytes"],
            "method": report["method"], "errors": len(report["errors"])}


def preload(c: cascade.Cascade) -> None:
    """Load what this arm reads before the first query (resident mode), and nothing else."""
    c.type_pairs
    direct = c.spec.get("pruning") == "solver only"
    if direct or not c.label_filter:
        c.full_target()
    if not direct and c.label_filter:
        c.label_index
    if c.bank is not None:
        for name in RANKER_INPUTS.get(c.spec["ranker"], ()):
            getattr(c.bank, name)


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
@dataclass
class Plan:
    experiment: str | None
    arms: list[Arm]
    T: float
    regimes: list[str]
    select: dict[str, Any]


def resolve_plan(args: argparse.Namespace) -> Plan:
    if args.experiment:
        try:
            spec = experiment_plan(args.experiment, args.dataset)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        default_arms = plan_partitions(spec).get(args.partition, spec["arms"])
        arms = [Arm.parse(a) for a in (args.arms or default_arms)]
        select = dict(spec["select"])
        T = float(args.T if args.T is not None else spec["T"])
        regimes = list(args.regimes or spec["regimes"])
    else:
        if not args.arms or args.T is None or not args.regimes:
            raise SystemExit("without --experiment give --arms, --T and --regimes")
        arms, select, T, regimes = [Arm.parse(a) for a in args.arms], {}, float(args.T), list(args.regimes)
    for key in ("kinds", "families", "sizes", "per_cell", "limit"):
        value = getattr(args, key)
        if value is not None:
            select[key] = value
    if select.pop("needs_uids", False) and not args.uids:
        raise SystemExit(f"{args.experiment} runs on a list of query uids: pass --uids FILE")
    if args.backend:
        arms = [a.with_setting("backend", args.backend) for a in arms]
    if len({a.name for a in arms}) != len(arms):
        raise SystemExit("an arm is listed twice")
    unknown = [r for r in regimes if r not in protocol.TRAINED_REGIMES]
    if unknown:
        raise SystemExit(f"unknown regimes {unknown}")
    return Plan(args.experiment, arms, T, regimes, select)


def build_cascades(store: GraphStore, regime: str, arms: Sequence[Arm], args: argparse.Namespace,
                   solvers: dict[str, Any]) -> list[cascade.Cascade]:
    out = []
    for arm in arms:
        if arm.backend not in solvers:
            kwargs = {"binary": args.solver_bin} if arm.backend == "glasgow" and args.solver_bin else {}
            solvers[arm.backend] = backends.make_backend(arm.backend, **kwargs)
        model_id = arm.model_id(store.dataset, regime)
        kwargs = {k: arm.get(k) for k in CASCADE_KEYS if arm.get(k) is not None}
        if arm.get("halo_hops") is not None:
            kwargs["halo"] = int(arm.get("halo_hops"))
        if arm.get("seed_count") is not None:
            kwargs["policy_params"] = {"seed_count": int(arm.get("seed_count"))}
        if args.budgets:
            kwargs["budgets"] = [int(b) for b in args.budgets]
        out.append(serving.make_cascade(store, regime, arm.method, solvers[arm.backend],
                                        model_dir=Path(args.models_root) / model_id if model_id else None,
                                        router_tau=args.router_tau, **kwargs))
    return out


def run_config(args: argparse.Namespace, plan: Plan, shard: tuple[int, int] | None,
               sets: dict[str, queries.QuerySet], selected: dict[str, list[queries.Query]],
               cascades: dict[str, list[cascade.Cascade]]) -> dict[str, Any]:
    """Every setting that changes the records; a resumed run must match it exactly."""
    models = {}
    options = {}
    for regime, row in cascades.items():
        options[regime] = {}
        for arm, c in zip(plan.arms, row):
            options[regime][arm.name] = {**c.options(), "budgets": list(c.budgets), "step_cap": c.step_cap,
                                         "router_tau": getattr(c.bank, "router_tau", None)}
            if c.embed is not None:
                models[f"{regime}/{arm.name}"] = serving.model_identity(c.embed.encoder)
    return {
        "schema": SCHEMA, "experiment": plan.experiment, "dataset": args.dataset, "partition": args.partition,
        "regimes": plan.regimes, "arms": [a.name for a in plan.arms], "T": plan.T, "mode": args.mode,
        "cache": args.cache, "warm_T": args.warm_T if args.cache == "warm" else None,
        "budgets": args.budgets, "router_tau": args.router_tau, "solver_bin": args.solver_bin,
        "select": plan.select, "uids_file": args.uids, "shard": list(shard) if shard else None,
        "query_sets": {r: {"path": str(Path(sets[r].path).resolve()), "split": sets[r].meta.get("split"),
                           "master": sets[r].meta.get("master"), "selected": len(selected[r])} for r in plan.regimes},
        "cascade_options": options, "models": models, "threads": int(protocol.SERVING["threads"]),
    }


def run_inputs(args: argparse.Namespace, plan: Plan, store: GraphStore,
               sets: dict[str, queries.QuerySet]) -> dict[str, Any]:
    inputs: dict[str, Any] = {"store_meta": store.dir / "meta.json",
                              "partition": Path(store.part.dir) / "partition.json",
                              "rankers/topology": rankers.artifact_dir(store, "topology") / "info.json"}
    for regime in plan.regimes:
        inputs[f"queries/{regime}"] = sets[regime].path
        inputs[f"label_index/{regime}"] = cascade.label_index_dir(store, regime) / "info.json"
        inputs[f"rankers/{regime}"] = rankers.artifact_dir(store, regime) / "info.json"
        for arm in plan.arms:
            model_id = arm.model_id(args.dataset, regime)
            if model_id:
                inputs[f"model/{model_id}"] = Path(args.models_root) / model_id / "model.json"
                emb = rankers.embeddings_dir(store, model_id, regime)
                inputs[f"embeddings/{model_id}/{regime}"] = emb / "block_emb.npy"
                inputs[f"embeddings/{model_id}/{regime}/info"] = emb / "info.json"
    return {k: v for k, v in inputs.items() if Path(v).exists()}


def check_resume(out: Path, config: dict[str, Any]) -> int:
    """Attempt number of this run; archives the previous manifest after checking its settings."""
    path = out / manifest.MANIFEST_NAME
    attempts = sorted(out.glob("manifest.attempt*.json"))
    if not path.exists():
        return len(attempts) + 1
    previous = json.loads(path.read_text(encoding="utf-8"))
    old = json.loads(json.dumps(previous.get("config", {}), default=_json_default))
    new = json.loads(json.dumps(config, default=_json_default))
    if old != new:
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        raise SystemExit(f"{out} holds a run with other settings ({', '.join(changed)}); use another --out")
    if previous.get("protocol_hash") != protocol.protocol_hash():
        raise SystemExit(f"{out} was run under protocol {previous.get('protocol_hash')}; use another --out")
    path.replace(out / f"manifest.attempt{len(attempts) + 1}.json")
    return len(attempts) + 2


def record_name(arm: Arm, shard: tuple[int, int] | None) -> str:
    safe = arm.name.replace("/", "_")
    return f"{safe}.shard{shard[0]}of{shard[1]}.jsonl" if shard else f"{safe}.jsonl"


def io_snapshot() -> dict[str, int]:
    """Cumulative I/O and page-fault counters of this process and its reaped children (the solver
    subprocesses). Linux: /proc/self/io and getrusage; elsewhere whatever getrusage offers."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/io", "rb") as handle:
            for line in handle:
                key, _, value = line.decode("ascii").partition(":")
                if key in ("rchar", "read_bytes"):
                    out[key] = int(value)
    except OSError:
        pass
    try:
        import resource
    except ImportError:
        return out
    for who, prefix in ((resource.RUSAGE_SELF, ""), (resource.RUSAGE_CHILDREN, "children_")):
        usage = resource.getrusage(who)
        out[prefix + "major_faults"] = int(usage.ru_majflt)
        out[prefix + "minor_faults"] = int(usage.ru_minflt)
        out[prefix + "block_input_bytes"] = int(usage.ru_inblock) * 512
    return out


def run_one(c: cascade.Cascade, q: queries.Query, T: float) -> dict[str, Any]:
    io_before = io_snapshot()
    t0 = time.perf_counter()
    try:
        rec = c.run(q, time_limit=T)
    except Exception as exc:  # a bug: recorded and counted by the reducer, the run goes on
        rec = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind, "regime": c.regime,
               "method": c.method, "dataset": c.store.dataset, "T": T, "certified": q.certified,
               "status": backends.ERROR, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(limit=8), "time_found": None, "time_decided": None,
               "decided_step": None, "steps": []}
    rec["wall_seconds"] = time.perf_counter() - t0
    io_after = io_snapshot()
    rec["io"] = {k: io_after[k] - io_before[k] for k in io_after if k in io_before}
    return rec


def load_sets(args: argparse.Namespace, plan: Plan) -> dict[str, queries.QuerySet]:
    sets = {}
    for regime in plan.regimes:
        qs = queries.QuerySet.load(Path(args.queries) / f"{regime}.npz")
        if qs.meta.get("dataset") not in (None, args.dataset) or qs.regime != regime:
            raise SystemExit(f"{qs.path} holds {qs.meta.get('dataset')}/{qs.regime}, not {args.dataset}/{regime}")
        sets[regime] = qs
    splits = {str(qs.meta.get("split")) for qs in sets.values()}
    if len(splits) > 1:
        raise SystemExit(f"the query sets come from different splits: {sorted(splits)}")
    return sets


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = resolve_plan(args)
    shard = None
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit("--shard i/n needs 0 <= i < n")
        shard = (index, count)
    if args.cache == "cold" and not hasattr(os, "posix_fadvise"):
        raise SystemExit("cold runs evict the store files from the page cache: they need posix_fadvise (Linux)")
    sets = load_sets(args, plan)
    split = str(next(iter(sets.values())).meta.get("split", "dev"))
    if split in queries.TEST_SPLITS:
        check_sealed_plan(args, plan)  # before the gate, so a refused run leaves no unseal entry
        seal.authorize_sealed_run(unseal=args.unseal, reason=args.reason or f"run {plan.experiment or 'arms'}")
    uids = None
    if args.uids:
        uids = [line.strip() for line in Path(args.uids).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = {r: select_queries(sets[r].queries, uids=uids, shard=shard, **plan.select) for r in plan.regimes}

    threads, limiter = pin_threads(int(protocol.SERVING["threads"]))
    try:
        store = GraphStore(args.store_root, args.dataset, mmap=args.mode == "streaming", partition=args.partition)
        solvers: dict[str, Any] = {}
        # every arm is assembled before the manifest, so a missing model or threshold writes nothing
        cascades = {r: build_cascades(store, r, plan.arms, args, solvers) for r in plan.regimes}
        out = Path(args.out)
        config = run_config(args, plan, shard, sets, selected, cascades)
        attempt = check_resume(out, config)
        glasgow = args.solver_bin or os.environ.get("GLASGOW_SOLVER_BIN")
        solver_paths = {"glasgow": glasgow} if "glasgow" in solvers else {}
        summary: dict[str, Any] = {"attempt": attempt, "regimes": {}}
        with manifest.RunManifest(out, experiment=plan.experiment or "arms", config=config, split=split,
                                  inputs=run_inputs(args, plan, store, sets), solvers=solver_paths,
                                  fast_input_hashes=args.fast_hashes) as run:
            run.note(threads=threads, attempt=attempt)
            for regime in plan.regimes:
                summary["regimes"][regime] = run_regime(store, regime, plan, cascades[regime], selected[regime],
                                                        args, out, shard, attempt)
            summary["torch_imported"] = "torch" in sys.modules
            summary["memory"] = peak_memory()
            if args.mode == "streaming" and summary["torch_imported"]:
                summary["warning"] = "torch was imported in streaming mode"
            run.note(**summary)
        return summary
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def run_regime(store: GraphStore, regime: str, plan: Plan, cascades: list[cascade.Cascade],
               qs: list[queries.Query], args: argparse.Namespace, out: Path, shard: tuple[int, int] | None,
               attempt: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    if args.mode == "resident":
        for c in cascades:
            preload(c)
    load_seconds = time.perf_counter() - t0
    files = [RecordFile(out / "records" / regime / record_name(arm, shard)) for arm in plan.arms]
    counts = {arm.name: {"run": 0, "recorded_before": len(f.done)} for arm, f in zip(plan.arms, files)}
    try:
        if args.cache == "warm":
            for q in qs:
                for c in cascades:
                    run_one(c, q, min(plan.T, float(args.warm_T)))
        for position, q in enumerate(qs):
            for slot, j in enumerate(arm_order(len(plan.arms), position)):
                arm, c, f = plan.arms[j], cascades[j], files[j]
                if q.uid in f.done:
                    continue
                cold = make_cold(store, cascades) if args.cache == "cold" else None
                rec = run_one(c, q, plan.T)
                rec.update({"schema": SCHEMA, "arm": arm.name, "arm_settings": dict(arm.settings),
                            "train_seed": arm.seed, "query_index": position, "arm_slot": slot,
                            "mode": args.mode, "cache": args.cache, "attempt": attempt,
                            "backend": c.backend.name, "model_id": arm.model_id(store.dataset, regime)})
                if cold is not None:
                    rec["cold"] = cold
                f.write(rec)
                counts[arm.name]["run"] += 1
    finally:
        for f in files:
            f.close()
    return {"queries": len(qs), "load_seconds": load_seconds, "arms": counts,
            "recovered_bytes": {arm.name: f.recovered_bytes for arm, f in zip(plan.arms, files) if f.recovered_bytes}}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=list(EXPERIMENTS))
    p.add_argument("--arms", nargs="*", help="arms to run (replaces the experiment's)")
    p.add_argument("--store-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default="primary")
    p.add_argument("--queries", required=True, help="directory of <regime>.npz query sets of one split master")
    p.add_argument("--regimes", nargs="*")
    p.add_argument("--models-root", default="models/canonical")
    p.add_argument("--out", required=True)
    p.add_argument("--T", type=float, default=None, help="time limit per query in seconds")
    p.add_argument("--budgets", type=int, nargs="*", help="smoke runs only: replaces the protocol ladder")
    p.add_argument("--router-tau", type=float, default=None, help="before the freeze: the router threshold")
    p.add_argument("--backend", default=None, help="verifier of every arm without +backend "
                   "(glasgow, networkx, gql, cfl, dpiso; default protocol.VERIFIERS)")
    p.add_argument("--solver-bin", default=None, help="Glasgow binary (default $GLASGOW_SOLVER_BIN)")
    p.add_argument("--mode", choices=["resident", "streaming"], default="resident")
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
                     default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
