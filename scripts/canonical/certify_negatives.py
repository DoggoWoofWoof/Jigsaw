"""E14: decide every negative query exactly before it counts.

    python -m scripts.canonical.certify_negatives run --store-root STORES --dataset mag \\
        --queries QUERIES/mag/dev/20260607 [--regimes R0 R1] [--out DIR] [--shard i/n] \\
        [--solver-bin GLASGOW] [--classical-binary SM ...] [--cache-dir DIR]
    python -m scripts.canonical.certify_negatives finalize --queries ... --out DIR [--regimes ...]

A generated negative is only a candidate negative: a changed label or an added
edge can still leave an embedding somewhere in the data graph (the legacy MAG
run counted such a true match as a false positive). Every negative of a query
set is therefore decided by complete search, in two stages
(protocol.QUERY_DESIGN["certification"]):

  stage 1  FilterAll on the label-filtered whole graph with sound pruning and no
           component cap, stage1_T_seconds (3600 s). Its single step is closed,
           so its NONE is a proof; the global certificates (label multiset, edge
           type pair) decide first.
  stage 2  on what stage 1 leaves open: classical GQL (SubgraphMatching) on the
           whole data graph, stage2_T_seconds (3600 s). Its NONE counts only when
           the search finished (run_external.none_certified).

Every witness either stage returns is checked against the raw store data
(verify.check_embedding). A negative with a checked witness is certified "found"
and counts as a positive with that witness; one that both stages leave open is
"undecided", excluded from the negative set and counted. A rejected witness is a
solver soundness failure: it decides nothing and is reported in the summary.

Output, next to nothing the runners read (the query set files never change, so
the sha256 in every run manifest stays valid):
    <out>[/shard<i>of<n>]/manifest.json            the run manifest (experiment E14) of one process
    <out>[/shard<i>of<n>]/<regime>.progress.jsonl  one line per decided query (resumable)
    <out>/<regime>.json                            jigsaw-certification-v1, written once every
                                                   negative has a line in some run; the runs
                                                   must agree on stages, partition and query set
The default <out> is <queries>/certification. jigsaw-certification-v1 holds
{schema, dataset, regime, split, partition, query_set {path, sha256, master},
protocol_hash, stages [{name, method, backend, T, options}], counts, queries
{uid: {family, size, certified none|found|undecided, by stage1|stage2|null,
certificate, status {stage: status}, seconds {stage: s}, witness, witness_check}}}.
The reducer joins it on uid and checks query_set.sha256 against the run manifest's
inputs["queries/<regime>"].

Sealed query sets run only with the protocol's stages (no --stage*-T, --backend,
--limit, --families, --skip-stage2) and pass the seal gate (--unseal).
"""

from __future__ import annotations

from . import run_experiment as rx  # first: it pins the BLAS thread variables before numpy loads

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from . import backends, cascade, classical, manifest, protocol, queries, seal, verify  # noqa: E402
from .backends import ERROR, FOUND, NONE  # noqa: E402
from .run_external import CACHE_DIRNAME, none_certified  # noqa: E402
from .store import GraphStore  # noqa: E402

SCHEMA = "jigsaw-certification-v1"
PROGRESS_SCHEMA = "jigsaw-certification-query-v1"
CERT = protocol.QUERY_DESIGN["certification"]
STAGE2_ALGORITHM = {"classical_gql": "gql", "classical_cfl": "cfl", "classical_dpiso": "dpiso"}


def default_out(queries_dir: str | os.PathLike) -> Path:
    return Path(queries_dir) / "certification"


def run_dir(out: Path, shard: tuple[int, int] | None) -> Path:
    """Where one process writes its manifest and progress: <out>, or <out>/shard<i>of<n>."""
    return out / f"shard{shard[0]}of{shard[1]}" if shard else out


def run_dirs(out: Path) -> list[Path]:
    return [d for d in (out, *sorted(out.glob("shard*of*"))) if (d / manifest.MANIFEST_NAME).exists()]


def _normal(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=rx._json_default))


# ----------------------------------------------------------------------------
# One query
# ----------------------------------------------------------------------------
def stage1_cascade(store: Any, regime: str, backend: Any) -> cascade.Cascade:
    c = cascade.Cascade(store, regime, "filterall", backend, record_coverage=False)
    options = c.options()
    if options.get("component_cap") is not None or not c.budgets or c.budgets[-1] != c.num_blocks:
        raise SystemExit(f"stage 1 needs FilterAll on every block without a component cap; got {options}")
    if not options.get("label_filter", True):
        raise SystemExit("stage 1 needs the label filter")
    return c


def certify_query(q: queries.Query, stage1: cascade.Cascade, stage2: Any, *, T1: float, T2: float) -> dict[str, Any]:
    """Decide one negative: stage 1, then stage 2 on what stage 1 leaves open."""
    entry: dict[str, Any] = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind,
                             "certified": "undecided", "by": None, "certificate": None, "status": {},
                             "seconds": {}, "witness": None, "witness_check": None, "rejected": []}
    rec = stage1.run(q, time_limit=T1)
    entry["status"]["stage1"] = rec["status"]
    entry["seconds"]["stage1"] = float(rec["elapsed"])
    if rec["status"] == FOUND:  # Cascade checks the witness against the store; a rejected one is an ERROR
        entry.update(certified="found", by="stage1", witness=rec["witness"], witness_check=rec["witness_check"])
        return entry
    if rec["status"] == NONE:  # the closed step, or a global certificate
        entry.update(certified="none", by="stage1", certificate=rec.get("certificate"))
        return entry
    if str(rec.get("error", "")).startswith("witness rejected"):
        entry["rejected"].append({"stage": "stage1", "reason": rec["error"], "witness": rec.get("witness")})
    elif rec.get("error"):
        entry.setdefault("errors", {})["stage1"] = rec["error"]
    if stage2 is None:
        return entry
    codes = stage1.table.query_codes(q.payload, stage1.store)
    out = stage2.solve(codes, np.asarray(q.edges, dtype=np.int64), time_limit=T2)
    entry["status"]["stage2"] = out.status
    entry["seconds"]["stage2"] = float(out.elapsed)
    if out.status == FOUND:
        phi = np.asarray(out.embeddings[0], dtype=np.int64)
        check = verify.check_embedding(stage1.store, q.payload, q.edges, phi, stage1.regime)
        if check.ok:
            entry.update(certified="found", by="stage2", witness=phi.astype(int).tolist(), witness_check="ok")
        else:
            entry["rejected"].append({"stage": "stage2", "reason": f"witness rejected: {check.reason}",
                                      "witness": phi.astype(int).tolist()})
    elif out.status == NONE and none_certified(out):
        entry.update(certified="none", by="stage2", certificate=out.detail.get("trivial"))
    elif out.status == ERROR:
        entry.setdefault("errors", {})["stage2"] = str(out.detail.get("error", ""))[-500:]
    return entry


# ----------------------------------------------------------------------------
# Assembling the certification file
# ----------------------------------------------------------------------------
def run_settings(out: Path, regime: str) -> tuple[list[dict[str, Any]], str]:
    """The stages and partition every run (shard) of a regime used; they must agree."""
    seen = []
    for d in run_dirs(out):
        config = json.loads((d / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))["config"]
        if regime in config["regimes"]:
            seen.append((d, _normal(config["stages"][regime]), str(config["partition"]),
                         _normal(config["query_sets"][regime]["path"])))
    if not seen:
        raise SystemExit(f"{out}: no certification run covers {regime}")
    first = seen[0]
    for d, stages, partition, path in seen[1:]:
        if (stages, partition, path) != first[1:]:
            raise SystemExit(f"{d} certified {regime} with other stages, partition or query set than {first[0]}")
    return first[1], first[2]


def read_progress(out: Path, regime: str) -> dict[str, dict[str, Any]]:
    """Every decided query of a regime over all runs (shards); a uid decided twice must agree."""
    entries: dict[str, dict[str, Any]] = {}
    for d in run_dirs(out):
        path = d / f"{regime}.progress.jsonl"
        if not path.exists():
            continue
        for rec in rx.read_records(path):
            old = entries.get(rec["uid"])
            if old is not None and old["certified"] != rec["certified"]:
                raise SystemExit(f"{rec['uid']} is certified {old['certified']} and {rec['certified']} in {out}")
            entries.setdefault(rec["uid"], rec)
    return entries


def summarize(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {"negatives": len(entries), "none": 0, "found": 0, "undecided": 0, "by": {},
                              "rejected_witnesses": 0, "errors": 0, "by_family": {}}
    for e in entries.values():
        counts[e["certified"]] += 1
        if e["by"]:
            key = e["by"] + (f":{e['certificate']}" if e.get("certificate") else "")
            counts["by"][key] = counts["by"].get(key, 0) + 1
        counts["rejected_witnesses"] += len(e.get("rejected") or [])
        counts["errors"] += len(e.get("errors") or {})
        fam = counts["by_family"].setdefault(e["family"], {"none": 0, "found": 0, "undecided": 0})
        fam[e["certified"]] += 1
    return counts


def finalize(out: Path, qs: queries.QuerySet, *, allow_partial: bool = False) -> dict[str, Any]:
    """Write <out>/<regime>.json from the progress files once every negative has a line."""
    regime = qs.regime
    negatives = [q for q in qs.queries if q.kind == "negative"]
    stages, partition = run_settings(out, regime)
    entries = read_progress(out, regime)
    missing = [q.uid for q in negatives if q.uid not in entries]
    foreign = sorted(set(entries) - {q.uid for q in negatives})
    if foreign:
        raise SystemExit(f"{out}: progress lines for queries outside {qs.path}: {foreign[:5]}")
    if missing and not allow_partial:
        return {"regime": regime, "complete": False, "recorded": len(entries), "negatives": len(negatives),
                "missing": len(missing)}
    queries_out = {}
    for q in negatives:
        e = entries.get(q.uid)
        if e is None:
            continue
        queries_out[q.uid] = {k: e.get(k) for k in ("family", "size", "certified", "by", "certificate", "status",
                                                     "seconds", "witness", "witness_check", "rejected", "errors")}
    doc = {
        "schema": SCHEMA, "dataset": qs.meta.get("dataset"), "regime": regime, "split": qs.meta.get("split"),
        "partition": partition,
        "query_set": {"path": str(Path(qs.path).resolve()), "sha256": manifest.sha256_file(qs.path)["sha256"],
                      "master": qs.meta.get("master")},
        "protocol_hash": protocol.protocol_hash(), "stages": stages, "complete": not missing,
        "counts": summarize({u: entries[u] for u in queries_out}),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "queries": queries_out,
    }
    tmp = out / f"{regime}.json.{os.getpid()}.tmp"  # one per process: shards of a query set share this directory
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out / f"{regime}.json")
    return {"regime": regime, "complete": not missing, "recorded": len(queries_out), "negatives": len(negatives),
            "counts": doc["counts"]}


def load_certification(path: str | os.PathLike, *, query_set_sha256: str | None = None) -> dict[str, Any]:
    """Read a certification file; with query_set_sha256, refuse one made for another query set."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {SCHEMA}")
    if not doc.get("complete"):
        raise ValueError(f"{path}: certification is partial")
    if query_set_sha256 is not None and doc["query_set"]["sha256"] != query_set_sha256:
        raise ValueError(f"{path} certifies another query set ({doc['query_set']['sha256'][:12]}...)")
    for uid, e in doc["queries"].items():
        if e["certified"] == "found" and e.get("witness_check") != "ok":
            raise ValueError(f"{path}: {uid} is certified found without a checked witness")
    return doc


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def check_sealed(args: argparse.Namespace) -> None:
    problems = [flag for flag, value in (("--stage1-T", args.stage1_T), ("--stage2-T", args.stage2_T),
                                         ("--backend", args.backend), ("--limit", args.limit),
                                         ("--families", args.families)) if value is not None]
    if args.skip_stage2:
        problems.append("--skip-stage2")
    if args.partition != "primary":
        problems.append(f"--partition {args.partition}")
    if problems:
        raise SystemExit(f"a sealed certification uses the protocol's stages; not allowed: {'; '.join(problems)}")


def stage_specs(stage1: cascade.Cascade, stage2: Any, T1: float, T2: float, args: argparse.Namespace,
                binary_sha: str | None) -> list[dict[str, Any]]:
    stages = [{"name": "stage1", "method": "filterall", "backend": stage1.backend.name, "T": T1,
               "options": {**stage1.options(), "budgets": list(stage1.budgets)}}]
    if stage2 is not None:
        stages.append({"name": "stage2", "method": str(CERT["stage2_method"]), "backend": stage2.name, "T": T2,
                       "options": {"preset": list(classical.PRESETS[stage2.algorithm]),
                                   "data_format": stage2.data_format, "binary_sha256": binary_sha}})
    return stages


def open_regime(args: argparse.Namespace, store: GraphStore, regime: str, solvers: dict[str, Any]):
    name = args.backend or str(protocol.VERIFIERS["default"])
    if name not in solvers:
        kwargs = {"binary": args.solver_bin} if name == "glasgow" and args.solver_bin else {}
        solvers[name] = backends.make_backend(name, **kwargs)
    stage1 = stage1_cascade(store, regime, solvers[name])
    stage2 = None
    if not args.skip_stage2:
        binary = None if not args.classical_binary else (
            args.classical_binary[0] if len(args.classical_binary) == 1 else list(args.classical_binary))
        stage2 = classical.FullGraphClassicalRunner(
            store, regime, STAGE2_ALGORITHM[str(CERT["stage2_method"])],
            cache_dir=args.cache_dir or Path(args.store_root) / CACHE_DIRNAME, binary=binary)
        if stage2.binary_path is None:
            raise SystemExit(f"stage 2: SubgraphMatching binary {stage2.backend.binary!r} not found; build it with "
                             "external/build_subgraphmatching.sh and set SUBGRAPHMATCHING_BIN, or pass "
                             "--classical-binary")
    return stage1, stage2


def execute(args: argparse.Namespace) -> dict[str, Any]:
    T1 = float(CERT["stage1_T_seconds"] if args.stage1_T is None else args.stage1_T)
    T2 = float(CERT["stage2_T_seconds"] if args.stage2_T is None else args.stage2_T)
    if T1 <= 0 or T2 <= 0:
        raise SystemExit("--stage1-T and --stage2-T must be positive")
    shard = None
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit("--shard i/n needs 0 <= i < n")
        shard = (index, count)
    regimes = list(args.regimes or [p.stem for p in sorted(Path(args.queries).glob("*.npz"))])
    if not regimes:
        raise SystemExit(f"no query sets in {args.queries}")
    plan = rx.Plan("E14", [], T1, regimes, {"kinds": ["negative"]})
    sets = rx.load_sets(args, plan)
    split = str(next(iter(sets.values())).meta.get("split", "dev"))
    if split in queries.TEST_SPLITS:
        check_sealed(args)  # before the gate, so a refused run leaves no unseal entry
        seal.authorize_sealed_run(unseal=args.unseal, reason=args.reason or "certify negatives (E14)")
    elif args.unseal:
        raise SystemExit("--unseal applies to sealed query sets only")
    selected = {r: rx.select_queries(sets[r].queries, kinds=["negative"], families=args.families, sizes=None,
                                     totals=None, limit=args.limit, shard=shard) for r in regimes}
    threads, limiter = rx.pin_threads(int(protocol.SERVING["threads"]))
    try:
        store = GraphStore(args.store_root, args.dataset, mmap=True, partition=args.partition)
        solvers: dict[str, Any] = {}
        opened = {r: open_regime(args, store, r, solvers) for r in regimes}
        binary_sha = None
        first2 = next(iter(opened.values()))[1]
        if first2 is not None:
            entry = manifest.sha256_file(first2.binary_path, fast=True)
            binary_sha = str(entry.get("sha256") or entry.get("fingerprint"))
        stages = {r: stage_specs(s1, s2, T1, T2, args, binary_sha) for r, (s1, s2) in opened.items()}
        out = Path(args.out) if args.out else default_out(args.queries)
        rdir = run_dir(out, shard)
        config = {
            "schema": SCHEMA, "experiment": "E14", "dataset": args.dataset, "partition": args.partition,
            "regimes": regimes, "stages": stages, "families": args.families, "limit": args.limit,
            "shard": list(shard) if shard else None, "solver_bin": args.solver_bin,
            "query_sets": {r: {"path": str(Path(sets[r].path).resolve()), "split": sets[r].meta.get("split"),
                               "master": sets[r].meta.get("master"), "selected": len(selected[r])} for r in regimes},
            "threads": int(protocol.SERVING["threads"]),
        }
        rdir.mkdir(parents=True, exist_ok=True)
        attempt = rx.check_resume(rdir, config)
        summary: dict[str, Any] = {"attempt": attempt, "regimes": {}}
        glasgow = args.solver_bin or os.environ.get("GLASGOW_SOLVER_BIN")
        with manifest.RunManifest(rdir, experiment="E14", config=config, split=split,
                                  inputs=rx.run_inputs(args, plan, store, sets),
                                  solvers={"glasgow": glasgow} if "glasgow" in solvers else {},
                                  fast_input_hashes=args.fast_hashes) as run:
            run.note(threads=threads, attempt=attempt)
            for regime in regimes:
                stage1, stage2 = opened[regime]
                if stage2 is not None:
                    run.note(**{f"stage2_data_graph/{regime}": stage2.prepare()})
                f = rx.RecordFile(rdir / f"{regime}.progress.jsonl")
                done_before = len(f.done)
                try:
                    for position, q in enumerate(selected[regime]):
                        if q.uid in f.done:
                            continue
                        t0 = time.perf_counter()
                        try:
                            entry = certify_query(q, stage1, stage2, T1=T1, T2=T2)
                        except Exception as exc:  # a bug: the query stays undecided, the run goes on
                            entry = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind,
                                     "certified": "undecided", "by": None, "certificate": None, "status": {},
                                     "seconds": {}, "witness": None, "witness_check": None, "rejected": [],
                                     "errors": {"certify": f"{type(exc).__name__}: {exc}"},
                                     "traceback": traceback.format_exc(limit=8)}
                        entry.update({"schema": PROGRESS_SCHEMA, "regime": regime, "query_index": position,
                                      "attempt": attempt, "wall_seconds": time.perf_counter() - t0})
                        f.write(entry)
                finally:
                    f.close()
                summary["regimes"][regime] = {
                    "selected": len(selected[regime]), "recorded_before": done_before,
                    "certification": finalize(out, sets[regime])}
            run.note(**summary)
        return summary
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def execute_finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the certification files of a sharded run (the shards may run on several machines)."""
    out = Path(args.out) if args.out else default_out(args.queries)
    dirs = run_dirs(out)
    if not dirs:
        raise SystemExit(f"{out} holds no certification run")
    covered = sorted({r for d in dirs for r in json.loads((d / manifest.MANIFEST_NAME).read_text(
        encoding="utf-8"))["config"]["regimes"]})
    summary = {}
    for regime in list(args.regimes or covered):
        qs = queries.QuerySet.load(Path(args.queries) / f"{regime}.npz")
        summary[regime] = finalize(out, qs, allow_partial=args.allow_partial)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="decide the negatives of one query-set directory")
    r.add_argument("--store-root", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--partition", default="primary")
    r.add_argument("--queries", required=True, help="directory of <regime>.npz query sets of one split master")
    r.add_argument("--regimes", nargs="*", help="default: every <regime>.npz in --queries")
    r.add_argument("--out", default=None, help="default <queries>/certification")
    r.add_argument("--solver-bin", default=None, help="Glasgow binary (default $GLASGOW_SOLVER_BIN)")
    r.add_argument("--backend", default=None, help="dev only: stage 1's solver instead of the protocol verifier")
    r.add_argument("--classical-binary", nargs="+", default=None,
                   help="SubgraphMatching binary or command prefix (default $SUBGRAPHMATCHING_BIN)")
    r.add_argument("--cache-dir", default=None, help=f"stage 2 data graph files (default <store-root>/{CACHE_DIRNAME})")
    r.add_argument("--stage1-T", type=float, default=None, help="dev only: stage 1 seconds per query")
    r.add_argument("--stage2-T", type=float, default=None, help="dev only: stage 2 seconds per query")
    r.add_argument("--skip-stage2", action="store_true", help="dev only: stage 1 alone")
    r.add_argument("--families", nargs="*", default=None, help="dev only: these negative families")
    r.add_argument("--limit", type=int, default=None, help="dev only: the first N negatives per regime")
    r.add_argument("--shard", default=None, help="i/n: every n-th negative from the i-th")
    r.add_argument("--models-root", default="models/canonical", help=argparse.SUPPRESS)
    r.add_argument("--unseal", action="store_true")
    r.add_argument("--reason", default="")
    r.add_argument("--fast-hashes", action="store_true", help="fingerprint large inputs instead of hashing them")
    f = sub.add_parser("finalize", help="assemble <regime>.json from the progress files of every shard")
    f.add_argument("--queries", required=True)
    f.add_argument("--out", default=None)
    f.add_argument("--regimes", nargs="*")
    f.add_argument("--allow-partial", action="store_true", help="write a partial file (complete: false)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = execute(args) if args.command == "run" else execute_finalize(args)
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
