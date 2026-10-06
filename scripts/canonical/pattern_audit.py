"""Pattern-overlap audit (E20): the queries of a set that a main encoder saw in training, up to isomorphism.

    python -m scripts.canonical.pattern_audit --store-root STORE --queries QUERIES/arxiv/test \\
        --run-root RUNS --out RUNS/audit/pattern/arxiv-test --unseal --reason "E20 arxiv" \\
        [--workers 8] [--held-out-root QUERIES]
    python -m scripts.canonical.pattern_audit ... --runs RUNS/train/<model_id> ... [--allow-smoke]

On a test split the audit runs only with --unseal, under a frozen protocol and committed code
(seal.authorize_sealed_run, after every check that needs no regeneration), and the unseal log records
it.

Training queries are a function of the run. train_encoder.TrainSampler with the run's dataset, trained
regime, seed and sampler settings, and the node sets the run held out, gives batch(epoch, step,
batch_size) for every epoch and step of the run's schedule. The sampler draws a family by its position
in TRAINING["sampler"]["family_probabilities"] and the manifest stores that table with sorted keys, so
the audit takes the settings from the protocol and checks that they equal the manifest's. The audit
regenerates every batch of every audited run: with --run-root, the main encoders of the query set's
dataset (every trained regime and training seed, RUNS/train/<model_id>); with --runs, the run
directories given. Before regenerating, it checks that a run's manifest is complete, that the run is a
main-arm encoder of the dataset (a smoke run only with --allow-smoke), and that the held-out node sets
rebuilt from the paths its manifest records (or, with --held-out-root, from the canonical layout under
that root) hash to the manifest's. After regenerating, the node-set hash of every training query must
equal the run's train_nodesets.txt, line for line and in number. A run that fails any check is refused
and no pattern_audit.json is written (a check after the manifest's start leaves it "failed").

Each training query is labelled in every regime its encoder serves (protocol.TRAINED_REGIMES: the R2-512
encoder serves R2-512, R2-128 and R2-32) with queries.payload_for, the labels a user would give with the
same nodes, and looked up in a queries.QueryIndex of the set's queries of that regime, negatives
included. A query of the set is seen by an encoder when some training query of that encoder is
isomorphic to it: a node bijection keeps every edge, type and regime label, so both have the same
matches in the data graph. Isomorphic queries of one set (older sets hold some) are seen together. The
R2-128 and R2-32 buckets coarsen the R2-512 ones (a bucket is the digest head mod 2^k), so a training
query isomorphic to a query at R2-512 stays isomorphic at R2-128 and R2-32, and labelling at the served
regime counts at least as many queries seen as labelling at the trained one. Node sets never repeat
between training and the query sets (the held-out validation masters, the sealed test masters), so a
seen query is the same pattern planted elsewhere.

OUT/pattern_audit.json (schema jigsaw-pattern-audit-v1):
  dataset, split, master, query_protocol_hash (the query sets'), protocol_hash (the code's), complete
  regimes.<regime>   query_set {path, sha256}, queries, positives, negatives, seen_positives,
                     seen_negatives (counts), uids (set order), encoders (the model ids that serve it),
                     seen {uid: {model_id: {first: [epoch, step, draw], hits}}}: for each encoder that saw
                     the query, its first matching training query (the draw index within the step) and
                     the number of matching training queries
  encoders.<id>      regime, serves, seed, smoke, run_dir, epochs, steps_per_epoch, batch_size, draws,
                     failed, held_out_skipped, accepted, train_nodesets_lines, held_out {nodesets, sha256},
                     seconds
and a manifest (experiment "pattern_audit", split = the query sets'). The CLI prints a JSON summary: per
regime the queries, the seen positives and negatives, and per encoder the seconds.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import manifest, protocol, queries, seal
from . import train_encoder as te
from .launch.train_matrix import VAL_TEMPLATE
from .store import GraphStore

SCHEMA = "jigsaw-pattern-audit-v1"
EXPERIMENT = "pattern_audit"
OUTPUT = "pattern_audit.json"


class AuditError(RuntimeError):
    """A run or query set the audit refuses; the message names it."""


# ----------------------------------------------------------------------------
# Encoders and query sets
# ----------------------------------------------------------------------------
def main_rows(dataset: str) -> list[tuple[str, int, str]]:
    """(trained regime, training seed, model id) of every main encoder of a dataset (train_matrix's main rows)."""
    return [(regime, int(seed), te.resolve_plan(dataset, regime, int(seed))["model_id"])
            for regime in te.trained_regimes_for(dataset) for seed in protocol.SEEDS["training"]]


def served_regimes(dataset: str, trained: str) -> list[str]:
    """The regimes of a dataset that the encoder trained in `trained` serves, in protocol order."""
    return [r for r in protocol.regimes_for(dataset) if te.trained_regime(r) == trained]


def load_query_sets(query_dir: str | Path) -> dict[str, queries.QuerySet]:
    """regime -> the query set of every <regime>.npz of the directory; one dataset, split and master."""
    query_dir = Path(query_dir)
    sets = {}
    for regime in protocol.LABEL_REGIMES:
        path = query_dir / f"{regime}.npz"
        if path.is_file():
            sets[regime] = queries.QuerySet.load(path)
    if not sets:
        raise AuditError(f"{query_dir}: no <regime>.npz query sets")
    origins = {(s.meta.get("dataset"), s.meta.get("split"), str(s.meta.get("master"))) for s in sets.values()}
    if len(origins) != 1:
        raise AuditError(f"{query_dir}: the query sets come from different datasets, splits or masters "
                         f"{sorted(origins)}")
    for regime, query_set in sets.items():
        if query_set.regime != regime:
            raise AuditError(f"{query_set.path}: holds regime {query_set.regime}")
    return sets


class SetIndex:
    """The queries of one set up to label-preserving isomorphism; isomorphic queries share an entry."""

    def __init__(self, query_set: queries.QuerySet) -> None:
        self.index = queries.QueryIndex(query_set.regime)
        self.members: dict[str, list[str]] = {}
        for q in query_set:
            item = self.index.item(q.edges, q.payload)
            first = self.index.find(item)
            if first is None:
                self.index.add(item, q.uid)
                self.members[q.uid] = [q.uid]
            else:
                self.members[first].append(q.uid)

    def lookup(self, edges: Any, payload: Mapping[str, Any]) -> list[str]:
        """The uids of the set's queries isomorphic to the query (edges, payload)."""
        name = self.index.find(self.index.item(edges, payload))
        return [] if name is None else self.members[name]


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
def read_run(run_dir: str | Path, dataset: str, *, allow_smoke: bool = False) -> dict[str, Any]:
    """The config of a complete main-arm train_encoder run of the dataset (its manifest's)."""
    run_dir = Path(run_dir)
    try:
        doc = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"{run_dir}: no readable {manifest.MANIFEST_NAME} ({exc})") from exc
    config = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    problems = []
    if doc.get("experiment") != "train_encoder":
        problems.append(f"experiment {doc.get('experiment')!r}, not 'train_encoder'")
    if doc.get("status") != "complete":
        problems.append(f"status {doc.get('status')!r}, not 'complete'")
    if config.get("dataset") != dataset:
        problems.append(f"dataset {config.get('dataset')!r}, the query sets' is {dataset!r}")
    if config.get("arm") != te.MAIN_ARM:
        problems.append(f"arm {config.get('arm')!r}; the audit covers the main encoders")
    if config.get("smoke") and not allow_smoke:
        problems.append(f"a smoke run ({config.get('smoke')}); pass --allow-smoke to audit it")
    if not (run_dir / te.TRAIN_NODESETS).is_file():
        problems.append(f"no {te.TRAIN_NODESETS}")
    if config.get("sampler") != te.sampler_settings():
        problems.append("sampler settings differ from protocol.TRAINING['sampler']")
    if problems:
        raise AuditError(f"{run_dir}: " + "; ".join(problems))
    return config


def held_out_of(config: Mapping[str, Any], held_out_root: str | Path | None = None) -> tuple[list[str], str]:
    """(sorted held-out node sets, their sha256) of a run, checked against its manifest.

    The paths are the manifest's (config paths.val_queries and paths.held_out), or with held_out_root
    the canonical layout under it (launch.train_matrix: the selection master's set of the trained
    regime and the directories of every validation master).
    """
    dataset, regime = config["dataset"], config["regime"]
    if held_out_root is None:
        paths = config.get("paths") or {}
        val_path = Path(paths.get("val_queries") or "")
        extra = [Path(p) for p in paths.get("held_out") or []]
    else:
        fields = {"queries_root": str(held_out_root), "dataset": dataset, "regime": regime}
        val_path = Path(VAL_TEMPLATE.format(**fields, master=te.SELECTION_MASTER))
        extra = [Path(VAL_TEMPLATE.format(**fields, master=m)).parent for m in protocol.SEEDS["validation"]]
    try:
        val_set, _ = te.load_validation_set(val_path, dataset, regime)
        held = sorted(te.held_out_nodesets(val_path, val_set, extra))
    except (OSError, ValueError) as exc:
        raise AuditError(f"{config.get('model_id')}: cannot rebuild the held-out node sets ({exc}); "
                         "--held-out-root names the query root of the validation masters") from exc
    digest = hashlib.sha256("\n".join(held).encode("ascii")).hexdigest()
    recorded = config.get("held_out") or {}
    if digest != recorded.get("sha256") or len(held) != recorded.get("nodesets"):
        raise AuditError(f"{config.get('model_id')}: the held-out node sets rebuild to {len(held)} "
                         f"({digest[:12]}...), the run held out {recorded.get('nodesets')} "
                         f"({str(recorded.get('sha256'))[:12]}...)")
    return held, digest


def audit_run(task: Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate one run's training queries, check them against its train_nodesets.txt and look each one
    up in the query sets of the regimes its encoder serves. A worker process runs one task."""
    t0 = time.perf_counter()
    config, run_dir = task["config"], Path(task["run_dir"])
    dataset, regime, seed = config["dataset"], config["regime"], int(config["seed"])
    store = GraphStore(task["store_root"], dataset, mmap=True, partition=str(config.get("partition") or "primary"))
    indexes = {r: SetIndex(queries.QuerySet.load(path)) for r, path in task["query_sets"].items()}
    # the protocol's settings (read_run checked they equal the manifest's): the manifest sorts the keys of
    # family_probabilities, and the sampler draws a family by its position
    sampler = te.TrainSampler(store, dataset, regime, seed, te.sampler_settings(), held_out=task["held_out"])
    schedule = config["schedule"]
    epochs, steps, batch = int(schedule["epochs"]), int(schedule["steps_per_epoch"]), int(schedule["batch_size"])
    expected = (run_dir / te.TRAIN_NODESETS).read_text(encoding="ascii").split()
    seen: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in indexes}
    line = 0
    for epoch in range(1, epochs + 1):
        for step in range(steps):
            for item in sampler.batch(epoch, step, batch):
                logged = expected[line] if line < len(expected) else None
                if logged != item.nodeset:
                    raise AuditError(f"{run_dir}: training query {line} (epoch {epoch}, step {step}, draw "
                                     f"{item.index}) regenerates as {item.nodeset}; {te.TRAIN_NODESETS} has "
                                     f"{logged or 'no line'}")
                line += 1
                for r, index in indexes.items():
                    for uid in index.lookup(item.edges, queries.payload_for(store, item.planted, r)):
                        entry = seen[r].setdefault(uid, {"first": [epoch, step, int(item.index)], "hits": 0})
                        entry["hits"] += 1
    if line != len(expected):
        raise AuditError(f"{run_dir}: {len(expected)} lines in {te.TRAIN_NODESETS}, {line} training queries "
                         "regenerated")
    stats = sampler.stats
    return {"model_id": config["model_id"], "regime": regime, "serves": list(indexes), "seed": seed,
            "smoke": dict(config.get("smoke") or {}), "run_dir": str(run_dir), "epochs": epochs,
            "steps_per_epoch": steps, "batch_size": batch, "draws": int(stats["draws"]),
            "failed": int(stats["failed"]), "held_out_skipped": int(stats["held_out"]), "accepted": line,
            "train_nodesets_lines": len(expected),
            "held_out": {"nodesets": len(task["held_out"]), "sha256": task["held_out_sha256"]},
            "seconds": round(time.perf_counter() - t0, 1), "seen": seen}


# ----------------------------------------------------------------------------
# The audit
# ----------------------------------------------------------------------------
def plan_tasks(*, store_root: str | Path, query_dir: str | Path, run_dirs: Sequence[str | Path] | None = None,
               run_root: str | Path | None = None, held_out_root: str | Path | None = None,
               allow_smoke: bool = False) -> tuple[dict[str, queries.QuerySet], list[dict[str, Any]]]:
    """(query sets, one task per run), every check before regeneration done."""
    if (run_dirs is None) == (run_root is None):
        raise AuditError("give either --runs or --run-root")
    sets = load_query_sets(query_dir)
    dataset = str(next(iter(sets.values())).meta["dataset"])
    if run_root is not None:
        rows = main_rows(dataset)
        run_dirs = [Path(run_root) / "train" / model_id for _, _, model_id in rows]
        missing = [str(p) for p in run_dirs if not p.is_dir()]
        if missing:
            raise AuditError(f"main encoders of {dataset} not found: {missing}")
    tasks, models = [], set()
    for run_dir in run_dirs:
        config = read_run(run_dir, dataset, allow_smoke=allow_smoke)
        if config["model_id"] in models:
            raise AuditError(f"{run_dir}: {config['model_id']} is audited twice")
        models.add(config["model_id"])
        serves = [r for r in served_regimes(dataset, config["regime"]) if r in sets]
        if not serves:
            raise AuditError(f"{run_dir}: {config['model_id']} serves none of the regimes {sorted(sets)}")
        held, digest = held_out_of(config, held_out_root)
        tasks.append({"config": config, "run_dir": str(run_dir), "store_root": str(store_root),
                      "query_sets": {r: str(sets[r].path) for r in serves}, "held_out": held,
                      "held_out_sha256": digest})
    unserved = sorted(set(sets) - {r for t in tasks for r in t["query_sets"]})
    if unserved:
        raise AuditError(f"no audited encoder serves {unserved}")
    return sets, tasks


def assemble(sets: Mapping[str, queries.QuerySet], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pattern_audit.json document of finished tasks."""
    first = next(iter(sets.values()))
    regimes: dict[str, Any] = {}
    for regime, query_set in sets.items():
        kinds = [q.kind for q in query_set]
        seen: dict[str, dict[str, Any]] = {}
        encoders = []
        for res in sorted(results, key=lambda r: r["model_id"]):
            if regime not in res["seen"]:
                continue
            encoders.append(res["model_id"])
            for uid, entry in res["seen"][regime].items():
                seen.setdefault(uid, {})[res["model_id"]] = entry
        order = {q.uid: i for i, q in enumerate(query_set)}
        kind_of = {q.uid: q.kind for q in query_set}
        regimes[regime] = {
            "query_set": {"path": str(query_set.path), "sha256": manifest.sha256_file(query_set.path)["sha256"]},
            "queries": len(query_set), "positives": kinds.count("positive"), "negatives": kinds.count("negative"),
            "seen_positives": sum(1 for u in seen if kind_of[u] == "positive"),
            "seen_negatives": sum(1 for u in seen if kind_of[u] == "negative"),
            "uids": [q.uid for q in query_set], "encoders": encoders,
            "seen": {uid: seen[uid] for uid in sorted(seen, key=order.__getitem__)},
        }
    encoders_doc = {res["model_id"]: {k: v for k, v in res.items() if k not in ("model_id", "seen")}
                    for res in sorted(results, key=lambda r: r["model_id"])}
    return {"schema": SCHEMA, "dataset": first.meta["dataset"], "split": first.meta["split"],
            "master": str(first.meta.get("master")), "query_protocol_hash": first.meta.get("protocol_hash"),
            "protocol_hash": protocol.protocol_hash(), "complete": True, "regimes": regimes,
            "encoders": encoders_doc}


def summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Per regime the queries and the seen positives and negatives; per encoder its training queries and seconds."""
    keys = ("queries", "positives", "negatives", "seen_positives", "seen_negatives")
    return {"dataset": doc["dataset"], "split": doc["split"], "master": doc["master"],
            "regimes": {r: {**{k: e[k] for k in keys}, "encoders": len(e["encoders"])}
                        for r, e in doc["regimes"].items()},
            "encoders": {m: {"accepted": e["accepted"], "seconds": e["seconds"]} for m, e in doc["encoders"].items()}}


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def run_audit(*, store_root: str | Path, query_dir: str | Path, out: str | Path,
              run_dirs: Sequence[str | Path] | None = None, run_root: str | Path | None = None,
              held_out_root: str | Path | None = None, allow_smoke: bool = False, workers: int = 1,
              unseal: bool = False, reason: str | None = None,
              argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Audit the query sets of query_dir against the runs; writes OUT/pattern_audit.json and a manifest."""
    out = Path(out)
    if (out / OUTPUT).exists():
        raise AuditError(f"{out / OUTPUT} exists; audit into a new directory")
    sets, tasks = plan_tasks(store_root=store_root, query_dir=query_dir, run_dirs=run_dirs, run_root=run_root,
                             held_out_root=held_out_root, allow_smoke=allow_smoke)
    first = next(iter(sets.values()))
    if first.meta["split"] in queries.TEST_SPLITS:  # after the checks, so a refused audit leaves no unseal entry
        try:
            seal.authorize_sealed_run(unseal=unseal, reason=reason or f"E20 pattern audit {first.meta['dataset']}")
        except seal.SealError as exc:
            raise AuditError(str(exc)) from exc
    inputs: dict[str, Any] = {f"queries/{r}": s.path for r, s in sets.items()}
    for t in tasks:
        model_id, run_dir = t["config"]["model_id"], Path(t["run_dir"])
        inputs[f"train/{model_id}/manifest"] = run_dir / manifest.MANIFEST_NAME
        inputs[f"train/{model_id}/{te.TRAIN_NODESETS}"] = run_dir / te.TRAIN_NODESETS
    config = {"schema": SCHEMA, "dataset": first.meta["dataset"], "master": str(first.meta.get("master")),
              "store_root": str(store_root), "query_dir": str(query_dir), "run_root": None if run_root is None
              else str(run_root), "runs": [t["run_dir"] for t in tasks], "held_out_root": None
              if held_out_root is None else str(held_out_root), "allow_smoke": bool(allow_smoke),
              "workers": int(workers)}
    with manifest.RunManifest(out, experiment=EXPERIMENT, config=config, inputs=inputs, split=str(first.meta["split"]),
                              argv=list(argv) if argv is not None else list(sys.argv)) as run_manifest:
        results = []
        if workers <= 1:
            done = map(audit_run, tasks)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=int(workers))
            done = pool.map(audit_run, tasks)
        try:
            for res in done:
                results.append(res)
                print(f"[pattern_audit] {res['model_id']}: {res['accepted']} training queries, {res['seconds']} s",
                      file=sys.stderr, flush=True)
        finally:
            if workers > 1:
                pool.shutdown(cancel_futures=True)
        doc = assemble(sets, results)
        _write_json(out / OUTPUT, doc)
        run_manifest.note(summary=summary(doc))
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--queries", required=True, help="the <regime>.npz files of one split master: "
                        "QUERIES/<dataset>/test, or QUERIES/<dataset>/<dev|validation>/<master>")
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument("--run-root", help="audit every main encoder of the dataset, RUN_ROOT/train/<model_id>")
    runs.add_argument("--runs", nargs="+", help="audit these train_encoder run directories")
    parser.add_argument("--out", required=True, help="a new directory")
    parser.add_argument("--held-out-root", default=None,
                        help="rebuild the held-out node sets from the validation masters under this query root")
    parser.add_argument("--allow-smoke", action="store_true", help="audit smoke runs (tests and previews)")
    parser.add_argument("--workers", type=int, default=1, help="runs regenerated at a time (processes)")
    parser.add_argument("--unseal", action="store_true", help="audit a sealed test split (logged)")
    parser.add_argument("--reason", default=None, help="the unseal log's reason")
    args = parser.parse_args(argv)
    try:
        doc = run_audit(store_root=args.store_root, query_dir=args.queries, out=args.out, run_dirs=args.runs,
                        run_root=args.run_root, held_out_root=args.held_out_root, allow_smoke=args.allow_smoke,
                        workers=args.workers, unseal=args.unseal, reason=args.reason,
                        argv=sys.argv if argv is None else ["pattern_audit", *argv])
    except AuditError as exc:
        print(f"pattern_audit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary(doc), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
