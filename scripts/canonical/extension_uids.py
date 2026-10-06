"""E3x input: the queries that E1 left undecided, for the 600 s extension.

    python -m scripts.canonical.extension_uids --e1 RUNS/E1/arxiv-test [RUNS/E1/arxiv-test-shard1 ...] \\
        --dataset arxiv --out RUNS/E3x/arxiv-test.uids

E3x reruns the arms of run_experiment.experiment_plan("E3x") at T = 600 s on
every query that at least one of those arms left undecided in E1 (T = 120 s),
so both arms run on the same queries and stay paired. A query is decided by a
FOUND whose witness passed the independent check or by a NONE that is a proof
(time_decided set); TIMEOUT and INCOMPLETE leave it open. An ERROR is a bug that
the reducer counts against PE1: it is listed only when another arm left the query
open.

The list is a function of the E1 records alone. <out> holds one uid per line
(run_experiment --uids); <out>.json records the E1 runs (manifest SHA-256), the
arms, T and the counts, so the reducer can check that E3x ran on exactly this
list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import manifest, protocol
from . import run_experiment as rx

SCHEMA = "jigsaw-extension-uids-v1"
DECIDED_NONE = "NONE"
OPEN = ("TIMEOUT", "INCOMPLETE")


def decided(record: dict[str, Any]) -> bool:
    if record.get("status") == "FOUND":
        return record.get("witness_check") == "ok"
    return record.get("status") == DECIDED_NONE and record.get("time_decided") is not None


def read_arm(e1_dirs: Sequence[Path], regime: str, arm: str) -> dict[str, dict[str, Any]]:
    """Every record of one (regime, arm) over the E1 runs and their shards; a uid must appear once."""
    safe = arm.replace("/", "_")
    out: dict[str, dict[str, Any]] = {}
    for d in e1_dirs:
        folder = d / "records" / regime
        for path in sorted(folder.glob(f"{safe}.jsonl")) + sorted(folder.glob(f"{safe}.shard*of*.jsonl")):
            for rec in rx.read_records(path):
                if rec["uid"] in out:
                    raise SystemExit(f"{rec['uid']} appears twice for {arm} in {regime}")
                out[rec["uid"]] = rec
    return out


def collect(e1_dirs: Sequence[Path], dataset: str) -> dict[str, Any]:
    e1 = rx.experiment_plan("E1", dataset)
    ext = rx.experiment_plan("E3x", dataset)
    arms = [rx.Arm.parse(a).name for a in ext["arms"]]
    runs = []
    for d in e1_dirs:
        info = json.loads((d / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
        config = info.get("config", {})
        if info.get("status") != "complete":
            raise SystemExit(f"{d}: the E1 run is {info.get('status')}, not complete")
        if config.get("experiment") != "E1" or config.get("dataset") != dataset:
            raise SystemExit(f"{d} is {config.get('experiment')} on {config.get('dataset')}, not E1 on {dataset}")
        if float(config.get("T", -1)) != float(e1["T"]):
            raise SystemExit(f"{d} ran at T = {config.get('T')}, not E1's {e1['T']}")
        runs.append({"path": str(d.resolve()), "manifest_sha256": manifest.sha256_file(d / manifest.MANIFEST_NAME)
                     ["sha256"], "split": info.get("split"), "shard": config.get("shard")})
    splits = {r["split"] for r in runs}
    if len(splits) != 1:
        raise SystemExit(f"the E1 runs come from different splits: {sorted(map(str, splits))}")
    uids: list[str] = []
    counts: dict[str, Any] = {}
    for regime in ext["regimes"]:
        per_arm = {arm: read_arm(e1_dirs, regime, arm) for arm in arms}
        sets = {arm: set(recs) for arm, recs in per_arm.items()}
        if not any(sets.values()):
            continue
        first = sets[arms[0]]
        for arm in arms[1:]:
            if sets[arm] != first:
                raise SystemExit(f"{regime}: the E1 records of {arms[0]} and {arm} cover different queries "
                                 f"({len(first ^ sets[arm])} differ); finish E1 first")
        open_by_arm = {arm: sorted(u for u, rec in recs.items() if rec.get("status") in OPEN)
                       for arm, recs in per_arm.items()}
        chosen = sorted(set().union(*open_by_arm.values()))
        counts[regime] = {"queries": len(first), "extension": len(chosen),
                          "open": {arm: len(v) for arm, v in open_by_arm.items()},
                          "errors": {arm: sum(rec.get("status") == "ERROR" for rec in recs.values())
                                     for arm, recs in per_arm.items()},
                          "decided": {arm: sum(decided(rec) for rec in recs.values())
                                      for arm, recs in per_arm.items()}}
        uids += chosen
    return {"schema": SCHEMA, "dataset": dataset, "split": splits.pop(), "arms": arms,
            "e1_T": float(e1["T"]), "extension_T": float(ext["T"]), "regimes": list(counts),
            "protocol_hash": protocol.protocol_hash(), "e1_runs": runs, "counts": counts, "uids": uids}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e1", nargs="+", required=True, help="E1 run directories (every shard) of one query set")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, help="uid file for run_experiment --uids; provenance in <out>.json")
    args = p.parse_args(argv)
    doc = collect([Path(d) for d in args.e1], args.dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{u}\n" for u in doc["uids"]), encoding="utf-8")
    doc["uids_sha256"] = manifest.sha256_file(out)["sha256"]
    Path(f"{out}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("dataset", "split", "arms", "counts")}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
