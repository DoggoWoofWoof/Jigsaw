"""The reducer's plan for the canonical layout (launch/README.md sections 5 to 11).

    python -m scripts.canonical.launch.reduce_plan --runs-root RUNS --store-root STORE --queries-root QUERIES \\
        [--datasets corafull arxiv mag products papers100m] --out PLAN.json

The plan (reducer.py) names every run directory of the layout below that exists, the test split's
certification files and the offline manifests, with absolute paths, so the reducer's record of the plan's
SHA-256 pins every input. e2_root and e10_root are RUNS/runs/E2 and RUNS/runs/E10 when they exist.

  RUNS/runs/E1/<d>/shard<i>         E1              RUNS/runs/E7/<d>/jigsaw       E7
  RUNS/runs/E3x/<d>                 E3x             RUNS/runs/E7/<d>/classical_*  E7-classical
  RUNS/runs/E8/<d>                  E8              RUNS/runs/E9/<d>              E9
  RUNS/runs/E16/<d>/<partition>     E16             RUNS/runs/E12/<d>/run         E12
  QUERIES/<d>/test/certification/<regime>.json      certifications (E14)
  offline (E17): prepare STORE/<d>/_prepare_run; label_index, rankers and embeddings
    RUNS/offline/<d>/{label_index,rankers-primary,embed-<id>-primary}; training RUNS/train/<id>;
    router_fit RUNS/router_fit/<d>. <id> is a main-arm encoder (serving.default_model_id, every
    training seed and served regime), the system whose offline cost E17 reports.
  RUNS/audit/pattern/<d>-test       pattern_audit (E20; pattern_audit.py on the test query sets)

Every entry of the layout that is absent is printed; the reducer refuses what is inconsistent. A run
directory without manifest.json is refused here: a run that never started is absent, not empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .. import manifest, protocol

DATASETS = ("corafull", "arxiv", "mag", "products", "papers100m")
RUN_ENTRIES = (  # (experiment, pattern under RUNS/runs with {d} the dataset)
    ("E1", "E1/{d}/shard*"),
    ("E3x", "E3x/{d}"),
    ("E7", "E7/{d}/jigsaw"),
    ("E7-classical", "E7/{d}/classical_*"),
    ("E8", "E8/{d}"),
    ("E9", "E9/{d}"),
    ("E16", "E16/{d}/*"),
    ("E12", "E12/{d}/run"),
)


def main_model_ids(dataset: str) -> list[str]:
    from .. import serving

    return sorted({serving.default_model_id(dataset, r, int(seed)) for r in protocol.regimes_for(dataset)
                   for seed in protocol.SEEDS["training"]})


def _dirs(root: Path, pattern: str) -> list[Path]:
    hits = sorted(p for p in root.glob(pattern) if p.is_dir())
    for hit in hits:
        if not (hit / manifest.MANIFEST_NAME).is_file():
            raise SystemExit(f"{hit}: no {manifest.MANIFEST_NAME}; remove the directory or finish its run")
    return [h.resolve() for h in hits]


def build_plan(*, runs_root: str | Path, store_root: str | Path, queries_root: str | Path,
               datasets: Sequence[str] = DATASETS) -> tuple[dict[str, Any], list[str]]:
    """The plan and the absent entries of the layout, as "<dataset> <entry>" lines."""
    runs_root, store_root, queries_root = Path(runs_root), Path(store_root), Path(queries_root)
    runs = runs_root / "runs"
    plan: dict[str, Any] = {"runs": [], "certifications": {}, "offline": {}, "pattern_audit": {}}
    absent: list[str] = []
    for d in datasets:
        if d not in protocol.DATASETS:
            raise SystemExit(f"unknown dataset {d!r}; known: {', '.join(protocol.DATASETS)}")
        for experiment, pattern in RUN_ENTRIES:
            dirs = _dirs(runs, pattern.format(d=d))
            if dirs:
                plan["runs"].append({"experiment": experiment, "dataset": d, "dirs": [str(p) for p in dirs]})
            else:
                absent.append(f"{d} {experiment} ({runs / pattern.format(d=d)})")
        certs = {}
        for regime in protocol.regimes_for(d):
            path = queries_root / d / "test" / "certification" / f"{regime}.json"
            if path.is_file():
                certs[regime] = str(path.resolve())
            else:
                absent.append(f"{d} certification {regime} ({path})")
        if certs:
            plan["certifications"][d] = certs
        ids = main_model_ids(d)
        stages = {"prepare": [store_root / d / "_prepare_run"],
                  "label_index": [runs_root / "offline" / d / "label_index"],
                  "rankers": [runs_root / "offline" / d / "rankers-primary"],
                  "embeddings": [runs_root / "offline" / d / f"embed-{i}-primary" for i in ids],
                  "training": [runs_root / "train" / i for i in ids],
                  "router_fit": [runs_root / "router_fit" / d]}
        offline = {}
        for stage, dirs in stages.items():
            found = [p for p in dirs if (p / manifest.MANIFEST_NAME).is_file()]
            absent += [f"{d} offline {stage} ({p})" for p in dirs if p not in found]
            if found:
                offline[stage] = [str(p.resolve()) for p in found]
        if offline:
            plan["offline"][d] = offline
        audit = runs_root / "audit" / "pattern" / f"{d}-test"
        if (audit / manifest.MANIFEST_NAME).is_file():
            plan["pattern_audit"][d] = [str(audit.resolve())]
        else:
            absent.append(f"{d} pattern_audit ({audit})")
    for name, experiment in (("e2_root", "E2"), ("e10_root", "E10")):
        if (runs / experiment).is_dir():
            plan[name] = str((runs / experiment).resolve())
        else:
            absent.append(f"{experiment} root ({runs / experiment})")
    if not (plan["runs"] or plan["certifications"] or plan["offline"] or plan["pattern_audit"] or "e2_root" in plan
            or "e10_root" in plan):
        raise SystemExit(f"{runs_root} holds none of the layout's runs")
    return plan, absent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", required=True, help="$JIGSAW_RUNS_ROOT")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--queries-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    plan, absent = build_plan(runs_root=args.runs_root, store_root=args.store_root, queries_root=args.queries_root,
                              datasets=args.datasets)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    entries = sum(len(r["dirs"]) for r in plan["runs"])
    print(f"{out}: {entries} run directories, {sum(map(len, plan['certifications'].values()))} certifications, "
          f"{sum(len(v) for s in plan['offline'].values() for v in s.values())} offline manifests, "
          f"{len(plan['pattern_audit'])} pattern audits")
    for line in absent:
        print(f"absent: {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
