"""README section 6 as a resumable driver: the dev and validation query sets of every dataset.

    python -m scripts.canonical.launch.build_queries --store-root STORE --queries-root QUERIES \\
        [--datasets corafull arxiv mag] [--dry-run]

Per dataset, every master of protocol.SEEDS["dev"] and then of protocol.SEEDS["validation"] is
built into QUERIES/<dataset>/<split>/<master> and excludes the node sets of every directory
built before it, so node sets never repeat across masters or splits (the bash loop of
launch/README.md section 6). A directory whose manifest is complete is kept; `queries build`
rewrites any other one. The first failure stops the driver, so a later master never excludes
an unfinished one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .. import manifest, protocol

SPLITS = ("dev", "validation")


def plan(queries_root: str | Path, datasets: Sequence[str]) -> list[dict[str, Any]]:
    steps = []
    for dataset in datasets:
        built: list[str] = []
        for split in SPLITS:
            for index, master in enumerate(protocol.SEEDS[split]):
                out = str(Path(queries_root) / dataset / split / str(master))
                steps.append({"dataset": dataset, "split": split, "index": index, "master": str(master), "out": out,
                              "exclude": list(built)})
                built.append(out)
    return steps


def complete(out: str | Path) -> bool:
    try:
        record = json.loads((Path(out) / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("status") == "complete"


def command(step: dict[str, Any], store_root: str, python: str = sys.executable) -> list[str]:
    argv = [python, "-m", "scripts.canonical.queries", "build", "--store-root", store_root,
            "--dataset", step["dataset"], "--split", step["split"], "--master-index", str(step["index"]),
            "--out", step["out"]]
    return argv + (["--exclude", *step["exclude"]] if step["exclude"] else [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--queries-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["corafull", "arxiv", "mag"], choices=sorted(protocol.DATASETS))
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args(argv)
    for step in plan(args.queries_root, args.datasets):
        cmd = command(step, args.store_root)
        name = f"{step['dataset']}/{step['split']}/{step['master']}"
        if complete(step["out"]):
            print(f"keep {name}: complete", file=sys.stderr, flush=True)
            continue
        print(f"build {name}", file=sys.stderr, flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue
        code = subprocess.run(cmd, check=False).returncode
        if code != 0:
            print(f"failed {name}: exit {code}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
