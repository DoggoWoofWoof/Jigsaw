"""Prepare the canonical stores: corafull, arxiv and mag (products and papers100m on request).

    python -m scripts.canonical.launch.prepare_all [--with-products] [--with-papers100m]
        [--alternatives] [--skip-sigma] [--force] [--metis-tool auto|gpmetis|pymetis] [DATASET ...]

DATASET names replace the default list (corafull arxiv mag), for example to redo one store.
Environment: JIGSAW_STORE_ROOT (required), JIGSAW_DATA_ROOT (default <repo>/data),
JIGSAW_RUNS_ROOT (default $JIGSAW_STORE_ROOT/_logs). Every dataset runs under memprobe:
<runs>/prepare/<dataset>.memprobe.json holds the exit code, wall time and peak RSS of the whole
process tree, <dataset>.log the progress log (stderr) and <dataset>.stdout the summary. The first
failing dataset stops the driver with exit 1, and each finished store is re-hashed against its
manifest. A rerun skips the stages that already finished (see scripts/canonical/prepare_data.py).
launch/prepare_all.sh runs this module; it runs the same way on Linux and Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .. import manifest, memprobe

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ["corafull", "arxiv", "mag"]
KNOWN = DEFAULT + ["products", "papers100m"]


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", choices=KNOWN, metavar="DATASET")
    parser.add_argument("--with-products", action="store_true")
    parser.add_argument("--with-papers100m", action="store_true")
    parser.add_argument("--alternatives", action="store_true")
    parser.add_argument("--skip-sigma", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--metis-tool", default=None, choices=["auto", "gpmetis", "pymetis"])
    args = parser.parse_args(argv)
    store_root = os.environ.get("JIGSAW_STORE_ROOT")
    if not store_root:
        parser.error("set JIGSAW_STORE_ROOT (see scripts/canonical/launch/env.example)")
    data_root = os.environ.get("JIGSAW_DATA_ROOT") or str(REPO_ROOT / "data")
    log_dir = Path(os.environ.get("JIGSAW_RUNS_ROOT") or Path(store_root) / "_logs") / "prepare"
    datasets = args.datasets or DEFAULT + ["products"] * args.with_products + ["papers100m"] * args.with_papers100m
    extra = [flag for flag, on in (("--alternatives", args.alternatives), ("--skip-sigma", args.skip_sigma),
                                   ("--force", args.force)) if on]
    extra += ["--metis-tool", args.metis_tool] if args.metis_tool else []
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(store_root).mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        probe = log_dir / f"{dataset}.memprobe.json"
        print(f"[{stamp()}] {dataset}: preparing (log {log_dir / f'{dataset}.log'})", flush=True)
        cmd = [sys.executable, "-m", "scripts.canonical.prepare_data", "--dataset", dataset, "--data-root", data_root,
               "--store-root", store_root, *extra]
        result = memprobe.run_measured(cmd, cwd=REPO_ROOT, stdout_path=log_dir / f"{dataset}.stdout",
                                       stderr_path=log_dir / f"{dataset}.log")
        probe.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[{stamp()}] {dataset}: exit {result['exit_code']}, {result['wall_seconds']:.1f} s, "
              f"peak RSS {result['peak_rss_bytes'] / 2**30:.2f} GiB", flush=True)
        if result["exit_code"] != 0 or result["timed_out"]:
            lines = (log_dir / f"{dataset}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"{dataset} failed; last lines of {log_dir / f'{dataset}.log'}:", *lines[-20:], sep="\n",
                  file=sys.stderr)
            return 1
        problems = manifest.verify_manifest(Path(store_root) / dataset / "_prepare_run")
        if problems:
            print(f"{dataset}: the store does not match its manifest:", *problems, sep="\n", file=sys.stderr)
            return 1
    print(f"stores ready under {store_root}: {' '.join(datasets)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
