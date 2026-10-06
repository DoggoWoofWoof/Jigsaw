#!/usr/bin/env bash
# Prepare the canonical stores: corafull, arxiv and mag (products and papers100m on request).
#
#   bash scripts/canonical/launch/prepare_all.sh [--with-products] [--with-papers100m]
#       [--alternatives] [--skip-sigma] [--force] [--metis-tool auto|gpmetis|pymetis] [DATASET ...]
#
# DATASET names replace the default list (corafull arxiv mag), for example to redo one store.
# Environment: JIGSAW_STORE_ROOT (required), JIGSAW_DATA_ROOT (default <repo>/data),
# JIGSAW_RUNS_ROOT (default $JIGSAW_STORE_ROOT/_logs), PYTHON (default python3).
# Every dataset runs under memprobe: <runs>/prepare/<dataset>.memprobe.json holds the exit code,
# wall time and peak RSS of the whole process tree, <dataset>.log the progress log (stderr) and
# <dataset>.stdout the summary. The first failing dataset stops the script. A rerun skips the
# stages that already finished (see scripts/canonical/prepare_data.py).
# The work is done by scripts/canonical/launch/prepare_all.py, which also runs on Windows.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python3}" -m scripts.canonical.launch.prepare_all "$@"
