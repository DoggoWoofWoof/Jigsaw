#!/usr/bin/env bash
# Build the Glasgow Subgraph Solver at the commit pinned in external/PINS.json into
# external/bin/glasgow_subgraph_solver.
#
#   bash external/build_glasgow.sh
#
# Needs git, cmake >= 3.21, a C++20 compiler, GMP with gmpxx (libgmp-dev), python3, and network
# access at configure time: FetchContent pulls cxxopts v3.3.1 and, unless installed, Catch2 v3.4.0
# and nlohmann_json v3.12.0 (CMakeLists.txt lines 99-128 at the pinned commit).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

TOOL=glasgow
require_cmd git cmake "$CXX" "$PYTHON"

fetch_pinned "$TOOL"
apply_patches "$TOOL"

SRC="${SRC_ROOT}/${TOOL}"
BUILD="${SRC}/build"
BIN="${BIN_ROOT}/glasgow_subgraph_solver"
# Release is the upstream default build type (CMakeLists.txt lines 16-18): -O3 plus -march=native.
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$BUILD" --target glasgow_subgraph_solver --parallel "$JOBS"
rm -f "$BIN"
install_file "${BUILD}/glasgow_subgraph_solver" "$BIN"

# Smoke test with the flags GlasgowBackend passes: a labelled triangle into K4 has 4*3*2 = 24
# injective embeddings.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '3\n0 2 1 2\n0 2 0 2\n0 2 0 1\n' > "${tmp}/pattern.lad"
printf '4\n0 3 1 2 3\n0 3 0 2 3\n0 3 0 1 3\n0 3 0 1 2\n' > "${tmp}/target.lad"
one="$("$BIN" --format vertexlabelledlad --timeout 10 "${tmp}/pattern.lad" "${tmp}/target.lad")"
grep -qx "status = true" <<< "$one" || die "decision run did not report status = true"
grep -q "^mapping = " <<< "$one" || die "decision run printed no mapping"
all="$("$BIN" --format vertexlabelledlad --timeout 10 --count-solutions --print-all-solutions \
  --solution-limit 100 "${tmp}/pattern.lad" "${tmp}/target.lad")"
grep -qx "solution_count = 24" <<< "$all" || die "enumeration run did not count 24 solutions"
[[ "$(grep -c '^mapping = ' <<< "$all")" == 24 ]] || die "enumeration run did not print 24 mappings"
echo "smoke test: triangle into K4, 24 solutions"

write_build_json "$TOOL" "$BIN" "smoke_triangle_k4_solutions=24"
