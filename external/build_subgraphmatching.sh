#!/usr/bin/env bash
# Build SubgraphMatching (Sun and Luo, SIGMOD 2020) at the commit pinned in external/PINS.json,
# with the pinned patches, into external/bin/subgraphmatching/.
#
#   bash external/build_subgraphmatching.sh
#
# Needs git, cmake, make, a C++11 compiler with AVX2 (config.h SI 0) and python3.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

TOOL=subgraphmatching
require_cmd git cmake make "$CXX" "$PYTHON" ldd
grep -qw avx2 /proc/cpuinfo 2>/dev/null || die "this CPU lacks AVX2, which config.h SI 0 (the upstream default) needs"

fetch_pinned "$TOOL"
apply_patches "$TOOL"

SRC="${SRC_ROOT}/${TOOL}"
BUILD="${SRC}/build"
OUT="${BIN_ROOT}/${TOOL}"
# Upstream builds with a plain `cmake ..` (README "Compile"): no CMAKE_BUILD_TYPE, so the
# CMakeLists flags -std=c++11 -O3 -g -Wall -march=native -pthread apply and asserts stay on.
# cmake_minimum_required(VERSION 2.8) is rejected by CMake >= 4.0 without the policy floor.
# graph and utility are shared libraries; RPATH $ORIGIN lets the installed binary find the
# copies placed next to it.
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build "$BUILD" --target SubgraphMatching.out --parallel "$JOBS"

rm -f "${OUT}/SubgraphMatching.out" "${OUT}/libgraph.so" "${OUT}/libutility.so"
install_file "${BUILD}/matching/SubgraphMatching.out" "${OUT}/SubgraphMatching.out"
install_file "${BUILD}/graph/libgraph.so" "${OUT}/libgraph.so"
install_file "${BUILD}/utility/libutility.so" "${OUT}/libutility.so"

libs="$(ldd "${OUT}/SubgraphMatching.out" | grep -E 'libgraph|libutility' || true)"
echo "$libs"
if [[ "$(grep -cF "${OUT}/" <<< "$libs")" != 2 ]]; then
  die "SubgraphMatching.out does not load libgraph.so and libutility.so from ${OUT}"
fi

# Smoke test on the upstream sample: every preset must print one embedding line per counted
# embedding (0001-print-embeddings.patch) and agree on the count.
sample="${SRC}/test/sample_dataset"
counts=()
for preset in "GQL GQL LFTJ" "CFL GQL LFTJ" "DPiso GQL LFTJ"; do
  read -r f o e <<< "$preset"
  log="$("${OUT}/SubgraphMatching.out" -d "${sample}/test_case_1.graph" -q "${sample}/query1_positive.graph" \
    -filter "$f" -order "$o" -engine "$e" -num MAX -print_embeddings)"
  grep -qx "End." <<< "$log" || die "${preset}: no End. line"
  grep -q $'^\tPrint Embeddings: true$' <<< "$log" || die "${preset}: -print_embeddings not recognised"
  printed="$(grep -c '^embedding:' <<< "$log" || true)"
  counted="$(sed -n 's/^#Embeddings: //p' <<< "$log")"
  [[ "$printed" == "$counted" && "$counted" -gt 0 ]] || die "${preset}: printed ${printed}, counted ${counted}"
  counts+=("$counted")
done
[[ "${counts[0]}" == "${counts[1]}" && "${counts[1]}" == "${counts[2]}" ]] || die "presets disagree: ${counts[*]}"
echo "smoke test: ${counts[0]} embeddings from every preset"

write_build_json "$TOOL" "${OUT}/SubgraphMatching.out" "${OUT}/libgraph.so" "${OUT}/libutility.so" \
  "smoke_sample_embeddings=${counts[0]}"
