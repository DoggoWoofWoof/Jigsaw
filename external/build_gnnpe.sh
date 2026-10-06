#!/usr/bin/env bash
# Build GNN-PE (Ye, Lian and Chen, PVLDB 17(7), 2024) at the commit pinned in external/PINS.json,
# with the pinned patch, into external/bin/gnnpe/main.
#
#   bash external/build_gnnpe.sh
#
# Needs git, cmake, make, a C++17 compiler with OpenMP (the upstream README asks for g++ >= 12.2 and
# CMake >= 3.28), python3 and coreutils timeout. The repository has no license file at the pinned
# commit: keep the source, the patch and the binary private until the authors allow redistribution.
#
# Besides the build smoke test, two informational probes run the exactness hazards noted in
# PINS.json on tiny graphs and record what the binary does; they never fail the build.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

TOOL=gnnpe
require_cmd git cmake make "$CXX" "$PYTHON" timeout

version="$("$CXX" -dumpversion 2>/dev/null | head -n1 || true)"
major="${version%%.*}"
if [[ "$major" =~ ^[0-9]+$ ]] && (( major < 12 )); then
  echo "warning: ${CXX} ${version} is older than the g++ 12.2 the upstream README asks for" >&2
fi

fetch_pinned "$TOOL"
apply_patches "$TOOL"

REPO="${SRC_ROOT}/${TOOL}"
SRC="${REPO}/GNN-PE"
BUILD="${SRC}/build"
BIN="${BIN_ROOT}/${TOOL}/main"
# Upstream builds with a plain `cmake ..` (README lines 72-77); GNN-PE/CMakeLists.txt line 7 hard-codes
# -std=c++17 -O3 -g -march=native -pthread -fopenmp. cmake_minimum_required(VERSION 3.0) is rejected
# by CMake >= 4.0 without the policy floor. The libraries under libsrc are OBJECT libraries linked
# straight into main, so the prebuilt .so files committed under GNN-PE/lib are never used.
cmake -S "$SRC" -B "$BUILD" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$BUILD" --target main --parallel "$JOBS"
rm -f "$BIN"
install_file "${BUILD}/src/main" "$BIN"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "${tmp}/helper.py" <<'PY'
"""Helpers for build_gnnpe.sh: membership files, probe graphs, embedding checks, reference counts."""
import os
import sys
import time


def load(path):
    labels, adj = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "t":
                labels = [0] * int(parts[1])
                adj = [set() for _ in labels]
            elif parts[0] == "v":
                labels[int(parts[1])] = int(parts[2])
            elif parts[0] == "e":
                u, v = int(parts[1]), int(parts[2])
                adj[u].add(v)
                adj[v].add(u)
    return labels, adj


def write_graph(path, labels, edges):
    degree = [0] * len(labels)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"t {len(labels)} {len(edges)}\n")
        for v, label in enumerate(labels):
            handle.write(f"v {v} {label} {degree[v]}\n")
        for u, v in edges:
            handle.write(f"e {u} {v}\n")


def membership(graph, parts, out, order):
    """One "node part" line per vertex; parts are contiguous id ranges."""
    labels, adj = load(graph)
    n = len(labels)
    # gnnpe.py lines 71-76 write vertices by ascending degree, ties in id order (stable sort).
    nodes = [int(x) for x in order] if order else sorted(range(n), key=lambda v: len(adj[v]))
    assert sorted(nodes) == list(range(n)), "membership order must list every vertex once"
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for v in nodes:
            handle.write(f"{v} {v * parts // n}\n")


def probe(directory):
    """Key path x(0)-y(1)-z(2) with labels 0, 1, 2; the query is the same labelled path.

    data.graph pads the key path with a star of 12 leaves (label 3, 66 more stored paths) so the
    partition's R-tree has an internal root; bare.graph is the key path alone (a leaf root).
    """
    os.makedirs(directory, exist_ok=True)
    key = [(0, 1), (1, 2)]
    write_graph(os.path.join(directory, "query.graph"), [0, 1, 2], key)
    write_graph(os.path.join(directory, "bare.graph"), [0, 1, 2], key)
    star = [(3, leaf) for leaf in range(4, 16)]
    write_graph(os.path.join(directory, "data.graph"), [0, 1, 2] + [3] * 13, key + star)


def embeddings(log, n):
    found = []
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("embedding:"):
                continue
            pairs = dict(tuple(int(x) for x in item.split(":")) for item in line.split()[1:])
            found.append(tuple(pairs.get(u, -1) for u in range(n)))
    return found


def valid(emb, q_labels, q_adj, d_labels, d_adj):
    if min(emb) < 0 or len(set(emb)) != len(emb):
        return False
    for u, v in enumerate(emb):
        if d_labels[v] != q_labels[u] or any(emb[w] not in d_adj[v] for w in q_adj[u]):
            return False
    return True


def reference_count(q_labels, q_adj, d_labels, d_adj, budget):
    """Non-induced, injective, label-equal embeddings by backtracking; None when over budget."""
    n = len(q_labels)
    order = [max(range(n), key=lambda u: len(q_adj[u]))]
    while len(order) < n:
        placed = set(order)
        frontier = [u for u in range(n) if u not in placed and q_adj[u] & placed]
        if not frontier:
            return None
        order.append(max(frontier, key=lambda u: len(q_adj[u] & placed)))
    back = [[w for w in q_adj[u] if w in set(order[:i])] for i, u in enumerate(order)]
    by_label = {}
    for v, label in enumerate(d_labels):
        by_label.setdefault(label, []).append(v)
    emb = [-1] * n
    used = set()
    state = {"total": 0, "steps": 0}
    deadline = time.monotonic() + budget

    def extend(i):
        if i == n:
            state["total"] += 1
            return
        u = order[i]
        pool = d_adj[emb[back[i][0]]] if back[i] else by_label.get(q_labels[u], [])
        for v in pool:
            state["steps"] += 1
            if state["steps"] % 65536 == 0 and time.monotonic() > deadline:
                raise TimeoutError
            if v in used or d_labels[v] != q_labels[u] or len(d_adj[v]) < len(q_adj[u]):
                continue
            if any(emb[w] not in d_adj[v] for w in back[i][1:]):
                continue
            emb[u] = v
            used.add(v)
            extend(i + 1)
            used.discard(v)
            emb[u] = -1

    try:
        extend(0)
    except TimeoutError:
        return None
    return state["total"]


def check(data, query, log, budget):
    d_labels, d_adj = load(data)
    q_labels, q_adj = load(query)
    found = embeddings(log, len(q_labels))
    ok = sum(valid(e, q_labels, q_adj, d_labels, d_adj) for e in found)
    ref = reference_count(q_labels, q_adj, d_labels, d_adj, float(budget))
    print(f"printed={len(found)} valid={ok} distinct={len(set(found))} "
          f"reference={'unfinished' if ref is None else ref}")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "membership":
        membership(args[0], int(args[1]), args[2], args[3:])
    elif command == "probe":
        probe(args[0])
    elif command == "check":
        check(*args)
    else:
        sys.exit(f"unknown command {command}")
PY

# layout DIR PARTS: the directories GNN-PE writes into but never creates (gnnpe.py lines 60-64).
layout() {
  local i
  rm -rf "$1"
  for ((i = 0; i < $2; i++)); do
    mkdir -p "$1/gnn-pe/partitions/partition-${i}"
  done
}

# answers LOG: the count from "Answer Number: N Query Time (ms): T" (src/main.cpp line 179).
answers() {
  grep -m1 '^Answer Number: ' "$1" | sed 's/^Answer Number: \([0-9][0-9]*\) .*/\1/' || true
}

# Smoke test on the upstream sample (Test/, README lines 81-85) with contiguous partitions.
data="${REPO}/Test/data_graph.graph"
query="${REPO}/Test/query_graph.graph"
work="${tmp}/test"
layout "$work" 5
"$PYTHON" "${tmp}/helper.py" membership "$data" 5 "${work}/gnn-pe/membership.txt"
timeout 600 "$BIN" -f "${work}/" -d "$data" -m offline -p 5 -l 2 > "${tmp}/offline.log" \
  || die "offline run on the upstream sample failed"
# The first online run also builds each partition's R-tree (index.dat), which later runs reuse.
timeout 600 "$BIN" -f "${work}/" -d "$data" -q "$query" -m online -p 5 -l 2 -e 2 -n MAX \
  > "${tmp}/count.log" || die "online run on the upstream sample failed"
total="$(answers "${tmp}/count.log")"
[[ -n "$total" ]] || die "online run printed no Answer Number line"
timeout 600 "$BIN" -f "${work}/" -d "$data" -q "$query" -m online -p 5 -l 2 -e 2 -n 1000 \
  --print-embeddings > "${tmp}/print.log" || die "--print-embeddings run failed"
limited="$(answers "${tmp}/print.log")"
[[ -n "$limited" ]] || die "--print-embeddings run printed no Answer Number line"
result="$("$PYTHON" "${tmp}/helper.py" check "$data" "$query" "${tmp}/print.log" 60)"
read -r printed valid distinct reference <<< "$(sed 's/[a-z]*=//g' <<< "$result")"
[[ "$printed" == "$limited" ]] || die "printed ${printed} embeddings, the run counted ${limited}"
[[ "$valid" == "$printed" && "$distinct" == "$printed" ]] \
  || die "${printed} printed embeddings, ${valid} valid, ${distinct} distinct"
if [[ "$reference" == "unfinished" ]]; then
  exact=unknown
elif [[ "$reference" == "$total" ]]; then
  exact=true
else
  exact=false
  echo "warning: GNN-PE counts ${total} embeddings of the upstream sample query, backtracking counts ${reference}" >&2
fi
echo "smoke test: ${total} embeddings (reference ${reference}), ${printed} printed and valid"

# Orientation probe: GNN-PE stores each 3-vertex data path in one orientation (the endpoint listed
# first in membership.txt starts it, include/custom.h lines 66-79) and compares query and data paths
# position by position (line 412). Listing x first stores x-y-z, which reads like the query; listing
# z first stores z-y-x. Both runs have exactly one true embedding.
probe_run() {  # probe_run NAME DATA ORDER...: prints the answer count, or rc=N if the run failed
  local name="$1" graph="$2" dir="${tmp}/$1" rc=0
  shift 2
  layout "$dir" 1
  "$PYTHON" "${tmp}/helper.py" membership "$graph" 1 "${dir}/gnn-pe/membership.txt" "$@"
  timeout 60 "$BIN" -f "${dir}/" -d "$graph" -m offline -p 1 -l 2 > "${dir}.offline.log" 2>&1 || rc=$?
  if (( rc == 0 )); then
    timeout 60 "$BIN" -f "${dir}/" -d "$graph" -q "${tmp}/probe/query.graph" -m online -p 1 -l 2 -e 2 \
      -n MAX --print-embeddings > "${dir}.online.log" 2>&1 || rc=$?
  fi
  if (( rc != 0 )); then
    echo "rc=${rc}"
  else
    answers "${dir}.online.log" | grep . || echo "no-answer-line"
  fi
}
"$PYTHON" "${tmp}/helper.py" probe "${tmp}/probe"
padding="$(seq 3 15 | tr '\n' ' ')"
# shellcheck disable=SC2086
forward="$(probe_run forward "${tmp}/probe/data.graph" 0 2 1 $padding)"
# shellcheck disable=SC2086
reversed="$(probe_run reversed "${tmp}/probe/data.graph" 2 0 1 $padding)"
# The key path alone gives the partition a leaf root, which query() still enters with level 1
# (include/custom.h line 375) and walks as an internal node (lines 441-481).
root_leaf="$(probe_run root_leaf "${tmp}/probe/bare.graph" 0 2 1)"
echo "orientation probe (1 true embedding each): x first ${forward}, z first ${reversed}, leaf root ${root_leaf}"

write_build_json "$TOOL" "$BIN" \
  "smoke_answers=${total}" "smoke_reference=${reference}" "smoke_exact=${exact}" \
  "smoke_printed_valid=${printed}" \
  "probe_expected=1" "probe_forward=${forward}" "probe_reversed=${reversed}" "probe_root_leaf=${root_leaf}"
