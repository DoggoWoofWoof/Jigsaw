# External exact matchers

The canonical protocol compares Jigsaw with three external exact subgraph matchers. This directory
pins the source each one is built from, holds the patches we apply, and has one build script per
tool. The Python adapters live in `scripts/canonical/`.

| Tool | Adapter | Role in the protocol |
| --- | --- | --- |
| SubgraphMatching (Sun and Luo, SIGMOD 2020) | `scripts/canonical/classical.py` | `classical_gql`, `classical_cfl`, `classical_dpiso` baselines on the whole data graph; stage 2 of negative certification |
| Glasgow Subgraph Solver | `GlasgowBackend` in `scripts/canonical/backends.py` | per-partition solver in the cascade |
| GNN-PE (Ye, Lian and Chen, PVLDB 2024) | `scripts/canonical/gnnpe_adapter.py` | learned exact-matching baseline |

```
external/
  PINS.json                    one entry per tool: repo, commit, license, build, patches, binary, notes
  build_common.sh              shared steps (fetch at the pin, apply patches, install, build record)
  build_subgraphmatching.sh    -> bin/subgraphmatching/SubgraphMatching.out (+ libgraph.so, libutility.so)
  build_glasgow.sh             -> bin/glasgow_subgraph_solver
  build_gnnpe.sh               -> bin/gnnpe/main
  patches/<tool>/NNNN-*.patch  applied in PINS.json order with git apply
  src/<tool>/                  git clones made by the build scripts (not tracked)
  bin/                         built binaries and bin/<tool>.build.json (not tracked)
```

`.gitattributes` keeps the patches byte-exact and the scripts LF-only, because `core.autocrlf` is on
in the Windows working copy.

## Pins

| Tool | Repository | Pinned commit | License |
| --- | --- | --- | --- |
| subgraphmatching | https://github.com/RapidsAtHKUST/SubgraphMatching | `bcaa935c83709b0220e950634dbf5215ae2a526e` | MIT |
| glasgow | https://github.com/ciaranm/glasgow-subgraph-solver | `b62cc77777467e9fe19d9b558d706adb92989130` | MIT |
| gnnpe | https://github.com/JamesWhiteSnow/GNN-PE | `21c6953b423bb67254954e9a1c0aa8da6de03e15` | none at the pin |

Each pin is the full SHA of the commit whose source was read to write the patches and the adapters
(`verified_utc` says when). Every line number below refers to that commit. The fields of each
PINS.json entry:

| Field | Meaning |
| --- | --- |
| `repo` | clone URL |
| `commit` | full 40-hex SHA, or `null` with a "todo" note; a build script refuses a `null` pin |
| `verified_utc` | when the source at `commit` was read |
| `license` | SPDX id of the license file at `commit`, or `null` when there is none |
| `build` | the script that builds this tool |
| `patches` | repo-relative patch paths, applied in this order |
| `binary` | repo-relative path of the installed binary |
| `notes` | provenance, flags, semantics and known hazards |

To move a pin: read the new commit's source, update or regenerate the patches, set `commit` and
`verified_utc`, rebuild, and rerun the adapter tests. Never point a pin at a branch name.

### How manifests use PINS.json

`scripts.canonical.manifest.write_manifest(run_dir, ..., solvers={name: path})` calls
`solver_info(path)` for every solver. `solver_info(path, pins_file=None)` returns
`{"available": False}` for a missing path; otherwise it records the path, its full SHA-256 and byte
count, and embeds the whole of `external/PINS.json` (or `pins_file`) under `info["pins"]`
(`"unreadable"` when the JSON does not parse). A run manifest therefore names the exact source
commit and patch list next to the hash of the binary it ran. To tie that hash to a build, compare it
with `binary_sha256` in `external/bin/<tool>.build.json`. The adapters expose the resolved binary
path for this (`SubgraphMatchingBackend.binary_path`, `FullGraphClassicalRunner.binary_path`,
`GNNPERunner.binary_path`); pass that path, not a bare command name, because `solver_info` only
hashes a path that exists. `CODE_PATHS` in `manifest.py` includes `external/patches`, so a local
edit to a patch changes the manifest's `code_diff_sha256`.

SubgraphMatching links two shared libraries of its own; their hashes are in the build record's
`extra_files`, not in the run manifest.

## Building

On the Linux machine that will run the experiments:

```
bash external/build_subgraphmatching.sh
bash external/build_glasgow.sh
bash external/build_gnnpe.sh
```

Environment: `CXX` (default `g++`), `JOBS` (default `nproc`), `PYTHON` (default `python3`). Each
script clones into `external/src/<tool>` (or reuses the clone), checks out the pinned commit with
`--force`, runs `git clean -fdx`, verifies `git rev-parse HEAD` equals the pin, runs
`git apply --check` and then `git apply` for every pinned patch, builds, copies the binary into
`external/bin/`, prints its SHA-256, runs a smoke test, and writes `external/bin/<tool>.build.json`:
tool, repo, commit, `source_head`, patch paths with SHA-256, compiler version and path, the CPU that
`-march=native` resolved to, CMake version, binary path and SHA-256, extra files with SHA-256, smoke
test results under `checks`, build time, platform and machine.

All three upstream builds use `-march=native`, so a binary is tied to the CPU family it was built on.
Build on the machine (or instance type) that runs the experiments.

Smoke tests:

* SubgraphMatching: the upstream sample (`test/sample_dataset/test_case_1.graph` with
  `query1_positive.graph`) under the three presets; each must print one `embedding:` line per counted
  embedding, and the presets must agree on the count.
* Glasgow: a labelled triangle into K4, once as a decision run (`status = true` and a mapping) and
  once with `--count-solutions --print-all-solutions --solution-limit 100` (24 solutions, 24 mapping
  lines).
* GNN-PE: the upstream sample (`Test/data_graph.graph`, `Test/query_graph.graph`) with 5 contiguous
  partitions. The count run (`-n MAX`) must print an `Answer Number` line; a second run with
  `-n 1000 --print-embeddings` must print as many valid, distinct embeddings as it counts. A Python
  backtracking count (60 s budget) is recorded as `smoke_reference`, and `smoke_exact` says whether
  GNN-PE agreed; a disagreement warns but does not fail the build. Two probes then run the hazards
  described under GNN-PE below and record their answers (`probe_forward`, `probe_reversed`,
  `probe_root_leaf`, each with one true embedding); they never fail the build.

## SubgraphMatching

### Presets

The adapter runs one filter/order/engine triple per algorithm name. The defaults follow the
upstream "Recommendation" section: README lines 176-177 recommend GQL, CFL or DPiso as the filter,
GQL or RI as the order, and LFTJ as the engine; line 178 recommends failing-set pruning for large
queries; lines 181-182 describe the baseline setup (failing sets compiled in, 100000 results, 300 s),
and line 193 is the GQL command.

| Name | -filter | -order | -engine | Source |
| --- | --- | --- | --- | --- |
| `gql` | GQL | GQL | LFTJ | README 176-177, 193 |
| `cfl` | CFL | GQL | LFTJ | README 176-177 |
| `dpiso` | DPiso | GQL | LFTJ | README 176-177 |
| `ri` (sensitivity) | GQL | RI | LFTJ | README 187 |
| `gql_orig` (sensitivity) | GQL | GQL | GQL | README 138 |
| `cfl_orig` (sensitivity) | CFL | CFL | EXPLORE | README 129 (CFL's own engine) |
| `dpiso_orig` (sensitivity) | DPiso | DPiso | DPiso | README 130 |

The adapter passes `-num <max_solutions>` (the protocol asks for 1 witness) and `-print_embeddings`.

### Patches

Applied in this order (SHA-256 in `bin/subgraphmatching.build.json`):

1. `0001-print-embeddings.patch`. Upstream only counts embeddings; the protocol needs a witness it
   can verify. The patch adds `-print_embeddings` (read in `matching/StudyPerformance.cpp` next to
   the other options at line 90, echoed as `\tPrint Embeddings: true|false` after line 104) and
   prints `embedding: 0:v0 1:v1 ...` (query vertex id : data vertex id, `fflush` after every line)
   right after `embedding_cnt += 1` at every reachable increment site in
   `matching/EvaluateQuery.cpp`: `exploreGraph` (EXPLORE, line 127), `LFTJ` (340),
   `exploreGraphQLStyle` (GQL, 637), `exploreQuickSIStyle` (QSI, 752), `exploreDPisoStyle` (DPiso,
   924) and `exploreCECIStyle` (CECI, 1281). `exploreDPisoRecursiveStyle` is not patched because
   its only call is commented out (StudyPerformance.cpp line 330). Each site indexes `embedding` by
   query vertex, so the printed pairs are a mapping, not a matching order. Output and counts are
   unchanged without the flag.
2. `0002-max-query-graph-size-128.patch`. `configuration/config.h` line 12 sets
   `MAXIMUM_QUERY_GRAPH_SIZE 64`; the protocol's largest queries need more headroom. The macro only
   sizes failing-set bitsets and a reserve: `std::bitset<MAXIMUM_QUERY_GRAPH_SIZE>` for `ancestors`
   and `vec_failing_set` in `LFTJ` (EvaluateQuery.cpp lines 298-300), `exploreDPisoStyle` (861-863),
   `exploreDPisoRecursiveStyle` (1100, 1133-1168) and `exploreCECIStyle` (1252-1254), their
   declarations (EvaluateQuery.h 116-127), and `reverse_embedding.reserve(MAXIMUM_QUERY_GRAPH_SIZE * 2)`
   (lines 302, 865, 1104, 1256). No array is indexed by it elsewhere, so 128 changes memory only.
   The adapter refuses queries above 128 vertices.
3. `0003-enable-failing-set.patch`. Uncomments `#define ENABLE_FAILING_SET` (config.h line 48), as
   README lines 178 and 181 ask for the baseline. It only prunes branches that cannot lead to an
   embedding, so counts are unchanged; a NONE from a failing-set build is still worth cross-checking
   against Glasgow on a sample of queries.
4. `0004-invalid-vertex-sentinel.patch`. `matching/FilterVertices.cpp` line 11 defines
   `INVALID_VERTEX_ID` as 100000000 and uses it to mark pruned candidates and frontiers (lines 98,
   104, 307, 354, 396, 434, 464, 490, 506, 696, 803, 874, 899, 903, 996, 1020, 1024, 1043). On a
   data graph with more than 1e8 vertices (papers100M has 111M) vertex 100000000 is real and would
   be dropped as pruned. The patch uses `0xFFFFFFFFu`, which no 32-bit vertex id reaches.
5. `0005-steady-clock-timers.patch`. Every timer (`matching/StudyPerformance.cpp`, `graph/graph.cpp`,
   `matching/EdgeListToCSR.cpp`) reads `std::chrono::high_resolution_clock`, which libstdc++ defines
   as `system_clock`, the realtime clock. A step of that clock (NTP or a VM's time sync) lands in the
   reported stage times that `run_external.reported_stages` reads (`Load graphs time`, `Total time`).
   The patch reads `steady_clock` instead; only the clock changes, on all 25 lines.

### Labels

The loader (`graph/graph.cpp` lines 107-146) sets
`labels_count_ = max(#distinct labels, max label + 1)`, and `BuildReverseIndex` (lines 12-26)
allocates offsets for every label id up to that bound; a filter that looks up a query label at or
above the data graph's `labels_count_` reads out of bounds. So labels must be dense small integers,
and a query label that the data graph lacks must never reach the binary:

* `SubgraphMatchingBackend` (candidate-partition use) remaps the labels of each (query, target) pair
  to `0..L-1` with one shared map, and answers NONE without running when a query label is absent
  from the target.
* `FullGraphClassicalRunner` writes the data graph with the dense sigma codes of the regime
  (`0..num_codes-1`), and answers NONE without running when a query code is fresh
  (`>= num_codes`) or when the query needs more vertices of a code than the data graph has (`df`).

### File formats and limits

* Text format (`-d`, `-q`): `t N M`, then `v id label degree` for every vertex in id order, then
  `e u v` once per undirected edge. The loader sizes each adjacency list from the `v` line's degree
  (line 115) and fills it from the `e` lines (lines 129-142), so vertex lines must come first and
  degrees must be exact. Edge lines carry no label.
* Binary format (`-csr PREFIX`, `loadGraphFromFileCompressed`, lines 184-300): `PREFIX_deg.bin`
  (int32 4, uint32 N, uint32 M, uint32 degree[N]), `PREFIX_adj.bin` (uint32 neighbour lists in id
  order), `PREFIX_label.bin` (uint32 label[N]). `FullGraphClassicalRunner(data_format="csr")` writes
  this; it avoids parsing a multi-GB text file per query.
* Vertex ids, degrees and CSR offsets are 32-bit (`ui`), so `2|E| < 2^32`. papers100M
  (2|E| = 3.23e9) fits, without much headroom.
* The build keeps asserts on (no `CMAKE_BUILD_TYPE`, as upstream documents). The query plan check
  asserts a connected query, so the adapter returns ERROR for a disconnected query instead of
  running. A one-vertex query indexes `vec_failing_set[-1]` in `LFTJ` (lines 347-351) with failing
  sets on, so the adapter returns ERROR for it too.
* One query per process: every run re-reads the data graph. The binary reports `Load graphs time`
  apart from `Total time` (which excludes loading); the adapter's wall time includes loading, and the
  detail records both.

### Memory at scale (unverified, to measure)

`OPTIMIZED_LABELED_GRAPH 1` (config.h line 21, the default) builds one `unordered_map` of neighbour
label frequencies per data vertex (`BuildNLF`, graph.cpp lines 30-46). At papers100M scale that is
on the order of 100 GB. The GQL filter keeps a query-by-data bitmap. Measure peak RSS on the largest
graph before scheduling full runs, and consider `OPTIMIZED_LABELED_GRAPH 0` as a declared variant if
it does not fit.

### Output

`StudyPerformance.cpp` echoes its options, prints the phases, then (lines 403-413) `Load graphs time
(seconds)`, `Filter vertices time`, `Build table time`, `Generate query plan time`, `Enumerate time`,
`Preprocessing time`, `Total time`, `Memory cost (MB)`, `#Embeddings`, `Call Count`,
`Per Call Count Time (nanoseconds)`, and `End.` (line 414). `parse_subgraphmatching_output` reads
all of them; a run is complete only when both `#Embeddings` and `End.` appear. An unsupported
filter or engine prints `The specified ... type ... is not supported.` and exits with -1. There is no
time limit outside the spectrum mode, so the adapter kills the process at the cap.

## Glasgow Subgraph Solver

One patch, `0001-steady-clock-timeout.patch`: `--timeout` sleeps until `system_clock::now() + limit`
(`gss/timeout.cc` lines 25 and 42), so a step of the realtime clock ends a run early or late. The
patch reads `steady_clock` on both lines; `started_at` (`src/glasgow_subgraph_solver.cc` line 292)
stays on the wall clock because it prints a date. Every flag `GlasgowBackend` passes exists at the pin
(`src/glasgow_subgraph_solver.cc`): `--timeout` (integer seconds, line 53), `--count-solutions`
(60), `--print-all-solutions` (61), `--solution-limit` (62, implies counting) and
`--format vertexlabelledlad` (65). Output: `status = true|false|aborted` (lines 372-379; `aborted`
also when `--solution-limit` is reached), `solution_count` (382), `mapping = ...` (313-316 with
`--print-all-solutions`, 387-391 otherwise), `runtime` in ms (394); line 399 re-verifies each
mapping. `GlasgowBackend` reads mappings before status, so a limit-stop with mappings is FOUND.
The build needs CMake >= 3.21, C++20, GMP with gmpxx, and network access at configure time
(FetchContent pulls cxxopts and, unless installed, Catch2 and nlohmann_json).

## GNN-PE

### Pipeline

The adapter (`gnnpe_adapter.GNNPERunner`) reproduces the upstream workflow (README lines 54-96):

1. `membership.txt`: one `node part` line per vertex, vertices in ascending degree order with ties in
   id order (`gnnpe.py` lines 71-76), parts from METIS (`pymetis.part_graph(parts, ...,
   recursive=True)`, line 69). The directories `gnn-pe/` and `gnn-pe/partitions/partition-i/` must
   exist before the offline run (gnnpe.py lines 60-64); the binary never creates them.
2. Offline: `main -f DIR/ -d DATA -m offline -p P -l 2` writes every 3-vertex path
   (`gnn-pe/all_paths.txt`) and each partition's path list. `-f` is concatenated with file names, so
   it must end with `/`.
3. Online: `main -f DIR/ -d DATA -q QUERY -m online -p P -l 2 -e 2 -n K --print-embeddings`. The
   first online run builds each partition's R-tree (`index.dat`, include/custom.h lines 218-257) and
   later runs reuse it, so the adapter runs one warm-up query after the offline stage.

Data and query files use the SubgraphMatching text format (`libsrc/graph/graph.cpp` lines 163-242);
the adapter reuses the classical writer. The paths per partition are computed with
`all_paths_set` deduplication, so the input must not repeat an edge: a repeated neighbour makes
`dfs` (custom.h line 68) skip the stop condition and extend paths without bound.

### Semantics read from the source

* Non-induced: a candidate only needs edges to the already-matched query neighbours
  (`generateValidCandidates`, custom.h lines 779-789).
* Injective: `visited_vertices` (lines 774, 843, 850).
* Label equality (774; path filter 412) and a degree filter (775).
* Undirected: every `e u v` line is stored in both adjacency lists (graph.cpp lines 206-218).
* Upstream README line 3 says "isomorphic"; the code enumerates non-induced embeddings, like our
  protocol.

### Witness patch

`0001-print-embeddings.patch` adds `--print-embeddings` (src/main.cpp, next to the options at lines
47-54) and prints `embedding: 0:v0 1:v1 ...` after `embedding_cnt += 1` in `exploreQuickSIStyle`
(custom.h line 848), the only enumeration GNN-PE runs. `embedding` is indexed by query vertex (line
842). Upstream prints only `Answer Number: N Query Time (ms): T` (main.cpp line 179).

### Exactness hazards (from reading the code; the build probes record what the binary does)

1. One orientation per path. `dfs` stores each undirected 3-vertex path once, starting from the
   endpoint that comes first in `membership.txt` (custom.h lines 66-79); `dfs_query` does the same
   for the query in id order (lines 94-106). The R-tree leaf test then compares query and data paths
   position by position (labels and degrees, line 412), so a data path stored in the opposite
   orientation to the query path matches only when the reversed labels and degrees also fit (never
   when its two end labels differ). Only the start vertex of the refinement uses these
   candidate sets (lines 827-830; later vertices come from data neighbours, lines 757-797), so a
   missing start candidate drops true embeddings. The `probe_forward` / `probe_reversed` build probes
   store the same labelled path in the two orientations.
2. Leaf root. `query()` pushes the root with `level = 1` whatever its real level (line 375). When a
   partition's R-tree is a single leaf, the leaf is walked as an internal node (lines 441-481): path
   ids index `auxiliary_index` and `Q_map`, which have one element per R-tree node (lines 264, 379).
   A node holds `(4096 - 5) / (16 * 6 + 4)` = 40 entries at `-e 2` (rtnode.cpp line 28, entry.cpp
   line 59), and a root leaf splits once an insert brings it to 39 (capacity - 1) entries
   (rtnode.cpp lines 572-647, rtree.cpp lines 246-274), so a leaf root holds 1 to 38 paths. The
   adapter reads each partition's path count after the offline stage and refuses partitions with 1
   to 38 paths (`gnnpe_adapter.leaf_root_max_paths()`; `probe_root_leaf` records the binary's
   behaviour on a 1-path partition, and the padded probe graph stores 67 paths, above the bound).
3. Queries with fewer than 3 vertices have no 3-vertex path, so every candidate set is empty and the
   answer is 0. The adapter returns ERROR for them.
4. Only `-l 2` is consistent: main.cpp line 58 adds 1 to `path_length`, but `dfs` always emits 3-vertex
   paths (start depth `path_length - 2`, stop at `path_length`), while later loops index
   `path_length` positions. The adapter fixes `-l 2`.
5. Disconnected queries leave `next_vertex` uninitialized in `generateGQLQueryPlan` (lines 684-705).
   The adapter returns ERROR for them.
6. No trained GNN is loaded at this commit: vertex embeddings are a label-seeded `mt19937` vector plus
   the sum over neighbours (lines 492-544). Dominance still holds for true matches because the
   values are non-negative, but results should be described as "GNN-PE code at commit ..." rather
   than as the paper's trained model.
7. Early stop. `query()` leaves the whole traversal (`break`, not `continue`) at the first heap
   entry whose plan key is below the entry's key (lines 396-399). An entry admitted through the
   epsilon slack of line 460 can trigger it and skip entries still queued, so start candidates can
   be lost as in 1.
8. Heap limit. The search heap holds `MAX_HEAP_SIZE = 100000` entries (include/heap/heap.h line 3);
   a larger frontier prints `heap exceeded...` and exits with status 0 (libsrc/heap/heap.cpp lines
   84-89). No `Answer Number` line follows, so the adapter reports ERROR. Large partitions of
   million-node graphs may reach this limit.

Because of 1 and 7, a NONE from GNN-PE is not a certificate; the adapter marks it with
`detail["none_unverified"] = True`. The adapter's `GNNPE_SEMANTICS` and `SEMANTIC_DIFFERENCES`
dicts carry the same notes with their source lines.

### Costs and timing

* The offline stage enumerates every 3-vertex path: the sum over vertices of C(deg, 2).
  `gnnpe_adapter.estimate_path_count(indptr)` computes it before running; each path then gets a
  6-dimensional embedding and an R-tree entry of 100 bytes.
* Every online run re-reads the data graph and `all_paths.txt` and recomputes all vertex and path
  embeddings (main.cpp lines 124-134) before it starts its own clock (line 148). The reported
  `Query Time (ms)` excludes that work; the adapter charges wall time and records both.
* `0002-steady-clock-timers.patch` moves every GNN-PE timer (`src/main.cpp`, `include/custom.h`,
  `libsrc/graph/graph.cpp`) from `high_resolution_clock` (`system_clock` under libstdc++) to
  `steady_clock`, so `Query Time (ms)` cannot absorb a step of the realtime clock. GNN-PGE is not
  built and keeps its timers.
* Partitions are searched in an OpenMP loop (main.cpp lines 160-164); the adapter sets
  `OMP_NUM_THREADS` (default 1).
* `error()` in `libsrc/func/functions.cpp` (lines 24-29) exits with status 0, and an unreadable
  graph file exits with -1 (graph.cpp lines 166-168). The adapter treats a run without an
  `Answer Number` line as ERROR whatever the exit status.

### License

The repository has no license file at the pinned commit. Do not redistribute the patched source,
the patch, or a built binary without the authors' permission; the build script only fetches from the
upstream repository.
