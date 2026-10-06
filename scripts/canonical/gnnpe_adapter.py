"""GNN-PE (Ye, Lian and Chen, PVLDB 17(7), 2024) as a learned exact-matching baseline.

Exporters for the data graph (dense sigma codes as labels, in the SubgraphMatching text format
that GNN-PE's Static_Graph loader shares) and for query sets, the offline and online commands,
an output parser, and GNNPERunner, whose solve() returns backends.SolveOutcome with embeddings
in global node ids.

The binary is external/bin/gnnpe/main, built by external/build_gnnpe.sh at the commit pinned in
external/PINS.json with 0001-print-embeddings.patch: upstream prints only a count, the patch adds
--print-embeddings so the protocol gets witnesses. GNNPE_SEMANTICS records what the code
computes and SEMANTIC_DIFFERENCES where it departs from the protocol's exact semantics, with
file and line references at the pinned commit before the patch. Because of the orientation and
early-break hazards, a NONE from GNN-PE is not a certificate: it carries
detail["none_unverified"] = True.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import classical
from .backends import ERROR, FOUND, NONE, TIMEOUT, SolveOutcome
from .store import LocalGraph, local_graph_from_edges

PATH_LENGTH = 2  # -l; the only consistent value (SEMANTIC_DIFFERENCES["path_length"])
EMBEDDING_DIM = 2  # -e; upstream default vde_dim (include/custom.h line 47)
NODE_BYTES = 4096  # R-tree block size (include/custom.h line 239)
PREPARED_SCHEMA = "jigsaw-gnnpe-prepared-v1"
METIS_RECURSIVE = "metis_recursive"  # membership source when none is given (gnnpe.py line 69)

GNNPE_SEMANTICS: dict[str, str] = {
    "matching": "non-induced: a candidate needs edges only to already matched query neighbours "
                "(generateValidCandidates, include/custom.h lines 779-789)",
    "injective": "visited_vertices excludes used data vertices (include/custom.h lines 774, 843, 850)",
    "labels": "label equality (include/custom.h line 774; path filter line 412) plus a degree filter (line 775)",
    "direction": "undirected: every `e u v` line fills both adjacency lists (libsrc/graph/graph.cpp lines 206-218)",
    "input_format": "`t N M`, `v id label degree` in id order, `e u v` (libsrc/graph/graph.cpp lines 163-242); "
                    "the same text format as SubgraphMatching",
    "enumeration": "exploreQuickSIStyle (include/custom.h lines 799-888) in a GraphQL order (generateGQLQueryPlan, "
                   "lines 670-722); it stops after -n answers (lines 851-853)",
    "candidates": "only the start vertex draws from the R-tree candidate sets (lines 827-830); later vertices come "
                  "from data neighbours of their pivot (lines 757-797)",
    "query_plan": "query paths sorted by degree sum, kept greedily while they cover a new query vertex "
                  "(include/custom.h lines 601-628); the plan size is printed as a bare line (line 630)",
    "answer_line": "`Answer Number: N Query Time (ms): T` (src/main.cpp line 179); N is embedding_cnt "
                   "(include/custom.h line 920)",
    "query_time": "T covers the query's vertex and path embeddings, the slowest partition's R-tree search and the "
                  "refinement (src/main.cpp lines 148-179); loading the data graph and all_paths.txt, recomputing "
                  "every data vertex and path embedding, loading the R-trees, loading the query and enumerating its "
                  "paths (lines 71-146) happen outside it",
    "witnesses": "0001-print-embeddings.patch prints `embedding: 0:v0 1:v1 ...` after `embedding_cnt += 1` "
                 "(include/custom.h line 848); embedding is indexed by query vertex (line 842)",
    "gnn": "no trained model is loaded at this commit: a vertex embedding is a label-seeded mt19937 vector plus "
           "the sum over its neighbours (include/custom.h lines 492-544); values are non-negative, so a true "
           "match still dominates",
    "offline": "every 3-vertex path is enumerated from each vertex in membership.txt order and stored once "
               "(src/main.cpp lines 75-120; dfs, include/custom.h lines 66-92); a path belongs to the partition "
               "of its first vertex (src/main.cpp line 95)",
    "index": "the first online run builds each partition's R-tree (index.dat, include/custom.h lines 218-257); "
             "later runs load any index.dat they find (lines 222-233), even one left by an older offline stage",
    "exit_status": "error() exits with status 0 (libsrc/func/functions.cpp lines 24-29); an unreadable graph file "
                   "prints `Can not open the graph file` and exits with -1 (libsrc/graph/graph.cpp lines 166-168)",
}

SEMANTIC_DIFFERENCES: dict[str, str] = {
    "orientation": "each 3-vertex path is stored in one orientation (dfs, include/custom.h lines 66-79; dfs_query, "
                   "lines 94-106) and compared position by position (line 412); a data path stored against the "
                   "query path's orientation passes only when the reversed labels and degrees also fit, so the "
                   "start vertex can lose its true image (lines 827-830) and true embeddings are dropped "
                   "(false NONE)",
    "early_break": "query() ends the whole traversal (break, not continue) at the first heap entry whose plan key is "
                   "below the entry's key (include/custom.h lines 396-399); an entry admitted through the epsilon "
                   "slack of line 460 can trigger it and skip the entries still queued",
    "leaf_root": "query() pushes the root with level 1 (include/custom.h line 375); when a partition's R-tree is "
                 "one leaf (1 to leaf_root_max_paths() paths) its path ids are used as node ids in "
                 "auxiliary_index and Q_map (lines 264, 379, 441-481): out-of-bounds accesses",
    "heap_limit": "the search heap holds MAX_HEAP_SIZE = 100000 entries (include/heap/heap.h line 3); a larger "
                  "frontier prints `heap exceeded...` and exits with status 0 (libsrc/heap/heap.cpp lines 84-89)",
    "small_queries": "dfs_query finds no 3-vertex path in a query under 3 vertices (include/custom.h lines 94-106), "
                     "so the plan and every candidate set stay empty and the answer is 0 whatever the data",
    "path_length": "src/main.cpp line 58 adds 1 to -l while dfs always emits 3-vertex paths and later loops index "
                   "path_length positions: only -l 2 is consistent",
    "disconnected": "generateGQLQueryPlan leaves next_vertex uninitialized for a disconnected query "
                    "(include/custom.h lines 684-705)",
    "duplicate_edges": "a repeated neighbour makes dfs miss its stop test (include/custom.h line 68) and extend "
                       "paths without bound; the writers never repeat an edge",
    "cost_accounting": "Query Time excludes the per-run reload and recomputation of all data embeddings "
                       "(src/main.cpp lines 71-146); the adapter charges wall time and records both",
}


class GNNPEError(RuntimeError):
    """The offline stage or the warm-up failed, or the data graph is outside what GNN-PE handles."""


def node_capacity(embedding_dim: int = EMBEDDING_DIM) -> int:
    """R-tree entries per node: (4096 - 5) // (2 * dim * 8 + 4) with dim = 3 * embedding_dim (40 at -e 2).

    Header and entry sizes from libsrc/rtree/rtnode.cpp line 28 and libsrc/rtree/entry.cpp line 59.
    """
    dim = (PATH_LENGTH + 1) * int(embedding_dim)
    return (NODE_BYTES - 5) // (2 * dim * 8 + 4)


def leaf_root_max_paths(embedding_dim: int = EMBEDDING_DIM) -> int:
    """Most paths a partition can hold while its R-tree root is still a leaf (38 at -e 2).

    A root leaf splits when an insert brings it to capacity - 1 entries (libsrc/rtree/rtnode.cpp
    lines 572-647, libsrc/rtree/rtree.cpp lines 246-274).
    """
    return node_capacity(embedding_dim) - 2


def leaf_root_partitions(counts: Sequence[int], embedding_dim: int = EMBEDDING_DIM) -> list[int]:
    """Partitions whose R-tree is a single leaf with entries (SEMANTIC_DIFFERENCES["leaf_root"])."""
    top = leaf_root_max_paths(embedding_dim)
    return [i for i, c in enumerate(counts) if 1 <= int(c) <= top]


# ----------------------------------------------------------------------------
# Data graph and query exporters
# ----------------------------------------------------------------------------
def estimate_path_count(indptr: np.ndarray, chunk: int = 1 << 22) -> int:
    """3-vertex paths the offline stage stores: the sum over vertices of C(degree, 2)."""
    n = int(indptr.shape[0]) - 1
    chunk = max(1, int(chunk))
    total = 0
    for start in range(0, n, chunk):
        d = np.diff(np.asarray(indptr[start:min(n, start + chunk) + 1], dtype=np.int64))
        total += int((d * (d - 1) // 2).sum())
    return total


def degree_order(indptr: np.ndarray) -> np.ndarray:
    """Vertices in ascending degree, ties in id order (gnnpe.py lines 71-72, a stable sort)."""
    return np.argsort(np.diff(np.asarray(indptr, dtype=np.int64)), kind="stable")


def write_membership(indptr: np.ndarray, membership: np.ndarray, path: str | os.PathLike,
                     chunk: int = 1 << 20) -> None:
    """gnn-pe/membership.txt: `node part` per vertex in degree_order (gnnpe.py lines 74-76)."""
    order = degree_order(indptr)
    membership = np.asarray(membership, dtype=np.int64)
    if membership.shape[0] != order.shape[0]:
        raise ValueError("membership must give one part per vertex")
    chunk = max(1, int(chunk))
    with open(path, "wb") as handle:
        for start in range(0, order.shape[0], chunk):
            nodes = order[start:start + chunk]
            handle.write(classical.format_int_rows(b"", [nodes, membership[nodes]]))


def metis_membership(indptr: np.ndarray, indices: np.ndarray, parts: int) -> np.ndarray:
    """Parts as gnnpe.py line 69 computes them: pymetis.part_graph(parts, ..., recursive=True), in process.

    Where in-process METIS is a risk (MAG-scale graphs), compute the parts elsewhere and pass them
    as GNNPERunner(membership=...); scripts.canonical.partition runs METIS in a child process.
    """
    n = int(indptr.shape[0]) - 1
    if parts <= 1 or n <= 1:
        return np.zeros(max(n, 0), dtype=np.int64)
    import pymetis

    xadj = np.asarray(indptr, dtype=np.int64)
    adjncy = np.asarray(indices, dtype=np.int64)
    if hasattr(pymetis, "CSRAdjacency"):
        _, membership = pymetis.part_graph(parts, adjacency=pymetis.CSRAdjacency(xadj, adjncy), recursive=True)
    else:
        _, membership = pymetis.part_graph(parts, xadj=xadj, adjncy=adjncy, recursive=True)
    return np.asarray(membership, dtype=np.int64)


def partition_dirs(dataset_dir: str | os.PathLike, partitions: int) -> list[Path]:
    """gnn-pe/partitions/partition-i/ (src/main.cpp lines 61 and 100); the binary never creates them."""
    base = Path(dataset_dir) / "gnn-pe" / "partitions"
    return [base / f"partition-{i}" for i in range(int(partitions))]


def _membership_sha256(membership: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(membership, dtype="<i8").tobytes()).hexdigest()


def export_data_graph(store: Any, regime: str, out_dir: str | os.PathLike, *, partitions: int = 5,
                      membership: np.ndarray | None = None) -> dict[str, Any]:
    """The dataset directory an offline run expects: data graph, gnn-pe/membership.txt, partition dirs.

    The data graph is the classical writer's `<dataset>.<regime>.graph` (reused while its sidecar
    matches). gnn-pe/ is recreated, as gnnpe.py does (lines 60-64), so no stale all_paths.txt or
    index.dat survives. membership defaults to metis_membership().
    """
    out_dir = Path(out_dir)
    data_path, data_info = classical.ensure_data_graph(store, regime, out_dir)
    if membership is None:
        membership = metis_membership(store.indptr, store.indices, partitions)
    membership = np.asarray(membership, dtype=np.int64)
    if membership.shape[0] != store.num_nodes:
        raise ValueError("membership must give one part per data vertex")
    if membership.shape[0] and (int(membership.min()) < 0 or int(membership.max()) >= partitions):
        raise ValueError(f"membership parts must lie in [0, {partitions})")
    gdir = out_dir / "gnn-pe"
    shutil.rmtree(gdir, ignore_errors=True)
    for d in partition_dirs(out_dir, partitions):
        d.mkdir(parents=True)
    write_membership(store.indptr, membership, gdir / "membership.txt")
    return {
        "data_graph": str(data_path),
        "data": data_info,
        "membership": str(gdir / "membership.txt"),
        "membership_sha256": _membership_sha256(membership),
        "partitions": int(partitions),
        "part_sizes": np.bincount(membership, minlength=partitions).tolist(),
        "path_count_estimate": estimate_path_count(store.indptr),
    }


def unsupported_query(query: LocalGraph) -> str | None:
    """Why the pinned GNN-PE code cannot answer this query exactly, or None."""
    if query.n < PATH_LENGTH + 1:
        return f"query has {query.n} vertices; GNN-PE needs 3 or more (no 3-vertex path, the answer is always 0)"
    if query.components()[0] > 1:
        return "disconnected query: generateGQLQueryPlan leaves next_vertex uninitialized"
    return None


def _safe_name(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", uid)


def export_query_set(store: Any, queries: Iterable[Any], regime: str,
                     out_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """One query file per queries.Query (labels = sigma codes of `regime`) plus out_dir/queries.json.

    Each entry notes whether the query is trivially NONE on the whole graph and whether GNN-PE
    cannot answer it (fewer than 3 vertices, disconnected); such queries are written anyway.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = store.sigma(regime)
    entries = []
    for q in queries:
        if q.regime != regime:
            raise ValueError(f"query {q.uid} is for regime {q.regime}, not {regime}")
        graph = q.local_graph(np.asarray(table.query_codes(q.payload, store), dtype=np.int64))
        path = out_dir / f"{_safe_name(q.uid)}.graph"
        classical.write_subgraphmatching_graph(graph, path)
        entries.append({"uid": q.uid, "file": path.name, "family": q.family, "size": int(q.size), "kind": q.kind,
                        "n": int(graph.n), "m": int(graph.num_undirected_edges),
                        "trivial_none": classical.whole_graph_none(store, regime, graph),
                        "unsupported": unsupported_query(graph)})
    manifest = {"dataset": store.dataset, "regime": regime, "path_length": PATH_LENGTH, "queries": entries}
    (out_dir / "queries.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return entries


# ----------------------------------------------------------------------------
# Commands and output
# ----------------------------------------------------------------------------
def _dir_arg(dataset_dir: str | os.PathLike) -> str:
    # -f is concatenated with file names (src/main.cpp lines 61, 80, 110), so it must end with "/".
    return Path(dataset_dir).as_posix().rstrip("/") + "/"


def offline_command(command: Sequence[str], dataset_dir: str | os.PathLike, data_path: str | os.PathLike, *,
                    partitions: int, embedding_dim: int = EMBEDDING_DIM) -> list[str]:
    """`main -f DIR/ -d DATA -m offline -p P -l 2 -e E`."""
    return [*command, "-f", _dir_arg(dataset_dir), "-d", str(data_path), "-m", "offline", "-p", str(int(partitions)),
            "-l", str(PATH_LENGTH), "-e", str(int(embedding_dim))]


def online_command(command: Sequence[str], dataset_dir: str | os.PathLike, data_path: str | os.PathLike,
                   query_path: str | os.PathLike, *, partitions: int, embedding_dim: int = EMBEDDING_DIM,
                   max_solutions: int | None = 1, print_embeddings: bool = True) -> list[str]:
    """`main -f DIR/ -d DATA -q QUERY -m online -p P -l 2 -e E -n K [--print-embeddings]`; None means -n MAX."""
    cmd = [*command, "-f", _dir_arg(dataset_dir), "-d", str(data_path), "-q", str(query_path), "-m", "online",
           "-p", str(int(partitions)), "-l", str(PATH_LENGTH), "-e", str(int(embedding_dim)),
           "-n", "MAX" if max_solutions is None else str(int(max_solutions))]
    return cmd + (["--print-embeddings"] if print_embeddings else [])


_ANSWER = re.compile(r"^Answer Number: (\d+) Query Time \(ms\): (\S+)$")


def parse_gnnpe_output(text: str, query_n: int) -> tuple[str | None, list[np.ndarray], dict[str, Any]]:
    """(status, embeddings, detail) from an online run.

    status is "complete" when the `Answer Number` line was printed, otherwise None. Embeddings come
    from `embedding: u:v ...` lines; a line without one distinct pair per query vertex counts in
    detail["malformed_lines"] and the unterminated last line of a killed run is skipped. detail also
    holds answer_number, query_time_ms, data_graph ({vertices, edges, labels}) and query_plan_paths.
    """
    detail: dict[str, Any] = {"malformed_lines": 0}
    embeddings: list[np.ndarray] = []
    lines = text.split("\n")
    if lines and lines[-1].strip():
        detail["unterminated_tail"] = True
    status = None
    for raw in lines[:-1]:
        line = raw.strip()
        if line.startswith("embedding:"):
            phi = classical.parse_embedding_pairs(line[len("embedding:"):], query_n)
            if phi is None:
                detail["malformed_lines"] += 1
            else:
                embeddings.append(phi)
        elif line.startswith("Answer Number:"):
            match = _ANSWER.match(line)
            if match:
                status = "complete"
                detail["answer_number"] = int(match.group(1))
                try:
                    detail["query_time_ms"] = float(match.group(2))
                except ValueError:
                    detail["query_time_ms"] = match.group(2)
        elif line.startswith("|V|:"):
            meta = classical.parse_graph_meta(line)
            if meta is not None and "data_graph" not in detail:
                detail["data_graph"] = meta
        elif line.isdigit():
            detail["query_plan_paths"] = int(line)
        elif line.startswith("Can not open"):
            detail["unreadable"] = line
    return status, embeddings, detail


def classify_output(name: str, text: str, returncode: int, killed: bool, query_n: int, max_solutions: int,
                    elapsed: float, detail: dict[str, Any], check: Callable[[np.ndarray], str | None]) -> SolveOutcome:
    """SolveOutcome of an online run: printed embeddings first, then the kill, then the Answer line."""
    status, embeddings, parsed = parse_gnnpe_output(text, query_n)
    detail = {**parsed, **detail, "returncode": returncode, "killed_at_cap": killed}
    if embeddings:
        chosen = embeddings[:max_solutions]
        for phi in chosen:
            problem = check(phi)
            if problem:
                detail.update({"error": f"the solver printed an invalid embedding: {problem}",
                               "output_tail": text[-2000:]})
                return SolveOutcome(ERROR, [], elapsed, name, detail)
        return SolveOutcome(FOUND, chosen, elapsed, name, detail)
    if killed:
        return SolveOutcome(TIMEOUT, [], elapsed, name, detail)
    if status == "complete" and returncode == 0:
        if int(parsed["answer_number"]) == 0:
            return SolveOutcome(NONE, [], elapsed, name, {**detail, "none_unverified": True})
        detail["error"] = "answers counted but not printed (binary without 0001-print-embeddings.patch?)"
    elif status is None:
        detail["error"] = "no Answer Number line (error() exits with status 0; GNNPE_SEMANTICS['exit_status'])"
    detail["output_tail"] = text[-2000:]
    return SolveOutcome(ERROR, [], elapsed, name, detail)


def read_partition_path_counts(dataset_dir: str | os.PathLike, partitions: int) -> list[int]:
    """First line of every partition_paths.txt (src/main.cpp line 102); -1 when a file is missing."""
    counts = []
    for d in partition_dirs(dataset_dir, partitions):
        try:
            with open(d / "partition_paths.txt", encoding="ascii") as handle:
                counts.append(int(handle.readline().strip() or -1))
        except (OSError, ValueError):
            counts.append(-1)
    return counts


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
class GNNPERunner:
    """GNN-PE on the whole data graph of one (store, regime); FOUND carries global node ids.

    prepare() exports <cache_dir>/<dataset>.<regime>.gnnpe-p<P>-e<E>/, runs the offline stage,
    refuses partitions whose R-tree root would be a single leaf, and runs one warm-up query so the
    R-trees exist before any timed query. Its result is kept in prepared.json and reused while the
    store, sigma table, membership, options and binary are unchanged; none of it is charged to a
    query. One writer per directory: a runner with another membership rewrites it.
    """

    def __init__(self, store: Any, regime: str, *, cache_dir: str | os.PathLike,
                 binary: str | os.PathLike | Sequence[str] | None = None, partitions: int = 5,
                 embedding_dim: int = EMBEDDING_DIM, threads: int = 1, membership: np.ndarray | None = None,
                 startup_allowance: float = 0.0, max_paths: int | None = None,
                 offline_time_limit: float = 7 * 24 * 3600.0, workdir: str | None = None) -> None:
        """binary: GNNPE_BIN or "main" by default; a list is a command prefix. threads sets OMP_NUM_THREADS.

        membership: one part per data vertex (default: metis_membership, as gnnpe.py computes it).
        startup_allowance extends each kill cap (sensitivity runs only; 0 charges everything).
        max_paths refuses a data graph whose offline stage would store more 3-vertex paths.
        """
        binary = binary or os.environ.get("GNNPE_BIN", "main")
        self.command = [str(binary)] if isinstance(binary, (str, os.PathLike)) else [str(b) for b in binary]
        self.binary = self.command[-1]
        self.name = "gnnpe"
        self.store = store
        self.regime = regime
        self.partitions = int(partitions)
        self.embedding_dim = int(embedding_dim)
        self.threads = int(threads)
        self.membership = None if membership is None else np.asarray(membership, dtype=np.int64)
        self.startup_allowance = float(startup_allowance)
        self.max_paths = max_paths
        self.offline_time_limit = float(offline_time_limit)
        self.workdir = workdir
        self.root = Path(cache_dir) / f"{store.dataset}.{regime}.gnnpe-p{self.partitions}-e{self.embedding_dim}"
        self.info: dict[str, Any] | None = None

    @property
    def binary_path(self) -> str | None:
        return classical.resolve_binary(self.binary)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(self.threads)
        return env

    def _binary_digest(self) -> str | None:
        from .manifest import sha256_file

        path = self.binary_path
        if not path:
            return None
        entry = sha256_file(path, fast=True)
        return str(entry.get("sha256") or entry.get("fingerprint"))

    def _signature(self, data_info: dict[str, Any]) -> dict[str, Any]:
        keys = ("dataset", "regime", "store_meta_sha256", "sigma_json_sha256", "fingerprints", "num_nodes",
                "num_edges", "num_codes", "writer_version")
        source = METIS_RECURSIVE if self.membership is None else _membership_sha256(self.membership)
        return {"schema": PREPARED_SCHEMA, "data": {k: data_info.get(k) for k in keys},
                "partitions": self.partitions, "path_length": PATH_LENGTH, "embedding_dim": self.embedding_dim,
                "membership_source": source, "binary_sha256": self._binary_digest()}

    def prepare(self, warmup: bool = True) -> dict[str, Any]:
        """Offline stage plus warm-up, or the recorded result when nothing it depends on changed.

        Raises GNNPEError when the offline stage or the warm-up fails, when a partition would have a
        leaf root, or when the data graph has more than max_paths 3-vertex paths.
        """
        if self.info is not None:
            return self.info
        data_path, data_info = classical.ensure_data_graph(self.store, self.regime, self.root)
        expected = self._signature(data_info)
        sidecar = self.root / "prepared.json"
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = None
        if (recorded is not None and all(recorded.get(k) == v for k, v in expected.items())
                and (recorded.get("warmup_seconds") is not None or not warmup)
                and (self.root / "gnn-pe" / "all_paths.txt").is_file()):
            self.info = {**recorded, "reused": True}
            return self.info
        sidecar.unlink(missing_ok=True)
        estimate = estimate_path_count(self.store.indptr)
        if self.max_paths is not None and estimate > self.max_paths:
            raise GNNPEError(f"the offline stage would store {estimate} 3-vertex paths (max_paths={self.max_paths})")
        membership = self.membership
        if membership is None:
            membership = metis_membership(self.store.indptr, self.store.indices, self.partitions)
        exported = export_data_graph(self.store, self.regime, self.root, partitions=self.partitions,
                                     membership=membership)
        gdir = self.root / "gnn-pe"
        log = gdir / "offline.log"
        cmd = offline_command(self.command, self.root, data_path, partitions=self.partitions,
                              embedding_dim=self.embedding_dim)
        try:
            returncode, killed, seconds = classical.run_capped(cmd, self.offline_time_limit, log, env=self._env(),
                                                               cwd=self.root)
        except OSError as exc:
            raise GNNPEError(f"cannot start {self.command[0]}: {exc}") from exc
        if killed or returncode != 0 or not (gdir / "all_paths.txt").is_file():
            raise GNNPEError(f"offline stage failed (returncode {returncode}, killed {killed}): "
                             f"{_read(log)[-2000:]}")
        counts = read_partition_path_counts(self.root, self.partitions)
        missing = [i for i, c in enumerate(counts) if c < 0]
        if missing:
            raise GNNPEError(f"the offline stage wrote no partition_paths.txt for partitions {missing}")
        bad = leaf_root_partitions(counts, self.embedding_dim)
        if bad:
            raise GNNPEError(f"partitions {bad} hold 1 to {leaf_root_max_paths(self.embedding_dim)} paths, so their "
                             f"R-tree root is a single leaf that query() walks out of bounds "
                             f"(SEMANTIC_DIFFERENCES['leaf_root']); use fewer partitions or another membership")
        with open(gdir / "all_paths.txt", encoding="ascii") as handle:
            all_paths = int(handle.readline().strip())
            first_path = handle.readline().split()
        info: dict[str, Any] = {**expected, "membership_sha256": exported["membership_sha256"],
                                "part_sizes": exported["part_sizes"], "path_count_estimate": estimate,
                                "all_paths": all_paths, "partition_path_counts": counts, "offline_seconds": seconds,
                                "data_graph": str(data_path), "warmup_seconds": None, "prepared_utc": _utc_now()}
        if warmup and len(first_path) == PATH_LENGTH + 1:
            info.update(self._warmup(data_path, [int(v) for v in first_path]))
        elif warmup:
            info.update({"warmup_seconds": 0.0, "warmup_note": "no 3-vertex path in the data graph"})
        info["index_files"] = sum((d / "index.dat").is_file() for d in partition_dirs(self.root, self.partitions))
        tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, sidecar)
        self.info = {**info, "reused": False}
        return self.info

    def _warmup(self, data_path: Path, nodes: list[int]) -> dict[str, Any]:
        """One online query (the first stored data path) so each partition writes index.dat before timed runs."""
        codes = np.asarray(self.store.sigma(self.regime).codes[np.asarray(nodes)], dtype=np.int64)
        query = local_graph_from_edges(3, np.asarray([[0, 1], [1, 2]]), labels=codes)
        query_path = self.root / "gnn-pe" / "warmup.graph"
        classical.write_subgraphmatching_graph(query, query_path)
        log = self.root / "gnn-pe" / "warmup.log"
        cmd = online_command(self.command, self.root, data_path, query_path, partitions=self.partitions,
                             embedding_dim=self.embedding_dim, max_solutions=1, print_embeddings=False)
        try:
            returncode, killed, seconds = classical.run_capped(cmd, self.offline_time_limit, log, env=self._env(),
                                                               cwd=self.root)
        except OSError as exc:
            raise GNNPEError(f"cannot start {self.command[0]}: {exc}") from exc
        text = _read(log)
        status, _, parsed = parse_gnnpe_output(text, 3)
        if killed or returncode != 0 or status != "complete":
            raise GNNPEError(f"warm-up query failed (returncode {returncode}, killed {killed}): {text[-2000:]}")
        return {"warmup_seconds": seconds, "warmup_answer_number": parsed.get("answer_number"),
                "warmup_query_time_ms": parsed.get("query_time_ms")}

    def solve(self, query_codes: np.ndarray, query_edges: np.ndarray, *, time_limit: float,
              max_solutions: int = 1) -> SolveOutcome:
        """One online run with -n max_solutions; prepare() runs first (uncharged) and may raise GNNPEError."""
        codes = np.asarray(query_codes, dtype=np.int64).reshape(-1)
        query = local_graph_from_edges(codes.shape[0], np.asarray(query_edges, dtype=np.int64).reshape(-1, 2),
                                       labels=codes)
        t0 = time.perf_counter()
        reason = classical.whole_graph_none(self.store, self.regime, query)
        if reason:
            return SolveOutcome(NONE, [], time.perf_counter() - t0, self.name, {"trivial": reason})
        problem = unsupported_query(query)
        if problem:
            return SolveOutcome(ERROR, [], time.perf_counter() - t0, self.name, {"error": problem})
        if time_limit <= 0:
            return SolveOutcome(TIMEOUT, [], 0.0, self.name, {"reason": "no time left"})
        self.prepare()
        data_path = Path(self.info["data_graph"])
        t0 = time.perf_counter()
        tmp = tempfile.mkdtemp(prefix="gnnpe-", dir=self.workdir)
        try:
            query_path = os.path.join(tmp, "query.graph")
            classical.write_subgraphmatching_graph(query, query_path)
            io_seconds = time.perf_counter() - t0
            remaining = time_limit - io_seconds
            detail: dict[str, Any] = {"io_seconds": io_seconds, "global_ids": True, "partitions": self.partitions,
                                      "threads": self.threads, "startup_allowance": self.startup_allowance}
            if remaining <= 0:
                return SolveOutcome(TIMEOUT, [], time.perf_counter() - t0, self.name,
                                    {**detail, "reason": "input writing used the whole cap"})
            cmd = online_command(self.command, self.root, data_path, query_path, partitions=self.partitions,
                                 embedding_dim=self.embedding_dim, max_solutions=max_solutions)
            out_path = os.path.join(tmp, "stdout.txt")
            try:
                returncode, killed, _ = classical.run_capped(cmd, remaining + self.startup_allowance, out_path,
                                                             env=self._env(), cwd=self.root)
            except OSError as exc:
                return SolveOutcome(ERROR, [], time.perf_counter() - t0, self.name,
                                    {**detail, "error": f"cannot start {self.command[0]}: {exc}"})
            text = Path(out_path).read_text(encoding="utf-8", errors="replace")
            table = self.store.sigma(self.regime)
            q_edges = query.edge_list()

            def check(phi: np.ndarray) -> str | None:
                return classical.embedding_problem(phi, codes, q_edges, table.codes, self.store.indptr,
                                                   self.store.indices)

            return classify_output(self.name, text, returncode, killed, query.n, max_solutions,
                                   time.perf_counter() - t0, detail, check)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
