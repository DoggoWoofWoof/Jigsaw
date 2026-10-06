"""SubgraphMatching (Sun and Luo, SIGMOD 2020) as an exact solver: GQL, CFL and DP-iso.

Two uses, both returning backends.SolveOutcome with the GlasgowBackend semantics:
  SubgraphMatchingBackend    one (query, target) pair of LocalGraphs; the verifier alternatives
                             "gql", "cfl" and "dpiso". FOUND carries local target ids.
  FullGraphClassicalRunner   the whole data graph, written once per (store, regime) and reused;
                             the classical_* baselines and stage 2 of negative certification.
                             FOUND carries global node ids.
FOUND needs at least one printed embedding; NONE only when the search finished without one;
TIMEOUT when the process was killed at the cap or the solver reports its own limit; ERROR
otherwise, with the output tail.

The binary is external/bin/subgraphmatching/SubgraphMatching.out, built by
external/build_subgraphmatching.sh at the commit and with the patches pinned in
external/PINS.json. 0001-print-embeddings.patch adds the -print_embeddings flag this module
relies on; external/README.md explains every patch and the input constraints handled here.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from .backends import ERROR, FOUND, NONE, TIMEOUT, SolveOutcome, trivially_none
from .store import LocalGraph, local_graph_from_edges

# 0002-max-query-graph-size-128.patch raises MAXIMUM_QUERY_GRAPH_SIZE (config.h line 12) from 64.
MAX_QUERY_VERTICES = 128
WRITER_VERSION = 1
DATA_SCHEMA = "jigsaw-subgraphmatching-data-v1"

# name -> (-filter, -order, -engine). The first three follow the upstream README recommendation
# (lines 176-178: GQL, CFL or DPiso filtering, GQL or RI ordering, LFTJ enumeration, failing sets
# on) and its GQL baseline command (line 193). The rest are sensitivity presets: the RI order
# (line 187) and each algorithm's own ordering and enumeration (lines 127-138).
PRESETS: dict[str, tuple[str, str, str]] = {
    "gql": ("GQL", "GQL", "LFTJ"),
    "cfl": ("CFL", "GQL", "LFTJ"),
    "dpiso": ("DPiso", "GQL", "LFTJ"),
    "ri": ("GQL", "RI", "LFTJ"),
    "gql_orig": ("GQL", "GQL", "GQL"),
    "cfl_orig": ("CFL", "CFL", "EXPLORE"),
    "dpiso_orig": ("DPiso", "DPiso", "DPiso"),
}
DEFAULT_ALGORITHMS = ("gql", "cfl", "dpiso")

_EDGE_BUDGET = 1 << 22  # adjacency entries per edge chunk of the data writers
_UI_MAX = 2**32  # SubgraphMatching stores ids, degrees and CSR offsets as 32-bit ui
_POW10 = np.array([10**k for k in range(1, 20)], dtype=np.uint64)


# ----------------------------------------------------------------------------
# Integer text without per-row Python objects
# ----------------------------------------------------------------------------
def _digit_counts(values: np.ndarray) -> np.ndarray:
    return np.searchsorted(_POW10, values.astype(np.uint64), side="right").astype(np.int64) + 1


def format_int_rows(prefix: bytes, columns: Sequence[np.ndarray]) -> bytes:
    """b"<prefix> c0 c1 ...\\n" for every row (no prefix: b"c0 c1 ...\\n"), built in numpy."""
    cols = [np.asarray(c, dtype=np.int64).reshape(-1) for c in columns]
    rows = int(cols[0].shape[0]) if cols else 0
    if rows == 0:
        return b""
    for c in cols:
        if c.shape[0] != rows:
            raise ValueError("columns differ in length")
        if int(c.min()) < 0:
            raise ValueError("format_int_rows writes non-negative integers only")
    widths = [_digit_counts(c) for c in cols]
    # One space before every column (none before the first without a prefix) and the newline.
    length = np.full(rows, len(prefix) + len(cols) + (1 if prefix else 0), dtype=np.int64)
    for w in widths:
        length += w
    end = np.cumsum(length)
    pos = end - length
    buf = np.empty(int(end[-1]), dtype=np.uint8)
    for j, byte in enumerate(prefix):
        buf[pos + j] = byte
    pos = pos + len(prefix)
    for i, (c, w) in enumerate(zip(cols, widths)):
        if prefix or i:
            buf[pos] = 32
            pos = pos + 1
        value = c.copy()
        for k in range(int(w.max())):
            live = w > k
            buf[pos[live] + w[live] - 1 - k] = (value[live] % 10 + 48).astype(np.uint8)
            value //= 10
        pos = pos + w
    buf[pos] = 10
    return buf.tobytes()


def _row_ranges(indptr: np.ndarray, max_rows: int, max_entries: int) -> Iterator[tuple[int, int]]:
    """Consecutive row ranges with at most max_rows rows and (unless one row is larger) max_entries entries."""
    n = int(indptr.shape[0]) - 1
    start = 0
    while start < n:
        stop = min(n, start + max_rows)
        base = int(indptr[start])
        if int(indptr[stop]) - base > max_entries:
            local = np.asarray(indptr[start:stop + 1], dtype=np.int64)
            cut = int(np.searchsorted(local, base + max_entries, side="right")) - 1
            stop = start + max(1, cut)
        yield start, stop
        start = stop


def _check_limits(num_nodes: int, adjacency: int) -> None:
    if num_nodes >= _UI_MAX or adjacency >= _UI_MAX:
        raise ValueError(f"SubgraphMatching uses 32-bit ids and offsets: {num_nodes} nodes, "
                         f"{adjacency} adjacency entries")


def _write_text(handle: Any, indptr: np.ndarray, indices: np.ndarray, labels: Any, chunk_nodes: int) -> int:
    """SubgraphMatching text: `t N M`, `v id label degree` in id order, then `e u v` once per edge."""
    n = max(0, int(indptr.shape[0]) - 1)
    adjacency = int(indptr[-1]) if indptr.shape[0] else 0
    _check_limits(n, adjacency)
    handle.write(f"t {n} {adjacency // 2}\n".encode("ascii"))
    chunk_nodes = max(1, int(chunk_nodes))
    for start in range(0, n, chunk_nodes):
        stop = min(n, start + chunk_nodes)
        ptr = np.asarray(indptr[start:stop + 1], dtype=np.int64)
        lab = np.asarray(labels[start:stop], dtype=np.int64)
        handle.write(format_int_rows(b"v", [np.arange(start, stop, dtype=np.int64), lab, np.diff(ptr)]))
    written = 0
    for start, stop in _row_ranges(indptr, chunk_nodes, _EDGE_BUDGET):
        ptr = np.asarray(indptr[start:stop + 1], dtype=np.int64)
        cols = np.asarray(indices[int(ptr[0]):int(ptr[-1])], dtype=np.int64)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), np.diff(ptr))
        keep = rows < cols
        handle.write(format_int_rows(b"e", [rows[keep], cols[keep]]))
        written += int(keep.sum())
    if 2 * written != adjacency:
        raise ValueError("the CSR is not symmetric or has self loops; SubgraphMatching needs each edge once")
    return written


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def dense_label_map(*graphs: LocalGraph) -> np.ndarray:
    """Sorted distinct labels of the graphs; label x is written as its position in this array."""
    parts = [np.asarray(g.labels, dtype=np.int64) for g in graphs if g.labels is not None]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def _mapped_labels(graph: LocalGraph, label_map: np.ndarray | None) -> np.ndarray:
    if graph.labels is None:
        raise ValueError("the graph has no labels")
    labels = np.asarray(graph.labels, dtype=np.int64)
    if label_map is None:
        if labels.shape[0] and int(labels.min()) < 0:
            raise ValueError("negative label; pass a label_map")
        return labels
    label_map = np.asarray(label_map, dtype=np.int64)
    pos = np.searchsorted(label_map, labels)
    ok = pos < label_map.shape[0]
    ok[ok] = label_map[pos[ok]] == labels[ok]
    if not ok.all():
        raise ValueError(f"label {int(labels[~ok][0])} is not in the label map")
    return pos.astype(np.int64)


def write_subgraphmatching_graph(graph: LocalGraph, path: str | os.PathLike, label_map: np.ndarray | None = None,
                                 chunk: int = 1 << 18) -> None:
    """A LocalGraph in SubgraphMatching's text format; labels are remapped through label_map when given.

    Vertex lines come first and in id order because the loader sizes each adjacency list from
    the `v` line's degree (graph/graph.cpp line 115); edge lines carry no label.
    """
    labels = _mapped_labels(graph, label_map)
    with open(path, "wb") as handle:
        _write_text(handle, np.asarray(graph.indptr, dtype=np.int64), graph.indices, labels, chunk)


def sidecar_path(path: str | os.PathLike) -> Path:
    return Path(str(path) + ".json")


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_graph_signature(store: Any, regime: str, data_format: str = "text") -> dict[str, Any]:
    """What a cached data graph file depends on: store meta, sigma table, array fingerprints, counts."""
    from .manifest import sha256_file

    def digest(path: Path) -> str:
        entry = sha256_file(path, fast=True)
        return str(entry.get("sha256") or entry.get("fingerprint"))

    table = store.sigma(regime)
    sigma_dir = store.dir / "sigma" / regime
    return {
        "schema": DATA_SCHEMA,
        "format": data_format,
        "writer_version": WRITER_VERSION,
        "dataset": store.dataset,
        "regime": regime,
        "store_meta_sha256": _sha256_bytes(store.dir / "meta.json"),
        "sigma_json_sha256": _sha256_bytes(sigma_dir / "sigma.json"),
        "fingerprints": {
            "graph/indptr.npy": digest(store.dir / "graph" / "indptr.npy"),
            "graph/indices.npy": digest(store.dir / "graph" / "indices.npy"),
            f"sigma/{regime}/codes.npy": digest(sigma_dir / "codes.npy"),
        },
        "num_nodes": int(store.num_nodes),
        "num_edges": int(store.indices.shape[0]) // 2,
        "num_codes": int(table.num_codes),
    }


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _finish(path: Path, store: Any, regime: str, data_format: str, files: list[Path], t0: float) -> dict[str, Any]:
    info = {
        **data_graph_signature(store, regime, data_format),
        "files": {p.name: p.stat().st_size for p in files},
        "written_utc": _utc_now(),
        "write_seconds": round(time.perf_counter() - t0, 3),
    }
    _write_json(sidecar_path(path), info)
    return info


def write_data_graph(store: Any, regime: str, path: str | os.PathLike, chunk_nodes: int = 1 << 20) -> dict[str, Any]:
    """The whole data graph in SubgraphMatching's text format, labels = dense sigma codes of `regime`.

    Streams `chunk_nodes` vertices at a time (edges in chunks of at most 4M adjacency entries),
    so papers100M (111M nodes, 1.6B edges) never becomes per-edge Python objects. Writes a
    temporary file, renames it into place, and records the store and sigma fingerprints in the
    sidecar `<path>.json`. Returns the sidecar.
    """
    t0 = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            _write_text(handle, store.indptr, store.indices, store.sigma(regime).codes, chunk_nodes)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return _finish(path, store, regime, "text", [path], t0)


def csr_files(prefix: str | os.PathLike) -> tuple[Path, Path, Path]:
    """The three files of `-csr PREFIX` (StudyPerformance.cpp lines 124-126)."""
    prefix = str(prefix)
    return Path(prefix + "_deg.bin"), Path(prefix + "_adj.bin"), Path(prefix + "_label.bin")


def write_data_graph_csr(store: Any, regime: str, prefix: str | os.PathLike, chunk_nodes: int = 1 << 20) -> dict[str, Any]:
    """The whole data graph in the binary format of `-csr PREFIX` (graph/graph.cpp lines 184-300).

    PREFIX_deg.bin: int32 4, uint32 N, uint32 M (undirected edges), uint32 degree[N];
    PREFIX_adj.bin: uint32 neighbour lists in id order; PREFIX_label.bin: uint32 label[N].
    Loading it skips the text parse, which dominates a run on a large graph.
    """
    t0 = time.perf_counter()
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    indptr, indices = store.indptr, store.indices
    codes = store.sigma(regime).codes
    n = int(store.num_nodes)
    adjacency = int(indptr[-1])
    _check_limits(n, adjacency)
    chunk_nodes = max(1, int(chunk_nodes))
    paths = csr_files(prefix)
    tmps = [p.with_name(f"{p.name}.{os.getpid()}.tmp") for p in paths]
    try:
        with open(tmps[0], "wb") as handle:
            handle.write(np.asarray([4, n, adjacency // 2], dtype="<u4").tobytes())
            for start in range(0, n, chunk_nodes):
                ptr = np.asarray(indptr[start:min(n, start + chunk_nodes) + 1], dtype=np.int64)
                handle.write(np.diff(ptr).astype("<u4").tobytes())
        with open(tmps[1], "wb") as handle:
            step = max(chunk_nodes, _EDGE_BUDGET)
            for lo in range(0, adjacency, step):
                handle.write(np.asarray(indices[lo:min(adjacency, lo + step)]).astype("<u4").tobytes())
        with open(tmps[2], "wb") as handle:
            for start in range(0, n, chunk_nodes):
                handle.write(np.asarray(codes[start:min(n, start + chunk_nodes)]).astype("<u4").tobytes())
        for tmp, path in zip(tmps, paths):
            os.replace(tmp, path)
    finally:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
    return _finish(prefix, store, regime, "csr", list(paths), t0)


def data_graph_path(cache_dir: str | os.PathLike, store: Any, regime: str, data_format: str = "text") -> Path:
    """The text file, or the `-csr` prefix, of one (store, regime) data graph in cache_dir."""
    suffix = ".graph" if data_format == "text" else ".csr"
    return Path(cache_dir) / f"{store.dataset}.{regime}{suffix}"


def ensure_data_graph(store: Any, regime: str, cache_dir: str | os.PathLike, data_format: str = "text",
                      chunk_nodes: int = 1 << 20) -> tuple[Path, dict[str, Any]]:
    """Write the data graph once per (store, regime, format); reuse it while its sidecar matches."""
    if data_format not in ("text", "csr"):
        raise ValueError(f"unknown data format {data_format!r}")
    path = data_graph_path(cache_dir, store, regime, data_format)
    expected = data_graph_signature(store, regime, data_format)
    try:
        info = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = None
    if info is not None and all(info.get(k) == v for k, v in expected.items()):
        files = info.get("files") or {}
        if files and all((path.parent / name).is_file() and (path.parent / name).stat().st_size == size
                         for name, size in files.items()):
            return path, {**info, "reused": True}
    writer = write_data_graph if data_format == "text" else write_data_graph_csr
    return path, {**writer(store, regime, path, chunk_nodes), "reused": False}


# ----------------------------------------------------------------------------
# Output parser
# ----------------------------------------------------------------------------
_SUMMARY = {  # StudyPerformance.cpp lines 403-413
    "Load graphs time (seconds)": "load_seconds",
    "Filter vertices time (seconds)": "filter_seconds",
    "Build table time (seconds)": "build_table_seconds",
    "Generate query plan time (seconds)": "plan_seconds",
    "Enumerate time (seconds)": "enumerate_seconds",
    "Preprocessing time (seconds)": "preprocessing_seconds",
    "Total time (seconds)": "total_seconds",
    "Memory cost (MB)": "memory_mb",
    "#Embeddings": "embedding_count",
    "Call Count": "call_count",
    "Per Call Count Time (nanoseconds)": "per_call_ns",
}
_COUNTS = {"embedding_count", "call_count"}
_ECHO = {  # StudyPerformance.cpp lines 94-104 and 0001-print-embeddings.patch
    "Filter Type": "filter",
    "Order Type": "order",
    "Engine Type": "engine",
    "Output Limit": "output_limit",
    "Print Embeddings": "print_embeddings",
}


def parse_embedding_pairs(body: str, query_n: int) -> np.ndarray | None:
    """phi from "0:v0 1:v1 ..." (query vertex : data vertex), or None unless each query vertex occurs once."""
    parts = body.split()
    if len(parts) != query_n:
        return None
    phi = np.full(query_n, -1, dtype=np.int64)
    for part in parts:
        u, sep, v = part.partition(":")
        if not sep or not u.isdigit() or not v.isdigit():
            return None
        q = int(u)
        if q >= query_n or phi[q] >= 0:
            return None
        phi[q] = int(v)
    return phi


def parse_graph_meta(line: str) -> dict[str, int] | None:
    """{vertices, edges, labels} from printGraphMetaData's "|V|: N, |E|: M, |<sigma>|: L" (graph.cpp line 169)."""
    fields = [part.split(":", 1) for part in line.split(",")]
    try:
        values = [int(value) for _, value in fields]
    except ValueError:
        return None
    return dict(zip(("vertices", "edges", "labels"), values)) if len(values) == 3 else None


def parse_subgraphmatching_output(text: str, query_n: int) -> tuple[str | None, list[np.ndarray], dict[str, Any]]:
    """(status, embeddings, detail) from the patched binary's output.

    status is "complete" when both #Embeddings and End. were printed, "limit" when the solver
    reports its own time limit (spectrum mode), and None otherwise. Embeddings come from
    `embedding: u:v ...` lines (query vertex id : data vertex id); a line without exactly one
    distinct pair per query vertex counts in detail["malformed_lines"], and the unterminated
    last line of a killed run is skipped (detail["unterminated_tail"]).
    """
    detail: dict[str, Any] = {"malformed_lines": 0}
    embeddings: list[np.ndarray] = []
    lines = text.split("\n")
    if lines and lines[-1].strip():
        detail["unterminated_tail"] = True
    ended = limit = False
    metas: list[dict[str, int]] = []
    for raw in lines[:-1]:
        line = raw.strip()
        if line.startswith("embedding:"):
            phi = parse_embedding_pairs(line[len("embedding:"):], query_n)
            if phi is None:
                detail["malformed_lines"] += 1
            else:
                embeddings.append(phi)
        elif line == "End.":
            ended = True
        elif line.startswith("|V|:"):
            meta = parse_graph_meta(line)
            if meta is not None:
                metas.append(meta)
        elif line.startswith("Spectrum Order") and "Timeout" in line:
            limit = True
        elif "is not supported" in line:
            detail["unsupported"] = line
        elif ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if key in _SUMMARY:
                try:
                    detail[_SUMMARY[key]] = int(value) if _SUMMARY[key] in _COUNTS else float(value)
                except ValueError:
                    pass
            elif key in _ECHO:
                detail[_ECHO[key]] = (value == "true") if key == "Print Embeddings" else value
    if metas:
        detail["query_graph"] = metas[0]
    if len(metas) > 1:
        detail["data_graph"] = metas[1]
    if limit:  # "Spectrum Order N status: Timeout" (StudyPerformance.cpp line 36); never proves absence
        status = "limit"
    elif ended and "embedding_count" in detail:
        status = "complete"
    else:
        status = None
    return status, embeddings, detail


# ----------------------------------------------------------------------------
# Process handling and checks shared with gnnpe_adapter
# ----------------------------------------------------------------------------
def resolve_binary(binary: str | os.PathLike) -> str | None:
    """The binary as an existing path (looked up on PATH), for manifest.solver_info; None if absent."""
    found = shutil.which(str(binary))
    if found:
        return found
    return str(binary) if Path(binary).exists() else None


def run_capped(cmd: Sequence[str], time_limit: float, out_path: str | os.PathLike, *,
               env: dict[str, str] | None = None, cwd: str | os.PathLike | None = None) -> tuple[int, bool, float]:
    """Run cmd with stdout and stderr in out_path; SIGKILL its process group after time_limit seconds.

    Returns (returncode, killed_at_cap, seconds). Raises OSError when cmd cannot start.
    """
    t0 = time.perf_counter()
    with open(out_path, "wb") as out:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                start_new_session=(os.name == "posix"))
        killed = threading.Event()

        def kill() -> None:
            killed.set()
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass

        timer = threading.Timer(max(0.0, float(time_limit)), kill)
        timer.daemon = True
        timer.start()
        try:
            proc.wait()
        except BaseException:
            kill()
            proc.wait()
            raise
        finally:
            timer.cancel()
    return int(proc.returncode), killed.is_set(), time.perf_counter() - t0


def unsupported_query(query: LocalGraph, max_vertices: int = MAX_QUERY_VERTICES) -> str | None:
    """Why the pinned SubgraphMatching build cannot run this query, or None."""
    if query.n == 0:
        return "empty query"
    if query.n == 1:
        return ("one-vertex query: LFTJ with failing sets reads vec_failing_set[-1] "
                "(EvaluateQuery.cpp lines 347-351)")
    if query.n > max_vertices:
        return f"query has {query.n} vertices; the build supports at most {max_vertices} (MAXIMUM_QUERY_GRAPH_SIZE)"
    if query.components()[0] > 1:
        return "disconnected query: checkQueryPlanCorrectness asserts a connected query"
    return None


def embedding_problem(phi: np.ndarray, query_labels: np.ndarray, query_edges: np.ndarray, labels: Any,
                      indptr: Any, indices: Any) -> str | None:
    """Why phi (query vertex -> data vertex) is not an embedding of the query, or None."""
    phi = np.asarray(phi, dtype=np.int64)
    n = int(np.asarray(query_labels).shape[0])
    num_nodes = int(indptr.shape[0]) - 1
    if phi.shape[0] != n or (phi < 0).any() or (phi >= num_nodes).any():
        return "maps outside the data graph"
    if np.unique(phi).shape[0] != n:
        return "not injective"
    if not np.array_equal(np.asarray(labels[phi], dtype=np.int64), np.asarray(query_labels, dtype=np.int64)):
        return "a label differs"
    for a, b in np.asarray(query_edges, dtype=np.int64).reshape(-1, 2):
        u, v = int(phi[a]), int(phi[b])
        row = np.asarray(indices[int(indptr[u]):int(indptr[u + 1])])
        pos = int(np.searchsorted(row, v))
        if pos >= row.shape[0] or int(row[pos]) != v:
            return f"query edge ({int(a)},{int(b)}) is not a data edge"
    return None


def _classify(name: str, text: str, returncode: int, killed: bool, query_n: int, max_solutions: int,
              elapsed: float, detail: dict[str, Any], check: Callable[[np.ndarray], str | None]) -> SolveOutcome:
    status, embeddings, parsed = parse_subgraphmatching_output(text, query_n)
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
    if killed or status == "limit":
        return SolveOutcome(TIMEOUT, [], elapsed, name, detail)
    if status == "complete" and returncode == 0:
        if int(parsed.get("embedding_count", 0)) == 0:
            return SolveOutcome(NONE, [], elapsed, name, detail)
        detail["error"] = "embeddings counted but not printed (binary without 0001-print-embeddings.patch?)"
    detail["output_tail"] = text[-2000:]
    return SolveOutcome(ERROR, [], elapsed, name, detail)


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
class SubgraphMatchingBackend:
    """One SubgraphMatching preset on a (query, target) pair; the GlasgowBackend interface."""

    def __init__(self, algorithm: str, binary: str | os.PathLike | Sequence[str] | None = None,
                 workdir: str | None = None, extra_args: Sequence[str] = ()) -> None:
        """`binary` is the solver path, or a command prefix such as [python, wrapper.py]."""
        if algorithm not in PRESETS:
            raise ValueError(f"unknown SubgraphMatching preset {algorithm!r}; choose from {sorted(PRESETS)}")
        binary = binary or os.environ.get("SUBGRAPHMATCHING_BIN", "SubgraphMatching.out")
        self.command = [str(binary)] if isinstance(binary, (str, os.PathLike)) else [str(b) for b in binary]
        self.binary = self.command[-1]
        self.algorithm = algorithm
        self.name = algorithm
        self.filter, self.order, self.engine = PRESETS[algorithm]
        self.workdir = workdir
        self.extra_args = tuple(str(a) for a in extra_args)

    def available(self) -> bool:
        head = self.command[0]
        return bool(shutil.which(head) or Path(head).exists())

    @property
    def binary_path(self) -> str | None:
        return resolve_binary(self.binary)

    def arguments(self, max_solutions: int) -> list[str]:
        """Everything after the graph arguments: the preset, -num and -print_embeddings."""
        return ["-filter", self.filter, "-order", self.order, "-engine", self.engine,
                "-num", str(int(max_solutions)), "-print_embeddings", *self.extra_args]

    def run(self, graph_args: Sequence[str], query_path: str, *, query_labels: np.ndarray, query_edges: np.ndarray,
            time_limit: float, max_solutions: int, t0: float, detail: dict[str, Any],
            check: Callable[[np.ndarray], str | None], name: str | None = None) -> SolveOutcome:
        """Run one query file against graph_args (-d FILE or -csr PREFIX) and classify the output."""
        name = name or self.name
        cmd = self.command + list(graph_args) + ["-q", query_path] + self.arguments(max_solutions)
        out_path = os.path.join(os.path.dirname(query_path), "stdout.txt")
        try:
            returncode, killed, _ = run_capped(cmd, time_limit, out_path)
        except OSError as exc:
            return SolveOutcome(ERROR, [], time.perf_counter() - t0, name,
                                {**detail, "error": f"cannot start {self.command[0]}: {exc}"})
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
        return _classify(name, text, returncode, killed, int(np.asarray(query_labels).shape[0]), max_solutions,
                         time.perf_counter() - t0, detail, check)

    def solve(self, query: LocalGraph, target: LocalGraph, *, time_limit: float,
              max_solutions: int = 1) -> SolveOutcome:
        t0 = time.perf_counter()
        reason = trivially_none(query, target)
        if reason:
            return SolveOutcome(NONE, [], time.perf_counter() - t0, self.name, {"trivial": reason})
        problem = unsupported_query(query)
        if problem:
            return SolveOutcome(ERROR, [], time.perf_counter() - t0, self.name, {"error": problem})
        if time_limit <= 0:
            return SolveOutcome(TIMEOUT, [], 0.0, self.name, {"reason": "no time left"})
        tmp = tempfile.mkdtemp(prefix="subgraphmatching-", dir=self.workdir)
        try:
            # One shared dense map: the loader indexes per-label arrays by label id (graph.cpp line 146).
            label_map = dense_label_map(query, target)
            query_path = os.path.join(tmp, "query.graph")
            data_path = os.path.join(tmp, "data.graph")
            write_subgraphmatching_graph(query, query_path, label_map)
            write_subgraphmatching_graph(target, data_path, label_map)
            io_seconds = time.perf_counter() - t0
            remaining = time_limit - io_seconds
            detail = {"io_seconds": io_seconds, "target_nodes": target.n,
                      "target_edges": target.num_undirected_edges, "preset": list(PRESETS[self.algorithm])}
            if remaining <= 0:
                return SolveOutcome(TIMEOUT, [], time.perf_counter() - t0, self.name,
                                    {**detail, "reason": "input writing used the whole cap"})
            q_labels = np.asarray(query.labels, dtype=np.int64)
            q_edges = query.edge_list()

            def check(phi: np.ndarray) -> str | None:
                return embedding_problem(phi, q_labels, q_edges, np.asarray(target.labels), target.indptr,
                                         target.indices)

            return self.run(["-d", data_path], query_path, query_labels=q_labels, query_edges=q_edges,
                            time_limit=remaining, max_solutions=max_solutions, t0=t0, detail=detail, check=check)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def whole_graph_none(store: Any, regime: str, query: LocalGraph) -> str | None:
    """Sound reasons why the whole data graph holds no embedding: size, fresh codes, label counts."""
    table = store.sigma(regime)
    codes = np.asarray(query.labels, dtype=np.int64)
    if query.n > store.num_nodes:
        return "query larger than the data graph"
    if query.num_undirected_edges > int(store.indices.shape[0]) // 2:
        return "query has more edges than the data graph"
    if ((codes < 0) | (codes >= table.num_codes)).any():
        return "a query label does not occur in the data graph"
    uniq, counts = np.unique(codes, return_counts=True)
    if (table.df[uniq] < counts).any():
        return "the data graph holds a query label fewer times than the query needs"
    return None


class FullGraphClassicalRunner:
    """A SubgraphMatching preset on the whole data graph: classical_* baselines, certification stage 2.

    The data graph is written once per (store, regime, format) into cache_dir with dense sigma
    codes as labels and reused while its sidecar matches (prepare(); its time is not charged to
    any query). Each solve writes the query with the same codes, so no remap is needed, and
    returns embeddings in global node ids.
    """

    def __init__(self, store: Any, regime: str, algorithm: str, *, cache_dir: str | os.PathLike,
                 binary: str | os.PathLike | Sequence[str] | None = None, data_format: str = "text",
                 workdir: str | None = None, extra_args: Sequence[str] = ()) -> None:
        if data_format not in ("text", "csr"):
            raise ValueError(f"unknown data format {data_format!r}")
        self.store = store
        self.regime = regime
        self.algorithm = algorithm
        self.name = f"classical_{algorithm}"
        self.cache_dir = Path(cache_dir)
        self.data_format = data_format
        self.workdir = workdir
        self.backend = SubgraphMatchingBackend(algorithm, binary=binary, workdir=workdir, extra_args=extra_args)
        self.data_path = data_graph_path(self.cache_dir, store, regime, data_format)
        self.data_info: dict[str, Any] | None = None

    @property
    def binary_path(self) -> str | None:
        return self.backend.binary_path

    def prepare(self) -> dict[str, Any]:
        """Write (or reuse) the data graph file; returns its sidecar plus "reused"."""
        if self.data_info is None:
            self.data_path, self.data_info = ensure_data_graph(self.store, self.regime, self.cache_dir,
                                                               self.data_format)
        return self.data_info

    def solve(self, query_codes: np.ndarray, query_edges: np.ndarray, *, time_limit: float,
              max_solutions: int = 1) -> SolveOutcome:
        codes = np.asarray(query_codes, dtype=np.int64).reshape(-1)
        query = local_graph_from_edges(codes.shape[0], np.asarray(query_edges, dtype=np.int64).reshape(-1, 2),
                                       labels=codes)
        t0 = time.perf_counter()
        reason = whole_graph_none(self.store, self.regime, query)
        if reason:
            return SolveOutcome(NONE, [], time.perf_counter() - t0, self.name, {"trivial": reason})
        problem = unsupported_query(query)
        if problem:
            return SolveOutcome(ERROR, [], time.perf_counter() - t0, self.name, {"error": problem})
        if time_limit <= 0:
            return SolveOutcome(TIMEOUT, [], 0.0, self.name, {"reason": "no time left"})
        self.prepare()
        t0 = time.perf_counter()
        graph_args = ["-d", str(self.data_path)] if self.data_format == "text" else ["-csr", str(self.data_path)]
        tmp = tempfile.mkdtemp(prefix="classical-", dir=self.workdir)
        try:
            query_path = os.path.join(tmp, "query.graph")
            write_subgraphmatching_graph(query, query_path)
            io_seconds = time.perf_counter() - t0
            remaining = time_limit - io_seconds
            detail = {"io_seconds": io_seconds, "data_format": self.data_format, "data_file": str(self.data_path),
                      "preset": list(PRESETS[self.algorithm]), "global_ids": True}
            if remaining <= 0:
                return SolveOutcome(TIMEOUT, [], time.perf_counter() - t0, self.name,
                                    {**detail, "reason": "input writing used the whole cap"})
            q_labels = codes
            q_edges = query.edge_list()
            table = self.store.sigma(self.regime)

            def check(phi: np.ndarray) -> str | None:
                return embedding_problem(phi, q_labels, q_edges, table.codes, self.store.indptr, self.store.indices)

            return self.backend.run(graph_args, query_path, query_labels=q_labels, query_edges=q_edges,
                                    time_limit=remaining, max_solutions=max_solutions, t0=t0, detail=detail,
                                    check=check, name=self.name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
