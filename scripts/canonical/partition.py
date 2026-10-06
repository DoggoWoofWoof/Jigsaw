"""Graph partitioners for the canonical store (protocol.PARTITIONING).

Every partitioner takes the normalised graph as a symmetric CSR (sorted rows, no
self loops or duplicate entries; see store.csr_from_edges) and returns block ids
that are consecutive and non-empty, plus an info dict for the run manifest.

  metis_partition         METIS k-way, edge-cut objective, ufactor 30, seed 1.
                          gpmetis when it is on PATH (or GPMETIS_BIN), otherwise
                          pymetis with the same options: k-way for every nparts
                          (pymetis switches to recursive bisection for
                          nparts <= 8 unless recursive=False) and the seed and
                          ufactor passed through pymetis.Options. METIS may
                          leave parts empty; they are renumbered away, or with
                          fill_empty=True refilled by fill_empty_parts.
  fill_empty_parts        EMPTY_PART_RULE: every empty part takes one side of a
                          2-way METIS split of a largest part.
  fine_partition          METIS k-way inside every block, min(fine_per_block,
                          block size) parts; fine ids consecutive per block.
  kahip_partition         KaHIP kaffpa (KAHIP_BIN or PATH), sensitivity runs only.
  bfs_random_partition    balanced parts grown by BFS from random seeds.
  import_partition        a node-to-block array from the legacy pipeline.
  block_contiguous_order  the canonical node order: (block, fine, original id).

pymetis always runs in a child process, `python -m scripts.canonical.partition
pymetis-worker <job_dir>`: in-process METIS crashed on MAG in the legacy
pipeline (src/data.py run_pymetis_in_subprocess), and a child can be killed at a
timeout without taking the driver down. One job holds one or many disjoint
graphs, so the fine level of many blocks is a single child call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from . import protocol
from .store import gather_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 1
DEFAULT_UFACTOR = 30
FINE_BATCH_ENTRIES = 1 << 25  # adjacency entries per fine-level pymetis job
STATS_CHUNK_ENTRIES = 1 << 24
TEXT_CHUNK_ENTRIES = 1 << 20
EMPTY_PART_RULE = str(protocol.PARTITIONING["empty_parts"])  # in the protocol hash


class PartitionError(RuntimeError):
    """A partitioner failed, crashed or timed out."""


class PartitionerUnavailable(PartitionError):
    """The requested external partitioner is not installed."""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def row_chunks(indptr: np.ndarray, chunk_entries: int) -> Iterator[tuple[int, int]]:
    """Row ranges [a, b) holding about chunk_entries adjacency entries each (at least one row)."""
    n = int(indptr.shape[0]) - 1
    a = 0
    while a < n:
        target = int(indptr[a]) + int(chunk_entries)
        b = int(np.searchsorted(indptr, target, side="right")) - 1
        b = min(max(b, a + 1), n)
        yield a, b
        a = b


def compact_labels(labels: np.ndarray, nparts: int) -> tuple[np.ndarray, int]:
    """Renumber part labels to 0..k-1 in label order; returns (ids, number of empty parts)."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] and (int(labels.min()) < 0 or int(labels.max()) >= nparts):
        raise PartitionError(f"part labels outside [0, {nparts})")
    used, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(-1).astype(np.int32), int(nparts) - int(used.shape[0])


def partition_stats(indptr: np.ndarray, indices: np.ndarray, node_block: np.ndarray,
                    chunk_entries: int = STATS_CHUNK_ENTRIES) -> dict[str, Any]:
    """Part sizes, METIS imbalance (max size * k / n) and edge cut of a partition."""
    node_block = np.asarray(node_block, dtype=np.int64)
    n = int(node_block.shape[0])
    sizes = np.bincount(node_block, minlength=int(node_block.max()) + 1 if n else 0)
    cut_entries = 0
    for a, b in row_chunks(indptr, chunk_entries):
        lo, hi = int(indptr[a]), int(indptr[b])
        lens = np.diff(np.asarray(indptr[a:b + 1], dtype=np.int64))
        own = np.repeat(node_block[a:b], lens)
        other = node_block[np.asarray(indices[lo:hi], dtype=np.int64)]
        cut_entries += int(np.count_nonzero(own != other))
    m = int(indptr[-1]) // 2
    k = int(sizes.shape[0])
    return {
        "num_parts": k,
        "part_size_min": int(sizes.min()) if k else 0,
        "part_size_max": int(sizes.max()) if k else 0,
        "part_size_mean": float(sizes.mean()) if k else 0.0,
        "imbalance": float(sizes.max()) * k / n if n else 0.0,
        "edge_cut": cut_entries // 2,
        "cut_fraction": (cut_entries // 2) / m if m else 0.0,
    }


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _search(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _describe_exit(code: int) -> str:
    if code < 0:
        return f"{code} (signal {-code})"
    if code > 0xFFFF:
        return f"{code} (0x{code & 0xFFFFFFFF:08X})"
    return str(code)


def _tail(data: bytes | None, limit: int = 2000) -> str:
    text = (data or b"").decode("utf-8", "replace").strip()
    return text[-limit:] if text else "(no output)"


def _command(binary: str | Sequence[str] | None, env_name: str, program: str) -> list[str] | None:
    """Command prefix from an explicit binary, an environment variable, or PATH."""
    if binary:
        return [binary] if isinstance(binary, str) else list(binary)
    configured = os.environ.get(env_name, "").strip()
    if configured:
        if not (shutil.which(configured) or Path(configured).exists()):
            raise PartitionerUnavailable(f"{env_name}={configured} does not exist")
        return [configured]
    found = shutil.which(program)
    return [found] if found else None


def _binary_info(cmd: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {"command": list(cmd)}
    path = Path(shutil.which(cmd[-1]) or cmd[-1])
    if path.is_file():
        info["binary_sha256"] = file_sha256(path)
    return info


def self_peak_memory() -> dict[str, int]:
    """Peak memory of this process as the OS reports it (Linux ru_maxrss; Windows peak working set and commit)."""
    out: dict[str, int] = {}
    try:
        import resource

        scale = 1 if sys.platform == "darwin" else 1024
        out["maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
    except ImportError:
        pass
    try:
        import psutil

        info = psutil.Process().memory_info()
        for name in ("peak_wset", "peak_pagefile"):
            if hasattr(info, name):
                out[f"{name}_bytes"] = int(getattr(info, name))
    except Exception:
        pass
    return out


def pymetis_available() -> bool:
    return importlib.util.find_spec("pymetis") is not None


def resolve_metis_tool(tool: str = "auto", binary: str | Sequence[str] | None = None) -> tuple[str, list[str] | None]:
    """('gpmetis', command) or ('pymetis', None). 'auto' prefers gpmetis."""
    if tool not in ("auto", "gpmetis", "pymetis"):
        raise ValueError(f"unknown METIS tool {tool!r}")
    if tool in ("auto", "gpmetis"):
        cmd = _command(binary, "GPMETIS_BIN", "gpmetis")
        if cmd:
            return "gpmetis", cmd
        if tool == "gpmetis":
            raise PartitionerUnavailable("gpmetis not found: set GPMETIS_BIN or put gpmetis on PATH")
    if not pymetis_available():
        raise PartitionerUnavailable("neither gpmetis nor pymetis is available")
    return "pymetis", None


def available_tools() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, env_name, program in (("gpmetis", "GPMETIS_BIN", "gpmetis"), ("kaffpa", "KAHIP_BIN", "kaffpa")):
        try:
            cmd = _command(None, env_name, program)
        except PartitionerUnavailable as exc:
            cmd, out[f"{name}_error"] = None, str(exc)
        out[name] = cmd
    if pymetis_available():
        import importlib.metadata as md

        out["pymetis"] = md.version("pymetis")
    else:
        out["pymetis"] = None
    return out


# ----------------------------------------------------------------------------
# METIS graph files (gpmetis and kaffpa)
# ----------------------------------------------------------------------------
def write_metis_graph(path: str | os.PathLike, indptr: np.ndarray, indices: np.ndarray,
                      chunk_entries: int = TEXT_CHUNK_ENTRIES) -> None:
    """METIS graph file: header 'n m', then the 1-based neighbours of vertex i on line i+1."""
    n = int(indptr.shape[0]) - 1
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{n} {int(indptr[-1]) // 2}\n")
        for a, b in row_chunks(indptr, chunk_entries):
            base = int(indptr[a])
            ptr = np.asarray(indptr[a:b + 1], dtype=np.int64) - base
            words = list(map(str, (np.asarray(indices[base:int(indptr[b])], dtype=np.int64) + 1).tolist()))
            handle.write("\n".join(" ".join(words[ptr[i]:ptr[i + 1]]) for i in range(b - a)))
            handle.write("\n")


def read_metis_graph(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of write_metis_graph (unweighted graphs only)."""
    with open(path, encoding="ascii") as handle:
        lines = [line for line in handle.read().split("\n") if not line.startswith("%")]
    n = int(lines[0].split()[0])
    rows = [np.asarray(line.split(), dtype=np.int64) - 1 for line in lines[1:n + 1]]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([row.shape[0] for row in rows], out=indptr[1:])
    indices = np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, dtype=np.int32)
    return indptr, indices


def read_part_file(path: str | os.PathLike, num_nodes: int) -> np.ndarray:
    """A METIS/KaHIP .part file: one part id per line."""
    labels = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if labels.shape[0] != num_nodes:
        raise PartitionError(f"{path}: {labels.shape[0]} part ids for {num_nodes} vertices")
    return labels


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
def _save_int64(path: Path, array: np.ndarray, chunk: int = 1 << 24) -> None:
    """np.save of array as int64 without an in-memory int64 copy of the whole array."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=(int(array.shape[0]),))
    for start in range(0, int(array.shape[0]), chunk):
        stop = min(int(array.shape[0]), start + chunk)
        out[start:stop] = np.asarray(array[start:stop], dtype=np.int64)
    out.flush()
    del out


def run_pymetis_job(indptr: np.ndarray, indices: np.ndarray, graph_ptr: np.ndarray, nparts: np.ndarray, *,
                    seed: int, ufactor: int, timeout: float | None = None,
                    workdir: str | os.PathLike | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Partition disjoint graphs (node ranges graph_ptr of one CSR) with pymetis in a child process.

    Returns the per-graph local part labels of every node and the worker's report.
    """
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jigsaw-pymetis-", dir=workdir) as tmp:
        job = Path(tmp)
        _save_int64(job / "indptr.npy", indptr)
        _save_int64(job / "indices.npy", indices)
        np.save(job / "graph_ptr.npy", np.asarray(graph_ptr, dtype=np.int64))
        np.save(job / "nparts.npy", np.asarray(nparts, dtype=np.int64))
        (job / "job.json").write_text(json.dumps({"seed": int(seed), "ufactor": int(ufactor)}), encoding="utf-8")
        cmd = [sys.executable, "-m", "scripts.canonical.partition", "pymetis-worker", str(job)]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PartitionError(f"pymetis worker timed out after {timeout} s") from exc
        wall = time.perf_counter() - t0
        result_path = job / "membership.npy"
        if proc.returncode != 0 or not result_path.exists():
            raise PartitionError(
                f"pymetis worker failed (exit code {_describe_exit(proc.returncode)}); stderr: {_tail(proc.stderr)}"
            )
        membership = np.load(result_path)
        report = json.loads((job / "result.json").read_text(encoding="utf-8"))
    report["subprocess_wall_seconds"] = wall
    return membership, report


def _pymetis_worker(job_dir: Path) -> int:
    """Child-process entry point: partition every graph of a job directory."""
    import importlib.metadata as md

    import pymetis

    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    indptr = np.load(job_dir / "indptr.npy", mmap_mode="r")  # pymetis reads memmaps without a copy
    indices = np.load(job_dir / "indices.npy", mmap_mode="r")
    graph_ptr = np.load(job_dir / "graph_ptr.npy")
    nparts = np.load(job_dir / "nparts.npy")
    membership = np.zeros(int(indptr.shape[0]) - 1, dtype=np.int64)
    cuts = np.zeros(int(nparts.shape[0]), dtype=np.int64)
    use_csr = hasattr(pymetis, "CSRAdjacency")
    t0 = time.perf_counter()
    for g in range(int(nparts.shape[0])):
        start, stop, k = int(graph_ptr[g]), int(graph_ptr[g + 1]), int(nparts[g])
        if k <= 1 or stop - start <= 1:
            continue
        if start == 0 and stop == membership.shape[0]:
            xadj, adjncy = indptr, indices
        else:
            xadj = indptr[start:stop + 1] - indptr[start]
            adjncy = indices[int(indptr[start]):int(indptr[stop])] - start
        options = pymetis.Options(seed=int(job["seed"]), ufactor=int(job["ufactor"]),
                                  objtype=int(pymetis.ObjType.CUT))
        if use_csr:
            cut, parts = pymetis.part_graph(k, adjacency=pymetis.CSRAdjacency(xadj, adjncy),
                                            recursive=False, options=options)
        else:
            cut, parts = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy, recursive=False, options=options)
        membership[start:stop] = np.asarray(parts).astype(np.int64, copy=False)
        cuts[g] = int(cut)
    wall = time.perf_counter() - t0
    try:
        from pymetis._internal import _idx_type_width

        idx_width = int(_idx_type_width())
    except Exception:
        idx_width = None
    tmp = job_dir / "membership.tmp.npy"
    np.save(tmp, membership)
    report = {
        "backend": "pymetis",
        "pymetis_version": md.version("pymetis"),
        "metis_idx_width_bits": idx_width,
        "function": "METIS_PartGraphKway",
        "recursive": False,
        "graphs": int(nparts.shape[0]),
        "metis_edge_cut_total": int(cuts.sum()),
        "metis_wall_seconds": wall,
        "worker_peak_memory": self_peak_memory(),
    }
    (job_dir / "result.json").write_text(json.dumps(report), encoding="utf-8")
    os.replace(tmp, job_dir / "membership.npy")
    return 0


def run_gpmetis(cmd: list[str], indptr: np.ndarray, indices: np.ndarray, nparts: int, *, seed: int,
                ufactor: int, timeout: float | None = None,
                workdir: str | os.PathLike | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """gpmetis -ptype=kway -objtype=cut -ufactor=U -seed=S on one graph; returns (labels, report)."""
    n = int(indptr.shape[0]) - 1
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jigsaw-gpmetis-", dir=workdir) as tmp:
        graph = Path(tmp) / "graph.metis"
        write_metis_graph(graph, indptr, indices)
        options = ["-ptype=kway", "-objtype=cut", f"-ufactor={int(ufactor)}", f"-seed={int(seed)}"]
        args = list(cmd) + options + [str(graph), str(int(nparts))]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(args, cwd=tmp, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PartitionError(f"gpmetis timed out after {timeout} s") from exc
        wall = time.perf_counter() - t0
        part_file = Path(f"{graph}.part.{int(nparts)}")
        if proc.returncode != 0 or not part_file.exists():
            raise PartitionError(f"gpmetis failed (exit code {_describe_exit(proc.returncode)}): "
                                 f"{_tail(proc.stdout)} {_tail(proc.stderr)}")
        labels = read_part_file(part_file, n)
    text = proc.stdout.decode("utf-8", "replace")
    report = {
        "backend": "gpmetis",
        "metis_version": _search(r"METIS\s+([0-9][0-9.]*)", text),
        "metis_idx_width_bits": _search(r"size of idx_t:\s*(\d+)", text),
        "function": "gpmetis -ptype=kway",
        "recursive": False,
        "arguments": options,
        "metis_edge_cut_total": int(_search(r"Edgecut:\s*(\d+)", text) or -1),
        "subprocess_wall_seconds": wall,
    }
    return labels, report


# ----------------------------------------------------------------------------
# Primary and fine partitions
# ----------------------------------------------------------------------------
def fill_empty_parts(indptr: np.ndarray, indices: np.ndarray, labels: np.ndarray, nparts: int, *,
                     seed: int = DEFAULT_SEED, ufactor: int = DEFAULT_UFACTOR, tool: str = "auto",
                     timeout: float | None = None, binary: str | Sequence[str] | None = None,
                     workdir: str | os.PathLike | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Refill the empty parts of labels in [0, nparts) by EMPTY_PART_RULE: (labels int64, one record per fill).

    METIS k-way does not promise non-empty parts (ogbn-mag with P=2000, pymetis
    2025.2.1 and seed 1 leaves one empty). Parts other than the split ones keep
    their METIS ids and nodes.
    """
    labels = np.array(labels, dtype=np.int64).reshape(-1)
    nparts = int(nparts)
    if labels.shape[0] and (int(labels.min()) < 0 or int(labels.max()) >= nparts):
        raise PartitionError(f"part labels outside [0, {nparts})")
    sizes = np.bincount(labels, minlength=nparts)
    empty = np.flatnonzero(sizes == 0)
    e = int(empty.shape[0])
    if e == 0:
        return labels, []
    donors = np.lexsort((np.arange(nparts), -sizes))[:e]
    if int(sizes[donors].min()) < 2:
        raise PartitionError(f"cannot fill {e} empty parts: too few parts with two or more nodes")
    tool_name, cmd = resolve_metis_tool(tool, binary)
    rank = np.full(nparts, -1, dtype=np.int64)
    rank[donors] = np.arange(e, dtype=np.int64)
    cand = np.flatnonzero(rank[labels] >= 0)  # ascending ids
    perm = np.argsort(rank[labels[cand]], kind="stable")
    nodes = cand[perm]  # grouped by donor rank, ascending id inside a donor
    where = np.empty(cand.shape[0], dtype=np.int64)
    where[perm] = np.arange(cand.shape[0], dtype=np.int64)
    graph_ptr = np.zeros(e + 1, dtype=np.int64)
    np.cumsum(sizes[donors], out=graph_ptr[1:])
    lens, nbrs = gather_rows(indptr, indices, nodes)
    nbrs = nbrs.astype(np.int64, copy=False)
    keep = labels[nbrs] == np.repeat(labels[nodes], lens)
    row = np.repeat(np.arange(nodes.shape[0], dtype=np.int64), lens)[keep]
    cols = where[np.searchsorted(cand, nbrs[keep])]
    ptr = np.zeros(nodes.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(row, minlength=nodes.shape[0]), out=ptr[1:])
    if tool_name == "gpmetis":
        assert cmd is not None
        sides = np.zeros(nodes.shape[0], dtype=np.int64)
        for g in range(e):
            s, t = int(graph_ptr[g]), int(graph_ptr[g + 1])
            sides[s:t], _ = run_gpmetis(cmd, ptr[s:t + 1] - ptr[s], cols[int(ptr[s]):int(ptr[t])] - s, 2,
                                        seed=seed, ufactor=ufactor, timeout=timeout, workdir=workdir)
    else:
        sides, _ = run_pymetis_job(ptr, cols, graph_ptr, np.full(e, 2, dtype=np.int64), seed=seed,
                                   ufactor=ufactor, timeout=timeout, workdir=workdir)
    records: list[dict[str, Any]] = []
    for g in range(e):
        s, t = int(graph_ptr[g]), int(graph_ptr[g + 1])
        side = np.asarray(sides[s:t]) == 1
        method = "metis_kway_2"
        if side.all() or not side.any():
            side = np.arange(t - s) >= (t - s) // 2
            method = "node_id_halves"
        labels[nodes[s:t][side]] = empty[g]
        records.append({"empty_part": int(empty[g]), "donor_part": int(donors[g]), "donor_size": t - s,
                        "sizes_after": [int(np.count_nonzero(~side)), int(np.count_nonzero(side))],
                        "method": method, "tool": tool_name})
    return labels, records


def metis_partition(indptr: np.ndarray, indices: np.ndarray, nparts: int, *, seed: int = DEFAULT_SEED,
                    ufactor: int = DEFAULT_UFACTOR, tool: str = "auto", timeout: float | None = None,
                    binary: str | Sequence[str] | None = None, workdir: str | os.PathLike | None = None,
                    fill_empty: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """METIS k-way edge-cut partition into nparts blocks: (node_block int32, info).

    Parts METIS leaves empty are renumbered away (fewer blocks), or with
    fill_empty=True refilled by fill_empty_parts (exactly nparts blocks).
    """
    n = int(indptr.shape[0]) - 1
    nparts = int(nparts)
    if not 1 <= nparts <= n:
        raise ValueError(f"nparts={nparts} must lie in [1, {n}]")
    tool_name, cmd = resolve_metis_tool(tool, binary)
    t0 = time.perf_counter()
    if nparts == 1:
        labels, backend = np.zeros(n, dtype=np.int64), {"backend": "trivial", "note": "nparts=1"}
    elif tool_name == "gpmetis":
        assert cmd is not None
        labels, backend = run_gpmetis(cmd, indptr, indices, nparts, seed=seed, ufactor=ufactor,
                                      timeout=timeout, workdir=workdir)
        backend.update(_binary_info(cmd))
    else:
        labels, backend = run_pymetis_job(indptr, indices, np.array([0, n]), np.array([nparts]), seed=seed,
                                          ufactor=ufactor, timeout=timeout, workdir=workdir)
    metis_empty = int(nparts - np.unique(labels).shape[0])
    filled: list[dict[str, Any]] = []
    if fill_empty and metis_empty:
        labels, filled = fill_empty_parts(indptr, indices, labels, nparts, seed=seed, ufactor=ufactor,
                                          tool=tool_name, timeout=timeout, binary=binary, workdir=workdir)
    node_block, empty = compact_labels(labels, nparts)
    info: dict[str, Any] = {
        "tool": tool_name,
        "version": backend.get("pymetis_version") or backend.get("metis_version"),
        "method": "kway",
        "recursive_bisection": False,
        "objective": "edge_cut",
        "options": {"ptype": "kway", "objtype": "cut", "ufactor": int(ufactor), "seed": int(seed)},
        "nparts_requested": nparts,
        "metis_empty_parts": metis_empty,
        "empty_part_rule": EMPTY_PART_RULE if fill_empty else "renumber",
        "empty_parts_filled": filled,
        "empty_parts_renumbered": empty,
        "backend": backend,
    }
    info.update(partition_stats(indptr, indices, node_block))
    info["wall_seconds"] = time.perf_counter() - t0
    return node_block, info


def _batch_induced(indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, node_block: np.ndarray,
                   pos_in_block: np.ndarray, block_ptr: np.ndarray, b0: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR of the disjoint union of the induced subgraphs of consecutive blocks.

    `nodes` lists the blocks' nodes grouped by block (ascending id inside a
    block); batch id of a node = its position in `nodes`.
    """
    lens, nbrs = gather_rows(indptr, indices, nodes)
    nbrs = nbrs.astype(np.int64, copy=False)
    own = np.repeat(node_block[nodes], lens)
    other = node_block[nbrs]
    keep = own == other
    row = np.repeat(np.arange(nodes.shape[0], dtype=np.int64), lens)[keep]
    nb = nbrs[keep]
    cols = block_ptr[node_block[nb]] - block_ptr[b0] + pos_in_block[nb]
    counts = np.bincount(row, minlength=nodes.shape[0]).astype(np.int64)
    ptr = np.zeros(nodes.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, cols


def fine_partition(indptr: np.ndarray, indices: np.ndarray, node_block: np.ndarray, fine_per_block: int, *,
                   seed: int = DEFAULT_SEED, ufactor: int = DEFAULT_UFACTOR, tool: str = "auto",
                   timeout: float | None = None, binary: str | Sequence[str] | None = None,
                   workdir: str | os.PathLike | None = None, batch_entries: int = FINE_BATCH_ENTRIES,
                   info: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """METIS k-way on every block's induced subgraph: (node_fine int32, fine_block int32).

    Block b gets min(fine_per_block, |b|) parts. Fine ids are consecutive per
    block in block order; parts METIS leaves empty are dropped by renumbering
    and counted in `info` (filled in place when given).
    """
    t0 = time.perf_counter()
    node_block = np.asarray(node_block, dtype=np.int64)
    n = int(node_block.shape[0])
    num_blocks = int(node_block.max()) + 1
    sizes = np.bincount(node_block, minlength=num_blocks).astype(np.int64)
    if (sizes == 0).any():
        raise ValueError("node_block has empty blocks")
    tool_name, cmd = resolve_metis_tool(tool, binary)
    order = np.argsort(node_block, kind="stable")
    block_ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    np.cumsum(sizes, out=block_ptr[1:])
    pos_in_block = np.empty(n, dtype=np.int64)
    pos_in_block[order] = np.arange(n, dtype=np.int64) - np.repeat(block_ptr[:-1], sizes)
    k_block = np.minimum(int(fine_per_block), sizes)
    degree = np.diff(np.asarray(indptr, dtype=np.int64))
    block_entries = np.bincount(node_block, weights=degree, minlength=num_blocks)
    local = np.zeros(n, dtype=np.int64)  # local part label, indexed like `order`
    metis_calls = 0
    b0 = 0
    while b0 < num_blocks:
        cumulative = np.cumsum(block_entries[b0:])
        b1 = b0 + max(1, int(np.searchsorted(cumulative, batch_entries, side="right")))
        nodes = order[block_ptr[b0]:block_ptr[b1]]
        ptr, cols = _batch_induced(indptr, indices, nodes, node_block, pos_in_block, block_ptr, b0)
        graph_ptr = block_ptr[b0:b1 + 1] - block_ptr[b0]
        if tool_name == "gpmetis":
            assert cmd is not None
            labels = np.zeros(nodes.shape[0], dtype=np.int64)
            for g in range(b1 - b0):
                s, e, k = int(graph_ptr[g]), int(graph_ptr[g + 1]), int(k_block[b0 + g])
                if k <= 1 or e - s <= 1:
                    continue
                sub_ptr = ptr[s:e + 1] - ptr[s]
                sub_idx = cols[int(ptr[s]):int(ptr[e])] - s
                labels[s:e], _ = run_gpmetis(cmd, sub_ptr, sub_idx, k, seed=seed, ufactor=ufactor,
                                             timeout=timeout, workdir=workdir)
                metis_calls += 1
        else:
            labels, _ = run_pymetis_job(ptr, cols, graph_ptr, k_block[b0:b1], seed=seed, ufactor=ufactor,
                                        timeout=timeout, workdir=workdir)
            metis_calls += 1
        local[block_ptr[b0]:block_ptr[b1]] = labels
        b0 = b1
    if (local >= np.repeat(k_block, sizes)).any() or (local < 0).any():
        raise PartitionError("fine labels outside [0, k) for some block")
    width = int(k_block.max())
    keys = np.repeat(np.arange(num_blocks, dtype=np.int64), sizes) * width + local
    uniq, fine_of_pos = np.unique(keys, return_inverse=True)
    node_fine = np.empty(n, dtype=np.int32)
    node_fine[order] = fine_of_pos.reshape(-1).astype(np.int32)
    fine_block = (uniq // width).astype(np.int32)
    empty = k_block - np.bincount(fine_block, minlength=num_blocks)
    if info is not None:
        fine_sizes = np.bincount(node_fine)
        info.update({
            "tool": tool_name,
            "method": "kway",
            "recursive_bisection": False,
            "options": {"ptype": "kway", "objtype": "cut", "ufactor": int(ufactor), "seed": int(seed)},
            "fine_per_block": int(fine_per_block),
            "num_fine": int(uniq.shape[0]),
            "fine_parts_requested": int(k_block.sum()),
            "empty_parts_renumbered": int(empty.sum()),
            "blocks_with_empty_parts": int((empty > 0).sum()),
            "blocks_smaller_than_fine_per_block": int((sizes < int(fine_per_block)).sum()),
            "fine_size_min": int(fine_sizes.min()),
            "fine_size_max": int(fine_sizes.max()),
            "fine_size_mean": float(fine_sizes.mean()),
            "metis_calls": metis_calls,
            "wall_seconds": time.perf_counter() - t0,
        })
    return node_fine, fine_block


# ----------------------------------------------------------------------------
# Alternatives for sensitivity runs
# ----------------------------------------------------------------------------
def scaled_nparts(P: int, scale: float) -> int:
    """round(scale * P) blocks, at least one (PARTITIONING['partition_scale_for_sensitivity'])."""
    return max(1, int(round(float(scale) * int(P))))


def bfs_random_partition(indptr: np.ndarray, indices: np.ndarray, nparts: int, *,
                         seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Balanced parts (sizes differ by at most one) grown one at a time by BFS from random seeds.

    A part grows level by level over unassigned nodes; when the last level is
    larger than the remaining capacity a random subset is taken, and when the
    frontier dies out the part restarts from the next unassigned node of a
    seeded permutation.
    """
    t0 = time.perf_counter()
    n = int(indptr.shape[0]) - 1
    nparts = int(nparts)
    if not 1 <= nparts <= n:
        raise ValueError(f"nparts={nparts} must lie in [1, {n}]")
    rng = np.random.default_rng(seed)
    base, extra = divmod(n, nparts)
    block = np.full(n, -1, dtype=np.int32)
    starts = rng.permutation(n)
    cursor = 0
    bfs_starts = 0
    for b in range(nparts):
        need = base + (1 if b < extra else 0)
        while need > 0:
            while block[starts[cursor]] >= 0:
                cursor += 1
            start = int(starts[cursor])
            block[start] = b
            need -= 1
            bfs_starts += 1
            frontier = np.array([start], dtype=np.int64)
            while need > 0 and frontier.shape[0]:
                _, nbrs = gather_rows(indptr, indices, frontier)
                cand = np.unique(nbrs.astype(np.int64, copy=False))
                cand = cand[block[cand] < 0]
                if cand.shape[0] > need:
                    cand = np.sort(rng.choice(cand, size=need, replace=False))
                block[cand] = b
                need -= int(cand.shape[0])
                frontier = cand
    info: dict[str, Any] = {"tool": "bfs_random", "seed": int(seed), "nparts_requested": nparts,
                            "bfs_starts": bfs_starts}
    info.update(partition_stats(indptr, indices, block))
    info["wall_seconds"] = time.perf_counter() - t0
    return block, info


def kahip_partition(indptr: np.ndarray, indices: np.ndarray, nparts: int, *, seed: int = DEFAULT_SEED,
                    preconfiguration: str = "fast", imbalance: float = DEFAULT_UFACTOR / 1000.0,
                    binary: str | Sequence[str] | None = None, timeout: float | None = None,
                    workdir: str | os.PathLike | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """KaHIP kaffpa --preconfiguration=fast into nparts blocks (imbalance 3 % like ufactor 30)."""
    cmd = _command(binary, "KAHIP_BIN", "kaffpa")
    if cmd is None:
        raise PartitionerUnavailable("KaHIP kaffpa not found: set KAHIP_BIN to the kaffpa binary "
                                     "or put kaffpa on PATH")
    n = int(indptr.shape[0]) - 1
    nparts = int(nparts)
    t0 = time.perf_counter()
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jigsaw-kahip-", dir=workdir) as tmp:
        graph = Path(tmp) / "graph.metis"
        out = Path(tmp) / "graph.part"
        write_metis_graph(graph, indptr, indices)
        options = [f"--k={nparts}", f"--preconfiguration={preconfiguration}", f"--seed={int(seed)}",
                   f"--imbalance={imbalance * 100:g}"]
        args = list(cmd) + [str(graph)] + options + [f"--output_filename={out}"]
        try:
            proc = subprocess.run(args, cwd=tmp, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PartitionError(f"kaffpa timed out after {timeout} s") from exc
        if proc.returncode != 0 or not out.exists():
            raise PartitionError(f"kaffpa failed (exit code {_describe_exit(proc.returncode)}): "
                                 f"{_tail(proc.stdout)} {_tail(proc.stderr)}")
        labels = read_part_file(out, n)
    node_block, empty = compact_labels(labels, nparts)
    info: dict[str, Any] = {"tool": "kahip", "preconfiguration": preconfiguration, "options": options,
                            "nparts_requested": nparts, "empty_parts_renumbered": empty, **_binary_info(cmd)}
    info.update(partition_stats(indptr, indices, node_block))
    info["wall_seconds"] = time.perf_counter() - t0
    return node_block, info


# ----------------------------------------------------------------------------
# Imported partitions
# ----------------------------------------------------------------------------
_LABEL_KEYS = ("node_block", "membership", "node_to_block", "parts", "part", "node_to_coarse")


def _labels_from_object(obj: Any, num_nodes: int, source: Path) -> np.ndarray:
    """A node-to-block array from a tensor, an array, or a legacy hierarchy dict."""
    if hasattr(obj, "detach") and hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)) and obj:
        if isinstance(obj[0], dict):
            return _labels_from_object(obj[0], num_nodes, source)  # hierarchy 0 is the exact target
        return np.asarray(obj)
    if isinstance(obj, dict):
        for key in _LABEL_KEYS:
            if key in obj:
                return _labels_from_object(obj[key], num_nodes, source)
        if "hierarchies" in obj:
            return _labels_from_object(obj["hierarchies"], num_nodes, source)
        if "coarse_part_nodes_map" in obj:
            obj = obj["coarse_part_nodes_map"]
        elif "node_to_coarse_map" in obj:
            mapping = obj["node_to_coarse_map"]
            labels = np.full(num_nodes, -1, dtype=np.int64)
            keys = np.fromiter((int(k) for k in mapping.keys()), dtype=np.int64, count=len(mapping))
            vals = np.fromiter((int(v) for v in mapping.values()), dtype=np.int64, count=len(mapping))
            if keys.shape[0] and (keys.min() < 0 or keys.max() >= num_nodes):
                raise ValueError(f"{source}: node ids outside [0, {num_nodes})")
            labels[keys] = vals
            return labels
        if obj and all(isinstance(k, (int, np.integer)) for k in obj):
            labels = np.full(num_nodes, -1, dtype=np.int64)
            for part, nodes in obj.items():
                members = np.asarray(nodes.detach().cpu().numpy() if hasattr(nodes, "detach") else nodes,
                                     dtype=np.int64)
                if members.shape[0] and (members.min() < 0 or members.max() >= num_nodes):
                    raise ValueError(f"{source}: node ids outside [0, {num_nodes})")
                labels[members] = int(part)
            return labels
    raise ValueError(f"{source}: no node-to-block array found (expected a tensor or array, or a dict with one "
                     f"of {', '.join(_LABEL_KEYS)}, coarse_part_nodes_map or node_to_coarse_map)")


def import_partition(path: str | os.PathLike, num_nodes: int, *, nparts: int | None = None,
                     trusted_pickle: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """A node-to-block array from a .npy, a text file (one id per line, e.g. a METIS .part) or a .pt.

    Validates the length, the id range ([0, nparts) when nparts is given) and
    that no block is empty. A .pt is read with torch.load(weights_only=True);
    trusted_pickle=True allows a full unpickle of your own legacy file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"partition file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif suffix in (".pt", ".pth"):
        import torch

        try:
            obj = torch.load(path, map_location="cpu", weights_only=not trusted_pickle)
        except Exception as exc:
            if trusted_pickle:
                raise
            raise ValueError(f"{path}: torch.load(weights_only=True) failed ({exc}); convert it to a .npy "
                             "node-to-block array or pass trusted_pickle=True for your own file") from exc
        raw = _labels_from_object(obj, num_nodes, path)
    else:
        raw = np.loadtxt(path, dtype=np.int64, ndmin=1)
    labels = np.asarray(raw)
    if labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D node-to-block array, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if np.issubdtype(labels.dtype, np.floating) and np.all(np.isfinite(labels)) \
                and np.array_equal(labels, np.round(labels)):
            labels = labels.astype(np.int64)
        else:
            raise ValueError(f"{path}: block ids must be integers, got {labels.dtype}")
    labels = labels.astype(np.int64, copy=False)
    if labels.shape[0] != int(num_nodes):
        raise ValueError(f"{path}: {labels.shape[0]} block ids for {num_nodes} nodes")
    if labels.shape[0] and int(labels.min()) < 0:
        raise ValueError(f"{path}: negative block id {int(labels.min())} (or a node without a block)")
    k = int(nparts) if nparts is not None else (int(labels.max()) + 1 if labels.shape[0] else 0)
    if labels.shape[0] and int(labels.max()) >= k:
        raise ValueError(f"{path}: block id {int(labels.max())} outside [0, {k})")
    sizes = np.bincount(labels, minlength=k)
    empty = np.flatnonzero(sizes == 0)
    if empty.shape[0]:
        raise ValueError(f"{path}: {empty.shape[0]} empty block(s), for example {empty[:10].tolist()}")
    info = {
        "tool": "import",
        "source": str(path),
        "source_sha256": file_sha256(path),
        "source_bytes": path.stat().st_size,
        "nparts_requested": k,
        "empty_parts_renumbered": 0,
    }
    return labels.astype(np.int32), info


# ----------------------------------------------------------------------------
# Canonical order
# ----------------------------------------------------------------------------
def block_contiguous_order(node_block: np.ndarray, node_fine: np.ndarray | None = None) -> np.ndarray:
    """perm with perm[new_id] = original id, sorted by (block, fine, original id).

    Canonical ids are positions in perm, so every block (and every fine part
    inside it) is a contiguous id range.
    """
    node_block = np.asarray(node_block, dtype=np.int64)
    if node_fine is None:
        return np.argsort(node_block, kind="stable").astype(np.int64)
    return np.lexsort((np.asarray(node_fine, dtype=np.int64), node_block)).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("pymetis-worker", help="internal: partition the graphs of one job directory")
    worker.add_argument("job_dir")
    sub.add_parser("tools", help="report the available partitioners")
    args = parser.parse_args(argv)
    if args.command == "pymetis-worker":
        return _pymetis_worker(Path(args.job_dir))
    print(json.dumps(available_tools(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
