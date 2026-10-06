"""Out-of-core helpers for graphs that do not fit in RAM (ogbn-papers100M).

Large arrays live in .npy files opened as memory maps and are processed in
chunks, so resident memory is bounded by the chunk size plus a few [N] vectors,
never by the edge count:

  npz_member_to_npy     one member of an .npz (stored or deflated) streamed to a
                        .npy file, optionally cast chunk by chunk
  npz_member_memmap     a read-only memmap of a stored (uncompressed) member
  csr_from_edge_index   COO [2, E] -> the normalised CSR of store.csr_from_edges
                        (symmetric, sorted rows, no self loops or duplicates) by a
                        chunked counting sort: degree count, scatter into a
                        memmap, then per-row sort and dedupe compacted in place
  permute_csr_chunked   store.permute_csr into memmaps
  take_rows             out[i] = src[rows[i]] chunk by chunk (sorted reads)
  block_graph_chunked   store.block_graph accumulated over row chunks
  write_store_ooc       store.write_store for a graph given as a normalised CSR;
                        the files it writes are byte-identical
  add_partition_ooc     store.add_partition without loading the CSR into RAM
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import store as store_mod
from .partition import row_chunks
from .store import PRIMARY, STORE_VERSION, gather_rows

CHUNK_EDGES = 1 << 25  # COO edges per chunk (about 2-3 GB of temporaries)
CHUNK_ENTRIES = 1 << 25  # CSR entries per chunk
CHUNK_ROWS = 1 << 18  # attribute rows per chunk
STREAM_BYTES = 1 << 26


def open_npy(path: str | os.PathLike, dtype: Any, shape: tuple[int, ...], *,
             fortran_order: bool = False) -> np.memmap:
    """A new .npy file opened as a writable memmap (same header as np.save)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=tuple(int(s) for s in shape),
                                     fortran_order=fortran_order)


def close_memmap(array: Any) -> None:
    """Flush and unmap a memmap now (Windows cannot delete or replace a mapped file)."""
    if isinstance(array, np.memmap):
        array.flush()
        mm = getattr(array, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass  # still exported to a live view; it closes with the last reference


# ----------------------------------------------------------------------------
# .npz members
# ----------------------------------------------------------------------------
def _read_npy_header(handle: Any) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"unsupported .npy format version {version}")
    if dtype.hasobject:
        raise ValueError("object arrays are not loaded (they would need pickle)")
    return tuple(int(s) for s in shape), bool(fortran), dtype


def _member_name(zf: zipfile.ZipFile, member: str) -> str:
    names = zf.namelist()
    for name in (member, member + ".npy"):
        if name in names:
            return name
    raise KeyError(f"{zf.filename}: no member {member!r} (members: {', '.join(sorted(names))})")


def npz_member_info(npz_path: str | os.PathLike, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(_member_name(zf, member))
        with zf.open(info) as handle:
            shape, fortran, dtype = _read_npy_header(handle)
    return {"member": info.filename, "shape": shape, "fortran_order": fortran, "dtype": str(dtype),
            "stored": info.compress_type == zipfile.ZIP_STORED, "compressed_bytes": info.compress_size}


def npz_member_memmap(npz_path: str | os.PathLike, member: str) -> np.memmap | None:
    """Read-only memmap of a stored member, or None when the member is compressed."""
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(_member_name(zf, member))
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        with zf.open(info) as handle:
            shape, fortran, dtype = _read_npy_header(handle)
            header_len = handle.tell()
    with open(npz_path, "rb") as raw:
        raw.seek(info.header_offset)
        local = raw.read(30)
    if local[:4] != b"PK\x03\x04":
        raise ValueError(f"{npz_path}: bad local header for {info.filename}")
    name_len = int.from_bytes(local[26:28], "little")
    extra_len = int.from_bytes(local[28:30], "little")
    offset = info.header_offset + 30 + name_len + extra_len + header_len
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def npz_member_to_npy(npz_path: str | os.PathLike, member: str, out_path: str | os.PathLike, *,
                      dtype: Any = None, chunk_bytes: int = STREAM_BYTES) -> np.memmap:
    """Stream one .npz member (deflated or stored) into a .npy file, cast to dtype per chunk."""
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(_member_name(zf, member))
        with zf.open(info) as handle:
            shape, fortran, src_dtype = _read_npy_header(handle)
            out_dtype = np.dtype(dtype) if dtype is not None else src_dtype
            out = open_npy(out_path, out_dtype, shape, fortran_order=fortran)
            flat = out.reshape(-1, order="F" if fortran else "C")
            total = int(np.prod(shape, dtype=np.int64))
            item = src_dtype.itemsize
            step = max(item, int(chunk_bytes) - int(chunk_bytes) % item)
            pos, carry = 0, b""
            while pos < total:
                data = carry + handle.read(step)
                count = min(len(data) // item, total - pos)
                if count == 0:
                    raise EOFError(f"{npz_path}:{info.filename} ended after {pos} of {total} values")
                flat[pos:pos + count] = np.frombuffer(data, dtype=src_dtype, count=count)
                carry = data[count * item:]
                pos += count
    out.flush()
    return out


# ----------------------------------------------------------------------------
# CSR construction and permutation
# ----------------------------------------------------------------------------
def _edge_chunks(edge_index: Any, num_nodes: int, chunk_edges: int):
    total = int(edge_index.shape[1])
    for a in range(0, total, chunk_edges):
        b = min(total, a + chunk_edges)
        u = np.asarray(edge_index[0, a:b], dtype=np.int64)
        v = np.asarray(edge_index[1, a:b], dtype=np.int64)
        if u.shape[0] and (min(int(u.min()), int(v.min())) < 0 or max(int(u.max()), int(v.max())) >= num_nodes):
            raise ValueError(f"edge endpoints outside [0, {num_nodes}) in edges {a}..{b}")
        keep = u != v
        yield u[keep], v[keep], int(b - a - int(keep.sum()))


def csr_from_edge_index(edge_index: Any, num_nodes: int, out_dir: str | os.PathLike, *,
                        chunk_edges: int = CHUNK_EDGES, index_dtype: Any = None
                        ) -> tuple[np.ndarray, np.memmap, dict[str, Any]]:
    """Normalised CSR of a COO edge_index [2, E] (memmap or array) with bounded memory.

    Writes out_dir/indptr.npy and out_dir/indices.npy; returns (indptr in RAM,
    indices as a read-only memmap, report). Equal to store.csr_from_edges.
    """
    out_dir = Path(out_dir)
    n = int(num_nodes)
    if len(edge_index.shape) != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    dtype = np.dtype(index_dtype) if index_dtype is not None else np.dtype(np.int32 if n < 2**31 else np.int64)
    deg = np.zeros(n, dtype=np.int64)
    loops = 0
    for u, v, dropped in _edge_chunks(edge_index, n, chunk_edges):
        loops += dropped
        deg += np.bincount(u, minlength=n)
        deg += np.bincount(v, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=ptr[1:])
    directed_entries = int(ptr[-1])
    raw_path = out_dir / "indices.raw.npy"
    raw = open_npy(raw_path, dtype, (directed_entries,))
    cursor = ptr[:-1].copy()
    for u, v, _ in _edge_chunks(edge_index, n, chunk_edges):
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        del u, v
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
        del order
        if rows.shape[0] == 0:
            continue
        first = np.flatnonzero(np.concatenate([[True], rows[1:] != rows[:-1]]))
        counts = np.diff(np.append(first, rows.shape[0]))
        rank = np.arange(rows.shape[0], dtype=np.int64) - np.repeat(first, counts)
        raw[cursor[rows] + rank] = cols
        cursor[rows[first]] += counts
    if not np.array_equal(cursor, ptr[1:]):
        raise AssertionError("edge scatter did not fill every row")
    new_deg = np.zeros(n, dtype=np.int64)
    write = 0
    for a, b in row_chunks(ptr, 2 * chunk_edges):
        seg = np.array(raw[int(ptr[a]):int(ptr[b])], dtype=np.int64)
        local = np.repeat(np.arange(b - a, dtype=np.int64), deg[a:b])
        keys = np.unique(local * n + seg)
        del seg, local
        rows_k = keys // n
        new_deg[a:b] = np.bincount(rows_k, minlength=b - a)
        raw[write:write + keys.shape[0]] = keys - rows_k * n  # write <= ptr[a]: never ahead of unread data
        write += int(keys.shape[0])
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(new_deg, out=indptr[1:])
    np.save(out_dir / "indptr.npy", indptr)
    final_path = out_dir / "indices.npy"
    if write == directed_entries:
        close_memmap(raw)
        del raw
        os.replace(raw_path, final_path)
    else:
        final = open_npy(final_path, dtype, (write,))
        for a in range(0, write, 2 * chunk_edges):
            b = min(write, a + 2 * chunk_edges)
            final[a:b] = raw[a:b]
        close_memmap(final)
        close_memmap(raw)
        del final, raw
        os.remove(raw_path)
    report = {
        "input_edges": int(edge_index.shape[1]),
        "self_loops_dropped": int(loops),
        "directed_entries": directed_entries,
        "duplicates_merged": (directed_entries - write) // 2,
        "undirected_edges": write // 2,
        "adjacency_entries": write,
        "isolated_nodes": int(np.count_nonzero(new_deg == 0)),
        "chunk_edges": int(chunk_edges),
    }
    return indptr, np.load(final_path, mmap_mode="r"), report


def permute_csr_chunked(indptr: np.ndarray, indices: np.ndarray, new_to_old: np.ndarray,
                        out_dir: str | os.PathLike, *, chunk_entries: int = CHUNK_ENTRIES
                        ) -> tuple[np.ndarray, np.memmap]:
    """store.permute_csr written to out_dir/indptr.npy and out_dir/indices.npy chunk by chunk."""
    out_dir = Path(out_dir)
    new_to_old = np.asarray(new_to_old, dtype=np.int64)
    n = int(new_to_old.shape[0])
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(n, dtype=np.int64)
    lens_new = np.diff(np.asarray(indptr, dtype=np.int64))[new_to_old]
    new_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens_new, out=new_ptr[1:])
    del lens_new
    np.save(out_dir / "indptr.npy", new_ptr)
    out = open_npy(out_dir / "indices.npy", indices.dtype, (int(new_ptr[-1]),))
    for a, b in row_chunks(new_ptr, chunk_entries):
        lens, nbrs = gather_rows(indptr, indices, new_to_old[a:b])
        cols = old_to_new[nbrs.astype(np.int64, copy=False)]
        local = np.repeat(np.arange(b - a, dtype=np.int64), lens)
        order = np.lexsort((cols, local))
        out[int(new_ptr[a]):int(new_ptr[b])] = cols[order]
    out.flush()
    return new_ptr, out


def take_rows(src: Any, rows: np.ndarray, out_path: str | os.PathLike, *, dtype: Any = None,
              chunk_rows: int = CHUNK_ROWS) -> np.memmap:
    """out[i] = src[rows[i]] into a new .npy, reading every chunk in ascending row order."""
    rows = np.asarray(rows, dtype=np.int64)
    out = open_npy(out_path, dtype if dtype is not None else src.dtype, (rows.shape[0],) + tuple(src.shape[1:]))
    for a in range(0, rows.shape[0], chunk_rows):
        b = min(rows.shape[0], a + chunk_rows)
        idx = rows[a:b]
        order = np.argsort(idx, kind="stable")
        block = np.empty((b - a,) + tuple(src.shape[1:]), dtype=out.dtype)
        block[order] = np.asarray(src[idx[order]])
        out[a:b] = block
    out.flush()
    return out


def block_graph_chunked(indptr: np.ndarray, indices: np.ndarray, node_block: np.ndarray, num_blocks: int, *,
                        chunk_entries: int = CHUNK_ENTRIES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """store.block_graph accumulated over row chunks."""
    node_block = np.asarray(node_block, dtype=np.int64)
    keys_parts, count_parts = [], []
    for a, b in row_chunks(indptr, chunk_entries):
        lens = np.diff(np.asarray(indptr[a:b + 1], dtype=np.int64))
        bu = np.repeat(node_block[a:b], lens)
        bv = node_block[np.asarray(indices[int(indptr[a]):int(indptr[b])], dtype=np.int64)]
        cut = bu != bv
        keys, counts = np.unique(bu[cut] * np.int64(num_blocks) + bv[cut], return_counts=True)
        keys_parts.append(keys)
        count_parts.append(counts.astype(np.int64))
    keys = np.concatenate(keys_parts) if keys_parts else np.zeros(0, dtype=np.int64)
    counts = np.concatenate(count_parts) if count_parts else np.zeros(0, dtype=np.int64)
    if keys.shape[0]:
        order = np.argsort(keys, kind="stable")
        keys, counts = keys[order], counts[order]
        first = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
        keys, counts = keys[first], np.add.reduceat(counts, first)
    src, dst = keys // num_blocks, keys % num_blocks
    ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=num_blocks), out=ptr[1:])
    return ptr, dst.astype(np.int32), counts.astype(np.int64)


# ----------------------------------------------------------------------------
# Store writers (same files as store.write_store / store.add_partition)
# ----------------------------------------------------------------------------
def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def write_partition_ooc(part_dir: Path, *, node_block: np.ndarray, node_fine: np.ndarray | None,
                        indptr: np.ndarray, indices: np.ndarray, info: Mapping[str, Any],
                        chunk_entries: int = CHUNK_ENTRIES) -> None:
    """store.write_partition with the block graph computed in chunks."""
    node_block = np.asarray(node_block, dtype=np.int32)
    num_blocks = int(node_block.max()) + 1
    order = np.argsort(node_block, kind="stable")
    counts = np.bincount(node_block, minlength=num_blocks).astype(np.int64)
    block_ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    np.cumsum(counts, out=block_ptr[1:])
    _save(part_dir / "block_ptr.npy", block_ptr)
    _save(part_dir / "block_nodes.npy", order.astype(np.int32))
    _save(part_dir / "node_block.npy", node_block)
    del order
    if node_fine is not None:
        node_fine = np.asarray(node_fine, dtype=np.int32)
        num_fine = int(node_fine.max()) + 1
        forder = np.argsort(node_fine, kind="stable")
        fcounts = np.bincount(node_fine, minlength=num_fine).astype(np.int64)
        fine_ptr = np.zeros(num_fine + 1, dtype=np.int64)
        np.cumsum(fcounts, out=fine_ptr[1:])
        fine_block = np.full(num_fine, -1, dtype=np.int32)
        fine_block[node_fine] = node_block
        _save(part_dir / "fine_ptr.npy", fine_ptr)
        _save(part_dir / "fine_nodes.npy", forder.astype(np.int32))
        _save(part_dir / "node_fine.npy", node_fine)
        _save(part_dir / "fine_block.npy", fine_block)
        del forder
    bptr, bidx, bw = block_graph_chunked(indptr, indices, node_block, num_blocks, chunk_entries=chunk_entries)
    _save(part_dir / "bgraph_ptr.npy", bptr)
    _save(part_dir / "bgraph_idx.npy", bidx)
    _save(part_dir / "bgraph_w.npy", bw)
    meta = dict(info)
    meta.update({
        "num_blocks": num_blocks,
        "num_fine": int(node_fine.max()) + 1 if node_fine is not None else 0,
        "block_size_min": int(counts.min()),
        "block_size_max": int(counts.max()),
        "block_size_mean": float(counts.mean()),
        "cut_edges": int(bw.sum() // 2),
    })
    (part_dir / "partition.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_store_ooc(root: str | os.PathLike, dataset: str, *, indptr: np.ndarray, indices: np.ndarray,
                    node_type: np.ndarray, attrs: Any, node_block: np.ndarray, node_fine: np.ndarray | None,
                    classes: np.ndarray | None = None, orig_id: np.ndarray | None = None,
                    meta: Mapping[str, Any] | None = None, partition_info: Mapping[str, Any] | None = None,
                    new_to_old: np.ndarray | None = None, attrs_dtype: Any = None,
                    chunk_entries: int = CHUNK_ENTRIES, chunk_rows: int = CHUNK_ROWS) -> Path:
    """store.write_store for a normalised CSR (source ids), streaming the large arrays.

    With new_to_old=None the node order is store.block_order(node_block), as in
    write_store; otherwise new_to_old lists the source ids block by block. The
    files equal those of write_store on the same graph.
    """
    out = Path(root) / dataset
    graph_dir = out / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = int(indptr.shape[0]) - 1
    node_block = np.asarray(node_block, dtype=np.int64)
    if new_to_old is None:
        new_to_old, _, _ = store_mod.block_order(node_block)
    new_to_old = np.asarray(new_to_old, dtype=np.int64)
    new_block = node_block[new_to_old].astype(np.int32)
    if new_to_old.shape[0] != num_nodes or np.any(np.diff(new_block) < 0):
        raise ValueError("new_to_old must list every node, block by block")
    new_ptr, new_idx = permute_csr_chunked(indptr, indices, new_to_old, graph_dir, chunk_entries=chunk_entries)
    _save(graph_dir / "node_type.npy", np.asarray(node_type, dtype=np.int8)[new_to_old])
    attrs_out = take_rows(attrs, new_to_old, graph_dir / "attrs.npy", dtype=attrs_dtype, chunk_rows=chunk_rows)
    attr_dim = int(attrs_out.shape[1]) if attrs_out.ndim == 2 else 0
    attr_dtype = str(attrs_out.dtype)
    close_memmap(attrs_out)
    del attrs_out
    if classes is not None:
        _save(graph_dir / "class.npy", np.asarray(classes, dtype=np.int32)[new_to_old])
    source_ids = np.arange(num_nodes, dtype=np.int64) if orig_id is None else np.asarray(orig_id, dtype=np.int64)
    _save(graph_dir / "orig_id.npy", source_ids[new_to_old])
    new_fine = None if node_fine is None else np.asarray(node_fine, dtype=np.int32)[new_to_old]
    write_partition_ooc(out / "part" / PRIMARY, node_block=new_block, node_fine=new_fine, indptr=new_ptr,
                        indices=new_idx, info=dict(partition_info or {}), chunk_entries=chunk_entries)
    full_meta = {
        "store_version": STORE_VERSION,
        "dataset": dataset,
        "num_nodes": num_nodes,
        "num_undirected_edges": int(new_idx.shape[0] // 2),
        "attr_dim": attr_dim,
        "attr_dtype": attr_dtype,
        "num_types": int(np.asarray(node_type).max()) + 1,
        "has_class": classes is not None,
        "partitions": [PRIMARY],
    }
    close_memmap(new_idx)
    full_meta.update(dict(meta or {}))
    (out / "meta.json").write_text(json.dumps(full_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def add_partition_ooc(store_dir: str | os.PathLike, name: str, *, node_block: np.ndarray,
                      node_fine: np.ndarray | None, info: Mapping[str, Any],
                      chunk_entries: int = CHUNK_ENTRIES) -> None:
    """store.add_partition with the store CSR memory-mapped."""
    store_dir = Path(store_dir)
    indptr = np.load(store_dir / "graph" / "indptr.npy", mmap_mode="r")
    indices = np.load(store_dir / "graph" / "indices.npy", mmap_mode="r")
    write_partition_ooc(store_dir / "part" / name, node_block=node_block, node_fine=node_fine, indptr=indptr,
                        indices=indices, info=info, chunk_entries=chunk_entries)
    meta_path = store_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if name not in meta["partitions"]:
        meta["partitions"].append(name)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
