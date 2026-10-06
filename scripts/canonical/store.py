"""Canonical on-disk graph store, its writer and its loader.

One set of .npy arrays serves both serving modes: `resident` loads them into
RAM, `streaming` memory-maps them. The candidate code below is identical in the
two modes, so the modes differ only in where bytes come from.

Layout under <root>/<dataset>/ (canonical node ids are contiguous per block of
the primary partition, so reading a block's adjacency is one sequential slice):

  meta.json
  graph/indptr.npy        int64 [N+1]  symmetric CSR, sorted rows, no self loops or duplicates
  graph/indices.npy       int32 [2E]
  graph/node_type.npy     int8  [N]
  graph/attrs.npy         [N, d] intrinsic attributes (dtype from the schema)
  graph/class.npy         int32 [N] class or venue, -1 when absent (optional)
  graph/orig_id.npy       int64 [N] id in the source dataset
  part/<name>/block_ptr.npy    int64 [P+1] offsets into block_nodes
  part/<name>/block_nodes.npy  int32 [N]  nodes grouped by block (arange(N) for "primary")
  part/<name>/node_block.npy   int32 [N]
  part/<name>/fine_ptr.npy, fine_nodes.npy, node_fine.npy, fine_block.npy  (fine level)
  part/<name>/bgraph_ptr.npy, bgraph_idx.npy, bgraph_w.npy  block graph, w = cut edges
  sigma/<regime>/...      written by sigma.py
  rankers/<regime>/...    written by rankers.py
  models/<model_id>/...   block embeddings written by the trainer

Candidate for a block set S (paper definition):
  C(S) = { v in V_S u N(V_S) : sigma(v) in sigma(V_Q) },  G[C(S)] induced.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

STORE_VERSION = 1
PRIMARY = "primary"


# ----------------------------------------------------------------------------
# Local graphs
# ----------------------------------------------------------------------------
@dataclass
class LocalGraph:
    """An induced subgraph with local ids 0..n-1 (sorted by global id)."""

    global_ids: np.ndarray  # int64 [n], strictly increasing
    indptr: np.ndarray  # int64 [n+1]
    indices: np.ndarray  # int32 local neighbour ids, sorted within each row
    labels: np.ndarray | None = None  # int32 [n] sigma codes
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.global_ids.shape[0])

    @property
    def num_undirected_edges(self) -> int:
        return int(self.indices.shape[0]) // 2

    def degrees(self) -> np.ndarray:
        return np.diff(self.indptr)

    def subgraph(self, keep: np.ndarray) -> "LocalGraph":
        """Induced subgraph on the local nodes where keep is True."""
        keep = np.asarray(keep, dtype=bool)
        new_id = np.full(self.n, -1, dtype=np.int64)
        kept = np.flatnonzero(keep)
        new_id[kept] = np.arange(kept.shape[0], dtype=np.int64)
        lens = np.diff(self.indptr)
        rows = np.repeat(np.arange(self.n, dtype=np.int64), lens)
        mask = keep[rows] & keep[self.indices]
        new_rows = new_id[rows[mask]]
        new_cols = new_id[self.indices[mask]].astype(np.int32)
        counts = np.bincount(new_rows, minlength=kept.shape[0]).astype(np.int64)
        indptr = np.zeros(kept.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return LocalGraph(
            global_ids=self.global_ids[kept],
            indptr=indptr,
            indices=new_cols,
            labels=None if self.labels is None else self.labels[kept],
        )

    def edge_list(self) -> np.ndarray:
        """Undirected edges (u < v) in local ids, shape [m, 2]."""
        lens = np.diff(self.indptr)
        rows = np.repeat(np.arange(self.n, dtype=np.int64), lens)
        cols = self.indices.astype(np.int64)
        mask = rows < cols
        return np.stack([rows[mask], cols[mask]], axis=1)

    def to_scipy(self):
        from scipy import sparse

        data = np.ones(self.indices.shape[0], dtype=np.int8)
        return sparse.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

    def components(self) -> tuple[int, np.ndarray]:
        from scipy.sparse.csgraph import connected_components

        if self.n == 0:
            return 0, np.zeros(0, dtype=np.int32)
        count, labels = connected_components(self.to_scipy(), directed=False)
        return int(count), labels.astype(np.int32)


def local_graph_from_edges(num_nodes: int, edges: np.ndarray, global_ids: np.ndarray | None = None,
                           labels: np.ndarray | None = None) -> LocalGraph:
    """Build a LocalGraph from an undirected edge list over 0..num_nodes-1."""
    indptr, indices = csr_from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), num_nodes)
    gids = np.arange(num_nodes, dtype=np.int64) if global_ids is None else np.asarray(global_ids, dtype=np.int64)
    return LocalGraph(global_ids=gids, indptr=indptr, indices=indices,
                      labels=None if labels is None else np.asarray(labels, dtype=np.int32))


# ----------------------------------------------------------------------------
# CSR helpers
# ----------------------------------------------------------------------------
def csr_from_edges(edges: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric, sorted, duplicate-free CSR without self loops from any edge list."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    u, v = edges[:, 0], edges[:, 1]
    keep = u != v
    u, v = u[keep], v[keep]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    keys = np.unique(lo * np.int64(num_nodes) + hi)
    lo, hi = keys // num_nodes, keys % num_nodes
    rows = np.concatenate([lo, hi])
    cols = np.concatenate([hi, lo])
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    counts = np.bincount(rows, minlength=num_nodes).astype(np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    index_dtype = np.int32 if num_nodes < 2**31 else np.int64
    return indptr, cols.astype(index_dtype)


def ranges_of_sorted(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a sorted node array into maximal runs of consecutive ids: (starts, ends)."""
    if nodes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(nodes) != 1) + 1
    starts = nodes[np.concatenate([[0], breaks])].astype(np.int64)
    ends = nodes[np.concatenate([breaks - 1, [nodes.shape[0] - 1]])].astype(np.int64) + 1
    return starts, ends


def gather_rows(indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray,
                max_ranges_for_slicing: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour lists of `nodes` (sorted): returns (row lengths, concatenated neighbours).

    When the nodes form few runs of consecutive ids (whole blocks of the primary
    partition), each run is read as one contiguous slice, which keeps memory-
    mapped reads sequential.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=indices.dtype)
    starts_ptr = np.asarray(indptr[nodes], dtype=np.int64)
    ends_ptr = np.asarray(indptr[nodes + 1], dtype=np.int64)
    lens = ends_ptr - starts_ptr
    run_starts, run_ends = ranges_of_sorted(nodes)
    if run_starts.shape[0] <= max_ranges_for_slicing:
        parts = [np.asarray(indices[int(indptr[a]):int(indptr[b])]) for a, b in zip(run_starts, run_ends)]
        nbrs = np.concatenate(parts) if parts else np.zeros(0, dtype=indices.dtype)
        return lens, nbrs
    total = int(lens.sum())
    offsets = np.cumsum(lens) - lens
    flat = np.arange(total, dtype=np.int64) - np.repeat(offsets, lens) + np.repeat(starts_ptr, lens)
    return lens, np.asarray(indices[flat])


def isin_sorted(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    """Vectorised membership test of values in a sorted unique array."""
    if sorted_set.shape[0] == 0:
        return np.zeros(values.shape[0], dtype=bool)
    pos = np.searchsorted(sorted_set, values)
    pos = np.minimum(pos, sorted_set.shape[0] - 1)
    return sorted_set[pos] == values


def induced_from_rows(nodes: np.ndarray, lens: np.ndarray, nbrs: np.ndarray,
                      labels: np.ndarray | None = None) -> LocalGraph:
    """Induced LocalGraph on sorted `nodes` from their gathered neighbour rows."""
    nodes = np.asarray(nodes, dtype=np.int64)
    nbrs64 = nbrs.astype(np.int64, copy=False)
    inside = isin_sorted(nbrs64, nodes)
    local = np.searchsorted(nodes, nbrs64[inside]).astype(np.int32)
    rows = np.repeat(np.arange(nodes.shape[0], dtype=np.int64), lens)[inside]
    counts = np.bincount(rows, minlength=nodes.shape[0]).astype(np.int64)
    indptr = np.zeros(nodes.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return LocalGraph(global_ids=nodes, indptr=indptr, indices=local, labels=labels)


# ----------------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------------
def block_order(node_block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable renumbering that makes blocks contiguous.

    Returns (new_to_old, old_to_new, block_ptr).
    """
    node_block = np.asarray(node_block, dtype=np.int64)
    num_blocks = int(node_block.max()) + 1 if node_block.shape[0] else 0
    new_to_old = np.argsort(node_block, kind="stable").astype(np.int64)
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(new_to_old.shape[0], dtype=np.int64)
    counts = np.bincount(node_block, minlength=num_blocks).astype(np.int64)
    block_ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    np.cumsum(counts, out=block_ptr[1:])
    return new_to_old, old_to_new, block_ptr


def permute_csr(indptr: np.ndarray, indices: np.ndarray, new_to_old: np.ndarray,
                old_to_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lens_old = np.diff(indptr)
    lens_new = lens_old[new_to_old]
    new_indptr = np.zeros(indptr.shape[0], dtype=np.int64)
    np.cumsum(lens_new, out=new_indptr[1:])
    rows_new = np.repeat(np.arange(new_to_old.shape[0], dtype=np.int64), lens_new)
    lens, nbrs = gather_rows(indptr, indices, new_to_old, max_ranges_for_slicing=0)
    cols_new = old_to_new[nbrs.astype(np.int64)]
    order = np.lexsort((cols_new, rows_new))
    return new_indptr, cols_new[order].astype(indices.dtype)


def block_graph(indptr: np.ndarray, indices: np.ndarray, node_block: np.ndarray,
                num_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block adjacency with w = number of cut edges between two blocks (CSR, symmetric)."""
    lens = np.diff(indptr)
    rows = np.repeat(np.arange(lens.shape[0], dtype=np.int64), lens)
    bu = np.asarray(node_block, dtype=np.int64)[rows]
    bv = np.asarray(node_block, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    cut = bu != bv
    keys = bu[cut] * np.int64(num_blocks) + bv[cut]
    uniq, counts = np.unique(keys, return_counts=True)
    src, dst = uniq // num_blocks, uniq % num_blocks
    ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=num_blocks), out=ptr[1:])
    return ptr, dst.astype(np.int32), counts.astype(np.int64)


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def write_partition(part_dir: Path, *, node_block: np.ndarray, node_fine: np.ndarray | None,
                    indptr: np.ndarray, indices: np.ndarray, info: Mapping[str, Any]) -> None:
    node_block = np.asarray(node_block, dtype=np.int32)
    num_blocks = int(node_block.max()) + 1
    order = np.argsort(node_block, kind="stable")
    counts = np.bincount(node_block, minlength=num_blocks).astype(np.int64)
    block_ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    np.cumsum(counts, out=block_ptr[1:])
    _save(part_dir / "block_ptr.npy", block_ptr)
    _save(part_dir / "block_nodes.npy", order.astype(np.int32))
    _save(part_dir / "node_block.npy", node_block)
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
    bptr, bidx, bw = block_graph(indptr, indices, node_block, num_blocks)
    _save(part_dir / "bgraph_ptr.npy", bptr)
    _save(part_dir / "bgraph_idx.npy", bidx)
    _save(part_dir / "bgraph_w.npy", bw)
    sizes = counts
    meta = dict(info)
    meta.update({
        "num_blocks": num_blocks,
        "num_fine": int(node_fine.max()) + 1 if node_fine is not None else 0,
        "block_size_min": int(sizes.min()),
        "block_size_max": int(sizes.max()),
        "block_size_mean": float(sizes.mean()),
        "cut_edges": int(bw.sum() // 2),
    })
    (part_dir / "partition.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_store(
    root: str | os.PathLike,
    dataset: str,
    *,
    edges: np.ndarray,
    num_nodes: int,
    node_type: np.ndarray,
    attrs: np.ndarray,
    node_block: np.ndarray,
    node_fine: np.ndarray | None,
    classes: np.ndarray | None = None,
    orig_id: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
    partition_info: Mapping[str, Any] | None = None,
) -> Path:
    """Normalise a graph, renumber it block-contiguously and write the store.

    `edges` may be directed, duplicated or contain self loops; the store holds
    the simple undirected graph. All per-node inputs use the source ids.
    """
    out = Path(root) / dataset
    graph_dir = out / "graph"
    indptr_src, indices_src = csr_from_edges(edges, num_nodes)
    new_to_old, old_to_new, block_ptr = block_order(node_block)
    indptr, indices = permute_csr(indptr_src, indices_src, new_to_old, old_to_new)
    _save(graph_dir / "indptr.npy", indptr)
    _save(graph_dir / "indices.npy", indices)
    _save(graph_dir / "node_type.npy", np.asarray(node_type, dtype=np.int8)[new_to_old])
    _save(graph_dir / "attrs.npy", np.asarray(attrs)[new_to_old])
    if classes is not None:
        _save(graph_dir / "class.npy", np.asarray(classes, dtype=np.int32)[new_to_old])
    source_ids = np.arange(num_nodes, dtype=np.int64) if orig_id is None else np.asarray(orig_id, dtype=np.int64)
    _save(graph_dir / "orig_id.npy", source_ids[new_to_old])
    new_block = np.asarray(node_block, dtype=np.int32)[new_to_old]
    new_fine = None if node_fine is None else np.asarray(node_fine, dtype=np.int32)[new_to_old]
    write_partition(out / "part" / PRIMARY, node_block=new_block, node_fine=new_fine,
                    indptr=indptr, indices=indices, info=dict(partition_info or {}))
    full_meta = {
        "store_version": STORE_VERSION,
        "dataset": dataset,
        "num_nodes": int(num_nodes),
        "num_undirected_edges": int(indices.shape[0] // 2),
        "attr_dim": int(np.asarray(attrs).shape[1]) if np.asarray(attrs).ndim == 2 else 0,
        "attr_dtype": str(np.asarray(attrs).dtype),
        "num_types": int(np.asarray(node_type).max()) + 1,
        "has_class": classes is not None,
        "partitions": [PRIMARY],
    }
    full_meta.update(dict(meta or {}))
    (out / "meta.json").write_text(json.dumps(full_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def add_partition(store_dir: str | os.PathLike, name: str, *, node_block: np.ndarray,
                  node_fine: np.ndarray | None, info: Mapping[str, Any]) -> None:
    """Add an alternative partition (canonical ids) for sensitivity runs."""
    store_dir = Path(store_dir)
    indptr = np.load(store_dir / "graph" / "indptr.npy")
    indices = np.load(store_dir / "graph" / "indices.npy")
    write_partition(store_dir / "part" / name, node_block=node_block, node_fine=node_fine,
                    indptr=indptr, indices=indices, info=info)
    meta_path = store_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if name not in meta["partitions"]:
        meta["partitions"].append(name)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------
class Partition:
    def __init__(self, part_dir: Path, mmap: bool) -> None:
        mode = "r" if mmap else None
        self.dir = part_dir
        self.info = json.loads((part_dir / "partition.json").read_text(encoding="utf-8"))
        self.block_ptr = np.load(part_dir / "block_ptr.npy", mmap_mode=mode)
        self.block_nodes = np.load(part_dir / "block_nodes.npy", mmap_mode=mode)
        self.node_block = np.load(part_dir / "node_block.npy", mmap_mode=mode)
        self.bgraph_ptr = np.load(part_dir / "bgraph_ptr.npy")
        self.bgraph_idx = np.load(part_dir / "bgraph_idx.npy")
        self.bgraph_w = np.load(part_dir / "bgraph_w.npy")
        fine = part_dir / "fine_ptr.npy"
        self.has_fine = fine.exists()
        if self.has_fine:
            self.fine_ptr = np.load(fine, mmap_mode=mode)
            self.fine_nodes = np.load(part_dir / "fine_nodes.npy", mmap_mode=mode)
            self.node_fine = np.load(part_dir / "node_fine.npy", mmap_mode=mode)
            self.fine_block = np.load(part_dir / "fine_block.npy", mmap_mode=mode)
        self.contiguous = bool(self.info.get("contiguous", part_dir.name == PRIMARY))

    @property
    def num_blocks(self) -> int:
        return int(self.block_ptr.shape[0]) - 1

    def block_neighbors(self, block: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.bgraph_ptr[block]), int(self.bgraph_ptr[block + 1])
        return self.bgraph_idx[a:b], self.bgraph_w[a:b]

    def nodes_of_blocks(self, blocks: Iterable[int]) -> np.ndarray:
        blocks = np.unique(np.asarray(list(blocks), dtype=np.int64))
        if blocks.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        if self.contiguous:
            parts = [np.arange(int(self.block_ptr[b]), int(self.block_ptr[b + 1]), dtype=np.int64) for b in blocks]
            return np.concatenate(parts)
        parts = [np.asarray(self.block_nodes[int(self.block_ptr[b]):int(self.block_ptr[b + 1])], dtype=np.int64)
                 for b in blocks]
        return np.sort(np.concatenate(parts))

    def fine_nodes_of(self, fine: int) -> np.ndarray:
        a, b = int(self.fine_ptr[fine]), int(self.fine_ptr[fine + 1])
        return np.sort(np.asarray(self.fine_nodes[a:b], dtype=np.int64))


class GraphStore:
    """Loader for one dataset. mmap=True is the streaming mode."""

    def __init__(self, root: str | os.PathLike, dataset: str, *, mmap: bool = False,
                 partition: str = PRIMARY) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.dir = self.root / dataset
        self.mmap = mmap
        mode = "r" if mmap else None
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        g = self.dir / "graph"
        self.indptr = np.load(g / "indptr.npy", mmap_mode=mode)
        self.indices = np.load(g / "indices.npy", mmap_mode=mode)
        self.node_type = np.load(g / "node_type.npy", mmap_mode=mode)
        self.attrs = np.load(g / "attrs.npy", mmap_mode=mode)
        cls = g / "class.npy"
        self.classes = np.load(cls, mmap_mode=mode) if cls.exists() else None
        self.partition_name = partition
        self.part = Partition(self.dir / "part" / partition, mmap)
        self._sigma: dict[str, Any] = {}

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.shape[0]) - 1

    @property
    def num_blocks(self) -> int:
        return self.part.num_blocks

    def degree(self, nodes: np.ndarray) -> np.ndarray:
        nodes = np.asarray(nodes, dtype=np.int64)
        return np.asarray(self.indptr[nodes + 1], dtype=np.int64) - np.asarray(self.indptr[nodes], dtype=np.int64)

    def neighbors(self, node: int) -> np.ndarray:
        return np.asarray(self.indices[int(self.indptr[node]):int(self.indptr[node + 1])])

    def has_edge(self, u: int, v: int) -> bool:
        row = self.neighbors(u)
        pos = np.searchsorted(row, v)
        return bool(pos < row.shape[0] and row[pos] == v)

    def induced(self, nodes: np.ndarray, regime: str | None = None) -> LocalGraph:
        """G[nodes] for sorted unique global ids; labels are sigma codes when a regime is given."""
        nodes = np.asarray(nodes, dtype=np.int64)
        lens, nbrs = gather_rows(self.indptr, self.indices, nodes)
        labels = None if regime is None else np.asarray(self.sigma(regime).codes[nodes], dtype=np.int32)
        return induced_from_rows(nodes, lens, nbrs, labels=labels)

    def block(self, block: int, regime: str | None = None) -> LocalGraph:
        """G[V_b], the graph a block embedding encodes."""
        return self.induced(self.part.nodes_of_blocks([block]), regime)

    def sigma(self, regime: str):
        from .sigma import SigmaTable

        if regime not in self._sigma:
            self._sigma[regime] = SigmaTable.load(self.dir / "sigma" / regime, mmap=self.mmap)
        return self._sigma[regime]

    def files(self) -> list[Path]:
        return sorted(p for p in self.dir.rglob("*.npy"))

    def open_arrays(self) -> list[np.ndarray]:
        arrays = [self.indptr, self.indices, self.node_type, self.attrs]
        if self.classes is not None:
            arrays.append(self.classes)
        arrays += [self.part.block_ptr, self.part.block_nodes, self.part.node_block]
        for table in self._sigma.values():
            arrays += table.open_arrays()
        return arrays

    def release(self) -> int:
        """Drop page-table references of memory-mapped arrays (for cold runs)."""
        if not self.mmap:
            return 0
        from .memprobe import release_mapping

        return sum(1 for array in self.open_arrays() if release_mapping(array))

    # ------------------------------------------------------------------ candidates
    def candidate(self, blocks: Iterable[int], query_codes: np.ndarray, regime: str,
                  stats: dict[str, Any] | None = None, *, halo: bool | int = True,
                  label_filter: bool = True) -> LocalGraph:
        """G[C(S)] for block set S: block nodes plus their halo, sigma-filtered.

        halo is the number of hops: True (1) is the canonical one-hop halo, False
        (0) drops it (operator ablation, E9) and 2 is the E16 sensitivity setting.
        Hop h+1 grows from the sigma-kept nodes of hop h. An embedding is connected
        and every node of it carries a query label, so growing through kept nodes
        reaches every node of an embedding that growing through all nodes reaches
        from the blocks, and it reads fewer rows. label_filter=False keeps nodes of
        every label (E9) and grows through all of them.
        """
        hops = int(halo)
        if hops < 0:
            raise ValueError(f"halo hops must be >= 0, got {halo!r}")
        codes = self.sigma(regime).codes
        allowed = np.unique(np.asarray(query_codes, dtype=np.int64))
        vs = self.part.nodes_of_blocks(blocks)
        lens_s, nbrs_s = gather_rows(self.indptr, self.indices, vs)

        def kept(nodes: np.ndarray) -> np.ndarray:
            if not label_filter or nodes.shape[0] == 0:
                return np.asarray(nodes, dtype=np.int64)
            return nodes[isin_sorted(np.asarray(codes[nodes], dtype=np.int64), allowed)]

        c_s = kept(vs)
        # Each ring is read once: its kept rows give the next ring and the candidate's edges.
        seen, frontier, ring_nodes = vs, nbrs_s, 0
        rings: list[np.ndarray] = []
        ring_rows: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(hops):
            ring = np.setdiff1d(np.unique(frontier.astype(np.int64)), seen, assume_unique=True)
            if ring.shape[0] == 0:
                break
            ring_nodes += int(ring.shape[0])
            seen = np.union1d(seen, ring)
            c_ring = kept(ring)
            rows = gather_rows(self.indptr, self.indices, c_ring)
            rings.append(c_ring)
            ring_rows.append(rows)
            frontier = rows[1]
        if rings:
            c_h = np.concatenate(rings)
            lens_ch = np.concatenate([r[0] for r in ring_rows])
            nbrs_ch = np.concatenate([r[1] for r in ring_rows])
        else:
            c_h = np.zeros(0, dtype=np.int64)
            lens_ch, nbrs_ch = gather_rows(self.indptr, self.indices, c_h)
        nodes = np.union1d(c_s, c_h)
        # Rows of kept block nodes come from the rows already read.
        keep_rows = isin_sorted(vs, c_s)
        row_owner = np.repeat(keep_rows, lens_s)
        lens_cs = lens_s[keep_rows]
        nbrs_cs = nbrs_s[row_owner]
        all_nodes = np.concatenate([c_s, c_h])
        all_lens = np.concatenate([lens_cs, lens_ch]).astype(np.int64)
        starts = np.cumsum(all_lens) - all_lens
        all_nbrs = np.concatenate([nbrs_cs, nbrs_ch])
        order = np.argsort(all_nodes, kind="stable")
        lens_sorted = all_lens[order]
        nbrs_sorted = all_nbrs[_reorder_segments(starts, all_lens, order)] if all_nbrs.shape[0] else all_nbrs
        if not np.array_equal(all_nodes[order], nodes):
            raise AssertionError("candidate node bookkeeping is inconsistent")
        graph = induced_from_rows(nodes, lens_sorted, nbrs_sorted,
                                  labels=np.asarray(codes[nodes], dtype=np.int32))
        if stats is not None:
            stats.update({
                "block_nodes": int(vs.shape[0]),
                "halo_nodes": ring_nodes,
                "sigma_kept_block": int(c_s.shape[0]),
                "sigma_kept_halo": int(c_h.shape[0]),
                "candidate_nodes": int(graph.n),
                "candidate_edges": int(graph.num_undirected_edges),
                "adjacency_entries_read": int(nbrs_s.shape[0] + nbrs_ch.shape[0]),
            })
        return graph


def _reorder_segments(starts: np.ndarray, lens: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Flat indices that concatenate segments (starts[i], lens[i]) in the given order."""
    seg_lens = lens[order]
    seg_starts = starts[order]
    total = int(seg_lens.sum())
    offsets = np.cumsum(seg_lens) - seg_lens
    return np.arange(total, dtype=np.int64) - np.repeat(offsets, seg_lens) + np.repeat(seg_starts, seg_lens)
