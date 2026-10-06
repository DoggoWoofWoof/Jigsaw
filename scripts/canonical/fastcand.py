"""Candidate construction in linear time (the cascade's two graph-building call sites).

GraphStore.candidate and GraphStore.induced build G[C(S)] with set operations that sort: unique,
union1d, argsort, and two binary searches per gathered adjacency entry. The same graphs fall out of
membership bitmaps over the node ids and counting-sort scatters, which cost a few passes over the
gathered entries each. candidate() and induced() return what the store methods return, array for
array (tests/canonical/test_fastcand.py), and fill the same stats. They only read the store; the
bitmaps are allocated per call (zeroed lazily by the allocator, so a call touches the pages of its own
nodes), and nothing of size |V| outlives it.

Used by the cascade for every arm that builds a candidate (FilterAll, the index and the learned
ladders), so a comparison between arms never mixes the two implementations.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .store import GraphStore, LocalGraph, ranges_of_sorted


def gather_rows(indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray,
                max_ranges_for_slicing: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """store.gather_rows with one repeat in the scattered path instead of two."""
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=indices.dtype)
    starts_ptr = np.asarray(indptr[nodes], dtype=np.int64)
    lens = np.asarray(indptr[nodes + 1], dtype=np.int64) - starts_ptr
    run_starts, run_ends = ranges_of_sorted(nodes)
    if run_starts.shape[0] <= max_ranges_for_slicing:
        parts = [np.asarray(indices[int(indptr[a]):int(indptr[b])]) for a, b in zip(run_starts, run_ends)]
        return lens, np.concatenate(parts) if parts else np.zeros(0, dtype=indices.dtype)
    total = int(lens.sum())
    offsets = np.cumsum(lens) - lens
    flat = np.arange(total, dtype=np.int64)
    flat += np.repeat(starts_ptr - offsets, lens)
    return lens, np.asarray(indices[flat])


def _allowed_table(codes_count: int, allowed: np.ndarray) -> np.ndarray:
    """Boolean table over sigma codes: True for the codes a query carries (fresh codes match no node)."""
    table = np.zeros(codes_count, dtype=bool)
    allowed = np.asarray(allowed, dtype=np.int64)
    table[allowed[(allowed >= 0) & (allowed < codes_count)]] = True
    return table


def _csr_rows(num_nodes: int, rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], member: np.ndarray,
              pos: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR (int64 indptr, int32 local indices) of G[nodes] from groups of gathered rows.

    rows: (row nodes, row lengths, concatenated neighbours) per group; the row nodes of all groups are
    exactly `nodes`. member marks `nodes` over the node ids and pos maps a node to its local id.
    Entries keep the order of the stored rows, so a row stays sorted when the stored row is.
    """
    n = int(nodes.shape[0])
    kept: list[np.ndarray] = []
    counts_by_group: list[np.ndarray] = []
    row_counts = np.zeros(n, dtype=np.int64)
    for row_nodes, lens, nbrs in rows:
        inside = member[nbrs]
        kept.append(nbrs[inside])
        ends = np.cumsum(lens)
        running = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(inside, dtype=np.int64)])
        counts = running[ends] - running[ends - lens]
        counts_by_group.append(counts)
        row_counts[pos[row_nodes]] = counts
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(row_counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for (row_nodes, _, _), entries, counts in zip(rows, kept, counts_by_group):
        if entries.shape[0] == 0:
            continue
        group_start = np.cumsum(counts) - counts
        dest = np.arange(entries.shape[0], dtype=np.int64)
        dest += np.repeat(indptr[pos[row_nodes]] - group_start, counts)
        indices[dest] = pos[entries]
    return indptr, indices


def induced(store: GraphStore, nodes: np.ndarray, regime: str | None = None) -> LocalGraph:
    """store.induced(nodes, regime): G[nodes] for sorted unique global ids."""
    nodes = np.asarray(nodes, dtype=np.int64)
    n, total = int(nodes.shape[0]), store.num_nodes
    lens, nbrs = gather_rows(store.indptr, store.indices, nodes)
    member = np.zeros(total, dtype=bool)
    member[nodes] = True
    pos = np.empty(total, dtype=np.int32)
    pos[nodes] = np.arange(n, dtype=np.int32)
    indptr, indices = _csr_rows(total, [(nodes, lens, nbrs)], member, pos, nodes)
    labels = None if regime is None else np.asarray(store.sigma(regime).codes[nodes], dtype=np.int32)
    return LocalGraph(global_ids=nodes, indptr=indptr, indices=indices, labels=labels)


def candidate(store: GraphStore, blocks: Iterable[int], query_codes: np.ndarray, regime: str,
              stats: dict[str, Any] | None = None, *, halo: bool | int = True,
              label_filter: bool = True) -> LocalGraph:
    """store.candidate(...): G[C(S)] for the block set S, block nodes plus their sigma-kept halo."""
    hops = int(halo)
    if hops < 0:
        raise ValueError(f"halo hops must be >= 0, got {halo!r}")
    table = store.sigma(regime)
    codes = table.codes
    total = store.num_nodes
    allowed = _allowed_table(table.num_codes, query_codes) if label_filter else None
    vs = store.part.nodes_of_blocks(blocks)
    lens_s, nbrs_s = gather_rows(store.indptr, store.indices, vs)

    def kept_of(nodes: np.ndarray) -> np.ndarray:
        if allowed is None or nodes.shape[0] == 0:
            return np.ones(nodes.shape[0], dtype=bool)
        return allowed[np.asarray(codes[nodes], dtype=np.int64)]

    keep_s = kept_of(vs)
    c_s = vs[keep_s]
    seen = np.zeros(total, dtype=bool)
    seen[vs] = True
    frontier, ring_nodes, entries_read = nbrs_s, 0, int(nbrs_s.shape[0])
    rings: list[np.ndarray] = []
    ring_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(hops):
        fresh = frontier[~seen[frontier]]
        if fresh.shape[0] == 0:
            break
        flag = np.zeros(total, dtype=bool)
        flag[fresh] = True
        ring = np.flatnonzero(flag)
        ring_nodes += int(ring.shape[0])
        seen |= flag
        c_ring = ring[kept_of(ring)]
        lens, nbrs = gather_rows(store.indptr, store.indices, c_ring)
        rings.append(c_ring)
        ring_rows.append((c_ring, lens, nbrs))
        entries_read += int(nbrs.shape[0])
        frontier = nbrs
    member = np.zeros(total, dtype=bool)
    member[c_s] = True
    for c_ring in rings:
        member[c_ring] = True
    nodes = np.flatnonzero(member)
    pos = np.empty(total, dtype=np.int32)
    pos[nodes] = np.arange(nodes.shape[0], dtype=np.int32)
    groups = [(c_s, lens_s[keep_s], nbrs_s[np.repeat(keep_s, lens_s)])] + ring_rows
    indptr, indices = _csr_rows(total, groups, member, pos, nodes)
    graph = LocalGraph(global_ids=nodes, indptr=indptr, indices=indices,
                       labels=np.asarray(codes[nodes], dtype=np.int32))
    if stats is not None:
        stats.update({
            "block_nodes": int(vs.shape[0]),
            "halo_nodes": ring_nodes,
            "sigma_kept_block": int(c_s.shape[0]),
            "sigma_kept_halo": int(sum(r.shape[0] for r in rings)),
            "candidate_nodes": int(graph.n),
            "candidate_edges": int(graph.num_undirected_edges),
            "adjacency_entries_read": entries_read,
        })
    return graph
