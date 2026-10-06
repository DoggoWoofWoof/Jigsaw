"""Planted query families, negative queries and query-set files.

A query is the induced subgraph of a connected planted node set of the data
graph (sparse sm_standard queries keep only the walk's edges), with the label
payload of one regime. The families port the legacy generators in
scripts/benchmark_glasgow.py and scripts/benchmark_overlap_glasgow_cascade.py
to the canonical store:

  single        BFS fragment inside one block
  multi_fine    fragments of 2-4 adjacent fine parts of one block, joined at bridge edges
  multi_coarse  fragments of adjacent fine parts spanning at least two blocks
  k_hop         BFS blob within 3 hops of a uniform anchor
  random_walk   first-visit order of a random walk
  degree_k_hop  k_hop from an anchor in one degree tertile (query index mod 3)
  sm_any, sm_dense, sm_sparse
                random-walk queries of 4-32 nodes in the style of Sun and Luo
                (SIGMOD 2020): dense = induced with average degree >= 3,
                sparse = walk edges with average degree < 3

Negatives start from a planted positive and change it so that it may have no
embedding. certify_negatives.py decides every negative exactly before use.
  neg_absent_label   one node gets a label key that no data node carries
  neg_swapped_label  one node gets the key of a same-type data node near its image
  neg_structure      one query non-edge is added between two node types that meet
                     on a data edge, preferring pairs at distance 2; info records
                     whether the pair's sigma labels meet on a data edge

Query i of a cell draws from numpy default_rng(derive_seed(master, dataset,
stream, family, size, i, attempt)), so each query is reproducible on its own and
independent of generation order; a failed or duplicate draw moves to the next
attempt. A draw is a duplicate when its node set is taken, or when its query is
isomorphic to an earlier query of the split in a regime whose set holds both (a
node bijection keeps every edge, type and label, so both have the same matches:
on mag, stars of featureless nodes do this). A negative whose modification is
isomorphic to an earlier query of its regime moves to its next modification
attempt; `queries duplicates PATH` checks saved sets. Query nodes are a random
permutation of the planted nodes: for a positive, query node j maps to
planted[j]. Node sets are unique within a split, and a later split excludes the
node sets of earlier ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from . import features, protocol, verify
from .store import GraphStore, LocalGraph, csr_from_edges, gather_rows, isin_sorted, local_graph_from_edges

GENERATOR_VERSION = "queries-v1"
MULTI_COARSE_CONFIGS = [(2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (5, 2), (5, 3), (5, 4), (6, 3), (6, 4)]
SM_FAMILIES = {"any": "sm_any", "dense": "sm_dense", "sparse": "sm_sparse"}
TEST_SPLITS = ("test", "reserve_a", "reserve_b")


class QueryGenerationError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def normalize_edges(edges: np.ndarray) -> np.ndarray:
    """Undirected edges as unique (u < v) rows in lexicographic order, no self loops."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    if e.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.sort(e, axis=1), axis=0)


def split_counts_for_fragments(total: int, parts: int, min_each: int = 3) -> list[int]:
    if parts <= 0:
        return []
    base, rem = divmod(total, parts)
    return [max(min_each, base + (1 if i < rem else 0)) for i in range(parts)]


def ordered_unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(v) for v in values))


def is_connected(n: int, edges: np.ndarray) -> bool:
    if n <= 1:
        return n == 1
    count, _ = local_graph_from_edges(n, edges).components()
    return count == 1


def nodeset_hash(nodes: np.ndarray) -> str:
    data = np.sort(np.asarray(nodes, dtype="<i8")).tobytes()
    return hashlib.blake2b(data, digest_size=16, person=b"jigsaw.nodeset").hexdigest()


def min_nodes_for(size: int) -> int:
    return max(5, int(size) - 10)


# ----------------------------------------------------------------------------
# Planted structures
# ----------------------------------------------------------------------------
@dataclass
class PlantedQuery:
    family: str
    size: int
    planted: np.ndarray  # int64 [n] data nodes; query node j -> planted[j]
    edges: np.ndarray  # int64 [m, 2] query edges, u < v, sorted
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.planted.shape[0])

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])


class QueryContext:
    """Samplers over one store. The trainer uses sample(); build_split() uses everything."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store
        self.indptr = store.indptr
        self.indices = store.indices
        self.num_nodes = store.num_nodes
        self.part = store.part
        self.node_block = self.part.node_block
        self.num_blocks = self.part.num_blocks
        self.has_fine = self.part.has_fine
        if self.has_fine:
            self.node_fine = self.part.node_fine
            fine_block = np.asarray(self.part.fine_block, dtype=np.int64)
            self._fine_order = np.argsort(fine_block, kind="stable")
            self._fine_ptr = np.zeros(self.num_blocks + 1, dtype=np.int64)
            np.cumsum(np.bincount(fine_block, minlength=self.num_blocks), out=self._fine_ptr[1:])
            self.fine_block = fine_block
        self._bgraph_ptr = np.asarray(self.part.bgraph_ptr, dtype=np.int64)
        self._bgraph_idx = np.asarray(self.part.bgraph_idx, dtype=np.int64)
        self._tertiles: list[np.ndarray] | None = None

    # ------------------------------------------------------------------ graph access
    def neighbors(self, u: int) -> np.ndarray:
        return np.asarray(self.indices[int(self.indptr[u]):int(self.indptr[u + 1])], dtype=np.int64)

    def induced_edges(self, nodes: np.ndarray) -> np.ndarray:
        """Edges of G[nodes] in the positions of `nodes` (u < v)."""
        nodes = np.asarray(nodes, dtype=np.int64)
        order = np.argsort(nodes, kind="stable")
        sorted_nodes = nodes[order]
        lens, nbrs = gather_rows(self.indptr, self.indices, sorted_nodes)
        nbrs = nbrs.astype(np.int64, copy=False)
        inside = isin_sorted(nbrs, sorted_nodes)
        rows = np.repeat(np.arange(nodes.shape[0], dtype=np.int64), lens)[inside]
        cols = np.searchsorted(sorted_nodes, nbrs[inside])
        keep = rows < cols
        return normalize_edges(np.stack([order[rows[keep]], order[cols[keep]]], axis=1))

    def bfs(self, start: int, target: int, rng: np.random.Generator, *,
            member: Callable[[np.ndarray], np.ndarray] | None = None, max_depth: int | None = None) -> list[int]:
        """BFS order from start with shuffled neighbour lists, restricted by member and depth."""
        out = [int(start)]
        seen = {int(start)}
        queue = deque([(int(start), 0)])
        while queue and len(out) < target:
            u, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            nb = self.neighbors(u)
            if member is not None and nb.shape[0]:
                nb = nb[member(nb)]
            if nb.shape[0] == 0:
                continue
            for v in nb[rng.permutation(nb.shape[0])].tolist():
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
                queue.append((v, depth + 1))
                if len(out) >= target:
                    break
        return out

    def fragment(self, member: Callable[[np.ndarray], np.ndarray], pool: np.ndarray, target: int, min_nodes: int,
                 rng: np.random.Generator, required: Iterable[int] = (), starts: int = 20) -> list[int] | None:
        """Connected BFS fragment inside a node pool that contains every required node."""
        if pool.shape[0] < min_nodes:
            return None
        req = [int(x) for x in required]
        tries = req + [int(pool[k]) for k in rng.integers(0, pool.shape[0], size=starts)]
        need = set(req)
        for s in tries:
            nodes = self.bfs(s, target, rng, member=member)
            if len(nodes) >= min_nodes and need.issubset(nodes):
                return nodes
        return None

    def finalize(self, family: str, size: int, nodes: Sequence[int], rng: np.random.Generator, *,
                 info: dict[str, Any] | None = None, edges: np.ndarray | None = None,
                 min_nodes: int | None = None, exact_size: bool = False) -> PlantedQuery | None:
        """Induce (or take the given edges), check the shape rules and permute the nodes."""
        nodes = np.asarray(ordered_unique(nodes), dtype=np.int64)
        if exact_size and nodes.shape[0] != size:
            return None
        if nodes.shape[0] > size:
            nodes = nodes[:size]
        if nodes.shape[0] < (min_nodes if min_nodes is not None else min_nodes_for(size)):
            return None
        e = self.induced_edges(nodes) if edges is None else normalize_edges(edges)
        if e.shape[0] < protocol.QUERY_DESIGN["min_edges"] or not is_connected(nodes.shape[0], e):
            return None
        n = nodes.shape[0]
        perm = rng.permutation(n)
        inverse = np.empty(n, dtype=np.int64)
        inverse[perm] = np.arange(n, dtype=np.int64)
        planted = nodes[perm]
        q_info = dict(info or {})
        q_info["blocks"] = np.unique(np.asarray(self.node_block[planted], dtype=np.int64)).tolist()
        if self.has_fine:
            q_info["fine"] = np.unique(np.asarray(self.node_fine[planted], dtype=np.int64)).tolist()
        return PlantedQuery(family, int(size), planted, normalize_edges(inverse[e]), q_info)

    # ------------------------------------------------------------------ fine parts
    def fines_of_block(self, block: int) -> np.ndarray:
        return self._fine_order[self._fine_ptr[block]:self._fine_ptr[block + 1]]

    def _fine_rows(self, fine: int, cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]):
        if fine not in cache:
            nodes = self.part.fine_nodes_of(fine)
            lens, nbrs = gather_rows(self.indptr, self.indices, nodes)
            src = np.repeat(nodes, lens)
            dst = nbrs.astype(np.int64, copy=False)
            fdst = np.asarray(self.node_fine[dst], dtype=np.int64)
            out = fdst != fine
            cache[fine] = (src[out], dst[out], fdst[out])
        return cache[fine]

    def _grow_fines(self, chosen: list[int], required: dict[int, set[int]], num_frags: int,
                    rng: np.random.Generator, cache: dict, block: int | None = None) -> None:
        visited = set(chosen)
        queue = deque(chosen)
        while queue and len(chosen) < num_frags:
            cur = queue.popleft()
            src, dst, fdst = self._fine_rows(cur, cache)
            cand = np.unique(fdst)
            if block is not None:
                cand = cand[self.fine_block[cand] == block]
            for g in cand[rng.permutation(cand.shape[0])].tolist():
                if g in visited:
                    continue
                idx = np.flatnonzero(fdst == g)
                k = int(idx[rng.integers(idx.shape[0])])
                required.setdefault(cur, set()).add(int(src[k]))
                required.setdefault(g, set()).add(int(dst[k]))
                visited.add(g)
                chosen.append(g)
                queue.append(g)
                if len(chosen) >= num_frags:
                    break

    def _fragments_of_fines(self, fines: list[int], required: dict[int, set[int]], size: int,
                            rng: np.random.Generator) -> list[int] | None:
        nodes: list[int] = []
        for fine, count in zip(fines, split_counts_for_fragments(size, len(fines), 3)):
            member = (lambda nb, f=fine: np.asarray(self.node_fine[nb]) == f)
            frag = self.fragment(member, self.part.fine_nodes_of(fine), count, max(3, count - 2), rng,
                                 required.get(fine, ()))
            if frag is None:
                return None
            nodes.extend(frag)
        return ordered_unique(nodes)

    # ------------------------------------------------------------------ families
    def single(self, size: int, rng: np.random.Generator) -> PlantedQuery | None:
        block = int(rng.integers(self.num_blocks))
        pool = self.part.nodes_of_blocks([block])
        member = (lambda nb: np.asarray(self.node_block[nb]) == block)
        nodes = self.fragment(member, pool, size, min_nodes_for(size), rng)
        if nodes is None:
            return None
        return self.finalize("single", size, nodes, rng, info={"anchor_block": block})

    def multi_fine(self, size: int, rng: np.random.Generator) -> PlantedQuery | None:
        if not self.has_fine:
            raise QueryGenerationError("multi_fine needs a fine partition level")
        block = int(rng.integers(self.num_blocks))
        fines = self.fines_of_block(block)
        if fines.shape[0] < 2:
            return None
        choices = [k for k in (2, 3, 4) if size // k >= 3]  # every choice is valid from size 12 up
        if not choices:
            return None
        num_frags = int(choices[int(rng.integers(len(choices)))])
        cache: dict = {}
        chosen = [int(fines[rng.integers(fines.shape[0])])]
        required: dict[int, set[int]] = {}
        self._grow_fines(chosen, required, num_frags, rng, cache, block=block)
        if len(chosen) < 2:
            return None
        nodes = self._fragments_of_fines(chosen, required, size, rng)
        if nodes is None:
            return None
        q = self.finalize("multi_fine", size, nodes, rng, info={"anchor_block": block, "fine_parts": chosen})
        if q is None or len(q.info.get("fine", [])) < 2:
            return None
        return q

    def _random_block_edge(self, rng: np.random.Generator) -> tuple[int, int] | None:
        total = int(self._bgraph_idx.shape[0])
        if total == 0:
            return None
        entry = int(rng.integers(total))
        c1 = int(np.searchsorted(self._bgraph_ptr, entry, side="right")) - 1
        return c1, int(self._bgraph_idx[entry])

    def _random_cut_edge(self, c1: int, c2: int, rng: np.random.Generator) -> tuple[int, int] | None:
        nodes = self.part.nodes_of_blocks([c1])
        lens, nbrs = gather_rows(self.indptr, self.indices, nodes)
        src = np.repeat(nodes, lens)
        hit = np.flatnonzero(np.asarray(self.node_block[nbrs.astype(np.int64)]) == c2)
        if hit.shape[0] == 0:
            return None
        k = int(hit[rng.integers(hit.shape[0])])
        return int(src[k]), int(nbrs[k])

    def multi_coarse(self, size: int, rng: np.random.Generator) -> PlantedQuery | None:
        edge = self._random_block_edge(rng)
        if edge is None:
            return None
        c1, c2 = edge
        bridge = self._random_cut_edge(c1, c2, rng)
        if bridge is None:
            return None
        u, v = bridge
        configs = [cfg for cfg in MULTI_COARSE_CONFIGS if size // cfg[0] >= 3]
        if self.has_fine and configs:
            num_frags, min_coarse = configs[int(rng.integers(len(configs)))]
            f1, f2 = int(self.node_fine[u]), int(self.node_fine[v])
            chosen = [f1, f2]
            required = {f1: {u}, f2: {v}}
            self._grow_fines(chosen, required, num_frags, rng, {})
            if len(chosen) >= num_frags and len({int(self.fine_block[f]) for f in chosen}) >= min_coarse:
                nodes = self._fragments_of_fines(chosen, required, size, rng)
                if nodes is not None:
                    q = self.finalize("multi_coarse", size, nodes, rng,
                                      info={"bridge_blocks": [c1, c2], "fine_parts": chosen,
                                            "config": [num_frags, min_coarse]})
                    if q is not None and len(q.info["blocks"]) >= 2:
                        return q
        # Two-block fallback of the legacy generator.
        counts = split_counts_for_fragments(size, 2, 5)
        frags = []
        for block, count, req in ((c1, counts[0], u), (c2, counts[1], v)):
            member = (lambda nb, b=block: np.asarray(self.node_block[nb]) == b)
            frag = self.fragment(member, self.part.nodes_of_blocks([block]), count, max(5, count - 2), rng, [req])
            if frag is None:
                return None
            frags.extend(frag)
        q = self.finalize("multi_coarse", size, frags, rng, info={"bridge_blocks": [c1, c2], "fallback": True})
        if q is None or len(q.info["blocks"]) < 2:
            return None
        return q

    def k_hop(self, size: int, rng: np.random.Generator) -> PlantedQuery | None:
        anchor = int(rng.integers(self.num_nodes))
        nodes = self.bfs(anchor, size, rng, max_depth=3)
        return self.finalize("k_hop", size, nodes, rng, info={"anchor": anchor})

    def random_walk(self, size: int, rng: np.random.Generator) -> PlantedQuery | None:
        cur = int(rng.integers(self.num_nodes))
        order = [cur]
        seen = {cur}
        for _ in range(size * 20):
            nb = self.neighbors(cur)
            if nb.shape[0] == 0:
                cur = order[int(rng.integers(len(order)))]
                continue
            cur = int(nb[rng.integers(nb.shape[0])])
            if cur not in seen:
                seen.add(cur)
                order.append(cur)
                if len(order) >= size:
                    break
        return self.finalize("random_walk", size, order, rng, info={"start": order[0]})

    def degree_tertiles(self) -> list[np.ndarray]:
        if self._tertiles is None:
            degrees = np.diff(np.asarray(self.indptr, dtype=np.int64))
            nonzero = np.flatnonzero(degrees > 0)
            ordered = nonzero[np.argsort(degrees[nonzero], kind="stable")]
            third = max(1, ordered.shape[0] // 3)
            self._tertiles = [ordered[:third], ordered[third:2 * third], ordered[2 * third:]]
        return self._tertiles

    def degree_k_hop(self, size: int, rng: np.random.Generator, tertile: int | None = None) -> PlantedQuery | None:
        tertile = int(rng.integers(3)) if tertile is None else int(tertile) % 3
        pool = self.degree_tertiles()[tertile]
        if pool.shape[0] == 0:
            return None
        anchor = int(pool[rng.integers(pool.shape[0])])
        nodes = self.bfs(anchor, size, rng, max_depth=3)
        return self.finalize("degree_k_hop", size, nodes, rng, info={"anchor": anchor, "tertile": tertile})

    def sm_standard(self, size: int, density: str, rng: np.random.Generator) -> PlantedQuery | None:
        spec = protocol.QUERY_DESIGN["standard_sm"]
        cur = int(rng.integers(self.num_nodes))
        order = [cur]
        seen = {cur}
        walk_edges: list[tuple[int, int]] = []
        for _ in range(size * 50):
            nb = self.neighbors(cur)
            if nb.shape[0] == 0:
                return None
            nxt = int(nb[rng.integers(nb.shape[0])])
            walk_edges.append((cur, nxt))
            cur = nxt
            if cur not in seen:
                seen.add(cur)
                order.append(cur)
                if len(order) >= size:
                    break
        if len(order) != size:
            return None
        nodes = np.asarray(order, dtype=np.int64)
        if density == "sparse":
            pos = {v: i for i, v in enumerate(order)}
            edges = normalize_edges(np.asarray([(pos[a], pos[b]) for a, b in walk_edges], dtype=np.int64))
        else:
            edges = self.induced_edges(nodes)
        avg_degree = 2.0 * edges.shape[0] / size
        if density == "dense" and avg_degree < spec["dense_min_avg_degree"]:
            return None
        if density == "sparse" and avg_degree >= spec["dense_min_avg_degree"]:
            return None
        return self.finalize(SM_FAMILIES[density], size, order, rng, edges=edges, min_nodes=size, exact_size=True,
                             info={"density": density, "avg_degree": avg_degree})

    def sample(self, family: str, size: int, rng: np.random.Generator, *, index: int | None = None) -> PlantedQuery | None:
        """One draw of a positive family (None when this draw fails)."""
        if family == "single":
            return self.single(size, rng)
        if family == "multi_fine":
            return self.multi_fine(size, rng)
        if family == "multi_coarse":
            return self.multi_coarse(size, rng)
        if family == "k_hop":
            return self.k_hop(size, rng)
        if family == "random_walk":
            return self.random_walk(size, rng)
        if family == "degree_k_hop":
            return self.degree_k_hop(size, rng, None if index is None else index % 3)
        for density, name in SM_FAMILIES.items():
            if family == name:
                return self.sm_standard(size, density, rng)
        raise ValueError(f"unknown family {family!r}")


# ----------------------------------------------------------------------------
# One-hop coverage (the FullCov criterion and the training coverage targets)
# ----------------------------------------------------------------------------
def cover_incidence(store: GraphStore, planted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (planted index i, block b) such that block b's candidate contains planted[i].

    A node lies in C(S) when its own block or a neighbour's block is in S.
    """
    planted = np.asarray(planted, dtype=np.int64)
    order = np.argsort(planted, kind="stable")
    lens, nbrs = gather_rows(store.indptr, store.indices, planted[order])
    node_block = store.part.node_block
    owner = np.concatenate([order, np.repeat(order, lens)])
    blocks = np.concatenate([np.asarray(node_block[planted[order]], dtype=np.int64),
                             np.asarray(node_block[nbrs.astype(np.int64)], dtype=np.int64)])
    keys = np.unique(owner * np.int64(store.num_blocks) + blocks)
    return keys // store.num_blocks, keys % store.num_blocks


def covers(store: GraphStore, planted: np.ndarray, blocks: Iterable[int]) -> bool:
    idx, blk = cover_incidence(store, planted)
    chosen = np.unique(np.asarray(list(blocks), dtype=np.int64))
    covered = np.zeros(np.asarray(planted).shape[0], dtype=bool)
    covered[idx[isin_sorted(blk, chosen)]] = True
    return bool(covered.all())


def greedy_cover(store: GraphStore, planted: np.ndarray) -> list[int]:
    """Greedy set cover of the planted nodes by one-hop block closures (ties to the lower id)."""
    idx, blk = cover_incidence(store, planted)
    uncovered = np.ones(np.asarray(planted).shape[0], dtype=bool)
    chosen: list[int] = []
    while uncovered.any():
        live = uncovered[idx]
        counts = np.bincount(blk[live], minlength=store.num_blocks)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            raise AssertionError("a planted node has no covering block")
        chosen.append(best)
        uncovered[idx[blk == best]] = False
    return chosen


# ----------------------------------------------------------------------------
# Payloads and negatives
# ----------------------------------------------------------------------------
def payload_for(store: GraphStore, nodes: np.ndarray, regime: str) -> dict[str, np.ndarray]:
    """The label payload a user supplies with the query, taken from data nodes."""
    from .sigma import hashed_buckets, row_digests

    nodes = np.asarray(nodes, dtype=np.int64)
    types = np.asarray(store.node_type[nodes], dtype=np.int8)
    out: dict[str, np.ndarray] = {"types": types}
    if regime == "R0":
        out["attrs"] = np.asarray(store.attrs[nodes])
    elif regime == "R1":
        if store.classes is None:
            raise QueryGenerationError("R1 needs classes in the store")
        out["classes"] = np.asarray(store.classes[nodes], dtype=np.int64)
    elif regime.startswith("R2-"):
        buckets = int(protocol.LABEL_REGIMES[regime]["buckets"])
        out["buckets"] = hashed_buckets(row_digests(types, np.asarray(store.attrs[nodes])), buckets).astype(np.int64)
    else:
        raise ValueError(f"unknown regime {regime}")
    return out


def _key_field(regime: str) -> str:
    return "attrs" if regime == "R0" else ("classes" if regime == "R1" else "buckets")


def _row_code(store: GraphStore, regime: str, types: np.ndarray, payload_rows: dict[str, np.ndarray]) -> np.ndarray:
    return store.sigma(regime).query_codes({"types": types, **payload_rows}, store)


def absent_label(store: GraphStore, regime: str, planted: PlantedQuery, payload: dict[str, np.ndarray],
                 rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Give one query node a key no data node carries; None when the regime has no such key."""
    table = store.sigma(regime)
    types = payload["types"]
    order = rng.permutation(planted.n)
    for j in order.tolist():
        new = {k: v.copy() for k, v in payload.items()}
        t = np.asarray(types[j:j + 1])
        if regime == "R0":
            row = new["attrs"][j].copy()
            for step in range(64):
                col = int(rng.integers(row.shape[0])) if row.shape[0] else 0
                if row.shape[0] == 0:
                    break
                if np.issubdtype(row.dtype, np.integer):
                    row[col] = 1 - row[col] if row[col] in (0, 1) else row[col] + 1
                else:
                    row[col] = row[col] + np.asarray(rng.normal() + 1e-3, dtype=row.dtype)
                code = _row_code(store, regime, t, {"attrs": row[None, :]})
                if int(code[0]) >= table.num_codes:
                    new["attrs"][j] = row
                    return new, {"node": j, "field": "attrs", "perturbation_steps": step + 1}
            continue
        if regime == "R1":
            present = table.keys[table.keys[:, 0] == int(t[0]), 1]
            value = int(present.max(initial=-1)) + 1 + int(rng.integers(1000))
            new["classes"][j] = value
            return new, {"node": j, "field": "classes", "value": value}
        buckets = int(protocol.LABEL_REGIMES[regime]["buckets"])
        present = np.unique(table.keys[table.keys[:, 0] == int(t[0]), 1])
        missing = np.setdiff1d(np.arange(buckets), present)
        if missing.shape[0]:
            value = int(missing[rng.integers(missing.shape[0])])
            new["buckets"][j] = value
            return new, {"node": j, "field": "buckets", "value": value}
    return None


def swapped_label(store: GraphStore, ctx: QueryContext, regime: str, planted: PlantedQuery,
                  payload: dict[str, np.ndarray], rng: np.random.Generator,
                  radius: int = 2, pool_cap: int = 2000) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Give one query node the key of another same-type data node close to its image."""
    codes = store.sigma(regime).codes
    types_all = store.node_type
    for j in rng.permutation(planted.n).tolist():
        image = int(planted.planted[j])
        t = int(types_all[image])
        own = int(codes[image])
        near = np.asarray(ctx.bfs(image, pool_cap, rng, max_depth=radius)[1:], dtype=np.int64)
        near = near[(np.asarray(types_all[near]) == t) & (np.asarray(codes[near]) != own)] if near.shape[0] else near
        source = "near"
        if near.shape[0] == 0:
            draws = rng.integers(0, store.num_nodes, size=4096)
            near = draws[(np.asarray(types_all[draws]) == t) & (np.asarray(codes[draws]) != own)]
            source = "uniform"
        if near.shape[0] == 0:
            continue
        donor = int(near[rng.integers(near.shape[0])])
        new = {k: v.copy() for k, v in payload.items()}
        donor_payload = payload_for(store, np.asarray([donor]), regime)
        field_name = _key_field(regime)
        new[field_name][j] = donor_payload[field_name][0]
        return new, {"node": j, "image": image, "donor": donor, "donor_source": source}
    return None


def structure_negative(planted: PlantedQuery, rng: np.random.Generator, *, node_types: np.ndarray | None = None,
                       type_pairs: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Add one non-edge, uniformly among pairs at distance 2 when there are any.

    With node_types (of the query nodes) and type_pairs ([T, T] bool, the type
    pairs of data edges), only pairs whose types meet on a data edge qualify: an
    edge between types that never meet is refuted by the types alone.
    """
    n = planted.n
    adj = np.zeros((n, n), dtype=bool)
    e = planted.edges
    adj[e[:, 0], e[:, 1]] = True
    adj[e[:, 1], e[:, 0]] = True
    two = (adj.astype(np.int32) @ adj.astype(np.int32)) > 0
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    if type_pairs is not None:
        t = np.asarray(node_types, dtype=np.int64)
        upper &= np.asarray(type_pairs, dtype=bool)[t[:, None], t[None, :]]
    at_two = np.argwhere(upper & two & ~adj)
    distance = 2
    pool = at_two
    if pool.shape[0] == 0:
        pool = np.argwhere(upper & ~adj)
        distance = None
    if pool.shape[0] == 0:
        return None
    a, b = (int(x) for x in pool[rng.integers(pool.shape[0])])
    return normalize_edges(np.vstack([e, [[a, b]]])), {"added_edge": [a, b], "distance": distance}


def label_pair_on_edge(store: GraphStore, regime: str, a: int, b: int) -> bool:
    """Whether a data edge joins a node with sigma code a to a node with code b."""
    table = store.sigma(regime)
    codes = np.asarray(table.codes)
    df = np.asarray(table.df)
    if df[a] > df[b]:
        a, b = b, a
    nodes = np.flatnonzero(codes == a)
    if nodes.shape[0] == 0:
        return False
    _, nbrs = gather_rows(store.indptr, store.indices, nodes)
    return bool((codes[np.asarray(nbrs, dtype=np.int64)] == b).any())


# ----------------------------------------------------------------------------
# Query records and files
# ----------------------------------------------------------------------------
@dataclass
class Query:
    uid: str
    family: str
    size: int
    kind: str  # "positive" or "negative"
    regime: str
    edges: np.ndarray
    payload: dict[str, np.ndarray]
    planted: np.ndarray
    seed: int
    attempt: int
    info: dict[str, Any] = field(default_factory=dict)
    certified: str = "planted"  # positives; negatives: uncertified, none, found, undecided
    _csr: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False, compare=False)

    @property
    def n(self) -> int:
        return int(self.payload["types"].shape[0])

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @property
    def is_negative(self) -> bool:
        return self.kind == "negative"

    def _ensure_csr(self) -> tuple[np.ndarray, np.ndarray]:
        if self._csr is None:
            self._csr = csr_from_edges(self.edges, self.n)
        return self._csr

    @property
    def indptr(self) -> np.ndarray:
        return self._ensure_csr()[0]

    @property
    def indices(self) -> np.ndarray:
        return self._ensure_csr()[1]

    def local_graph(self, codes: np.ndarray) -> LocalGraph:
        indptr, indices = self._ensure_csr()
        return LocalGraph(global_ids=np.arange(self.n, dtype=np.int64), indptr=indptr, indices=indices,
                          labels=np.asarray(codes, dtype=np.int32))

    def content_hash(self) -> str:
        return content_hash(self.regime, self.edges, self.payload)


def content_hash(regime: str, edges: np.ndarray, payload: dict[str, np.ndarray]) -> str:
    """SHA-256 of what a query shows a matcher: regime, size, edges, types and labels in node order."""
    from .sigma import canonical_rows

    h = hashlib.sha256()
    h.update(regime.encode("ascii"))
    h.update(np.asarray([payload["types"].shape[0]], dtype="<i8").tobytes())
    h.update(np.asarray(edges, dtype="<i8").tobytes())
    h.update(np.asarray(payload["types"], dtype="<i2").tobytes())
    key = _key_field(regime)
    values = payload[key]
    h.update(canonical_rows(values).tobytes() if key == "attrs" else np.asarray(values, dtype="<i8").tobytes())
    return h.hexdigest()


def node_labels(regime: str, payload: dict[str, np.ndarray]) -> list[bytes]:
    """Each query node's type and regime label as bytes; the matcher's label (sigma code) is a function of both."""
    from .sigma import canonical_rows

    types = np.asarray(payload["types"], dtype="<i2")
    key = _key_field(regime)
    values = canonical_rows(payload[key]) if key == "attrs" else np.asarray(payload[key], dtype="<i8")
    return [t.tobytes() + v.tobytes() for t, v in zip(types, values)]


@dataclass
class IndexedQuery:
    key: tuple
    graph: Any  # networkx.Graph with node attributes "digest" (for the WL hash) and "label"
    wl: str | None = None


def _same_label(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["label"] == b["label"]


class QueryIndex:
    """The queries of one regime set up to label-preserving isomorphism.

    A query is the undirected simple graph csr_from_edges makes of its edges, with (type, regime
    label) on every node. Two queries are the same query when a node bijection keeps every edge and
    every label: their matches in any data graph are then the same. Candidates share size, edge
    count, degree sequence, label multiset and Weisfeiler-Lehman hash; VF2 decides.
    """

    def __init__(self, regime: str) -> None:
        self.regime = regime
        self.entries: dict[tuple, list[tuple[IndexedQuery, str]]] = {}

    def item(self, edges: np.ndarray, payload: dict[str, np.ndarray]) -> IndexedQuery:
        import networkx as nx

        labels = node_labels(self.regime, payload)
        digests = [hashlib.blake2b(label, digest_size=16).hexdigest() for label in labels]
        n = len(labels)
        indptr, indices = csr_from_edges(edges, n)
        degrees = np.diff(indptr)
        rows = np.repeat(np.arange(n, dtype=np.int64), degrees)
        graph = nx.Graph()
        graph.add_nodes_from((i, {"digest": digests[i], "label": labels[i]}) for i in range(n))
        graph.add_edges_from((int(u), int(v)) for u, v in zip(rows, indices) if u < v)
        key = (n, int(indices.shape[0]) // 2, tuple(sorted(degrees.tolist())), tuple(sorted(digests)))
        return IndexedQuery(key, graph)

    @staticmethod
    def _wl(item: IndexedQuery) -> str:
        if item.wl is None:
            import networkx as nx

            item.wl = nx.weisfeiler_lehman_graph_hash(item.graph, node_attr="digest", iterations=3)
        return item.wl

    def find(self, item: IndexedQuery) -> str | None:
        """The name of an indexed query isomorphic to item, or None."""
        import networkx as nx

        for other, name in self.entries.get(item.key, ()):
            if self._wl(item) == self._wl(other) and nx.is_isomorphic(item.graph, other.graph, node_match=_same_label):
                return name
        return None

    def add(self, item: IndexedQuery, name: str) -> None:
        self.entries.setdefault(item.key, []).append((item, name))


def duplicate_pairs(query_set: Iterable[Query]) -> list[tuple[str, str]]:
    """(earlier uid, uid) for every query isomorphic to an earlier query of the set."""
    index: QueryIndex | None = None
    pairs = []
    for q in query_set:
        index = index or QueryIndex(q.regime)
        item = index.item(q.edges, q.payload)
        earlier = index.find(item)
        if earlier is None:
            index.add(item, q.uid)
        else:
            pairs.append((earlier, q.uid))
    return pairs


def save_query_set(path: str | Path, queries: Sequence[Query], meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not queries:
        raise ValueError("empty query set")
    regime = queries[0].regime
    key = _key_field(regime)
    n = np.asarray([q.n for q in queries], dtype=np.int64)
    m = np.asarray([q.m for q in queries], dtype=np.int64)
    node_ptr = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    edge_ptr = np.concatenate([[0], np.cumsum(m)]).astype(np.int64)
    arrays = {
        "uid": np.asarray([q.uid for q in queries]),
        "family": np.asarray([q.family for q in queries]),
        "kind": np.asarray([q.kind for q in queries]),
        "size": np.asarray([q.size for q in queries], dtype=np.int32),
        "seed": np.asarray([q.seed for q in queries], dtype=np.int64),
        "attempt": np.asarray([q.attempt for q in queries], dtype=np.int32),
        "certified": np.asarray([q.certified for q in queries]),
        "info": np.asarray([json.dumps(q.info, sort_keys=True, default=_json_default) for q in queries]),
        "content_hash": np.asarray([q.content_hash() for q in queries]),
        "nodeset_hash": np.asarray([nodeset_hash(q.planted) for q in queries]),
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
        "edges": np.concatenate([q.edges for q in queries]).astype(np.int32).reshape(-1, 2),
        "planted": np.concatenate([q.planted for q in queries]).astype(np.int64),
        "types": np.concatenate([q.payload["types"] for q in queries]).astype(np.int8),
        key: np.concatenate([q.payload[key] for q in queries]),
        "meta": np.asarray(json.dumps({**meta, "regime": regime, "generator": GENERATOR_VERSION}, sort_keys=True,
                                      default=_json_default)),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


@dataclass
class QuerySet:
    path: Path
    meta: dict[str, Any]
    queries: list[Query]

    def __iter__(self) -> Iterator[Query]:
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def regime(self) -> str:
        return str(self.meta["regime"])

    def select(self, *, kind: str | None = None, family: str | None = None, size: int | None = None) -> list[Query]:
        return [q for q in self.queries if (kind is None or q.kind == kind) and (family is None or q.family == family)
                and (size is None or q.size == size)]

    @classmethod
    def load(cls, path: str | Path, *, verify_hashes: bool = True) -> "QuerySet":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        meta = json.loads(str(data["meta"]))
        regime = meta["regime"]
        key = _key_field(regime)
        queries = []
        for i in range(data["uid"].shape[0]):
            a, b = int(data["node_ptr"][i]), int(data["node_ptr"][i + 1])
            c, d = int(data["edge_ptr"][i]), int(data["edge_ptr"][i + 1])
            payload = {"types": data["types"][a:b], key: data[key][a:b]}
            q = Query(uid=str(data["uid"][i]), family=str(data["family"][i]), size=int(data["size"][i]),
                      kind=str(data["kind"][i]), regime=regime, edges=data["edges"][c:d].astype(np.int64),
                      payload=payload, planted=data["planted"][a:b], seed=int(data["seed"][i]),
                      attempt=int(data["attempt"][i]), info=json.loads(str(data["info"][i])),
                      certified=str(data["certified"][i]))
            if verify_hashes and q.content_hash() != str(data["content_hash"][i]):
                raise QueryGenerationError(f"{path}: content hash mismatch for {q.uid}")
            queries.append(q)
        return cls(path, meta, queries)


# ----------------------------------------------------------------------------
# Building a split
# ----------------------------------------------------------------------------
def split_counts(split: str, regime: str) -> dict[str, int]:
    design = protocol.QUERY_DESIGN
    if split in TEST_SPLITS:
        counts = dict(design["per_cell"][protocol.regime_family(regime)])
        counts["sm_standard"] = int(design["standard_sm"]["per_cell"])
        return counts
    if split in design["split_counts"]:
        return dict(design["split_counts"][split])
    raise ValueError(f"unknown split {split!r}")


def sm_cells() -> list[tuple[str, int]]:
    spec = protocol.QUERY_DESIGN["standard_sm"]
    cells = []
    for size in spec["sizes"]:
        if size < spec["density_split_min_size"]:
            cells.append(("any", size))
        else:
            cells += [(density, size) for density in spec["densities"]]
    return cells


class SplitBuilder:
    """Generates the structures of one split master and the per-regime query sets."""

    def __init__(self, store: GraphStore, dataset: str, split: str, master: str, *,
                 exclude: Iterable[str] = (), sizes: Sequence[int] | None = None,
                 max_per_cell: int | None = None) -> None:
        self.store = store
        self.ctx = QueryContext(store)
        self.type_pairs = features.store_relation_table(store) >= 0
        self.dataset = dataset
        self.split = split
        self.master = master
        self.taken: set[str] = set(exclude)
        self.excluded = len(self.taken)
        self.sizes = list(sizes or protocol.QUERY_SIZES)
        self.max_attempts = int(protocol.QUERY_DESIGN["max_attempts_per_query"])
        self.attempt_log: dict[str, list[int]] = {}
        self.accepted: dict[str, QueryIndex] = {}  # regime -> the accepted positive and sm_standard queries
        self.duplicates: dict[str, int] = {}  # draws redrawn because an isomorphic query was taken
        # Smoke runs only: caps every per-cell count. Recorded in the query-set meta.
        self.max_per_cell = max_per_cell

    def counts_for(self, regime: str) -> dict[str, int]:
        counts = split_counts(self.split, regime)
        if self.max_per_cell is not None:
            counts = {k: min(int(v), int(self.max_per_cell)) for k, v in counts.items()}
        return counts

    def _draw(self, stream: tuple[Any, ...], make: Callable[[np.random.Generator], PlantedQuery | None],
              regimes: Sequence[str] = ()):
        """The first attempt whose node set is free and whose positive query is new in every regime given."""
        for attempt in range(self.max_attempts):
            seed = protocol.derive_seed(self.master, self.dataset, *stream, attempt)
            q = make(np.random.default_rng(seed))
            if q is None:
                continue
            h = nodeset_hash(q.planted)
            if h in self.taken:
                continue
            items = {r: self.accepted.setdefault(r, QueryIndex(r)).item(q.edges, payload_for(self.store, q.planted, r))
                     for r in regimes}
            if any(self.accepted[r].find(item) is not None for r, item in items.items()):
                cell = "/".join(str(s) for s in stream[:3])
                self.duplicates[cell] = self.duplicates.get(cell, 0) + 1
                continue
            self.taken.add(h)
            for r, item in items.items():
                self.accepted[r].add(item, "/".join(str(s) for s in stream))
            self.attempt_log.setdefault("/".join(str(s) for s in stream[:3]), []).append(attempt)
            return q, seed, attempt
        raise QueryGenerationError(f"{self.dataset}/{self.split}: no query for {stream} in {self.max_attempts} attempts")

    def structures(self, counts: dict[str, int],
                   by_regime: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
        """Positive structures, negative bases and sm_standard structures for the largest counts.

        With by_regime (the counts of every regime), a positive or sm_standard draw is also
        redrawn when its query is isomorphic to an earlier one in a regime whose set holds both.
        """
        by_regime = by_regime or {}

        def used_in(count_key: str, i: int) -> list[str]:
            return [r for r, c in by_regime.items() if i < int(c.get(count_key, 0))]

        out: dict[str, Any] = {"positive": {}, "negative": {}, "sm": {}}
        for family in protocol.POSITIVE_FAMILIES:
            for size in self.sizes:
                for i in range(counts.get("positive", 0)):
                    out["positive"][(family, size, i)] = self._draw(
                        ("pos", family, size, i), lambda rng, f=family, s=size, k=i: self.ctx.sample(f, s, rng, index=k),
                        used_in("positive", i))
        for kind in protocol.NEGATIVE_FAMILIES:
            for size in self.sizes:
                for i in range(counts.get(kind, 0)):
                    families = protocol.POSITIVE_FAMILIES
                    base_family = families[i % len(families)]
                    out["negative"][(kind, size, i)] = self._draw(
                        ("neg", kind, size, i),
                        lambda rng, f=base_family, s=size, k=i // len(families): self.ctx.sample(f, s, rng, index=k))
        for density, size in sm_cells():
            for i in range(counts.get("sm_standard", 0)):
                family = SM_FAMILIES[density]
                out["sm"][(family, size, i)] = self._draw(
                    ("sm", family, size, i), lambda rng, s=size, d=density: self.ctx.sm_standard(s, d, rng),
                    used_in("sm_standard", i))
        return out

    def regime_queries(self, structures: dict[str, Any], regime: str, counts: dict[str, int]) -> tuple[list[Query], dict[str, Any]]:
        store = self.store
        queries: list[Query] = []
        skipped: dict[str, int] = {}
        prefix = f"{self.dataset}/{self.split}"
        positives = [(key, value, "positive") for key, value in structures["positive"].items()]
        positives += [(key, value, "sm_standard") for key, value in structures["sm"].items()]
        for (family, size, i), (pq, seed, attempt), count_key in positives:
            if i >= counts.get(count_key, 0):
                continue
            info = dict(pq.info)
            info["cover_blocks"] = greedy_cover(store, pq.planted)
            payload = payload_for(store, pq.planted, regime)
            check = verify.check_embedding(store, payload, pq.edges, pq.planted, regime)
            if not check.ok:
                raise QueryGenerationError(f"{prefix}/{family}/{size}/{i}: planted witness fails: {check.reason}")
            queries.append(Query(f"{prefix}/{family}/{size}/{i:04d}", family, size, "positive", regime, pq.edges,
                                 payload, pq.planted, seed, attempt, info))
        index = QueryIndex(regime)
        for q in queries:  # structures(by_regime=...) already redrew duplicate positives
            item = index.item(q.edges, q.payload)
            earlier = index.find(item)
            if earlier is not None:
                raise QueryGenerationError(f"{q.uid} duplicates {earlier}")
            index.add(item, q.uid)
        duplicates: dict[str, int] = {}
        for (kind, size, i), (pq, seed, attempt) in structures["negative"].items():
            if i >= counts.get(kind, 0):
                continue
            base = payload_for(store, pq.planted, regime)
            made = None
            for mod_attempt in range(20):
                stream = ("negmod", kind, size, i, mod_attempt) if kind == "neg_structure" else \
                    ("negmod", kind, size, i, regime, mod_attempt)
                rng = np.random.default_rng(protocol.derive_seed(self.master, self.dataset, *stream))
                if kind == "neg_structure":
                    res = structure_negative(pq, rng, node_types=np.asarray(store.node_type[pq.planted]),
                                             type_pairs=self.type_pairs)
                    if res is not None:
                        edges, detail = res
                        a, b = (int(pq.planted[j]) for j in detail["added_edge"])
                        codes = store.sigma(regime).codes
                        detail["label_pair_in_data"] = label_pair_on_edge(store, regime, int(codes[a]), int(codes[b]))
                        made = (edges, base, detail)
                elif kind == "neg_absent_label":
                    res = absent_label(store, regime, pq, base, rng)
                    if res is not None:
                        made = (pq.edges, res[0], res[1])
                else:
                    res = swapped_label(store, self.ctx, regime, pq, base, rng)
                    if res is not None:
                        made = (pq.edges, res[0], res[1])
                if made is not None:
                    item = index.item(made[0], made[1])
                    if index.find(item) is not None:  # a copy of an earlier query (a positive: not a negative)
                        duplicates[f"{kind}/{size}"] = duplicates.get(f"{kind}/{size}", 0) + 1
                        made = None
                if made is not None:
                    detail = dict(made[2])
                    detail.update({"base_family": pq.family, "base_blocks": pq.info.get("blocks"),
                                   "mod_attempt": mod_attempt})
                    break
            if made is None:
                skipped[f"{kind}/{size}"] = skipped.get(f"{kind}/{size}", 0) + 1
                continue
            query = Query(f"{prefix}/{kind}/{size}/{i:04d}", kind, size, "negative", regime, made[0], made[1],
                          pq.planted, seed, attempt, detail, certified="uncertified")
            index.add(item, query.uid)
            queries.append(query)
        extra: dict[str, Any] = {"not_constructible": skipped}
        if duplicates:  # modification attempts that repeated an earlier query of this regime
            extra["negative_duplicates_redrawn"] = duplicates
        return queries, extra

    def build(self, regimes: Sequence[str], out_dir: str | Path, *, master_tag: str) -> dict[str, Any]:
        out_dir = Path(out_dir)
        counts_by_regime = {r: self.counts_for(r) for r in regimes}
        largest: dict[str, int] = {}
        for counts in counts_by_regime.values():
            for k, v in counts.items():
                largest[k] = max(largest.get(k, 0), int(v))
        structures = self.structures(largest, counts_by_regime)
        report: dict[str, Any] = {"dataset": self.dataset, "split": self.split, "master": master_tag,
                                  "excluded_nodesets": self.excluded, "sizes": self.sizes,
                                  "max_per_cell": self.max_per_cell, "regimes": {}}
        if self.duplicates:  # draws whose positive query was isomorphic to an earlier one
            report["duplicates_redrawn"] = dict(sorted(self.duplicates.items()))
        for regime in regimes:
            queries, extra = self.regime_queries(structures, regime, counts_by_regime[regime])
            meta = {"dataset": self.dataset, "split": self.split, "master": master_tag, "counts": counts_by_regime[regime],
                    "sizes": self.sizes, "max_per_cell": self.max_per_cell, "protocol_hash": protocol.protocol_hash(),
                    **extra}
            if self.duplicates:
                meta["duplicates_redrawn"] = report["duplicates_redrawn"]
            save_query_set(out_dir / f"{regime}.npz", queries, meta)
            report["regimes"][regime] = {"queries": len(queries), **extra}
        hashes = sorted(self.taken)
        (out_dir / "nodesets.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")
        attempts = {k: {"mean_attempt": float(np.mean(v)), "max_attempt": int(np.max(v)), "n": len(v)}
                    for k, v in self.attempt_log.items()}
        report["attempts"] = attempts
        (out_dir / "queries_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                    encoding="utf-8")
        return report


def read_exclusions(paths: Iterable[str | Path]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        p = Path(p)
        files = [p / "nodesets.txt"] if p.is_dir() else [p]
        for f in files:
            out.update(line.strip() for line in f.read_text(encoding="ascii").splitlines() if line.strip())
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate the query sets of one split master")
    b.add_argument("--store-root", required=True)
    b.add_argument("--dataset", required=True, choices=sorted(protocol.DATASETS))
    b.add_argument("--split", required=True, choices=["dev", "validation", *TEST_SPLITS])
    b.add_argument("--master-index", type=int, default=0, help="which dev or validation master")
    b.add_argument("--regimes", nargs="*", default=None)
    b.add_argument("--out", required=True, help="output directory for this split master")
    b.add_argument("--exclude", nargs="*", default=[], help="earlier split directories or nodesets.txt files")
    b.add_argument("--max-per-cell", type=int, default=None, help="smoke runs only: cap every per-cell count")
    b.add_argument("--sizes", type=int, nargs="*", default=None, help="smoke runs only: override QUERY_SIZES")
    b.add_argument("--unseal", action="store_true")
    b.add_argument("--reason", default="")
    i = sub.add_parser("inspect", help="summarise a query-set file")
    i.add_argument("path")
    d = sub.add_parser("duplicates", help="isomorphic query pairs of query-set files (exit 1 when any)")
    d.add_argument("paths", nargs="+", help="query-set files or directories searched for *.npz")
    args = parser.parse_args(argv)

    if args.cmd == "duplicates":
        files = sorted({f for p in map(Path, args.paths) for f in ([p] if p.is_file() else p.rglob("*.npz"))})
        found = {}
        for f in files:
            qs = QuerySet.load(f)
            found[str(f)] = {"queries": len(qs), "duplicates": duplicate_pairs(qs)}
        print(json.dumps(found, indent=1, sort_keys=True))
        pairs = sum(len(v["duplicates"]) for v in found.values())
        print(f"{pairs} isomorphic pairs in {len(files)} query sets", file=sys.stderr)
        return 1 if pairs or not files else 0

    if args.cmd == "inspect":
        qs = QuerySet.load(args.path)
        cells: dict[str, int] = {}
        for q in qs:
            key = f"{q.kind}/{q.family}/{q.size}"
            cells[key] = cells.get(key, 0) + 1
        print(json.dumps({"meta": qs.meta, "cells": cells}, indent=2, sort_keys=True))
        return 0

    from . import manifest, seal

    if args.split in TEST_SPLITS and (args.max_per_cell is not None or args.sizes):
        parser.error("a sealed split is always generated in full")
    masters = seal.split_masters(args.split, unseal=args.unseal, reason=args.reason or f"queries {args.dataset}")
    if args.split in TEST_SPLITS:
        master, tag = masters[0], f"sealed:{args.split}"
    else:
        master = masters[args.master_index]
        tag = master
    regimes = args.regimes or protocol.regimes_for(args.dataset)
    store = GraphStore(args.store_root, args.dataset, mmap=True)
    out = Path(args.out)
    config = {"dataset": args.dataset, "split": args.split, "master": tag, "regimes": regimes,
              "exclude": [str(p) for p in args.exclude], "max_per_cell": args.max_per_cell, "sizes": args.sizes}
    with manifest.RunManifest(out, experiment="queries", config=config, split=args.split,
                              inputs={"store_meta": str(Path(args.store_root) / args.dataset / "meta.json")}) as run:
        builder = SplitBuilder(store, args.dataset, args.split, master, exclude=read_exclusions(args.exclude),
                               sizes=args.sizes, max_per_cell=args.max_per_cell)
        report = builder.build(regimes, out, master_tag=tag)
        run.note(queries={r: v["queries"] for r, v in report["regimes"].items()})
    print(json.dumps({r: v["queries"] for r, v in report["regimes"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
