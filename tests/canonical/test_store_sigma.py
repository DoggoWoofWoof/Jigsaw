from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from scripts.canonical import store as store_mod
from scripts.canonical import verify
from scripts.canonical.sigma import hashed_buckets, row_digests


@pytest.fixture(params=[False, True], ids=["resident", "mmap"])
def gstore(request, synthetic_store):
    root, _ = synthetic_store
    return store_mod.GraphStore(root, "synth", mmap=request.param)


def _edge_set(indptr, indices):
    out = set()
    for u in range(len(indptr) - 1):
        for v in indices[indptr[u]:indptr[u + 1]]:
            out.add((u, int(v)))
    return out


def test_csr_is_simple_symmetric_sorted(gstore):
    indptr, indices = np.asarray(gstore.indptr), np.asarray(gstore.indices)
    edges = _edge_set(indptr, indices)
    assert all((v, u) in edges for u, v in edges)
    assert all(u != v for u, v in edges)
    for u in range(gstore.num_nodes):
        row = indices[indptr[u]:indptr[u + 1]]
        assert np.all(np.diff(row) > 0)


def test_renumbering_preserves_graph(synthetic_store, gstore):
    _, g = synthetic_store
    orig = np.load(gstore.dir / "graph" / "orig_id.npy")
    src = {(min(u, v), max(u, v)) for u, v in g["edges"].tolist() if u != v}
    got = {(min(int(orig[u]), int(orig[v])), max(int(orig[u]), int(orig[v])))
           for u, v in _edge_set(np.asarray(gstore.indptr), np.asarray(gstore.indices))}
    assert got == src
    assert np.array_equal(np.asarray(gstore.node_type), g["node_type"][orig])
    assert np.array_equal(np.asarray(gstore.attrs), g["attrs"][orig])


def test_blocks_are_contiguous(gstore):
    part = gstore.part
    nb = np.asarray(part.node_block)
    assert np.all(np.diff(nb) >= 0)
    for b in range(part.num_blocks):
        nodes = part.nodes_of_blocks([b])
        assert np.all(nb[nodes] == b)
        assert nodes.shape[0] == int(part.block_ptr[b + 1] - part.block_ptr[b])


def _groups(keys):
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    return sorted(sorted(v) for v in groups.values())


def test_r0_codes_equal_iff_keys_equal(gstore):
    codes = np.asarray(gstore.sigma("R0").codes)
    attrs = np.asarray(gstore.attrs)
    attrs = np.where(attrs == 0, 0.0, attrs)  # -0.0 and 0.0 are the same value
    keys = [(int(t), attrs[i].tobytes()) for i, t in enumerate(np.asarray(gstore.node_type))]
    assert _groups(keys) == _groups(codes.tolist())
    # first-appearance order
    first = {}
    for i, c in enumerate(codes.tolist()):
        first.setdefault(c, i)
    assert list(first) == sorted(first)


def test_negative_zero_is_zero(gstore):
    attrs = np.asarray(gstore.attrs)
    types = np.asarray(gstore.node_type)
    neg = [i for i in range(len(attrs)) if types[i] == 2 and np.signbit(attrs[i, 0])]
    pos = [i for i in range(len(attrs)) if types[i] == 2 and not np.signbit(attrs[i, 0])]
    assert neg and pos
    codes = np.asarray(gstore.sigma("R0").codes)
    assert codes[neg[0]] == codes[pos[0]]


def test_r0_lookup_known_and_fresh(gstore):
    table = gstore.sigma("R0")
    nodes = np.array([0, 5, 17, 300])
    got = table.lookup_exact(np.asarray(gstore.node_type)[nodes], np.asarray(gstore.attrs)[nodes],
                             gstore.node_type, gstore.attrs)
    assert np.array_equal(got, np.asarray(table.codes)[nodes])
    novel = np.asarray(gstore.attrs)[[0, 0]].copy()
    novel[:, 3] = 1234.5
    fresh = table.lookup_exact(np.asarray(gstore.node_type)[[0, 0]], novel, gstore.node_type, gstore.attrs)
    assert fresh[0] == fresh[1] >= table.num_codes


def test_r1_and_r2_codes(gstore):
    types = np.asarray(gstore.node_type)
    classes = np.asarray(gstore.classes)
    r1 = np.asarray(gstore.sigma("R1").codes)
    assert _groups(list(zip(types.tolist(), classes.tolist()))) == _groups(r1.tolist())
    digests = row_digests(types, np.asarray(gstore.attrs))
    for regime, k in (("R2-512", 512), ("R2-32", 32)):
        buckets = hashed_buckets(digests, k)
        codes = np.asarray(gstore.sigma(regime).codes)
        assert _groups(list(zip(types.tolist(), buckets.tolist()))) == _groups(codes.tolist())
        for i in range(0, len(types), 37):
            assert verify._bucket(int(types[i]), np.asarray(gstore.attrs[i]), k) == int(buckets[i])
        table = gstore.sigma(regime)
        assert table.centroids is not None and table.centroids.shape[0] == table.num_codes


@pytest.mark.parametrize("halo,label_filter", [(True, True), (False, True), (True, False), (False, False),
                                               (2, True), (2, False)])
def test_candidate_matches_brute_force(gstore, halo, label_filter):
    rng = np.random.default_rng(3)
    codes = np.asarray(gstore.sigma("R1").codes)
    indptr, indices = np.asarray(gstore.indptr), np.asarray(gstore.indices)
    adj = [set(indices[indptr[u]:indptr[u + 1]].tolist()) for u in range(gstore.num_nodes)]
    for trial in range(6):
        blocks = sorted(rng.choice(gstore.num_blocks, size=1 + trial % 3, replace=False).tolist())
        qcodes = rng.choice(np.unique(codes), size=4, replace=False)
        wanted = set(qcodes.tolist())
        stats = {}
        cand = gstore.candidate(blocks, qcodes, "R1", stats=stats, halo=halo, label_filter=label_filter)
        vs = set(gstore.part.nodes_of_blocks(blocks).tolist())
        # hop 1 grows from every block node, hop h + 1 from the kept nodes of hop h
        closure, frontier, halo_nodes = set(vs), set(vs), 0
        for _ in range(int(halo)):
            ring = set().union(*(adj[u] for u in frontier)) - closure if frontier else set()
            halo_nodes += len(ring)
            closure |= ring
            frontier = {v for v in ring if not label_filter or codes[v] in wanted}
        expect = sorted(v for v in closure if not label_filter or codes[v] in wanted)
        assert stats["halo_nodes"] == halo_nodes
        assert cand.global_ids.tolist() == expect
        want_edges = {(u, v) for u in expect for v in adj[u] if v in set(expect)}
        got_edges = set()
        for i in range(cand.n):
            for j in cand.indices[cand.indptr[i]:cand.indptr[i + 1]]:
                got_edges.add((int(cand.global_ids[i]), int(cand.global_ids[j])))
        assert got_edges == want_edges
        assert np.array_equal(cand.labels, codes[cand.global_ids])
        assert stats["candidate_nodes"] == len(expect)


def test_subgraph_and_components():
    edges = np.array([[0, 1], [1, 2], [3, 4], [5, 5]])
    g = store_mod.local_graph_from_edges(6, edges)
    count, labels = g.components()
    assert count == 3  # {0,1,2}, {3,4}, {5}
    sub = g.subgraph(np.array([True, True, False, True, True, False]))
    assert sub.n == 4 and sub.num_undirected_edges == 2


def test_checker_accepts_planted_and_rejects_corruptions(gstore):
    nodes = np.array([0, 1, 2])
    indptr, indices = np.asarray(gstore.indptr), np.asarray(gstore.indices)
    u = 0
    nbrs = indices[indptr[u]:indptr[u + 1]]
    v = int(nbrs[0])
    w = int(indices[indptr[v]:indptr[v + 1]][0])
    if w == u:
        w = int(indices[indptr[v]:indptr[v + 1]][-1])
    nodes = np.array([u, v, w])
    edges = np.array([[0, 1], [1, 2]])
    payload = {"types": np.asarray(gstore.node_type)[nodes], "attrs": np.asarray(gstore.attrs)[nodes]}
    assert verify.check_embedding(gstore, payload, edges, nodes, "R0").ok
    assert not verify.check_embedding(gstore, payload, edges, nodes[[0, 0, 2]], "R0").ok
    bad = dict(payload)
    bad["attrs"] = payload["attrs"].copy()
    bad["attrs"][1, 0] += 1.0
    assert not verify.check_embedding(gstore, bad, edges, nodes, "R0").ok
    non_edge = np.array([[0, 1], [1, 2], [0, 2]])
    if not gstore.has_edge(u, w):
        assert not verify.check_embedding(gstore, payload, non_edge, nodes, "R0").ok
