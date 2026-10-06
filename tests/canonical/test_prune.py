from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from scripts.canonical import backends, prune, queries, store as store_mod


def _graph(n, edges, labels):
    return store_mod.local_graph_from_edges(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                                            labels=np.asarray(labels))


def _adj(g):
    return [set(np.asarray(g.indices[g.indptr[v]:g.indptr[v + 1]]).tolist()) for v in range(g.n)]


def reference_alive(cand, query):
    """The greatest LDF/NLF fixpoint, by one-node-at-a-time removal."""
    qlab = np.asarray(query.labels).tolist()
    qadj = _adj(query)
    qreq = [(qlab[u], len(qadj[u]), Counter(qlab[w] for w in qadj[u])) for u in range(query.n)]
    clab = np.asarray(cand.labels).tolist()
    cadj = _adj(cand)
    alive = {v for v in range(cand.n) if clab[v] in set(qlab)}
    changed = True
    while changed:
        changed = False
        for v in sorted(alive):
            nb = [w for w in cadj[v] if w in alive]
            cnt = Counter(clab[w] for w in nb)
            if not any(lab == clab[v] and deg <= len(nb) and all(cnt[k] >= c for k, c in need.items())
                       for lab, deg, need in qreq):
                alive.discard(v)
                changed = True
    return alive


def _random_labelled(rng, n, p, num_labels):
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return _graph(n, edges, rng.integers(0, num_labels, size=n))


def _random_query(rng, target, size):
    """A connected induced subgraph of `target` (so at least one embedding exists)."""
    adj = _adj(target)
    start = int(rng.integers(target.n))
    nodes = [start]
    frontier = sorted(adj[start])
    while len(nodes) < size and frontier:
        v = frontier.pop(int(rng.integers(len(frontier))))
        if v not in nodes:
            nodes.append(v)
            frontier = sorted(set(frontier) | (adj[v] - set(nodes)))
    pos = {v: i for i, v in enumerate(nodes)}
    edges = [(pos[u], pos[w]) for u in nodes for w in adj[u] if w in pos and pos[u] < pos[w]]
    return _graph(len(nodes), edges, np.asarray(target.labels)[nodes]), np.asarray(nodes)


# ----------------------------------------------------------------------------- requirements
def test_pareto_minimal():
    vecs = np.asarray([[2, 1, 0], [3, 1, 0], [1, 2, 0], [2, 1, 0], [1, 1, 1]])
    got = sorted(map(tuple, prune.pareto_minimal(vecs).tolist()))
    assert got == [(1, 1, 1), (1, 2, 0), (2, 1, 0)]


def test_query_requirements():
    q = _graph(4, [[0, 1], [1, 2], [2, 3]], [7, 5, 7, 7])  # a path 7-5-7-7
    req = prune.query_requirements(q)
    assert req.labels.tolist() == [5, 7] and req.counts.tolist() == [1, 3] and req.connected
    rule7 = req.per_label[1]
    # label-7 nodes: node 0 (deg 1, one 5), node 2 (deg 2, one 5 and one 7), node 3 (deg 1, one 7)
    vecs = sorted(zip(rule7.degree.tolist(), map(tuple, rule7.need.tolist())))
    assert rule7.columns.tolist() == [0, 1]
    assert vecs == [(1, (0, 1)), (1, (1, 0))]  # node 2 dominates node 0 and is dropped
    assert req.label_index(np.asarray([7, 6, 5, 99])).tolist() == [1, -1, 0, -1]


# ----------------------------------------------------------------------------- fixpoint
@pytest.mark.parametrize("seed", range(12))
def test_fixpoint_matches_the_reference(seed):
    rng = np.random.default_rng(seed)
    cand = _random_labelled(rng, 60, 0.08, 4)
    query, _ = _random_query(rng, cand, int(rng.integers(3, 8)))
    req = prune.query_requirements(query)
    alive, info = prune.ldf_nlf_fixpoint(cand, req, max_iterations=1000)
    assert info["converged"]
    assert set(np.flatnonzero(alive).tolist()) == reference_alive(cand, query)
    tiny, _ = prune.ldf_nlf_fixpoint(cand, req, max_iterations=1000, chunk_entries=5)
    assert np.array_equal(alive, tiny)


def reference_pass(cand, query, alive, dirty):
    """One synchronous pass: every dirty alive node is tested against the alive set at the pass start."""
    qlab = np.asarray(query.labels).tolist()
    qadj = _adj(query)
    qreq = [(qlab[u], len(qadj[u]), Counter(qlab[w] for w in qadj[u])) for u in range(query.n)]
    clab = np.asarray(cand.labels).tolist()
    cadj = _adj(cand)
    failed = set()
    for v in sorted(dirty & alive):
        nb = [w for w in cadj[v] if w in alive]
        cnt = Counter(clab[w] for w in nb)
        if not any(lab == clab[v] and deg <= len(nb) and all(cnt[k] >= c for k, c in need.items())
                   for lab, deg, need in qreq):
            failed.add(v)
    return failed, {w for v in failed for w in cadj[v]}


@pytest.mark.parametrize("seed", range(10))
def test_each_pass_removes_what_the_synchronous_reference_removes(seed):
    """The degree shortcut leaves the pass semantics alone: same removals per pass, pass by pass."""
    rng = np.random.default_rng(300 + seed)
    cand = _random_labelled(rng, 70, 0.07, 4)
    query, _ = _random_query(rng, cand, int(rng.integers(3, 9)))
    req = prune.query_requirements(query)
    qlabels = set(np.asarray(query.labels).tolist())
    alive = {v for v, lab in enumerate(np.asarray(cand.labels).tolist()) if lab in qlabels}
    dirty = set(alive)
    removed_per_pass = []
    while dirty & alive:
        failed, touched = reference_pass(cand, query, alive, dirty)
        removed_per_pass.append(len(failed))
        if not failed:
            break
        alive -= failed
        dirty = touched
    for passes in range(1, len(removed_per_pass) + 1):
        got, info = prune.ldf_nlf_fixpoint(cand, req, max_iterations=passes)
        replay = {v for v, lab in enumerate(np.asarray(cand.labels).tolist()) if lab in qlabels}
        replay_dirty = set(replay)
        for _ in range(passes):
            failed, touched = reference_pass(cand, query, replay, replay_dirty)
            replay -= failed
            replay_dirty = touched
        assert set(np.flatnonzero(got).tolist()) == replay
        assert info["removed_per_pass"] == removed_per_pass[:passes][:len(info["removed_per_pass"])]


@pytest.mark.parametrize("seed", range(8))
def test_every_embedding_survives_in_one_target(seed):
    import networkx as nx
    from networkx.algorithms import isomorphism

    rng = np.random.default_rng(100 + seed)
    cand = _random_labelled(rng, 40, 0.12, 3)
    query, _ = _random_query(rng, cand, 5)
    for mode in ("separate", "merged", "none"):
        result = prune.prune(cand, query, mode=mode)
        assert not result.empty
        if mode == "none":
            alive, _ = prune.ldf_nlf_fixpoint(cand, prune.query_requirements(query))
            assert len(result.targets) == 1
            assert np.array_equal(result.targets[0].global_ids, cand.subgraph(alive).global_ids)
        gt, gq = nx.Graph(), nx.Graph()
        for g, h in ((cand, gt), (query, gq)):
            for i, lab in enumerate(np.asarray(g.labels).tolist()):
                h.add_node(i, label=lab)
            h.add_edges_from(map(tuple, g.edge_list().tolist()))
        matcher = isomorphism.GraphMatcher(gt, gq, node_match=lambda a, b: a["label"] == b["label"])
        targets = [set(np.asarray(t.global_ids).tolist()) for t in result.targets]
        for k, mapping in enumerate(matcher.subgraph_monomorphisms_iter()):
            image = {int(cand.global_ids[t]) for t in mapping}
            assert sum(image <= t for t in targets) == 1
            if k > 200:
                break


def test_cap_on_passes_is_sound_and_weaker():
    rng = np.random.default_rng(7)
    cand = _random_labelled(rng, 80, 0.06, 3)
    query, planted = _random_query(rng, cand, 6)
    req = prune.query_requirements(query)
    full, _ = prune.ldf_nlf_fixpoint(cand, req, max_iterations=1000)
    one, info = prune.ldf_nlf_fixpoint(cand, req, max_iterations=1)
    assert info["passes"] == 1
    assert (one | ~full).all()  # a capped run keeps a superset
    assert one[planted].all() and full[planted].all()


# ----------------------------------------------------------------------------- components
def test_components_filter_order_and_modes():
    # Two triangles labelled (1,1,2), one triangle labelled (1,1,1), one path (1,2), and an isolated 2.
    edges = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [6, 7], [7, 8], [6, 8], [9, 10],
             [11, 12], [12, 13], [11, 13], [13, 14]]
    labels = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2]
    cand = _graph(16, edges, labels)
    query = _graph(3, [[0, 1], [1, 2], [0, 2]], [1, 1, 2])
    sep = prune.prune(cand, query, mode="separate")
    assert [t.global_ids.tolist() for t in sep.targets] == [[0, 1, 2], [3, 4, 5], [11, 12, 13]]
    assert sep.stats["components_kept"] == 3
    merged = prune.prune(cand, query, mode="merged")
    assert len(merged.targets) == 1
    assert merged.targets[0].global_ids.tolist() == [0, 1, 2, 3, 4, 5, 11, 12, 13]
    for t in sep.targets:
        assert backends.NetworkXBackend().solve(query, t, time_limit=5.0).status == backends.FOUND
    capped = prune.prune(cand, query, mode="separate", component_cap=1)
    assert len(capped.targets) == 1 and capped.stats["components_capped"] == 2


def test_smallest_component_first():
    # A 4-cycle with a chord and a triangle, both labelled 1: the triangle comes first.
    cand = _graph(7, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [4, 5], [5, 6], [4, 6]], [1] * 7)
    query = _graph(3, [[0, 1], [1, 2], [0, 2]], [1, 1, 1])
    res = prune.prune(cand, query)
    assert [t.global_ids.tolist() for t in res.targets] == [[4, 5, 6], [0, 1, 2, 3]]
    assert res.targets[1].num_undirected_edges == 5


def test_absent_label_and_disconnected_query():
    cand = _graph(4, [[0, 1], [1, 2], [2, 3]], [1, 1, 1, 1])
    fresh = _graph(2, [[0, 1]], [1, 99])
    res = prune.prune(cand, fresh)
    assert res.empty and "multiset" in res.stats["reason_empty"]
    split = _graph(4, [[0, 1], [2, 3]], [1, 1, 1, 1])
    res = prune.prune(_graph(6, [[0, 1], [2, 3], [4, 5]], [1] * 6), split)
    assert len(res.targets) == 1 and res.targets[0].n == 6 and not res.stats["query_connected"]


def test_locate_follows_the_witness_through_every_mode():
    edges = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [6, 7], [7, 8], [6, 8], [9, 10],
             [11, 12], [12, 13], [11, 13], [13, 14]]
    labels = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2]
    cand = _graph(16, edges, labels)
    query = _graph(3, [[0, 1], [1, 2], [0, 2]], [1, 1, 2])
    second = np.asarray([3, 4, 5])  # the second of the three kept triangles
    sep = prune.prune(cand, query, mode="separate")
    got = prune.locate(sep, cand, second)
    assert got == {"nodes": 3, "in_candidate": 3, "alive": 3, "component": "kept", "target": 1}
    assert np.isin(second, sep.targets[got["target"]].global_ids).all()
    assert prune.locate(prune.prune(cand, query, mode="merged"), cand, second)["target"] == 0
    none = prune.locate(prune.prune(cand, query, mode="none"), cand, second)
    assert none["component"] == "unfiltered" and none["target"] == 0
    capped = prune.prune(cand, query, mode="separate", component_cap=1)
    assert prune.locate(capped, cand, second)["component"] == "capped"
    assert prune.locate(capped, cand, np.asarray([0, 1, 2]))["target"] == 0
    pruned = prune.locate(sep, cand, np.asarray([9, 10, 11]))  # 9 and 10 fail LDF
    assert pruned["alive"] == 1 and pruned["component"] is None
    missing = prune.locate(sep, cand.subgraph(np.arange(16) != 4), second)
    assert missing["in_candidate"] == 2 and missing["alive"] is None and missing["target"] is None
    raw = prune.locate(prune.PruneResult([cand], mode="unpruned"), cand, second)
    assert raw["component"] == "unfiltered" and raw["alive"] == 3
    # outcomes that soundness rules out are still named, so the reducer can count them
    alive = np.ones(16, dtype=bool)
    comp = np.repeat(np.arange(8), 2)
    fake = prune.PruneResult([cand], alive=alive, component=comp, passed=np.asarray([0]), kept=np.asarray([0]),
                             mode="separate")
    assert prune.locate(fake, cand, np.asarray([2, 3]))["component"] == "filtered"
    assert prune.locate(fake, cand, np.asarray([1, 2]))["component"] == "split"


@pytest.mark.parametrize("seed", range(4))
def test_locate_finds_every_embedding_in_its_target(seed):
    import networkx as nx
    from networkx.algorithms import isomorphism

    rng = np.random.default_rng(300 + seed)
    cand = _random_labelled(rng, 40, 0.12, 3)
    query, _ = _random_query(rng, cand, 5)
    gt, gq = nx.Graph(), nx.Graph()
    for g, h in ((cand, gt), (query, gq)):
        for i, lab in enumerate(np.asarray(g.labels).tolist()):
            h.add_node(i, label=lab)
        h.add_edges_from(map(tuple, g.edge_list().tolist()))
    matcher = isomorphism.GraphMatcher(gt, gq, node_match=lambda a, b: a["label"] == b["label"])
    images = [np.asarray(sorted(m), dtype=np.int64) for _, m in zip(range(50), matcher.subgraph_monomorphisms_iter())]
    assert images
    for mode in ("separate", "merged", "none"):
        result = prune.prune(cand, query, mode=mode)
        for image in images:
            got = prune.locate(result, cand, np.asarray(cand.global_ids)[image])
            assert got["alive"] == 5 and got["component"] in ("kept", "unfiltered")
            assert np.isin(np.asarray(cand.global_ids)[image], result.targets[got["target"]].global_ids).all()


def test_split_by_component_preserves_the_graph():
    rng = np.random.default_rng(5)
    g = _random_labelled(rng, 50, 0.05, 2)
    count, comp = g.components()
    order = np.arange(count)[::-1]
    parts = prune.split_by_component(g, comp, order)
    assert sum(p.n for p in parts) == g.n
    assert sum(p.num_undirected_edges for p in parts) == g.num_undirected_edges
    for c, p in zip(order.tolist(), parts):
        want = g.subgraph(comp == c)
        assert np.array_equal(p.global_ids, want.global_ids)
        assert np.array_equal(p.indptr, want.indptr) and np.array_equal(p.indices, want.indices)
        assert np.array_equal(p.labels, want.labels)


# ----------------------------------------------------------------------------- on the store
@pytest.mark.parametrize("regime", ["R0", "R1", "R2-32"])
def test_planted_queries_survive_on_the_store(clustered_store, regime):
    root, _ = clustered_store
    gstore = store_mod.GraphStore(root, "synth", mmap=True)
    ctx = queries.QueryContext(gstore)
    table = gstore.sigma(regime)
    solver = backends.NetworkXBackend()
    done = 0
    for k in range(40):
        pq = ctx.sample("random_walk" if k % 2 else "multi_coarse", 10, np.random.default_rng(k), index=k)
        if pq is None:
            continue
        payload = queries.payload_for(gstore, pq.planted, regime)
        codes = table.query_codes(payload, gstore)
        query = store_mod.local_graph_from_edges(pq.n, pq.edges, labels=codes)
        cand = gstore.candidate(queries.greedy_cover(gstore, pq.planted), codes, regime)
        res = prune.prune(cand, query)
        holders = [t for t in res.targets if np.isin(pq.planted, t.global_ids).all()]
        assert len(holders) == 1
        assert res.stats["pruned_nodes"] <= cand.n
        out = solver.solve(query, holders[0], time_limit=30.0)
        assert out.status == backends.FOUND
        done += 1
        if done == 6:
            break
    assert done == 6
