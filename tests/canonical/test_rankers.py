from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.canonical import features, queries, rankers, store as store_mod


class FakePart:
    """A random weighted block graph with the Partition interface the policies use."""

    def __init__(self, num_blocks: int, p: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        edges = {}
        for a in range(num_blocks):
            for b in range(a + 1, num_blocks):
                if rng.random() < p:
                    edges[(a, b)] = int(rng.integers(1, 50))
        self.num_blocks = num_blocks
        self.rows = [[] for _ in range(num_blocks)]
        for (a, b), w in sorted(edges.items()):
            self.rows[a].append((b, w))
            self.rows[b].append((a, w))
        self.rows = [sorted(r) for r in self.rows]
        self.edges = edges

    def block_neighbors(self, block):
        row = self.rows[block]
        return (np.asarray([b for b, _ in row], dtype=np.int64), np.asarray([w for _, w in row], dtype=np.int64))

    def to_networkx(self):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(self.num_blocks))
        for (a, b), w in self.edges.items():
            g.add_edge(a, b, weight=w)
        return g


@pytest.mark.parametrize("seed", range(6))
def test_hybrid_matches_the_legacy_policy(seed):
    from scripts.retrieval_strategies import hybrid_boundary_expand

    part = FakePart(80, 0.06 if seed % 2 else 0.15, seed)
    graph = part.to_networkx()
    rng = np.random.default_rng(seed)
    ranking = rng.permutation(80)
    for budget in (1, 5, 20, 21, 37, 60, 80):
        want = hybrid_boundary_expand(ranking.tolist(), ranking.tolist(), budget, graph,
                                      seed_count=20, model_weight=0.5, teleport_every=10)
        got = rankers.hybrid_order(ranking, part, budget, seed_count=20, model_weight=0.5, teleport_every=10)
        assert got.tolist() == want, budget


def test_hybrid_is_prefix_consistent_and_lazy():
    part = FakePart(120, 0.05, 11)
    ranking = np.random.default_rng(3).permutation(120)
    full = rankers.hybrid_order(ranking, part, 120)
    assert sorted(full.tolist()) == list(range(120))
    lazy = rankers.HybridSelector(ranking, part)
    for budget in (3, 10, 25, 26, 70, 120, 500):
        assert lazy.prefix(budget).tolist() == full[:budget].tolist()
    shrink = rankers.HybridSelector(ranking, part)
    assert shrink.prefix(50).tolist() == full[:50].tolist()
    assert shrink.prefix(10).tolist() == full[:10].tolist()


def test_hybrid_seeds_are_the_top_of_the_ranking():
    part = FakePart(50, 0.2, 5)
    ranking = np.random.default_rng(0).permutation(50)
    order = rankers.hybrid_order(ranking, part, 30, seed_count=20)
    assert order[:20].tolist() == ranking[:20].tolist()


def test_hybrid_on_a_disconnected_block_graph_takes_the_ranking():
    part = FakePart(30, 0.0, 1)
    ranking = np.random.default_rng(1).permutation(30)
    order = rankers.hybrid_order(ranking, part, 30, seed_count=4)
    assert order.tolist() == ranking.tolist()


def test_partial_and_duplicate_rankings():
    from scripts.retrieval_strategies import hybrid_boundary_expand

    part = FakePart(40, 0.1, 9)
    ranking = [5, 3, 5, 7, 3, 1]
    got = rankers.hybrid_order(np.asarray(ranking), part, 25, seed_count=2)
    want = hybrid_boundary_expand(ranking, ranking, 25, part.to_networkx(), seed_count=2)
    assert got.tolist() == want


def test_topk_rrf_and_score_ranking():
    scores = np.asarray([0.3, 0.9, 0.9, -1.0])
    assert rankers.ranking_from_scores(scores).tolist() == [1, 2, 0, 3]
    assert rankers.TopKSelector(np.asarray([2, 0, 1])).prefix(2).tolist() == [2, 0]
    fused = rankers.rrf_scores([np.asarray([0, 1, 2]), np.asarray([2, 1, 0])], 3, k=60)
    assert np.allclose(fused, [1 / 61 + 1 / 63, 2 / 62, 1 / 63 + 1 / 61])
    assert rankers.make_selector("topk", np.asarray([1, 0]), None).prefix(5).tolist() == [1, 0]


# ----------------------------------------------------------------------------- offline inputs and rankers
@pytest.fixture(scope="module")
def rstore(clustered_store, tmp_path_factory):
    """A private copy of the clustered store with the ranker inputs built."""
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("rankers")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    report = rankers.build_artifacts(gstore, ["R0", "R1", "R2-32"], chunk_nodes=64)
    return gstore, report


def _block_code_sets(gstore, regime):
    codes = np.asarray(gstore.sigma(regime).codes)
    return [set(codes[gstore.part.nodes_of_blocks([b])].tolist()) for b in range(gstore.num_blocks)]


@pytest.mark.parametrize("regime", ["R0", "R1", "R2-32"])
def test_postings_match_brute_force(rstore, regime):
    gstore, _ = rstore
    post = rankers.build_postings(gstore, regime, chunk_nodes=1 << 20)
    small = rankers.build_postings(gstore, regime, chunk_nodes=1)
    assert np.array_equal(post.ptr, small.ptr) and np.array_equal(post.blocks, small.blocks)
    sets = _block_code_sets(gstore, regime)
    for code in range(post.num_codes):
        want = [b for b in range(gstore.num_blocks) if code in sets[b]]
        assert post.blocks[post.ptr[code]:post.ptr[code + 1]].tolist() == want
    loaded = rankers.RankerBank(gstore, regime).postings
    assert np.array_equal(np.asarray(loaded.ptr), post.ptr)


@pytest.mark.parametrize("regime", ["R0", "R1", "R2-32"])
def test_index_scores_match_brute_force(rstore, regime):
    gstore, _ = rstore
    post = rankers.RankerBank(gstore, regime).postings
    sets = _block_code_sets(gstore, regime)
    num_blocks = gstore.num_blocks
    rng = np.random.default_rng(1)
    fresh = post.num_codes + 3
    for _ in range(20):
        codes = rng.integers(0, post.num_codes, size=int(rng.integers(1, 12)))
        codes = np.concatenate([codes, codes[:2], [fresh]])  # repeats and a fresh code
        distinct = set(codes.tolist()) - {fresh}
        bdf = {c: sum(c in s for s in sets) for c in distinct}
        for weighting in ("idf", "count"):
            got = post.scores(codes, weighting=weighting)
            for b in range(num_blocks):
                held = [c for c in sorted(distinct) if c in sets[b]]
                want = sum(np.log1p(num_blocks / bdf[c]) if weighting == "idf" else 1.0 for c in held)
                assert got[b] == pytest.approx(want, abs=1e-12)
    assert post.block_df(np.asarray([0, post.num_codes + 1])).tolist()[1] == 0
    assert post.scores(np.asarray([post.num_codes + 7])).tolist() == [0.0] * num_blocks


def test_equal_coverage_ties_to_the_lower_block():
    post = rankers.Postings(np.asarray([0, 3, 5, 6]), np.asarray([0, 2, 4, 2, 4, 1], dtype=np.int32), 5)
    scores = post.scores(np.asarray([0, 1]))
    assert scores[2] == scores[4] and scores[2] > scores[0]
    assert rankers.ranking_from_scores(scores).tolist()[:3] == [2, 4, 0]


def test_topo_stats_match_networkx():
    import networkx as nx

    for seed in range(5):
        g = nx.gnp_random_graph(40, 0.12, seed=seed)
        g.add_node(40)  # an isolated node
        lg = store_mod.local_graph_from_edges(41, np.asarray(list(g.edges()), dtype=np.int64))
        types = np.random.default_rng(seed).integers(0, 3, size=41)
        got = rankers.topo_stats(lg.indptr, lg.indices, types, 4)
        deg = np.asarray([d for _, d in sorted(g.degree())], dtype=float)
        assert got[0] == pytest.approx(deg.mean())
        assert got[1] == pytest.approx(deg.std() / deg.mean())
        assert got[2] == pytest.approx(float((deg <= 1).mean()))
        assert got[3] == pytest.approx(nx.transitivity(g))
        assert got[4:].tolist() == pytest.approx((np.bincount(types, minlength=4) / 41).tolist())
    empty = rankers.topo_stats(np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0), 2)
    assert empty.tolist() == [0.0] * 6


def test_mean_inputs_are_block_means(rstore):
    gstore, report = rstore
    mean = np.load(rankers.artifact_dir(gstore, "R1") / "mean_inputs.npy")
    assert mean.shape == (gstore.num_blocks, report["R1"]["input_dim"])
    for b in (0, gstore.num_blocks - 1):
        x = features.store_graph_inputs(gstore, gstore.block(b), "R1")["x"]
        assert np.allclose(mean[b], x.mean(axis=0), atol=1e-6)


def _query_inputs(gstore, regime, pq, uid, embedding=None):
    payload = queries.payload_for(gstore, pq.planted, regime)
    codes = gstore.sigma(regime).query_codes(payload, gstore)
    lg = store_mod.local_graph_from_edges(pq.n, pq.edges, labels=codes)
    view = SimpleNamespace(indptr=lg.indptr, indices=lg.indices, payload=payload)
    x = features.query_graph_inputs(gstore, view, regime, codes)["x"]
    return rankers.QueryInputs(uid, codes, np.asarray(payload["types"]), lg.indptr, lg.indices, x=x,
                               embedding=embedding)


def _some_query(gstore, family="multi_coarse", size=12):
    ctx = queries.QueryContext(gstore)
    for k in range(100):
        pq = ctx.sample(family, size, np.random.default_rng(k), index=k)
        if pq is not None:
            return pq
    raise AssertionError("no query")


def test_bank_rankings(rstore):
    gstore, _ = rstore
    num_blocks = gstore.num_blocks
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(num_blocks, 16))
    q = _query_inputs(gstore, "R1", _some_query(gstore), "uid-1", embedding=rng.normal(size=16))
    bank = rankers.RankerBank(gstore, "R1", block_emb=emb, router_tau=10.0)
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    learned = unit @ (q.embedding / np.linalg.norm(q.embedding))
    assert np.allclose(bank.scores("learned", q), learned)
    for name in rankers.RANKER_NAMES:
        order, info = bank.rank(name, q)
        assert sorted(order.tolist()) == list(range(num_blocks)), name
        assert info["ranker"] == name
    fused = rankers.rrf_scores([rankers.ranking_from_scores(learned),
                                rankers.ranking_from_scores(bank.scores("index", q))], num_blocks)
    assert bank.rank("fused_learned_index", q)[0].tolist() == rankers.ranking_from_scores(fused).tolist()
    mean_q = q.x.astype(np.float64).mean(axis=0)
    mean_b = np.load(rankers.artifact_dir(gstore, "R1") / "mean_inputs.npy").astype(np.float64)
    cos = mean_b @ mean_q / np.linalg.norm(mean_b, axis=1) / np.linalg.norm(mean_q)
    assert np.allclose(bank.scores("mean_feature", q), cos)
    sel = bank.selectivity(q.codes)
    df = gstore.sigma("R1").label_frequency(q.codes)
    assert sel == pytest.approx(float(np.mean(np.log10(np.maximum(df, 1)))))
    low = rankers.RankerBank(gstore, "R1", block_emb=emb, router_tau=sel)
    order, info = low.rank("router", q)
    assert info["route"] == "index"
    index = bank.scores("index", q)
    assert order.tolist() == np.lexsort((np.arange(num_blocks), -learned, -index)).tolist()
    high = rankers.RankerBank(gstore, "R1", block_emb=emb, router_tau=sel - 1e-9)
    order, info = high.rank("router", q)
    assert info["route"] == "fused_learned_index"
    assert order.tolist() == bank.rank("fused_learned_index", q)[0].tolist()
    other = rankers.QueryInputs("uid-2", q.codes, q.types, q.indptr, q.indices)
    a, b = bank.rank("random", q)[0], bank.rank("random", other)[0]
    assert a.tolist() == bank.rank("random", q)[0].tolist() and a.tolist() != b.tolist()
    with pytest.raises(ValueError):
        rankers.RankerBank(gstore, "R1").rank("router", q)
    with pytest.raises(ValueError):
        rankers.RankerBank(gstore, "R1", router_tau=0.0).rank("learned", q)


def test_topo_ranker_prefers_the_closest_block(rstore):
    gstore, _ = rstore
    bank = rankers.RankerBank(gstore, "R0")
    g = gstore.block(3)
    q = rankers.QueryInputs("t", np.zeros(g.n, dtype=np.int64), np.asarray(gstore.node_type[g.global_ids]),
                            g.indptr, g.indices)
    scores = bank.scores("topo_feature", q)
    assert int(np.argmax(scores)) == 3 and scores[3] == pytest.approx(0.0, abs=1e-9)


def test_stale_inputs_are_refused(rstore, tmp_path):
    gstore, _ = rstore
    copy = tmp_path / "synth"
    shutil.copytree(gstore.dir, copy)
    info_path = copy / "part" / store_mod.PRIMARY / "rankers" / "R1" / "info.json"
    info = json.loads(info_path.read_text())
    info["num_codes"] += 1
    info_path.write_text(json.dumps(info))
    other = store_mod.GraphStore(tmp_path, "synth", mmap=True)
    with pytest.raises(ValueError, match="stale"):
        rankers.RankerBank(other, "R1").postings
    with pytest.raises(FileNotFoundError):
        rankers.load_block_embeddings(other, "no-model", "R1")


def test_block_embeddings_round_trip(rstore):
    gstore, _ = rstore
    emb = np.random.default_rng(0).normal(size=(gstore.num_blocks, 8)).astype(np.float32)
    rankers.save_block_embeddings(gstore, "m1", "R0", emb, {"weights_sha256": "abc"})
    got, info = rankers.load_block_embeddings(gstore, "m1", "R0")
    assert np.array_equal(got, emb) and info["weights_sha256"] == "abc" and info["dim"] == 8
    with pytest.raises(ValueError):
        rankers.save_block_embeddings(gstore, "m1", "R0", emb[:-1])


def test_fit_router_tau():
    tau, info = rankers.fit_router_tau([0.1, 0.2, 0.3, 2.0, 2.5, 3.0], [1, 1, 1, 9, 9, 9], [5, 5, 5, 2, 2, 2])
    assert 0.3 < tau < 2.0 and info["cost"] == 9.0 and info["routed_to_index"] == 3
    tau, info = rankers.fit_router_tau([1.0, 2.0], [1, 1], [1, 1])
    assert info["optimal_interval"] == [0.0, 3.0] and tau == 1.5
    tau, _ = rankers.fit_router_tau([1.0, 2.0], [1, 1], [5, 5])
    assert tau >= 2.0
    tau, _ = rankers.fit_router_tau([1.0, 2.0], [5, 5], [1, 1])
    assert tau < 1.0


@pytest.mark.parametrize("policy", ["hybrid", "topk"])
def test_coverage_cost_matches_prefix_checks(rstore, policy):
    gstore, _ = rstore
    ctx = queries.QueryContext(gstore)
    part = gstore.part
    params = {"seed_count": 1} if policy == "hybrid" else {}
    checked = 0
    for k in range(30):
        pq = ctx.sample("random_walk" if k % 2 else "multi_coarse", 12, np.random.default_rng(100 + k), index=k)
        if pq is None:
            continue
        ranking = np.random.default_rng(k).permutation(gstore.num_blocks)
        sel = rankers.make_selector(policy, ranking, part, **params)
        idx, blk = queries.cover_incidence(gstore, pq.planted)
        cost = rankers.coverage_cost(sel, idx, blk, pq.n, gstore.num_blocks)
        fresh = rankers.make_selector(policy, ranking, part, **params)
        want = next(b for b in range(1, gstore.num_blocks + 1) if queries.covers(gstore, pq.planted, fresh.prefix(b)))
        assert cost == want
        assert rankers.fullcov_at(cost, [want - 1, want]) == [False, True]
        checked += 1
    assert checked >= 10
    assert rankers.fullcov_at(None, [1, 2]) == [False, False]
