from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

from scripts.canonical import backends, cascade, protocol, queries, rankers, store as store_mod

REGIMES = ["R0", "R1", "R2-32"]
BUDGETS = [1, 2, 4]


@pytest.fixture(scope="module")
def cstore(clustered_store, tmp_path_factory):
    """A private copy of the clustered store with ranker inputs, label indexes and a small dev query set."""
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("cascade")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    rankers.build_artifacts(gstore, REGIMES)
    for regime in REGIMES:
        cascade.build_label_index(gstore, regime)
    out = copy / "queries"
    builder = queries.SplitBuilder(gstore, "synth", "dev", "20260607", sizes=[8, 12], max_per_cell=2)
    builder.build(REGIMES, out, master_tag="20260607")
    sets = {r: queries.QuerySet.load(out / f"{r}.npz") for r in REGIMES}
    return gstore, sets


def _embed(inputs):
    return np.resize(np.asarray(inputs["x"], dtype=np.float64).mean(axis=0), 16)


def _make(gstore, regime, method, backend=None, **kw):
    emb = np.random.default_rng(0).normal(size=(gstore.num_blocks, 16))
    bank = rankers.RankerBank(gstore, regime, block_emb=emb, router_tau=1.0)
    kw.setdefault("budgets", BUDGETS)
    return cascade.Cascade(gstore, regime, method, backend or backends.NetworkXBackend(), bank=bank, embed=_embed, **kw)


def test_label_index_and_global_candidate(cstore):
    gstore, sets = cstore
    for regime in REGIMES:
        codes_all = np.asarray(gstore.sigma(regime).codes)
        index = cascade.LabelIndex(gstore, regime)
        c = _make(gstore, regime, "filterall")
        for q in sets[regime].queries[:10]:
            codes = gstore.sigma(regime).query_codes(q.payload, gstore)
            want = np.flatnonzero(np.isin(codes_all, codes))
            assert index.nodes_of(codes).tolist() == want.tolist()
            stats: dict = {}
            got = c.global_candidate(codes, stats)
            ref = gstore.candidate(range(gstore.num_blocks), codes, regime)
            assert np.array_equal(got.global_ids, ref.global_ids)
            assert np.array_equal(got.indptr, ref.indptr) and np.array_equal(got.indices, ref.indices)
            assert np.array_equal(got.labels, ref.labels)
            assert stats["candidate_nodes"] == ref.n


METHODS = ["jigsaw", "jigsaw_learned", "jigsaw_index", "jigsaw_fused", "mean_rrf", "mean_feature",
           "topo_feature", "random", "filterall", "direct"]


@pytest.mark.parametrize("regime", REGIMES)
def test_every_method_decides_the_same(cstore, regime):
    """Exact methods agree on every query they decide; VF2 decides every query of size <= 12 here."""
    gstore, sets = cstore
    qs = [q for q in sets[regime].queries if q.size <= 12]
    answers: dict[str, dict[str, str]] = {}
    for method in METHODS:
        c = _make(gstore, regime, method)
        answers[method] = {}
        for q in qs:
            rec = c.run(q, time_limit=30.0)
            assert rec["status"] != backends.ERROR, (method, q.uid, rec)
            answers[method][q.uid] = rec["status"]
            if rec["status"] == backends.FOUND:
                assert rec["witness_check"] == "ok"
                assert rec["time_found"] == rec["elapsed"]
            assert sum(rec["stages"].values()) <= rec["elapsed"] + 1e-6
    reference = answers["filterall"]
    assert all(status != backends.TIMEOUT for status in reference.values())
    for q in qs:
        if q.kind == "positive":
            assert reference[q.uid] == backends.FOUND, q.uid
    for method, got in answers.items():
        decided = {uid: status for uid, status in got.items() if status != backends.TIMEOUT}
        assert decided == {uid: reference[uid] for uid in decided}, method
        if method != "direct":
            assert len(decided) == len(qs), method
    negatives = [q for q in qs if q.kind == "negative"]
    assert negatives and any(reference[q.uid] == backends.NONE for q in negatives)


def test_absent_label_is_certified_before_any_step(cstore):
    gstore, sets = cstore
    c = _make(gstore, "R0", "jigsaw_index")
    absent = [q for q in sets["R0"].queries if q.family == "neg_absent_label"]
    assert absent
    for q in absent:
        rec = c.run(q, time_limit=10.0)
        assert rec["status"] == backends.NONE and rec["certificate"] == "global label multiset"
        assert rec["steps"] == [] and rec["time_decided"] is not None


def _unique_label_query(gstore, block, size, *, extra_edge):
    """A connected query on nodes of one block whose R0 codes occur once in the graph."""
    table = gstore.sigma("R0")
    codes = np.asarray(table.codes)
    unique = np.flatnonzero(np.asarray(table.df)[codes] == 1)
    members = set(np.intersect1d(unique, gstore.part.nodes_of_blocks([block])).tolist())
    for start in sorted(members):
        chosen, frontier = [start], [start]
        while frontier and len(chosen) < size:
            u = frontier.pop(0)
            for w in gstore.neighbors(u).tolist():
                if w in members and w not in chosen and len(chosen) < size:
                    chosen.append(w)
                    frontier.append(w)
        if len(chosen) < size:
            continue
        planted = np.asarray(chosen, dtype=np.int64)
        pos = {v: i for i, v in enumerate(chosen)}
        edges = sorted({(min(pos[u], pos[w]), max(pos[u], pos[w])) for u in chosen
                        for w in gstore.neighbors(u).tolist() if w in pos and w != u})
        if extra_edge:
            missing = [(a, b) for a in range(size) for b in range(a + 1, size) if (a, b) not in set(edges)]
            if not missing:
                continue
            edges = sorted(edges + [missing[0]])
        payload = queries.payload_for(gstore, planted, "R0")
        kind = "negative" if extra_edge else "positive"
        return queries.Query(f"test/unique/{kind}", "neg_structure" if extra_edge else "single", size, kind, "R0",
                             np.asarray(edges, dtype=np.int64), payload, planted, 0, 0, {},
                             certified="uncertified" if extra_edge else "planted")
    raise AssertionError("no unique-label query")


def test_closed_step_decides_negatives_early(cstore):
    """With unique labels the first step already holds every label node, so its NONE is final."""
    gstore, sets = cstore
    c = _make(gstore, "R0", "jigsaw_index")
    for block in range(gstore.num_blocks):
        q = _unique_label_query(gstore, block, 6, extra_edge=True)
        rec = c.run(q, time_limit=30.0)
        assert rec["status"] == backends.NONE and rec["decided_step"] == 0, rec
        step = rec["steps"][0]
        assert step["closed"] and not step["final"]
        assert step["candidate"]["candidate_nodes"] == rec["label_nodes_global"] == 6
        assert len(rec["steps"]) == 1
        pos = _unique_label_query(gstore, block, 6, extra_edge=False)
        assert c.run(pos, time_limit=30.0)["status"] == backends.FOUND
    # Every NONE on the dev set is decided at a closed step.
    for q in sets["R0"].queries:
        rec = c.run(q, time_limit=30.0)
        if rec["status"] == backends.NONE and rec["decided_step"] is not None:
            assert rec["steps"][rec["decided_step"]]["closed"]


@pytest.mark.parametrize("halo", [True, False, 2])
def test_positive_steps_record_coverage(cstore, halo):
    gstore, sets = cstore
    c = _make(gstore, "R1", "jigsaw_index", halo=halo)
    seen = 0
    for q in sets["R1"].select(kind="positive")[:8]:
        rec = c.run(q, time_limit=30.0)
        home = np.asarray(gstore.part.node_block[np.asarray(q.planted)], dtype=np.int64)
        codes = gstore.sigma("R1").query_codes(q.payload, gstore)
        for step in rec["steps"]:
            if step["final"] or "fullcov" not in step:  # the final step holds every node; not_started records none
                continue
            blocks = rankers.make_selector("hybrid", c.bank.rank("index", c.query_inputs(q, codes))[0],
                                           gstore.part).prefix(step["budget"])
            homecov = bool(np.isin(home, blocks).all())
            one_hop = queries.covers(gstore, q.planted, blocks)
            assert step["homecov"] == homecov
            if halo is True:
                assert step["fullcov"] == one_hop
            elif halo is False:
                assert step["fullcov"] == homecov
            else:
                two_hop = gstore.candidate(blocks, codes, "R1", halo=2)
                assert step["fullcov"] == bool(np.isin(q.planted, two_hop.global_ids).all())
                assert step["fullcov"] or not one_hop  # a second hop never covers less
            assert (step["covered"] == len(q.planted)) == step["fullcov"]
            assert step["fullcov"] or not step["homecov"]  # the blocks alone never cover more
            seen += 1
    assert seen


@pytest.mark.parametrize("method", ["jigsaw_index", "filterall", "direct"])
def test_steps_record_times_and_the_witness(cstore, method):
    """Step and solver-call times place every event on the query clock; on positives each pruned step says
    where the planted witness went, consistently with its coverage and with soundness."""
    gstore, sets = cstore
    c = _make(gstore, "R1", method)
    kept = 0
    for q in sets["R1"].queries[:16]:
        rec = c.run(q, time_limit=30.0)
        starts = [s["t_start"] for s in rec["steps"]]
        assert starts == sorted(starts)
        for step in rec["steps"]:
            if "t_end" in step:
                assert step["t_start"] <= step["t_end"] <= rec["elapsed"] + 1e-9
            for call in step.get("solver_calls", []):
                assert step["t_start"] <= call["t"] <= step["t_end"]
            if q.kind != "positive" or "prune" not in step:
                assert "witness" not in step
                continue
            w = step["witness"]
            n = len(q.planted)
            if "covered" in step:
                assert w["in_candidate"] == step["covered"]  # planted nodes carry query labels
            if step["final"]:
                assert w["in_candidate"] == n
            if w["in_candidate"] == n:
                assert w["alive"] == n and w["component"] == "kept"  # sound pruning, no cap
                kept += 1
            if step.get("status") == backends.FOUND and w["target"] is not None:
                assert step["solver_calls"][-1]["target"] <= w["target"]  # targets run in order
    assert kept or method == "direct"


class RecordingTimeout:
    """Times out after half of every limit it is given and records the limits."""

    name = "fake_timeout"

    def __init__(self) -> None:
        self.limits: list[float] = []

    def solve(self, query, target, *, time_limit, max_solutions=1):
        self.limits.append(time_limit)
        time.sleep(max(0.0, min(time_limit, 0.02)))
        return backends.SolveOutcome(backends.TIMEOUT, [], min(time_limit, 0.02), self.name, {})


def test_step_caps_and_final_remainder(cstore):
    gstore, sets = cstore
    fake = RecordingTimeout()
    c = _make(gstore, "R1", "jigsaw_index", backend=fake, step_cap=0.01, budgets=[1, 2, 3, 4, 5])
    q = sets["R1"].select(kind="positive")[0]
    rec = c.run(q, time_limit=5.0)
    assert rec["status"] == backends.TIMEOUT
    solved = [s for s in rec["steps"] if "solve_budget" in s]
    finals = [s for s in solved if s["closed"]]
    assert finals and finals[-1] is solved[-1]
    for s in solved[:-1]:
        assert s["solve_budget"] <= 0.01 + 1e-9
    assert solved[-1]["solve_budget"] > 1.0
    assert all(limit <= 0.01 + 1e-9 for limit in fake.limits[:-1])


def test_identical_candidates_are_skipped(cstore):
    gstore, sets = cstore
    c = _make(gstore, "R1", "jigsaw_index", budgets=[1, 1, 1])
    for q in sets["R1"].select(kind="negative"):
        rec = c.run(q, time_limit=30.0)
        steps = rec["steps"]
        if steps and steps[0].get("status") == backends.NONE and not steps[0]["closed"]:
            assert steps[1]["skipped"] == "identical candidate" and steps[1]["status"] == backends.NONE
            assert "solver_calls" not in steps[1]
            return
    pytest.skip("no negative left step 0 open")


class WrongWitness:
    name = "fake_wrong"

    def solve(self, query, target, *, time_limit, max_solutions=1):
        if target.n < query.n:
            return backends.SolveOutcome(backends.NONE, [], 0.0, self.name, {})
        return backends.SolveOutcome(backends.FOUND, [np.arange(query.n)[::-1].copy()], 0.0, self.name, {})


def test_rejected_witness_is_an_error(cstore):
    gstore, sets = cstore
    c = _make(gstore, "R1", "filterall", backend=WrongWitness())
    errors = 0
    for q in sets["R1"].select(kind="positive")[:6]:
        rec = c.run(q, time_limit=10.0)
        if rec["status"] == backends.ERROR:
            assert rec["error"].startswith("witness rejected") and rec["time_found"] is None
            errors += 1
    assert errors > 0


def test_enumeration_counts_every_embedding(cstore):
    import networkx as nx
    from networkx.algorithms import isomorphism

    gstore, sets = cstore
    c = _make(gstore, "R0", "filterall")
    q = sets["R0"].select(kind="positive", size=8)[0]
    res = cascade.enumerate_embeddings(c, q, time_limit=60.0, limit=10_000)
    assert res["status"] == "complete"
    codes = gstore.sigma("R0").query_codes(q.payload, gstore)
    stats: dict = {}
    cand = c.global_candidate(codes, stats)
    gt, gq = nx.Graph(), nx.Graph()
    for g, h in ((cand, gt), (q.local_graph(codes), gq)):
        for i, lab in enumerate(np.asarray(g.labels).tolist()):
            h.add_node(i, label=lab)
        h.add_edges_from(map(tuple, g.edge_list().tolist()))
    matcher = isomorphism.GraphMatcher(gt, gq, node_match=lambda a, b: a["label"] == b["label"])
    assert res["embeddings"] == sum(1 for _ in matcher.subgraph_monomorphisms_iter())
    capped = cascade.enumerate_embeddings(c, q, time_limit=60.0, limit=1)
    assert capped["embeddings"] == 1 and capped["status"] == "partial"


def test_setup_errors(cstore):
    gstore, _ = cstore
    with pytest.raises(ValueError):
        cascade.Cascade(gstore, "R1", "classical_gql", backends.NetworkXBackend())
    with pytest.raises(ValueError):
        cascade.Cascade(gstore, "R1", "jigsaw_index", backends.NetworkXBackend())
    bank = rankers.RankerBank(gstore, "R1")
    with pytest.raises(ValueError):
        cascade.Cascade(gstore, "R1", "jigsaw_learned", backends.NetworkXBackend(), bank=bank)
    c = cascade.Cascade(gstore, "R1", "filterall", backends.NetworkXBackend())
    assert c.budgets == [gstore.num_blocks]
    bad = [("filterall", {"policy": "topk"}, "selects no blocks"),
           ("filterall", {"policy_params": {"seed_count": 5}}, "selects no blocks"),
           ("jigsaw_index", {"policy": "greedy"}, "unknown selection policy"),
           ("jigsaw_index", {"policy": "topk", "policy_params": {"seed_count": 5}}, "hybrid policy only"),
           ("jigsaw_index", {"halo": -1}, "hops")]
    for method, kw, match in bad:
        with pytest.raises(ValueError, match=match):
            _make(gstore, "R1", method, **kw)


def test_sensitivity_partitions_scale_the_ladder():
    """On a sensitivity partition of B blocks a ladder step of K primary blocks (of P) takes ceil(K * B / P)."""
    from scripts.canonical import protocol

    class Store:
        dataset = "arxiv"

        def __init__(self, name: str, blocks: int) -> None:
            self.partition_name, self.num_blocks = name, blocks

        def sigma(self, regime):
            return None

    P = int(protocol.DATASETS["arxiv"]["partitions"])
    ladder = protocol.ladder_for("arxiv")

    def budgets(name, blocks, method="jigsaw_index"):
        return cascade.Cascade(Store(name, blocks), "R0", method, backends.NetworkXBackend(), bank=object()).budgets

    assert budgets("primary", P) == ladder and budgets("kahip_fast", P) == ladder
    assert budgets("metis_x0.5", P // 2) == sorted({max(1, -(-k * (P // 2) // P)) for k in ladder})
    assert budgets("metis_x2", 2 * P) == [2 * k for k in ladder]
    assert budgets("metis_x0.5", P // 2)[-1] == P // 2 and budgets("metis_x2", 2 * P)[-1] == 2 * P
    assert budgets("metis_x2", 2 * P, "filterall") == [2 * P]


def test_selection_settings_stay_sound(cstore, monkeypatch):
    """E16: top-k selection, other hybrid seed counts and a two-hop halo reorder or widen the candidates and
    keep every answer within the truth of FilterAll."""
    gstore, sets = cstore
    qs = [q for q in sets["R1"].queries if q.size <= 8]
    truth = {q.uid: _make(gstore, "R1", "filterall").run(q, time_limit=30.0)["status"] for q in qs}
    made = []
    real = rankers.make_selector
    monkeypatch.setattr(rankers, "make_selector",
                        lambda policy, ranking, part, **params: made.append((policy, params)) or
                        real(policy, ranking, part, **params))
    variants = [({"policy": "topk"}, ("topk", {})), ({"policy_params": {"seed_count": 1}}, ("hybrid", {"seed_count": 1})),
                ({"policy_params": {"seed_count": 40}}, ("hybrid", {"seed_count": 40})), ({"halo": 2}, ("hybrid", {}))]
    for kw, selector in variants:
        made.clear()
        c = _make(gstore, "R1", "jigsaw_index", **kw)
        assert c.options()["policy"] == selector[0] and c.options()["policy_params"] == selector[1]
        for q in qs:
            rec = c.run(q, time_limit=20.0)
            assert rec["options"] == c.options()
            assert rec["status"] != backends.ERROR, rec.get("error")
            if rec["status"] in (backends.FOUND, backends.NONE):
                assert rec["status"] == truth[q.uid], (kw, q.uid, rec["status"], truth[q.uid])
            if q.kind == "positive":
                assert rec["status"] != backends.NONE
        assert made and set(map(repr, made)) == {repr(selector)}


def test_edge_type_pair_certificate(tmp_path):
    """Leaves (type 1) attach only to hubs (type 0), so a query edge between two leaves is refuted by types."""
    from scripts.canonical import sigma

    n_hub, n_leaf = 12, 24
    n = n_hub + n_leaf
    edges = [(h, (h + 1) % n_hub) for h in range(n_hub)]
    edges += [(n_hub + i, i % n_hub) for i in range(n_leaf)] + [(n_hub + i, (i + 1) % n_hub) for i in range(n_leaf)]
    node_type = np.r_[np.zeros(n_hub), np.ones(n_leaf)].astype(np.int8)
    attrs = np.zeros((n, 2), dtype=np.float32)
    block = np.arange(n) % 2
    store_mod.write_store(tmp_path, "typed", edges=np.asarray(edges, dtype=np.int64), num_nodes=n,
                          node_type=node_type, attrs=attrs, node_block=block, node_fine=block * 3 + np.arange(n) % 3,
                          classes=np.zeros(n, dtype=np.int32), meta={"schema": "synthetic"},
                          partition_info={"tool": "planted", "seed": 0})
    sigma.build_store_regimes(tmp_path / "typed", ["R1"])
    gstore = store_mod.GraphStore(tmp_path, "typed", mmap=False)
    cascade.build_label_index(gstore, "R1")
    # leaves 12 and 13 share hub 1; the query closes the triangle 12-1-13 with a leaf-leaf edge
    planted = np.asarray([12, 1, 13], dtype=np.int64)
    payload = queries.payload_for(gstore, planted, "R1")
    for q_edges, want in (([(0, 1), (1, 2), (0, 2)], "edge type pair"), ([(0, 1), (1, 2)], None)):
        q = queries.Query("test/typed", "neg_structure", 3, "negative", "R1", np.asarray(q_edges, dtype=np.int64),
                          payload, planted, 0, 0, {}, certified="uncertified")
        for method in ("filterall", "jigsaw_index"):
            bank = rankers.RankerBank(gstore, "R1") if method != "filterall" else None
            c = cascade.Cascade(gstore, "R1", method, backends.NetworkXBackend(), bank=bank, budgets=[1])
            if want is None and method == "jigsaw_index":
                continue  # the positive needs ranker artifacts; filterall decides it
            rec = c.run(q, time_limit=10.0)
            if want:
                assert rec["status"] == backends.NONE and rec["certificate"] == want and rec["steps"] == []
            else:
                assert rec["status"] == backends.FOUND and "certificate" not in rec


ABLATIONS = [{"halo": False}, {"prune_passes": 0}, {"component_mode": "none"}, {"component_cap": 1},
             {"label_filter": False}]


@pytest.mark.parametrize("variant", ABLATIONS, ids=lambda v: "-".join(f"{k}={v[k]}" for k in v))
def test_operator_ablations_stay_sound(cstore, variant):
    """Every ablation answers within the truth of FilterAll; a capped component list never proves NONE."""
    gstore, sets = cstore
    qs = [q for q in sets["R1"].queries if q.size <= 8]
    truth = {q.uid: _make(gstore, "R1", "filterall").run(q, time_limit=30.0)["status"] for q in qs}
    c = _make(gstore, "R1", "jigsaw_index", **variant)
    assert all(c.options()[k] == v for k, v in variant.items())
    for q in qs:
        rec = c.run(q, time_limit=20.0)
        assert rec["options"] == c.options()
        assert rec["status"] != backends.ERROR, rec.get("error")
        if rec["status"] in (backends.FOUND, backends.NONE):
            assert rec["status"] == truth[q.uid], (q.uid, rec["status"], truth[q.uid])
        if q.kind == "positive":
            assert rec["status"] != backends.NONE
        for step in rec["steps"]:
            if variant.get("halo") is False and "halo_nodes" in step.get("candidate", {}):
                assert step["candidate"]["halo_nodes"] == 0 and step["candidate"]["sigma_kept_halo"] == 0
            if step.get("status") == backends.NONE and step.get("prune", {}).get("components_capped"):
                raise AssertionError("a capped step reported NONE")
            w = step.get("witness")
            if w and w["in_candidate"] == w["nodes"]:  # every ablation stays sound: only the cap may drop it
                assert w["alive"] == w["nodes"] and w["component"] in ("kept", "capped", "unfiltered"), w
        if variant.get("component_cap") == 1 and rec["status"] == cascade.INCOMPLETE:
            assert rec["time_decided"] is None and rec["decided_step"] is None


def test_capped_components_give_incomplete(tmp_path):
    """Two 6-cycles with alternating labels and a 4-cycle query: LDF/NLF keep both cycles, neither holds a
    4-cycle, so the full list proves NONE and a list capped at one component proves nothing."""
    from scripts.canonical import sigma

    n = 12
    edges = [(i, (i + 1) % 6) for i in range(6)] + [(6 + i, 6 + (i + 1) % 6) for i in range(6)]
    store_mod.write_store(tmp_path, "cycles", edges=np.asarray(edges, dtype=np.int64), num_nodes=n,
                          node_type=np.zeros(n, dtype=np.int8), attrs=np.zeros((n, 2), dtype=np.float32),
                          node_block=np.repeat([0, 1], 6), node_fine=np.repeat([0, 1, 2, 3], 3),
                          classes=(np.arange(n) % 2).astype(np.int32), meta={"schema": "synthetic"},
                          partition_info={"tool": "planted", "seed": 0})
    sigma.build_store_regimes(tmp_path / "cycles", ["R1"])
    gstore = store_mod.GraphStore(tmp_path, "cycles", mmap=False)
    cascade.build_label_index(gstore, "R1")
    planted = np.asarray([0, 1, 2, 3], dtype=np.int64)
    q = queries.Query("test/c4", "neg_structure", 4, "negative", "R1", np.asarray([(0, 1), (1, 2), (2, 3), (0, 3)]),
                      queries.payload_for(gstore, planted, "R1"), planted, 0, 0, {}, certified="uncertified")
    full = cascade.Cascade(gstore, "R1", "filterall", backends.NetworkXBackend()).run(q, time_limit=10.0)
    assert full["status"] == backends.NONE and full["decided_step"] == 0
    assert full["steps"][0]["prune"]["components_kept"] == 2
    capped = cascade.Cascade(gstore, "R1", "filterall", backends.NetworkXBackend(), component_cap=1).run(q, time_limit=10.0)
    assert capped["status"] == cascade.INCOMPLETE and capped["time_decided"] is None
    assert capped["steps"][0]["prune"]["components_capped"] == 1


def test_skip_below_sends_small_label_sets_straight_to_the_whole_graph_step(cstore):
    """A query whose label nodes are at or below the threshold skips retrieval: one final step, no ranking."""
    gstore, sets = cstore
    always = _make(gstore, "R1", "jigsaw_index", skip_below=10**9)
    never = _make(gstore, "R1", "jigsaw_index", skip_below=1)
    plain = _make(gstore, "R1", "jigsaw_index")
    full = _make(gstore, "R1", "filterall")
    assert always.options()["skip_below"] == 10**9 and plain.options()["skip_below"] is None
    seen = 0
    for q in sets["R1"].queries[:16]:
        rec, ref = always.run(q, time_limit=30.0), full.run(q, time_limit=30.0)
        assert rec["status"] == ref["status"]
        if rec["steps"]:
            assert rec["skipped_ladder"] and "ranker_info" not in rec and len(rec["steps"]) == 1
            step = rec["steps"][0]
            assert step["final"] and step["budget"] == gstore.num_blocks and "escalated" not in step
            seen += 1
        a, b = never.run(q, time_limit=30.0), plain.run(q, time_limit=30.0)
        assert a["status"] == b["status"] and [s["budget"] for s in a["steps"]] == [s["budget"] for s in b["steps"]]
        assert "skipped_ladder" not in a and "ranker_info" in a or not a["steps"]
    assert seen


def test_escalate_jumps_to_the_whole_graph_step(cstore):
    """After the escalation time the next step is the final one: the same decision as the full scan, one step."""
    gstore, sets = cstore
    jump = _make(gstore, "R1", "jigsaw_index", escalate=1e-9)
    full = _make(gstore, "R1", "filterall")
    never = _make(gstore, "R1", "jigsaw_index", escalate=1e9)
    plain = _make(gstore, "R1", "jigsaw_index")
    assert jump.options()["escalate"] == 1e-9 and plain.options()["escalate"] is None
    seen = 0
    for q in sets["R1"].queries[:16]:
        rec, ref = jump.run(q, time_limit=30.0), full.run(q, time_limit=30.0)
        assert rec["status"] == ref["status"]
        if rec["steps"]:
            assert len(rec["steps"]) == 1
            step = rec["steps"][0]
            assert step["escalated"] and step["final"] and step["budget"] == gstore.num_blocks
            seen += 1
        a, b = never.run(q, time_limit=30.0), plain.run(q, time_limit=30.0)
        assert [s["budget"] for s in a["steps"]] == [s["budget"] for s in b["steps"]]
        assert not any(s.get("escalated") for s in a["steps"])
    assert seen


def test_planner_settings_default_to_the_protocol_and_off_switches_them_off(cstore, monkeypatch):
    gstore, sets = cstore
    monkeypatch.setitem(protocol.LADDER, "skip_below_nodes", {gstore.dataset: {"R1": 10**9}})
    monkeypatch.setitem(protocol.LADDER, "escalate_seconds", {gstore.dataset: {"R1": 7, "R1-x": 9}})
    tuned = _make(gstore, "R1", "jigsaw_index")
    assert tuned.options()["skip_below"] == 10**9 and tuned.options()["escalate"] == 7.0
    explicit = _make(gstore, "R1", "jigsaw_index", skip_below=5, escalate=2)
    assert explicit.options()["skip_below"] == 5 and explicit.options()["escalate"] == 2.0
    off = _make(gstore, "R1", "jigsaw_index", skip_below=cascade.LADDER_OFF, escalate=cascade.LADDER_OFF)
    assert off.options()["skip_below"] is None and off.options()["escalate"] is None
    assert _make(gstore, "R0", "jigsaw_index").options()["skip_below"] is None  # no value for the regime
    assert _make(gstore, "R1", "filterall").options()["skip_below"] is None  # no ladder to plan
    with pytest.raises(ValueError):
        _make(gstore, "R1", "filterall", skip_below=5)
    rec = tuned.run(sets["R1"].queries[0], time_limit=30.0)
    assert rec["status"] == _make(gstore, "R1", "jigsaw_index", skip_below=cascade.LADDER_OFF).run(
        sets["R1"].queries[0], time_limit=30.0)["status"]
    if rec["steps"]:
        assert rec["skipped_ladder"]


def test_ladder_setting_prefers_the_exact_regime_over_its_family(monkeypatch):
    monkeypatch.setitem(protocol.LADDER, "escalate_seconds",
                        {"arxiv": {"R1": 3, "R2": 4, "R2-512": 6}, "mag": {"R0": None}})
    assert protocol.ladder_setting("escalate_seconds", "arxiv", "R1") == 3.0
    assert protocol.ladder_setting("escalate_seconds", "arxiv", "R2-512") == 6.0
    assert protocol.ladder_setting("escalate_seconds", "arxiv", "R2-128") == 4.0
    assert protocol.ladder_setting("escalate_seconds", "arxiv", "R0") is None
    assert protocol.ladder_setting("escalate_seconds", "mag", "R0") is None
    assert protocol.ladder_setting("escalate_seconds", "corafull", "R1") is None
    monkeypatch.setitem(protocol.LADDER, "escalate_seconds", None)
    assert protocol.ladder_setting("escalate_seconds", "arxiv", "R1") is None
