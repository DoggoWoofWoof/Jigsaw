from __future__ import annotations

import json
from collections import deque

import numpy as np
import pytest

from scripts.canonical import protocol, queries, store as store_mod, verify

FAMILIES = protocol.POSITIVE_FAMILIES
SM = [("sm_any", 4), ("sm_dense", 8), ("sm_sparse", 8), ("sm_dense", 16), ("sm_sparse", 16)]


@pytest.fixture(scope="module")
def gstore(clustered_store):
    root, _ = clustered_store
    return store_mod.GraphStore(root, "synth", mmap=True)


@pytest.fixture(scope="module")
def ctx(gstore):
    return queries.QueryContext(gstore)


def _adjacency(gstore):
    indptr, indices = np.asarray(gstore.indptr), np.asarray(gstore.indices)
    return [set(indices[indptr[u]:indptr[u + 1]].tolist()) for u in range(gstore.num_nodes)]


def _distances(adj, source, limit):
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if dist[u] >= limit:
            continue
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                queue.append(w)
    return dist


def _draws(ctx, family, size, count=20, cap=400):
    """The first `count` successful draws (a family may fail on some draws; cap bounds the search)."""
    out = []
    for k in range(cap):
        rng = np.random.default_rng(protocol.derive_seed("test", family, size, k))
        q = ctx.sample(family, size, rng, index=k)
        if q is not None:
            out.append(q)
            if len(out) == count:
                break
    return out


# ----------------------------------------------------------------------------- positive families
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("size", [8, 20])
def test_positive_families_are_valid_induced_queries(gstore, ctx, family, size):
    adj = _adjacency(gstore)
    made = _draws(ctx, family, size, count=12)
    assert len(made) == 12, f"{family}/{size}: only {len(made)} successes in 400 draws"
    tertiles = ctx.degree_tertiles()
    for q in made:
        nodes = q.planted.tolist()
        assert len(set(nodes)) == q.n and queries.min_nodes_for(size) <= q.n <= size
        brute = sorted((i, j) for i in range(q.n) for j in range(i + 1, q.n) if nodes[j] in adj[nodes[i]])
        assert q.edges.tolist() == [list(e) for e in brute]  # the induced subgraph, u < v, sorted
        assert queries.is_connected(q.n, q.edges)
        blocks = sorted(set(np.asarray(gstore.part.node_block)[q.planted].tolist()))
        assert q.info["blocks"] == blocks
        if family == "single":
            assert blocks == [q.info["anchor_block"]]
        elif family == "multi_fine":
            assert blocks == [q.info["anchor_block"]] and len(q.info["fine"]) >= 2
        elif family == "multi_coarse":
            assert len(blocks) >= 2
        elif family in ("k_hop", "degree_k_hop"):
            dist = _distances(adj, q.info["anchor"], 3)
            assert q.info["anchor"] in nodes and all(v in dist for v in nodes)
        if family == "degree_k_hop":
            assert q.info["anchor"] in set(tertiles[q.info["tertile"]].tolist())


@pytest.mark.parametrize("family,size", SM)
def test_sm_standard_density_rules(gstore, ctx, family, size):
    adj = _adjacency(gstore)
    made = _draws(ctx, family, size, count=8)
    assert len(made) == 8, f"{family}/{size}: only {len(made)} successes in 400 draws"
    for q in made:
        assert q.n == size and queries.is_connected(q.n, q.edges)
        nodes = q.planted.tolist()
        induced = {(i, j) for i in range(q.n) for j in range(i + 1, q.n) if nodes[j] in adj[nodes[i]]}
        got = {tuple(e) for e in q.edges.tolist()}
        avg = 2.0 * q.m / q.n
        if family == "sm_sparse":
            assert got <= induced and avg < 3.0
        else:
            assert got == induced
        if family == "sm_dense":
            assert avg >= 3.0


def test_degree_tertile_follows_the_index(ctx):
    for k in range(6):
        q = None
        for attempt in range(20):
            q = ctx.sample("degree_k_hop", 8, np.random.default_rng([k, attempt]), index=k)
            if q is not None:
                break
        assert q is not None and q.info["tertile"] == k % 3


def test_sampling_is_deterministic(ctx):
    for family in FAMILIES + ["sm_dense"]:
        a = ctx.sample(family, 12, np.random.default_rng(99), index=4)
        b = ctx.sample(family, 12, np.random.default_rng(99), index=4)
        assert (a is None) == (b is None)
        if a is not None:
            assert np.array_equal(a.planted, b.planted) and np.array_equal(a.edges, b.edges) and a.info == b.info


def test_small_sizes_do_not_crash(ctx):
    for family in FAMILIES:
        for size in (5, 6):
            for k in range(5):
                ctx.sample(family, size, np.random.default_rng(k), index=k)


def test_nodeset_hash_ignores_order():
    a = np.asarray([5, 1, 9], dtype=np.int64)
    assert queries.nodeset_hash(a) == queries.nodeset_hash(a[::-1])
    assert queries.nodeset_hash(a) != queries.nodeset_hash(np.asarray([5, 1, 8]))


# ----------------------------------------------------------------------------- coverage
def test_cover_incidence_matches_brute_force_and_candidate(gstore, ctx):
    adj = _adjacency(gstore)
    node_block = np.asarray(gstore.part.node_block)
    for family in ("single", "multi_coarse", "random_walk"):
        for q in _draws(ctx, family, 20, count=6):
            idx, blk = queries.cover_incidence(gstore, q.planted)
            got = {(int(i), int(b)) for i, b in zip(idx, blk)}
            want = {(i, int(node_block[v])) for i, v in enumerate(q.planted.tolist())}
            want |= {(i, int(node_block[w])) for i, v in enumerate(q.planted.tolist()) for w in adj[v]}
            assert got == want
            cover = queries.greedy_cover(gstore, q.planted)
            assert queries.covers(gstore, q.planted, cover)
            assert not queries.covers(gstore, q.planted, [])
            assert len(cover) == len(set(cover))
            for regime in ("R0", "R1", "R2-32"):
                payload = queries.payload_for(gstore, q.planted, regime)
                codes = gstore.sigma(regime).query_codes(payload, gstore)
                cand = gstore.candidate(cover, codes, regime)
                assert np.isin(q.planted, cand.global_ids).all()


def test_greedy_cover_breaks_ties_to_the_lower_block(gstore):
    node = int(np.flatnonzero(np.asarray(gstore.part.node_block) == 3)[0])
    lone = np.asarray([node], dtype=np.int64)
    _, blk = queries.cover_incidence(gstore, lone)
    assert queries.greedy_cover(gstore, lone) == [int(blk.min())]


# ----------------------------------------------------------------------------- payloads and negatives
@pytest.mark.parametrize("regime", ["R0", "R1", "R2-512", "R2-32"])
def test_payload_matches_store_codes_and_checker(gstore, ctx, regime):
    table = gstore.sigma(regime)
    for q in _draws(ctx, "k_hop", 12, count=5):
        payload = queries.payload_for(gstore, q.planted, regime)
        assert np.array_equal(table.query_codes(payload, gstore), np.asarray(table.codes)[q.planted])
        assert verify.check_embedding(gstore, payload, q.edges, q.planted, regime).ok


def _base(ctx, family="k_hop", size=12, seed=3):
    for k in range(50):
        q = ctx.sample(family, size, np.random.default_rng([seed, k]), index=k)
        if q is not None:
            return q
    raise AssertionError("no base query")


@pytest.mark.parametrize("regime", ["R0", "R1", "R2-512", "R2-32"])
def test_absent_label_gives_one_fresh_code(gstore, ctx, regime):
    table = gstore.sigma(regime)
    made = 0
    for seed in range(8):
        q = _base(ctx, seed=seed)
        payload = queries.payload_for(gstore, q.planted, regime)
        before = {k: v.copy() for k, v in payload.items()}
        res = queries.absent_label(gstore, regime, q, payload, np.random.default_rng(seed))
        for k in payload:
            assert np.array_equal(payload[k], before[k])  # the base payload is not mutated
        if res is None:
            # Only a hashed regime can run out of absent keys: every bucket of every query type is taken.
            assert regime.startswith("R2-")
            buckets = int(protocol.LABEL_REGIMES[regime]["buckets"])
            for t in np.unique(payload["types"]):
                assert np.unique(table.keys[table.keys[:, 0] == int(t), 1]).shape[0] == buckets
            continue
        made += 1
        new, info = res
        codes_old = table.query_codes(payload, gstore)
        codes_new = table.query_codes(new, gstore)
        j = info["node"]
        assert codes_new[j] >= table.num_codes
        others = np.arange(q.n) != j
        assert np.array_equal(codes_new[others], codes_old[others])
        assert np.array_equal(new["types"], payload["types"])
    if regime != "R2-32":
        assert made == 8


@pytest.mark.parametrize("regime", ["R0", "R1", "R2-32"])
def test_swapped_label_takes_a_nearby_same_type_code(gstore, ctx, regime):
    table = gstore.sigma(regime)
    adj = _adjacency(gstore)
    codes = np.asarray(table.codes)
    for seed in range(6):
        q = _base(ctx, seed=seed)
        payload = queries.payload_for(gstore, q.planted, regime)
        res = queries.swapped_label(gstore, ctx, regime, q, payload, np.random.default_rng(seed))
        assert res is not None
        new, info = res
        j, donor, image = info["node"], info["donor"], info["image"]
        assert image == int(q.planted[j])
        assert int(gstore.node_type[donor]) == int(payload["types"][j])
        new_codes = table.query_codes(new, gstore)
        assert new_codes[j] == codes[donor] != codes[image] and new_codes[j] < table.num_codes
        others = np.arange(q.n) != j
        assert np.array_equal(new_codes[others], codes[q.planted][others])
        if info["donor_source"] == "near":
            assert donor in _distances(adj, image, 2)


def test_structure_negative_adds_one_distance_two_edge(ctx):
    for seed in range(8):
        q = _base(ctx, family="random_walk", seed=seed)
        edges, info = queries.structure_negative(q, np.random.default_rng(seed))
        old = {tuple(e) for e in q.edges.tolist()}
        new = {tuple(e) for e in edges.tolist()}
        assert old < new and new - old == {tuple(info["added_edge"])}
        a, b = info["added_edge"]
        if info["distance"] == 2:
            nbr = {u: set() for u in range(q.n)}
            for u, v in old:
                nbr[u].add(v)
                nbr[v].add(u)
            assert nbr[a] & nbr[b]


def test_structure_negative_of_a_clique_is_none():
    clique = queries.PlantedQuery("k_hop", 4, np.arange(4), queries.normalize_edges(
        np.asarray([(i, j) for i in range(4) for j in range(i + 1, 4)])))
    assert queries.structure_negative(clique, np.random.default_rng(0)) is None


def test_structure_negative_skips_type_pairs_absent_from_the_data():
    # path 0-1-2-3 with types [1, 0, 1, 0]: the pairs at distance 2 are (0, 2) of types (1, 1)
    # and (1, 3) of types (0, 0); the non-edge at distance 3 is (0, 3) of types (1, 0)
    path = queries.PlantedQuery("k_hop", 4, np.arange(4), queries.normalize_edges(np.asarray([(0, 1), (1, 2), (2, 3)])))
    types = np.asarray([1, 0, 1, 0])
    no_11 = np.asarray([[True, True], [True, False]])
    for seed in range(12):
        edges, info = queries.structure_negative(path, np.random.default_rng(seed), node_types=types, type_pairs=no_11)
        assert info["added_edge"] == [1, 3] and info["distance"] == 2
    only_01 = np.asarray([[False, True], [True, False]])
    edges, info = queries.structure_negative(path, np.random.default_rng(0), node_types=types, type_pairs=only_01)
    assert info["added_edge"] == [0, 3] and info["distance"] is None
    nothing = np.zeros((2, 2), dtype=bool)
    assert queries.structure_negative(path, np.random.default_rng(0), node_types=types, type_pairs=nothing) is None


def test_label_pair_on_edge_matches_brute_force(gstore):
    for regime in ("R0", "R1", "R2-32"):
        codes = np.asarray(gstore.sigma(regime).codes)
        pairs = set()
        for u in range(gstore.num_nodes):
            for w in gstore.neighbors(u).tolist():
                pairs.add((int(codes[u]), int(codes[w])))
        rng = np.random.default_rng(3)
        for _ in range(40):
            a, b = (int(codes[x]) for x in rng.integers(0, gstore.num_nodes, size=2))
            assert queries.label_pair_on_edge(gstore, regime, a, b) == ((a, b) in pairs)


# ----------------------------------------------------------------------------- split builder and files
def _build(gstore, out, split="dev", master="20260607", exclude=(), regimes=("R0", "R1", "R2-32")):
    builder = queries.SplitBuilder(gstore, "synth", split, master, exclude=exclude, sizes=[8, 12], max_per_cell=2)
    report = builder.build(list(regimes), out, master_tag=master)
    return builder, report


@pytest.fixture(scope="module")
def built(gstore, tmp_path_factory):
    out = tmp_path_factory.mktemp("dev0")
    builder, report = _build(gstore, out)
    return out, builder, report


def test_build_writes_verified_query_sets(gstore, built):
    out, builder, report = built
    per_family = 2 * 2  # two sizes, two per cell
    for regime in ("R0", "R1", "R2-32"):
        qs = queries.QuerySet.load(out / f"{regime}.npz")
        assert qs.regime == regime and qs.meta["max_per_cell"] == 2 and qs.meta["sizes"] == [8, 12]
        positives = qs.select(kind="positive")
        for family in FAMILIES:
            assert len(qs.select(kind="positive", family=family)) == per_family
        sm = [q for q in positives if q.family.startswith("sm_")]
        assert len(sm) == 2 * len(queries.sm_cells())
        for q in positives:
            assert q.certified == "planted" and queries.covers(gstore, q.planted, q.info["cover_blocks"])
            assert verify.check_embedding(gstore, q.payload, q.edges, q.planted, regime).ok
        negatives = qs.select(kind="negative")
        skipped = sum(qs.meta["not_constructible"].values())
        assert len(negatives) + skipped == 3 * per_family
        for q in negatives:
            assert q.certified == "uncertified" and q.info["base_family"] in FAMILIES
        uids = [q.uid for q in qs]
        assert len(uids) == len(set(uids))
        assert len({q.content_hash() for q in qs}) == len(qs)
    assert report["regimes"]["R0"]["queries"] == len(queries.QuerySet.load(out / "R0.npz"))
    hashes = (out / "nodesets.txt").read_text(encoding="ascii").split()
    assert len(hashes) == len(set(hashes)) == len(builder.taken)
    json.loads((out / "queries_report.json").read_text(encoding="utf-8"))


def test_structure_negatives_record_the_label_pair(gstore, built):
    out, _, _ = built
    for regime in ("R0", "R1", "R2-32"):
        codes = np.asarray(gstore.sigma(regime).codes)
        seen = 0
        for q in queries.QuerySet.load(out / f"{regime}.npz"):
            if q.family != "neg_structure":
                continue
            a, b = (int(q.planted[j]) for j in q.info["added_edge"])
            assert not gstore.has_edge(a, b)
            assert q.info["label_pair_in_data"] == queries.label_pair_on_edge(gstore, regime, int(codes[a]), int(codes[b]))
            seen += 1
        assert seen


def test_structures_are_shared_across_regimes(built):
    out, _, _ = built
    by_regime = {r: {q.uid: q for q in queries.QuerySet.load(out / f"{r}.npz")} for r in ("R0", "R1", "R2-32")}
    for uid, q in by_regime["R0"].items():
        if q.kind == "positive" or q.family == "neg_structure":
            other = by_regime["R1"].get(uid)
            if other is not None:
                assert np.array_equal(q.planted, other.planted) and np.array_equal(q.edges, other.edges)


def test_negative_bases_cycle_through_the_tertiles(gstore):
    builder = queries.SplitBuilder(gstore, "synth", "dev", "20260607", sizes=[8])
    counts = {"positive": 0, "neg_absent_label": 18, "neg_swapped_label": 0, "neg_structure": 0, "sm_standard": 0}
    structures = builder.structures(counts)
    tertiles = [structures["negative"][("neg_absent_label", 8, i)][0].info["tertile"] for i in (5, 11, 17)]
    assert tertiles == [0, 1, 2]


def test_build_is_deterministic(gstore, built, tmp_path):
    out, _, _ = built
    _build(gstore, tmp_path)
    for regime in ("R0", "R1", "R2-32"):
        with np.load(out / f"{regime}.npz") as a, np.load(tmp_path / f"{regime}.npz") as b:
            assert np.array_equal(a["content_hash"], b["content_hash"])
            assert np.array_equal(a["planted"], b["planted"])
    assert (out / "nodesets.txt").read_text() == (tmp_path / "nodesets.txt").read_text()


def test_later_split_excludes_earlier_nodesets(gstore, built, tmp_path):
    out, first, _ = built
    exclude = queries.read_exclusions([out])
    second, report = _build(gstore, tmp_path, split="validation", master="31415", exclude=exclude,
                            regimes=("R0",))
    assert report["excluded_nodesets"] == len(exclude)
    mine = set((tmp_path / "nodesets.txt").read_text(encoding="ascii").split())
    assert exclude <= mine  # the file lists everything taken, the new sets included
    new = mine - exclude
    assert new and not (new & set(first.taken))
    for q in queries.QuerySet.load(tmp_path / "R0.npz"):
        if q.kind == "positive":
            assert queries.nodeset_hash(q.planted) in new


def test_load_rejects_tampered_content(built, tmp_path):
    out, _, _ = built
    with np.load(out / "R1.npz") as z:
        data = {k: z[k] for k in z.files}
    data["classes"] = data["classes"].copy()
    data["classes"][0] += 1
    np.savez_compressed(tmp_path / "bad.npz", **data)
    with pytest.raises(queries.QueryGenerationError):
        queries.QuerySet.load(tmp_path / "bad.npz")
    assert len(queries.QuerySet.load(tmp_path / "bad.npz", verify_hashes=False)) == len(data["uid"])


def test_split_counts_and_sm_cells():
    assert queries.split_counts("dev", "R0")["sm_standard"] == 10
    test_r0 = queries.split_counts("test", "R0")
    assert test_r0["positive"] == 100 and test_r0["sm_standard"] == 50
    assert queries.split_counts("reserve_a", "R2-32")["positive"] == 50
    with pytest.raises(ValueError):
        queries.split_counts("nope", "R0")
    cells = queries.sm_cells()
    assert cells[0] == ("any", 4) and ("dense", 32) in cells and ("sparse", 8) in cells and len(cells) == 7


def test_cli_build_and_inspect(clustered_store, tmp_path, monkeypatch, capsys):
    root, _ = clustered_store
    monkeypatch.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"]))
    out = tmp_path / "cli"
    assert queries.main(["build", "--store-root", str(root), "--dataset", "synth", "--split", "dev",
                         "--regimes", "R0", "--out", str(out), "--max-per-cell", "1", "--sizes", "8"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["config"]["max_per_cell"] == 1
    capsys.readouterr()
    assert queries.main(["inspect", str(out / "R0.npz")]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["meta"]["master"] == "20260607" and shown["cells"]["positive/single/8"] == 1
    with pytest.raises(SystemExit):
        queries.main(["build", "--store-root", str(root), "--dataset", "synth", "--split", "test",
                      "--out", str(tmp_path / "x"), "--max-per-cell", "1"])


def test_duplicate_queries_are_redrawn(gstore, monkeypatch):
    """A draw whose query repeats an earlier one moves on: positives redraw, negatives modify again."""
    builder = queries.SplitBuilder(gstore, "synth", "dev", "20260607", sizes=[8])

    def pair(family, size, rng, index=0):
        planted = rng.choice(gstore.num_nodes, size=2, replace=False).astype(np.int64)
        return queries.PlantedQuery(family, size, planted, np.asarray([[0, 1]], dtype=np.int64), {"tertile": 0})

    def few_labels(store, nodes, regime):  # six queries up to isomorphism, one per family: collisions are certain
        nodes = np.asarray(nodes, dtype=np.int64)
        return {"types": np.zeros(len(nodes), dtype=np.int8), "classes": nodes % 3}

    monkeypatch.setattr(builder.ctx, "sample", pair)
    monkeypatch.setattr(queries, "payload_for", few_labels)
    counts = {"positive": 1, "neg_absent_label": 0, "neg_swapped_label": 0, "neg_structure": 0, "sm_standard": 0}
    structures = builder.structures(counts, {"R1": counts})
    drawn = [pq for pq, _, _ in structures["positive"].values()]
    # an edge with labels (a, b) is isomorphic to one with (b, a): the classes are the label multisets
    classes = {tuple(sorted(few_labels(gstore, pq.planted, "R1")["classes"].tolist())) for pq in drawn}
    assert len(classes) == len(drawn) == len(FAMILIES) and sum(builder.duplicates.values()) >= 1

    counts = {"positive": 0, "neg_absent_label": 2, "neg_swapped_label": 0, "neg_structure": 0, "sm_standard": 0}
    structures = builder.structures(counts)  # two bases with the same edges and types
    labels = iter([np.array([5, 5]), np.array([5, 5]), np.array([6, 6])])
    monkeypatch.setattr(queries, "absent_label",
                        lambda store, regime, pq, payload, rng: ({"types": payload["types"], "classes": next(labels)}, {}))
    made, extra = builder.regime_queries(structures, "R1", counts)
    assert [q.info["mod_attempt"] for q in made] == [0, 1]
    assert extra["negative_duplicates_redrawn"] == {"neg_absent_label/8": 1}


def _query(uid, edges, types, values, regime="R1"):
    key = "attrs" if regime == "R0" else ("classes" if regime == "R1" else "buckets")
    types = np.asarray(types, dtype=np.int8)
    return queries.Query(uid, "k_hop", len(types), "positive", regime, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                         {"types": types, key: np.asarray(values)}, np.arange(len(types), dtype=np.int64), 0, 0)


def test_query_index_decides_isomorphism():
    """Copies are found whatever their node and edge order; WL-equal graphs that are not isomorphic are not."""
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 2]]
    types, classes = np.asarray([0, 0, 1, 0, 1]), np.asarray([5, 6, 5, 7, 5])
    perm = np.asarray([3, 0, 4, 1, 2])  # copy node j is original node perm[j]
    copy_edges = np.argsort(perm)[np.asarray(edges)][::-1, ::-1]  # renumbered, listed backwards, each pair flipped
    original = _query("a", edges, types, classes)
    copy = _query("b", copy_edges, types[perm], classes[perm])
    relabelled = _query("c", edges, types, [5, 6, 5, 8, 5])
    assert original.content_hash() != copy.content_hash()
    assert queries.duplicate_pairs([original, copy, relabelled]) == [("a", "b")]

    # two triangles and a hexagon are 2-regular with equal labels: WL cannot tell them apart, VF2 does
    triangles = _query("t", [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], [0] * 6, [1] * 6)
    hexagon = _query("h", [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], [0] * 6, [1] * 6)
    index = queries.QueryIndex("R1")
    a, b = index.item(triangles.edges, triangles.payload), index.item(hexagon.edges, hexagon.payload)
    assert a.key == b.key and index._wl(a) == index._wl(b)
    index.add(a, "t")
    assert index.find(b) is None

    # R0 labels are attribute rows compared exactly: -0.0 equals 0.0, any other change counts
    rows = np.asarray([[0.0, 1.5], [2.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    moved = np.asarray([[2.0, -0.0], [0.0, 1.5], [0.5, 0.5]], dtype=np.float32)  # rows 1, 0, 2; the centre is node 1
    changed = np.asarray([[0.0, 1.5], [2.0, 0.0], [0.5, 0.25]], dtype=np.float32)
    pairs = queries.duplicate_pairs([_query("s", [[0, 1], [0, 2]], [0, 0, 0], rows, regime="R0"),
                                     _query("m", [[1, 0], [1, 2]], [0, 0, 0], moved, regime="R0"),
                                     _query("x", [[0, 1], [0, 2]], [0, 0, 0], changed, regime="R0")])
    assert pairs == [("s", "m")]


def test_cli_duplicates(tmp_path, capsys):
    clean = [_query(f"q{i}", [[0, 1], [1, 2]], [0, 0, 0], [i, i + 1, i + 2]) for i in range(3)]
    queries.save_query_set(tmp_path / "clean" / "R1.npz", clean, {"split": "dev"})
    assert queries.main(["duplicates", str(tmp_path / "clean")]) == 0
    backwards = _query("q3", [[2, 1], [1, 0]], [0, 0, 0], [3, 2, 1])  # q1's path read from the other end
    queries.save_query_set(tmp_path / "dup" / "R1.npz", clean + [backwards], {"split": "dev"})
    capsys.readouterr()
    assert queries.main(["duplicates", str(tmp_path)]) == 1
    shown = json.loads(capsys.readouterr().out)
    assert shown[str(tmp_path / "dup" / "R1.npz")]["duplicates"] == [["q1", "q3"]]
    assert shown[str(tmp_path / "clean" / "R1.npz")]["duplicates"] == []
