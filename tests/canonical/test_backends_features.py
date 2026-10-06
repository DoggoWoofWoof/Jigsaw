from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import backends, features, store as store_mod, verify

FAKE = str(Path(__file__).with_name("fake_glasgow.py"))


def _graph(n, edges, labels):
    return store_mod.local_graph_from_edges(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                                            labels=np.asarray(labels))


# ----------------------------------------------------------------------------- trivially_none
def test_trivially_none_cases():
    tri = _graph(3, [[0, 1], [1, 2], [0, 2]], [1, 1, 2])
    path = _graph(3, [[0, 1], [1, 2]], [1, 1, 2])
    assert backends.trivially_none(path, tri) is None
    assert backends.trivially_none(tri, path) == "query has more edges than target"
    big = _graph(4, [[0, 1]], [1, 1, 1, 1])
    assert backends.trivially_none(big, tri) == "query larger than target"
    beyond = _graph(2, [[0, 1]], [1, 9])  # label above every target label
    assert backends.trivially_none(beyond, tri) is not None
    below = _graph(2, [[0, 1]], [0, 1])  # label below every target label
    assert backends.trivially_none(below, tri) is not None
    too_many = _graph(3, [[0, 1]], [2, 2, 1])  # target has one node labelled 2
    assert backends.trivially_none(too_many, tri) is not None
    empty = _graph(0, np.zeros((0, 2)), [])
    assert backends.trivially_none(empty, tri) is None
    assert backends.trivially_none(path, _graph(0, np.zeros((0, 2)), [])) == "query larger than target"


# ----------------------------------------------------------------------------- LAD writer and parser
def test_vertexlabelled_lad_format(tmp_path):
    g = _graph(4, [[0, 1], [1, 2], [0, 2]], [5, 6, 5, 7])
    out = tmp_path / "g.lad"
    backends.write_vertexlabelled_lad(g, out, chunk=2)
    lines = out.read_text(encoding="ascii").splitlines()
    assert lines == ["4", "5 2 1 2", "6 2 0 2", "5 2 0 1", "7 0"]


def test_parse_glasgow_output():
    text = "\n".join([
        "pattern_file = p.lad", "status = true", "runtime = 12", "nodes = 3",
        "mapping = (0 -> 7) (2 -> 1) (1 -> 4)",
        "mapping = (0 -> 8) (1 -> 9) (2 -> 3)",
        "solution_count = 2",
    ])
    status, embeddings, detail = backends.parse_glasgow_output(text, 3)
    assert status == "true"
    assert [e.tolist() for e in embeddings] == [[7, 4, 1], [8, 9, 3]]
    assert detail["solution_count"] == "2" and detail["runtime"] == "12"
    status, embeddings, _ = backends.parse_glasgow_output("status = aborted\n", 3)
    assert status == "aborted" and embeddings == []


# ----------------------------------------------------------------------------- Glasgow process handling
@pytest.fixture
def fake_glasgow(tmp_path):
    return backends.GlasgowBackend(binary=[sys.executable, FAKE], workdir=str(tmp_path))


@pytest.mark.parametrize("mode,status", [("found", backends.FOUND), ("none", backends.NONE),
                                         ("aborted", backends.TIMEOUT), ("crash", backends.ERROR)])
def test_glasgow_statuses(fake_glasgow, monkeypatch, mode, status):
    monkeypatch.setenv("FAKE_GLASGOW_MODE", mode)
    q = _graph(3, [[0, 1], [1, 2]], [1, 1, 1])
    t = _graph(3, [[0, 1], [1, 2]], [1, 1, 1])
    out = fake_glasgow.solve(q, t, time_limit=20.0)
    assert out.status == status, out.detail
    if status == backends.FOUND:
        assert out.embeddings[0].tolist() == [2, 1, 0]
    if status == backends.ERROR:
        assert "output_tail" in out.detail


def test_glasgow_enumeration_flags(fake_glasgow, monkeypatch):
    monkeypatch.setenv("FAKE_GLASGOW_MODE", "all")
    q = _graph(2, [[0, 1]], [1, 1])
    t = _graph(3, [[0, 1], [1, 2]], [1, 1, 1])
    out = fake_glasgow.solve(q, t, time_limit=20.0, max_solutions=5)
    assert out.status == backends.FOUND and len(out.embeddings) == 2
    monkeypatch.setenv("FAKE_GLASGOW_MODE", "echo_args")
    out = fake_glasgow.solve(q, t, time_limit=20.0, max_solutions=5)
    assert out.status == backends.ERROR
    tail = out.detail["output_tail"]
    assert "--print-all-solutions" in tail and "--solution-limit 5" in tail and "--timeout 20" in tail


def test_glasgow_is_killed_at_the_fractional_cap(fake_glasgow, monkeypatch):
    monkeypatch.setenv("FAKE_GLASGOW_MODE", "sleep")
    q = _graph(2, [[0, 1]], [1, 1])
    t = _graph(3, [[0, 1], [1, 2]], [1, 1, 1])
    t0 = time.perf_counter()
    out = fake_glasgow.solve(q, t, time_limit=1.5)
    wall = time.perf_counter() - t0
    assert out.status == backends.TIMEOUT and out.detail["killed_at_cap"]
    # killed long before the fake's 30 s sleep ends; the margin absorbs interpreter start-up on a loaded machine
    assert wall < 15.0
    assert out.detail["returncode"] != 0


def test_glasgow_trivial_none_skips_the_solver(tmp_path):
    missing = backends.GlasgowBackend(binary=str(tmp_path / "does-not-exist"))
    q = _graph(2, [[0, 1]], [1, 9])
    t = _graph(3, [[0, 1], [1, 2]], [1, 1, 1])
    out = missing.solve(q, t, time_limit=1.0)
    assert out.status == backends.NONE and "trivial" in out.detail


# ----------------------------------------------------------------------------- VF2 on the synthetic store
def _planted_query(gstore, rng, size=6):
    indptr, indices = np.asarray(gstore.indptr), np.asarray(gstore.indices)
    start = int(rng.integers(0, gstore.num_nodes))
    nodes = [start]
    frontier = list(indices[indptr[start]:indptr[start + 1]])
    while len(nodes) < size and frontier:
        v = int(frontier.pop(int(rng.integers(0, len(frontier)))))
        if v not in nodes:
            nodes.append(v)
            frontier.extend(indices[indptr[v]:indptr[v + 1]].tolist())
    nodes = np.asarray(nodes, dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(nodes)}
    edges = sorted({(min(pos[u], pos[int(w)]), max(pos[u], pos[int(w)]))
                    for u in nodes.tolist() for w in indices[indptr[u]:indptr[u + 1]] if int(w) in pos})
    return nodes, np.asarray(edges, dtype=np.int64).reshape(-1, 2)


@pytest.mark.parametrize("regime", ["R0", "R1", "R2-32"])
def test_vf2_finds_planted_queries_and_checker_agrees(synthetic_store, regime):
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth", mmap=False)
    rng = np.random.default_rng(11)
    solver = backends.NetworkXBackend()
    sigma = gstore.sigma(regime)
    for _ in range(5):
        nodes, edges = _planted_query(gstore, rng)
        types = np.asarray(gstore.node_type)[nodes]
        payload = {"types": types, "attrs": np.asarray(gstore.attrs)[nodes]}
        if regime == "R1":
            payload["classes"] = np.asarray(gstore.classes)[nodes]
        if regime.startswith("R2-"):
            payload["buckets"] = np.asarray([verify._bucket(int(t), np.asarray(gstore.attrs[v]), 32)
                                             for t, v in zip(types, nodes)])
        qcodes = sigma.query_codes(payload, gstore)
        assert np.array_equal(qcodes, np.asarray(sigma.codes)[nodes])
        query = store_mod.local_graph_from_edges(len(nodes), edges, labels=qcodes)
        blocks = np.unique(np.asarray(gstore.part.node_block)[nodes])
        target = gstore.candidate(blocks, qcodes, regime)
        out = solver.solve(query, target, time_limit=30.0)
        assert out.status == backends.FOUND
        mapping = target.global_ids[out.embeddings[0]]
        assert verify.check_embedding(gstore, payload, edges, mapping, regime).ok


def test_vf2_proves_absence():
    q = _graph(3, [[0, 1], [1, 2], [0, 2]], [1, 1, 1])
    t = _graph(4, [[0, 1], [1, 2], [2, 3], [3, 0]], [1, 1, 1, 1])
    out = backends.NetworkXBackend().solve(q, t, time_limit=10.0)
    assert out.status == backends.NONE


# ----------------------------------------------------------------------------- features
def test_neighbor_type_counts_and_relations(synthetic_store):
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth", mmap=True)
    indptr, indices = np.asarray(gstore.indptr), np.asarray(gstore.indices)
    types = np.asarray(gstore.node_type)
    counts = features.neighbor_type_counts(indptr, indices, types, 3)
    for u in range(0, gstore.num_nodes, 23):
        want = np.bincount(types[indices[indptr[u]:indptr[u + 1]]], minlength=3)
        assert np.array_equal(counts[u], want)
    pairs = features.relation_pairs_present(indptr, indices, types, 3, chunk=97)
    brute = sorted({(int(types[u]), int(types[v])) for u in range(gstore.num_nodes)
                    for v in indices[indptr[u]:indptr[u + 1]]})
    assert pairs == brute
    table = features.relation_table(pairs, 3)
    rel = features.edge_relations(indptr, indices, types, table)
    assert rel.shape[0] == indices.shape[0] and rel.min() >= 0
    lens = np.diff(indptr)
    rows = np.repeat(np.arange(gstore.num_nodes), lens)
    assert np.array_equal(table[types[indices], types[rows]], rel)


def test_encoder_inputs_shapes(synthetic_store):
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth")
    block = gstore.part.nodes_of_blocks([0])
    sub = gstore.candidate([0], np.unique(np.asarray(gstore.sigma("R0").codes)), "R0")
    keep = np.isin(sub.global_ids, block)
    g = sub.subgraph(keep)
    types = np.asarray(gstore.node_type)[g.global_ids]
    x0 = features.encoder_inputs("R0", g.indptr, g.indices, types, 3, attrs=np.asarray(gstore.attrs)[g.global_ids])
    assert x0.shape == (g.n, features.input_dim("R0", 8, 3)) and x0.dtype == np.float32
    assert np.allclose(x0[:, -3:], np.log1p(features.neighbor_type_counts(g.indptr, g.indices, types, 3)))
    table = gstore.sigma("R1")
    codes = np.asarray(table.codes)[g.global_ids]
    x1 = features.encoder_inputs("R1", g.indptr, g.indices, types, 3, codes=codes, sigma_table=table)
    assert x1.shape == (g.n, features.input_dim("R1", 8, 3))
    with pytest.raises(ValueError):
        features.encoder_inputs("R1", g.indptr, g.indices, types, 3)
