from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import backends, gnnpe_adapter as gp, queries, store as store_mod, verify

FAKE = [sys.executable, str(Path(__file__).with_name("fake_gnnpe.py"))]


def _store(synthetic_store):
    root, _ = synthetic_store
    return store_mod.GraphStore(root, "synth", mmap=False)


@pytest.fixture
def gstore(synthetic_store):
    return _store(synthetic_store)


def _planted(gstore, regime, seed, size=8):
    ctx = queries.QueryContext(gstore)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        planted = ctx.sample("k_hop", size, rng)
        if planted is not None:
            payload = queries.payload_for(gstore, planted.planted, regime)
            return planted, payload, gstore.sigma(regime).query_codes(payload, gstore)
    raise AssertionError("no k_hop query")


def _alternating(gstore):
    return np.arange(gstore.num_nodes, dtype=np.int64) % 2


@pytest.fixture(scope="module")
def prepared(synthetic_store, tmp_path_factory):
    """One offline stage and warm-up through the fake binary, shared by the online tests."""
    gstore = _store(synthetic_store)
    runner = gp.GNNPERunner(gstore, "R0", cache_dir=tmp_path_factory.mktemp("gnnpe"), binary=FAKE, partitions=2,
                            membership=_alternating(gstore))
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("FAKE_GNNPE_MODE", raising=False)
        info = runner.prepare()
    return gstore, runner, info


# ----------------------------------------------------------------------------- pure helpers
def test_semantics_cite_the_source():
    for text in [*gp.GNNPE_SEMANTICS.values(), *gp.SEMANTIC_DIFFERENCES.values()]:
        assert re.search(r"lines? \d+", text), text
    assert {"orientation", "early_break", "leaf_root", "small_queries", "path_length",
            "disconnected"} <= set(gp.SEMANTIC_DIFFERENCES)


def test_leaf_root_bounds():
    assert gp.node_capacity(2) == 40 and gp.leaf_root_max_paths(2) == 38
    assert gp.node_capacity(1) == (4096 - 5) // (16 * 3 + 4)
    assert gp.leaf_root_partitions([0, 1, 38, 39, 40, 500]) == [1, 2]


def test_estimate_path_count_matches_enumeration(gstore):
    indptr, indices = gstore.indptr, gstore.indices
    paths = set()
    for b in range(gstore.num_nodes):
        nbrs = indices[indptr[b]:indptr[b + 1]].tolist()
        paths.update((min(a, c), b, max(a, c)) for a in nbrs for c in nbrs if a != c)
    assert gp.estimate_path_count(indptr) == len(paths) == gp.estimate_path_count(indptr, chunk=7)


def test_write_membership_orders_by_degree(tmp_path):
    g = store_mod.local_graph_from_edges(5, np.array([[0, 2], [0, 3], [1, 2], [2, 3]]))  # degrees 2 1 3 2 0
    gp.write_membership(g.indptr, np.array([0, 1, 0, 1, 1]), tmp_path / "m.txt", chunk=2)
    assert (tmp_path / "m.txt").read_bytes() == b"4 1\n1 1\n0 0\n3 1\n2 0\n"
    with pytest.raises(ValueError):
        gp.write_membership(g.indptr, np.array([0, 1]), tmp_path / "bad.txt")


def test_metis_membership(gstore):
    assert not gp.metis_membership(gstore.indptr, gstore.indices, 1).any()
    parts = gp.metis_membership(gstore.indptr, gstore.indices, 3)
    assert parts.shape == (gstore.num_nodes,) and set(parts.tolist()) == {0, 1, 2}


def test_commands():
    off = gp.offline_command(["main"], "/data/x", "/data/x/g.graph", partitions=5)
    assert off == ["main", "-f", "/data/x/", "-d", "/data/x/g.graph", "-m", "offline", "-p", "5", "-l", "2",
                   "-e", "2"]
    on = gp.online_command(["main"], "/data/x/", "g", "q", partitions=3, max_solutions=None, print_embeddings=False)
    assert on == ["main", "-f", "/data/x/", "-d", "g", "-q", "q", "-m", "online", "-p", "3", "-l", "2", "-e", "2",
                  "-n", "MAX"]
    assert gp.online_command(["main"], "d", "g", "q", partitions=1, max_solutions=4)[-3:] == [
        "-n", "4", "--print-embeddings"]


def test_parse_realistic_output():
    text = ("|V|: 5, |E|: 4, |\u03a3|: 3\nMax Degree: 3, Max Label Frequency: 2\n2\n"
            "embedding: 0:4 1:2 2:3\nembedding: 0:1 1:2\nAnswer Number: 2 Query Time (ms): 1.23457e+06\n")
    status, embeddings, detail = gp.parse_gnnpe_output(text, 3)
    assert status == "complete" and [e.tolist() for e in embeddings] == [[4, 2, 3]]
    assert detail["malformed_lines"] == 1 and detail["answer_number"] == 2
    assert detail["query_time_ms"] == pytest.approx(1.23457e6)
    assert detail["data_graph"] == {"vertices": 5, "edges": 4, "labels": 3} and detail["query_plan_paths"] == 2
    status, embeddings, detail = gp.parse_gnnpe_output("|V|: 5, |E|: 4, |S|: 3\nembedding: 0:4 1:2 2:3\nembed", 3)
    assert status is None and len(embeddings) == 1 and detail["unterminated_tail"]
    status, _, detail = gp.parse_gnnpe_output("Can not open the graph file q.graph .\n", 3)
    assert status is None and detail["unreadable"].startswith("Can not open")


def test_classification_prefers_printed_embeddings():
    def ok(phi):
        return None

    out = gp.classify_output("gnnpe", "embedding: 0:1 1:2 2:3\n", -9, True, 3, 1, 1.0, {}, ok)
    assert out.status == backends.FOUND and out.embeddings[0].tolist() == [1, 2, 3]
    assert gp.classify_output("gnnpe", "2\n", -9, True, 3, 1, 1.0, {}, ok).status == backends.TIMEOUT
    out = gp.classify_output("gnnpe", "Answer Number: 0 Query Time (ms): 0.1\n", 139, False, 3, 1, 1.0, {}, ok)
    assert out.status == backends.ERROR and "Answer Number" in out.detail["output_tail"]


# ----------------------------------------------------------------------------- exporters
def test_export_data_graph_layout(gstore, tmp_path):
    stale = tmp_path / "gnn-pe" / "partitions" / "partition-7" / "index.dat"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    membership = np.arange(gstore.num_nodes) % 3
    info = gp.export_data_graph(gstore, "R1", tmp_path, partitions=3, membership=membership)
    assert not stale.exists() and all(d.is_dir() for d in gp.partition_dirs(tmp_path, 3))
    rows = (tmp_path / "gnn-pe" / "membership.txt").read_text(encoding="ascii").split()
    order = [int(x) for x in rows[0::2]]
    assert order == gp.degree_order(gstore.indptr).tolist()
    assert [int(x) for x in rows[1::2]] == membership[order].tolist()
    assert info["part_sizes"] == np.bincount(membership).tolist()
    assert Path(info["data_graph"]).name == "synth.R1.graph" and info["data"]["regime"] == "R1"
    with pytest.raises(ValueError):
        gp.export_data_graph(gstore, "R1", tmp_path, partitions=2, membership=membership)
    with pytest.raises(ValueError):
        gp.export_data_graph(gstore, "R1", tmp_path, partitions=3, membership=membership[:-1])


def test_export_query_set(gstore, tmp_path):
    planted, payload, codes = _planted(gstore, "R0", seed=5)

    def query(uid, edges, nodes, payload=None):
        nodes = np.asarray(nodes, dtype=np.int64)
        return queries.Query(uid=uid, family="k_hop", size=int(nodes.shape[0]), kind="positive", regime="R0",
                             edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                             payload=payload or queries.payload_for(gstore, nodes, "R0"), planted=nodes, seed=0,
                             attempt=0)

    a = int(np.flatnonzero(np.diff(gstore.indptr))[0])
    b = int(gstore.indices[gstore.indptr[a]])
    fresh = {k: np.array(v, copy=True) for k, v in payload.items()}
    fresh["attrs"][0] = 123.25  # a row the data graph does not hold
    qs = [query("pos/1", planted.edges, planted.planted), query("pair", [[0, 1]], [a, b]),
          query("fresh", planted.edges, planted.planted, fresh)]
    entries = gp.export_query_set(gstore, qs, "R0", tmp_path / "q")
    assert [e["file"] for e in entries] == ["pos_1.graph", "pair.graph", "fresh.graph"]
    assert entries[0]["trivial_none"] is None and entries[0]["unsupported"] is None
    assert "3 or more" in entries[1]["unsupported"] and "does not occur" in entries[2]["trivial_none"]
    lines = (tmp_path / "q" / "pos_1.graph").read_text(encoding="ascii").splitlines()
    assert lines[0] == f"t {planted.n} {planted.m}"
    assert [int(line.split()[2]) for line in lines[1:1 + planted.n]] == np.asarray(codes).tolist()
    manifest = json.loads((tmp_path / "q" / "queries.json").read_text(encoding="utf-8"))
    assert manifest["regime"] == "R0" and [e["uid"] for e in manifest["queries"]] == ["pos/1", "pair", "fresh"]
    with pytest.raises(ValueError):
        gp.export_query_set(gstore, qs, "R1", tmp_path / "r1")


# ----------------------------------------------------------------------------- runner, fake binary
def test_prepare_runs_offline_and_warmup(prepared):
    gstore, runner, info = prepared
    assert info["reused"] is False and Path(runner.binary_path).name == "fake_gnnpe.py"
    assert info["all_paths"] == info["path_count_estimate"] == gp.estimate_path_count(gstore.indptr)
    assert sum(info["partition_path_counts"]) == info["all_paths"]
    assert min(info["partition_path_counts"]) > gp.leaf_root_max_paths()
    assert info["index_files"] == 2 and info["warmup_answer_number"] == 1
    recorded = json.loads((runner.root / "prepared.json").read_text(encoding="utf-8"))
    assert recorded["membership_sha256"] == info["membership_sha256"] and recorded["path_length"] == 2


def test_prepare_is_reused_without_running_the_binary(prepared, monkeypatch):
    gstore, runner, info = prepared
    monkeypatch.setenv("FAKE_GNNPE_MODE", "offline_error0")  # a second offline stage would fail
    again = gp.GNNPERunner(gstore, "R0", cache_dir=runner.root.parent, binary=FAKE, partitions=2,
                           membership=_alternating(gstore))
    assert again.prepare()["reused"] is True and again.prepare()["offline_seconds"] == info["offline_seconds"]


def test_found_embeddings_are_global_and_valid(prepared, monkeypatch):
    gstore, runner, _ = prepared
    monkeypatch.delenv("FAKE_GNNPE_MODE", raising=False)
    planted, payload, codes = _planted(gstore, "R0", seed=5)
    out = runner.solve(codes, planted.edges, time_limit=30.0)
    assert out.status == backends.FOUND, out.detail
    assert out.solver == "gnnpe" and out.detail["global_ids"] and out.detail["threads"] == 1
    assert verify.check_embedding(gstore, payload, planted.edges, out.embeddings[0], "R0").ok
    assert out.detail["answer_number"] == 1 and isinstance(out.detail["query_time_ms"], float)
    assert out.detail["data_graph"]["vertices"] == gstore.num_nodes and out.detail["query_plan_paths"] >= 1
    many = runner.solve(codes, planted.edges, time_limit=30.0, max_solutions=3)
    assert many.status == backends.FOUND and len(many.embeddings) == many.detail["answer_number"] <= 3
    assert all(verify.check_embedding(gstore, payload, planted.edges, phi, "R0").ok for phi in many.embeddings)


@pytest.mark.parametrize("mode,status,needle", [
    ("none", backends.NONE, None),
    ("unpatched", backends.ERROR, "not printed"),
    ("invalid", backends.ERROR, "invalid embedding: not injective"),
    ("crash", backends.ERROR, "no Answer Number line"),
    ("error0", backends.ERROR, "no Answer Number line"),
])
def test_statuses(prepared, monkeypatch, mode, status, needle):
    gstore, runner, _ = prepared
    monkeypatch.setenv("FAKE_GNNPE_MODE", mode)
    planted, _, codes = _planted(gstore, "R0", seed=5)
    out = runner.solve(codes, planted.edges, time_limit=30.0)
    assert out.status == status, out.detail
    if needle:
        assert needle in out.detail["error"]
    if mode == "none":
        assert out.detail["none_unverified"] is True and out.detail["answer_number"] == 0
    if mode == "crash":
        assert out.detail["returncode"] == 139 and out.detail["unterminated_tail"]


def test_killed_at_the_cap(prepared, monkeypatch):
    gstore, runner, _ = prepared
    monkeypatch.setenv("FAKE_GNNPE_MODE", "sleep")
    planted, _, codes = _planted(gstore, "R0", seed=5)
    t0 = time.perf_counter()
    out = runner.solve(codes, planted.edges, time_limit=1.5)
    assert out.status == backends.TIMEOUT and out.detail["killed_at_cap"]
    assert time.perf_counter() - t0 < 5.0


def test_online_flags_and_threads(prepared, monkeypatch):
    gstore, runner, _ = prepared
    monkeypatch.setenv("FAKE_GNNPE_MODE", "echo_args")
    planted, _, codes = _planted(gstore, "R0", seed=5)
    out = runner.solve(codes, planted.edges, time_limit=30.0, max_solutions=3)
    assert out.status == backends.ERROR and out.detail["returncode"] == 1
    tail = out.detail["output_tail"]
    assert f"-f {runner.root.as_posix()}/ -d " in tail
    assert "-m online -p 2 -l 2 -e 2 -n 3 --print-embeddings" in tail and "OMP_NUM_THREADS=1" in tail


def test_trivial_unsupported_and_no_time_skip_the_binary(gstore, tmp_path):
    runner = gp.GNNPERunner(gstore, "R0", cache_dir=tmp_path / "cache", binary=str(tmp_path / "missing"))
    assert runner.binary_path is None
    planted, _, codes = _planted(gstore, "R0", seed=9)
    fresh = codes.copy()
    fresh[0] = gstore.sigma("R0").num_codes + 3
    out = runner.solve(fresh, planted.edges, time_limit=5.0)
    assert out.status == backends.NONE and "does not occur" in out.detail["trivial"]
    assert "none_unverified" not in out.detail
    out = runner.solve(codes[:2], np.array([[0, 1]]), time_limit=5.0)
    assert out.status == backends.ERROR and "3 or more" in out.detail["error"]
    out = runner.solve(codes[:4], np.array([[0, 1], [2, 3]]), time_limit=5.0)
    assert out.status == backends.ERROR and "disconnected" in out.detail["error"]
    assert runner.solve(codes, planted.edges, time_limit=0.0).status == backends.TIMEOUT
    assert not (tmp_path / "cache").exists()


def test_max_paths_refuses_before_running(gstore, tmp_path):
    runner = gp.GNNPERunner(gstore, "R0", cache_dir=tmp_path, binary=str(tmp_path / "missing"), max_paths=10)
    with pytest.raises(gp.GNNPEError, match="3-vertex paths"):
        runner.prepare()


def test_leaf_root_partition_is_refused(gstore, tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_GNNPE_MODE", raising=False)
    indptr, indices = gstore.indptr, gstore.indices
    degree = np.diff(indptr)
    order = gp.degree_order(indptr)
    v = int(order[degree[order] > 0][0])  # the first vertex whose dfs stores paths
    expected = int(sum(int(degree[a]) - 1 for a in indices[indptr[v]:indptr[v + 1]]))
    assert 1 <= expected <= gp.leaf_root_max_paths()
    membership = np.zeros(gstore.num_nodes, dtype=np.int64)
    membership[v] = 1
    runner = gp.GNNPERunner(gstore, "R0", cache_dir=tmp_path, binary=FAKE, partitions=2, membership=membership)
    with pytest.raises(gp.GNNPEError, match=r"partitions \[1\]"):
        runner.prepare()
    assert gp.read_partition_path_counts(runner.root, 2)[1] == expected
    assert not (runner.root / "prepared.json").exists()


@pytest.mark.parametrize("mode,needle", [("offline_error0", "offline stage failed"),
                                         ("error0", "warm-up query failed")])
def test_prepare_failures(gstore, tmp_path, monkeypatch, mode, needle):
    monkeypatch.setenv("FAKE_GNNPE_MODE", mode)
    runner = gp.GNNPERunner(gstore, "R0", cache_dir=tmp_path, binary=FAKE, partitions=2,
                            membership=_alternating(gstore))
    with pytest.raises(gp.GNNPEError, match=needle):
        runner.prepare()
    assert not (runner.root / "prepared.json").exists()
