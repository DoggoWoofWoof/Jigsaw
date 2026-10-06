from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import backends, classical, manifest, queries, store as store_mod, verify

FAKE = str(Path(__file__).with_name("fake_subgraphmatching.py"))
REPO = Path(__file__).resolve().parents[2]


def _graph(n, edges, labels):
    return store_mod.local_graph_from_edges(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                                            labels=np.asarray(labels))


def _fake(tmp_path, algorithm="gql", **kwargs):
    return classical.SubgraphMatchingBackend(algorithm, binary=[sys.executable, FAKE], workdir=str(tmp_path), **kwargs)


def _is_embedding(query, target, phi):
    phi = np.asarray(phi)
    if np.unique(phi).shape[0] != query.n or not np.array_equal(target.labels[phi], query.labels):
        return False
    t_edges = {tuple(e) for e in target.edge_list().tolist()}
    return all(tuple(sorted((int(phi[a]), int(phi[b])))) in t_edges for a, b in query.edge_list())


# ----------------------------------------------------------------------------- writers
def test_writer_format_exact_lines(tmp_path):
    g = _graph(4, [[0, 1], [1, 2], [0, 2]], [5, 6, 5, 7])
    out = tmp_path / "g.graph"
    classical.write_subgraphmatching_graph(g, out)
    assert out.read_bytes().decode("ascii").split("\n") == [
        "t 4 3", "v 0 5 2", "v 1 6 2", "v 2 5 2", "v 3 7 0", "e 0 1", "e 0 2", "e 1 2", ""]
    chunked = tmp_path / "chunked.graph"
    classical.write_subgraphmatching_graph(g, chunked, chunk=1)
    assert chunked.read_bytes() == out.read_bytes()
    dense = tmp_path / "dense.graph"
    classical.write_subgraphmatching_graph(g, dense, classical.dense_label_map(g))
    assert dense.read_text(encoding="ascii").splitlines()[1:5] == ["v 0 0 2", "v 1 1 2", "v 2 0 2", "v 3 2 0"]


def test_format_int_rows_matches_python():
    rng = np.random.default_rng(0)
    a = np.concatenate([[0, 9, 10, 99, 100, 2**40, 2**62], rng.integers(0, 10**9, size=500)])
    b = rng.integers(0, 10**6, size=a.shape[0])
    expected = "".join(f"e {x} {y}\n" for x, y in zip(a.tolist(), b.tolist())).encode("ascii")
    assert classical.format_int_rows(b"e", [a, b]) == expected
    assert classical.format_int_rows(b"", [a]) == "".join(f"{x}\n" for x in a.tolist()).encode("ascii")
    with pytest.raises(ValueError):
        classical.format_int_rows(b"v", [np.array([-1])])


def test_label_remap_is_shared_and_dense(tmp_path):
    query = _graph(2, [[0, 1]], [30, -4])
    target = _graph(4, [[0, 1], [1, 2], [2, 3]], [-4, 30, 99, -4])
    label_map = classical.dense_label_map(query, target)
    assert label_map.tolist() == [-4, 30, 99]
    q_path, t_path = tmp_path / "q.graph", tmp_path / "t.graph"
    classical.write_subgraphmatching_graph(query, q_path, label_map)
    classical.write_subgraphmatching_graph(target, t_path, label_map)
    assert [line.split()[2] for line in q_path.read_text().splitlines() if line[0] == "v"] == ["1", "0"]
    assert [line.split()[2] for line in t_path.read_text().splitlines() if line[0] == "v"] == ["0", "1", "2", "0"]
    with pytest.raises(ValueError):
        classical.write_subgraphmatching_graph(target, t_path, np.array([-4, 30]))  # 99 missing
    with pytest.raises(ValueError):
        classical.write_subgraphmatching_graph(target, t_path)  # negative label without a map


def test_backend_maps_embeddings_back_through_the_remap(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SM_MODE", "found")
    query = _graph(3, [[0, 1], [1, 2]], [70000, -3, 70000])
    target = _graph(5, [[0, 1], [1, 2], [2, 3], [3, 4]], [5, 70000, -3, 70000, 5])
    out = _fake(tmp_path).solve(query, target, time_limit=30.0, max_solutions=5)
    assert out.status == backends.FOUND, out.detail
    assert sorted(e.tolist() for e in out.embeddings) == [[1, 2, 3], [3, 2, 1]]
    assert all(_is_embedding(query, target, e) for e in out.embeddings)
    assert out.detail["embedding_count"] == 2 and out.detail["print_embeddings"] is True


# ----------------------------------------------------------------------------- parser
def _realistic_output(embedding_lines, count, tail="End.\n"):
    sep = "-" * 68
    lines = [
        "Command Line:", "\tData Graph CSR: ", "\tData Graph: /tmp/x/data.graph", "\tQuery Graph: /tmp/x/query.graph",
        "\tFilter Type: DPiso", "\tOrder Type: GQL", "\tEngine Type: LFTJ", "\tOutput Limit: 2",
        "\tTime Limit (seconds): 60", "\tOrder Num: 100", "\tDistribution File Path: temp.distribution",
        "\tPrint Embeddings: true", sep, "Load graphs...", "-----", "Query Graph Meta Information",
        "|V|: 3, |E|: 2, |\u03a3|: 2", "Max Degree: 2, Max Label Frequency: 2", "-----",
        "|V|: 5, |E|: 4, |\u03a3|: 3", "Max Degree: 2, Max Label Frequency: 2", sep, "Start queries...", "-----",
        "Filter candidates...", "-----", "Build indices...", "Index Info: CoreTable(0), TreeTable(0), LeafTable(2)",
        "LeafTable 0-1: 2", "LeafTable 1-2: 2", "Total Cardinality: 4.0", "-----", "Generate a matching order...",
        "Query Plan: 1 0 2 ", "-----", "Enumerate...", *embedding_lines, sep, "Release memories...", sep,
        "Load graphs time (seconds): 0.0021", "Filter vertices time (seconds): 0.0003",
        "Build table time (seconds): 0.0001", "Generate query plan time (seconds): 0.0000",
        "Enumerate time (seconds): 0.0004", "Preprocessing time (seconds): 0.0004", "Total time (seconds): 0.0008",
        "Memory cost (MB): 0.0010", f"#Embeddings: {count}", "Call Count: 7",
        "Per Call Count Time (nanoseconds): 57.1429",
    ]
    return "\n".join(lines) + "\n" + tail


def test_parse_realistic_output():
    text = _realistic_output(["embedding: 0:1 1:2 2:3", "embedding: 2:1 1:2 0:3", "embedding: 0:1 1:2",
                              "embedding: 0:1 0:2 2:3"], 2)
    status, embeddings, detail = classical.parse_subgraphmatching_output(text, 3)
    assert status == "complete"
    assert [e.tolist() for e in embeddings] == [[1, 2, 3], [3, 2, 1]]
    assert detail["malformed_lines"] == 2
    assert detail["embedding_count"] == 2 and detail["call_count"] == 7
    assert detail["load_seconds"] == pytest.approx(0.0021) and detail["total_seconds"] == pytest.approx(0.0008)
    assert (detail["filter"], detail["order"], detail["engine"], detail["output_limit"]) == ("DPiso", "GQL", "LFTJ", "2")
    assert detail["print_embeddings"] is True
    assert detail["query_graph"] == {"vertices": 3, "edges": 2, "labels": 2}
    assert detail["data_graph"] == {"vertices": 5, "edges": 4, "labels": 3}


def test_parse_incomplete_limit_and_unsupported_output():
    killed = _realistic_output(["embedding: 0:1 1:2 2:3"], 1, tail="").split("Release memories")[0] + "embedding: 0:4"
    status, embeddings, detail = classical.parse_subgraphmatching_output(killed, 3)
    assert status is None and len(embeddings) == 1 and detail["unterminated_tail"]
    spectrum = _realistic_output(["Spectrum Order 1 status: Timeout"], 0)
    assert classical.parse_subgraphmatching_output(spectrum, 3)[0] == "limit"
    bad = "Filter candidates...\nThe specified filter type 'XYZ' is not supported.\n"
    status, embeddings, detail = classical.parse_subgraphmatching_output(bad, 3)
    assert status is None and embeddings == [] and "XYZ" in detail["unsupported"]


# ----------------------------------------------------------------------------- process handling
Q_PATH = ([[0, 1], [1, 2]], [1, 2, 1])
T_PATH = ([[0, 1], [1, 2], [2, 3]], [3, 1, 2, 1])


@pytest.mark.parametrize("mode,status", [("none", backends.NONE), ("limit", backends.TIMEOUT),
                                         ("crash", backends.ERROR), ("unpatched", backends.ERROR),
                                         ("invalid", backends.ERROR)])
def test_statuses(tmp_path, monkeypatch, mode, status):
    monkeypatch.setenv("FAKE_SM_MODE", mode)
    q, t = _graph(3, *Q_PATH), _graph(4, *T_PATH)
    out = _fake(tmp_path).solve(q, t, time_limit=30.0)
    assert out.status == status, out.detail
    assert out.solver == "gql" and out.detail["returncode"] == (139 if mode == "crash" else 0)
    if status == backends.ERROR:
        assert "output_tail" in out.detail
    if mode == "unpatched":
        assert "not printed" in out.detail["error"]
    if mode == "invalid":
        assert "invalid embedding" in out.detail["error"]


def test_killed_at_the_fractional_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SM_MODE", "sleep")
    q, t = _graph(3, *Q_PATH), _graph(4, *T_PATH)
    t0 = time.perf_counter()
    out = _fake(tmp_path).solve(q, t, time_limit=1.5)
    wall = time.perf_counter() - t0
    assert out.status == backends.TIMEOUT and out.detail["killed_at_cap"]
    # killed long before the fake's 30 s sleep ends; the margin absorbs interpreter start-up on a loaded machine
    assert out.detail["returncode"] != 0 and wall < 15.0


def test_classification_prefers_printed_embeddings():
    q, t = _graph(3, *Q_PATH), _graph(4, *T_PATH)

    def check(phi):
        return None if _is_embedding(q, t, phi) else "bad"

    partial = "Enumerate...\nembedding: 0:1 1:2 2:3\nembedding: 0:3 1:"
    out = classical._classify("gql", partial, -9, True, 3, 1, 1.5, {}, check)
    assert out.status == backends.FOUND and out.detail["killed_at_cap"]  # killed after a find
    assert out.embeddings[0].tolist() == [1, 2, 3]
    assert classical._classify("gql", "Enumerate...\n", -9, True, 3, 1, 1.5, {}, check).status == backends.TIMEOUT
    done = _realistic_output([], 0)
    assert classical._classify("gql", done, 0, False, 3, 1, 0.1, {}, check).status == backends.NONE
    assert classical._classify("gql", done, 134, False, 3, 1, 0.1, {}, check).status == backends.ERROR


def test_preset_num_and_print_flags_are_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SM_MODE", "echo_args")
    q, t = _graph(3, *Q_PATH), _graph(4, *T_PATH)
    out = _fake(tmp_path, "cfl", extra_args=["-time_limit", "7"]).solve(q, t, time_limit=30.0, max_solutions=5)
    assert out.status == backends.ERROR
    args = out.detail["output_tail"].strip()
    assert re.search(r"-d \S+data\.graph -q \S+query\.graph ", args)
    assert "-filter CFL -order GQL -engine LFTJ -num 5 -print_embeddings -time_limit 7" in args
    assert classical.SubgraphMatchingBackend("dpiso").arguments(1)[:6] == ["-filter", "DPiso", "-order", "GQL",
                                                                          "-engine", "LFTJ"]


def test_trivial_none_and_unsupported_queries_skip_the_solver(tmp_path):
    missing = classical.SubgraphMatchingBackend("gql", binary=str(tmp_path / "does-not-exist"))
    assert not missing.available() and missing.binary_path is None
    t = _graph(4, *T_PATH)
    out = missing.solve(_graph(2, [[0, 1]], [1, 9]), t, time_limit=1.0)
    assert out.status == backends.NONE and "trivial" in out.detail
    one = missing.solve(_graph(1, np.zeros((0, 2)), [1]), t, time_limit=1.0)
    assert one.status == backends.ERROR and "one-vertex" in one.detail["error"]
    split = missing.solve(_graph(3, [[0, 1]], [1, 2, 1]), t, time_limit=1.0)
    assert split.status == backends.ERROR and "disconnected" in split.detail["error"]
    n = classical.MAX_QUERY_VERTICES + 1
    big_t = _graph(n + 1, [[i, i + 1] for i in range(n)], [1] * (n + 1))
    big = missing.solve(_graph(n, [[i, i + 1] for i in range(n - 1)], [1] * n), big_t, time_limit=1.0)
    assert big.status == backends.ERROR and "at most 128" in big.detail["error"]
    ran = missing.solve(_graph(3, *Q_PATH), t, time_limit=1.0)
    assert ran.status == backends.ERROR and "cannot start" in ran.detail["error"]
    assert missing.solve(_graph(3, *Q_PATH), t, time_limit=0.0).status == backends.TIMEOUT


def test_make_backend_presets_and_binary_path(tmp_path):
    for name, preset in [("gql", ("GQL", "GQL", "LFTJ")), ("cfl", ("CFL", "GQL", "LFTJ")),
                         ("dpiso", ("DPiso", "GQL", "LFTJ"))]:
        backend = backends.make_backend(name, binary=[sys.executable, FAKE])
        assert isinstance(backend, classical.SubgraphMatchingBackend) and backend.name == name
        assert (backend.filter, backend.order, backend.engine) == preset
    with pytest.raises(ValueError):
        classical.SubgraphMatchingBackend("vf3")
    backend = _fake(tmp_path)
    assert backend.available() and Path(backend.binary_path) == Path(FAKE)
    info = manifest.solver_info(backend.binary_path)
    assert info["available"] and info["pins"]["subgraphmatching"]["commit"] == classical_pin()


def classical_pin():
    return json.loads((REPO / "external" / "PINS.json").read_text(encoding="utf-8"))["subgraphmatching"]["commit"]


def test_pins_name_existing_patches_and_scripts():
    pins = json.loads((REPO / "external" / "PINS.json").read_text(encoding="utf-8"))
    assert set(pins) == {"subgraphmatching", "glasgow", "gnnpe"}
    for tool, pin in pins.items():
        assert re.fullmatch(r"[0-9a-f]{40}", pin["commit"]), tool
        assert (REPO / pin["build"]).is_file(), tool
        for patch in pin["patches"]:
            body = (REPO / patch).read_bytes()
            assert body.startswith(b"From ") or b"diff --git" in body, patch
            assert b"\r\n" not in body, patch


# ----------------------------------------------------------------------------- whole data graph
@pytest.fixture
def gstore(synthetic_store):
    root, _ = synthetic_store
    return store_mod.GraphStore(root, "synth", mmap=False)


def test_write_data_graph_equals_the_whole_graph_writer(gstore, tmp_path, monkeypatch):
    monkeypatch.setattr(classical, "_EDGE_BUDGET", 64)  # many edge chunks, some rows larger than one chunk
    regime = "R1"
    path = tmp_path / "data.graph"
    info = classical.write_data_graph(gstore, regime, path, chunk_nodes=97)
    whole = gstore.induced(np.arange(gstore.num_nodes, dtype=np.int64), regime)
    reference = tmp_path / "reference.graph"
    classical.write_subgraphmatching_graph(whole, reference)
    assert path.read_bytes() == reference.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
    sidecar = json.loads(classical.sidecar_path(path).read_text(encoding="utf-8"))
    assert sidecar == info
    assert sidecar["store_meta_sha256"] == hashlib.sha256((gstore.dir / "meta.json").read_bytes()).hexdigest()
    assert (sidecar["regime"], sidecar["dataset"]) == (regime, "synth")
    assert sidecar["num_nodes"] == gstore.num_nodes and sidecar["num_edges"] == whole.num_undirected_edges
    assert sidecar["num_codes"] == gstore.sigma(regime).num_codes
    assert path.read_text(encoding="ascii").split("\n", 1)[0] == f"t {gstore.num_nodes} {whole.num_undirected_edges}"


def test_write_data_graph_csr_layout(gstore, tmp_path):
    prefix = tmp_path / "data.csr"
    classical.write_data_graph_csr(gstore, "R0", prefix, chunk_nodes=50)
    deg, adj, lab = classical.csr_files(prefix)
    head = np.fromfile(deg, dtype="<u4")
    n, adjacency = gstore.num_nodes, int(gstore.indptr[-1])
    assert head[:3].tolist() == [4, n, adjacency // 2]
    assert np.array_equal(head[3:], np.diff(np.asarray(gstore.indptr)))
    assert np.array_equal(np.fromfile(adj, dtype="<u4"), np.asarray(gstore.indices))
    assert np.array_equal(np.fromfile(lab, dtype="<u4"), np.asarray(gstore.sigma("R0").codes))


def test_data_graph_is_written_once_and_rewritten_on_change(gstore, tmp_path):
    path, info = classical.ensure_data_graph(gstore, "R0", tmp_path)
    assert not info["reused"] and path == tmp_path / "synth.R0.graph"
    again_path, again = classical.ensure_data_graph(gstore, "R0", tmp_path)
    assert again["reused"] and again_path == path
    side = classical.sidecar_path(path)
    stale = json.loads(side.read_text(encoding="utf-8"))
    stale["store_meta_sha256"] = "0" * 64
    side.write_text(json.dumps(stale), encoding="utf-8")
    assert not classical.ensure_data_graph(gstore, "R0", tmp_path)[1]["reused"]
    with open(path, "ab") as handle:
        handle.write(b"e 0 1\n")
    assert not classical.ensure_data_graph(gstore, "R0", tmp_path)[1]["reused"]
    assert classical.ensure_data_graph(gstore, "R0", tmp_path)[1]["reused"]


def _planted(gstore, regime, seed):
    ctx = queries.QueryContext(gstore)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        planted = ctx.sample("k_hop", 8, rng)
        if planted is not None:
            payload = queries.payload_for(gstore, planted.planted, regime)
            return planted, payload, gstore.sigma(regime).query_codes(payload, gstore)
    raise AssertionError("no k_hop query")


@pytest.mark.parametrize("regime,data_format", [("R0", "text"), ("R1", "csr")])
def test_full_graph_runner_finds_planted_queries(gstore, tmp_path, monkeypatch, regime, data_format):
    monkeypatch.setenv("FAKE_SM_MODE", "found")
    planted, payload, codes = _planted(gstore, regime, seed=5)
    runner = classical.FullGraphClassicalRunner(gstore, regime, "dpiso", cache_dir=tmp_path / "cache",
                                                binary=[sys.executable, FAKE], data_format=data_format,
                                                workdir=str(tmp_path))
    out = runner.solve(codes, planted.edges, time_limit=30.0)
    assert out.status == backends.FOUND, out.detail
    assert out.solver == "classical_dpiso" and out.detail["global_ids"]
    assert out.detail["data_graph"]["vertices"] == gstore.num_nodes
    assert verify.check_embedding(gstore, payload, planted.edges, out.embeddings[0], regime).ok
    assert runner.prepare()["reused"] is False  # written by this runner, kept for the next one
    second = classical.FullGraphClassicalRunner(gstore, regime, "gql", cache_dir=tmp_path / "cache",
                                                binary=[sys.executable, FAKE], data_format=data_format)
    assert second.prepare()["reused"] is True


def test_full_graph_runner_trivial_none(gstore, tmp_path):
    runner = classical.FullGraphClassicalRunner(gstore, "R0", "gql", cache_dir=tmp_path / "cache",
                                                binary=str(tmp_path / "does-not-exist"))
    planted, payload, codes = _planted(gstore, "R0", seed=9)
    fresh = codes.copy()
    fresh[0] = gstore.sigma("R0").num_codes + 3  # a label the data graph does not hold
    out = runner.solve(fresh, planted.edges, time_limit=5.0)
    assert out.status == backends.NONE and "does not occur" in out.detail["trivial"]
    table = gstore.sigma("R0")
    rare = int(np.flatnonzero(np.asarray(table.df) == 1)[0])
    twice = np.array([rare, rare], dtype=np.int64)
    out = runner.solve(twice, np.array([[0, 1]]), time_limit=5.0)
    assert out.status == backends.NONE and "fewer times" in out.detail["trivial"]
    assert not (tmp_path / "cache").exists()  # no data file for trivially answered queries
