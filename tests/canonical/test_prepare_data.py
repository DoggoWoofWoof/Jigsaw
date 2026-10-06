"""prepare_data: loaders, normalisation, staged restartable preparation, meta and report."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import manifest, protocol
from scripts.canonical import partition as part_mod
from scripts.canonical import prepare_data as prep
from scripts.canonical.store import GraphStore, csr_from_edges

pytest.importorskip("pymetis")
MAG_TYPES = ["paper", "author", "institution", "field_of_study"]


@pytest.fixture(autouse=True)
def _no_external_tools(monkeypatch):
    for name in ("GPMETIS_BIN", "KAHIP_BIN", "JIGSAW_STORE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    real = shutil.which
    monkeypatch.setattr(part_mod.shutil, "which",
                        lambda cmd, *a, **k: None if Path(str(cmd)).stem in ("gpmetis", "kaffpa") else real(cmd, *a, **k))


# ----------------------------------------------------------------------------
# Synthetic raw files
# ----------------------------------------------------------------------------
def planted_edges(rng: np.random.Generator, n: int, k: int, per_node: int = 4) -> np.ndarray:
    """Directed edges, mostly inside the communities id % k."""
    u = rng.integers(0, n, size=n * per_node)
    v = (rng.integers(0, n // k, size=u.shape[0]) * k + u % k) % n
    cross = rng.random(u.shape[0]) < 0.05
    v[cross] = rng.integers(0, n, size=int(cross.sum()))
    return np.stack([u, v], axis=1)


def write_cora_npz(path: Path, *, n: int = 160, k: int = 4, d: int = 24, binary: bool = False, seed: int = 0) -> dict:
    """A cora.npz like CitationFull's: CSR adjacency and attributes plus pickled index maps."""
    rng = np.random.default_rng(seed)
    edges = planted_edges(rng, n, k)
    edges = np.concatenate([edges, [[3, 3], [5, 9], [9, 5], [5, 9], [7, 8]]])  # loop, duplicates, a zero entry
    data = np.ones(edges.shape[0], dtype=np.float32)
    data[-1] = 0.0  # explicit zero: not an edge
    order = np.argsort(edges[:, 0], kind="stable")
    edges, data = edges[order], data[order]
    adj_indptr = np.concatenate([[0], np.cumsum(np.bincount(edges[:, 0], minlength=n))]).astype(np.int32)
    dense = np.zeros((n, d), dtype=np.float32)
    mask = rng.random((n, d)) < 0.2
    mask[:, 0] |= ~mask.any(axis=1)
    if binary:
        dense[mask] = 1.0
    else:
        dense[mask] = rng.random(int(mask.sum())).astype(np.float32) + 0.01
        dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    attr_indptr = np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32)
    attr_indices = np.nonzero(mask)[1].astype(np.int32)
    labels = rng.integers(0, 5, size=n).astype(np.int64)
    pickled = np.empty(3, dtype=object)
    pickled[:] = [{"a": 1}, {"b": 2}, {"c": 3}]
    np.savez(path, adj_data=data, adj_indices=edges[:, 1].astype(np.int32), adj_indptr=adj_indptr,
             adj_shape=np.array([n, n]), attr_data=dense[mask], attr_indices=attr_indices, attr_indptr=attr_indptr,
             attr_shape=np.array([n, d]), labels=labels, idx_to_node=pickled, idx_to_attr=pickled,
             idx_to_class=pickled)
    keep = data != 0
    return {"n": n, "edges": edges[keep], "attrs": dense, "labels": labels, "k": k}


def cora_root(tmp: Path, **kwargs) -> tuple[Path, dict]:
    raw = tmp / "data" / "CoraFull" / "cora" / "raw"
    raw.mkdir(parents=True)
    return tmp / "data", write_cora_npz(raw / "cora.npz", **kwargs)


def undirected_set(edges: np.ndarray) -> set[tuple[int, int]]:
    edges = np.asarray(edges, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    lo, hi = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
    return set(zip(lo.tolist(), hi.tolist()))


def store_edges_in_source_ids(store_dir: Path) -> set[tuple[int, int]]:
    g = store_dir / "graph"
    indptr, indices = np.load(g / "indptr.npy"), np.load(g / "indices.npy")
    orig = np.load(g / "orig_id.npy")
    rows = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
    return undirected_set(np.stack([orig[rows], orig[indices]], axis=1))


def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    return {p.relative_to(directory).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def run_manifest(store_dir: Path) -> dict:
    return json.loads((store_dir / prep.RUN_DIR / "manifest.json").read_text(encoding="utf-8"))


def write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def write_rows(path: Path, rows) -> None:
    write_gz(path, "".join(",".join(repr(v) if isinstance(v, float) else str(v) for v in row) + "\n" for row in rows))


def write_mag_raw(root: Path, seed: int = 5) -> dict:
    """A tiny ogbn-mag raw directory (types listed out of schema order in num-node-dict)."""
    rng = np.random.default_rng(seed)
    counts = {"paper": 60, "author": 50, "institution": 6, "field_of_study": 12}
    raw = root / "ogbn_mag" / "ogbn_mag" / "raw"
    order = ["author", "field_of_study", "institution", "paper"]
    write_gz(raw / "num-node-dict.csv.gz", ",".join(order) + "\n" + ",".join(str(counts[t]) for t in order) + "\n")
    rel = {
        ("author", "affiliated_with", "institution"): np.stack([np.arange(50), rng.integers(0, 6, 50)], 1),
        ("author", "writes", "paper"): np.stack([rng.integers(0, 50, 120), np.arange(120) % 60], 1),
        ("paper", "cites", "paper"): planted_edges(rng, 60, 3, per_node=3),
        ("paper", "has_topic", "field_of_study"): np.stack([np.arange(90) % 60, rng.integers(0, 12, 90)], 1),
    }
    rel[("paper", "cites", "paper")] = np.concatenate([rel[("paper", "cites", "paper")], [[4, 4], [1, 2], [1, 2]]])
    rel[("author", "affiliated_with", "institution")] = np.concatenate(
        [rel[("author", "affiliated_with", "institution")], [[0, 1], [0, 1]]])
    write_gz(raw / "triplet-type-list.csv.gz", "".join(f"{s},{r},{t}\n" for s, r, t in rel))
    for (s, r, t), edges in rel.items():
        d = raw / "relations" / f"{s}___{r}___{t}"
        write_rows(d / "edge.csv.gz", edges.tolist())
        write_gz(d / "num-edge-list.csv.gz", f"{edges.shape[0]}\n")
        write_gz(d / "edge_reltype.csv.gz", "".join("0\n" for _ in range(edges.shape[0])))
    feats = rng.normal(size=(60, 8))
    write_rows(raw / "node-feat" / "paper" / "node-feat.csv.gz", feats.tolist())
    venue = rng.integers(0, 4, size=60)
    write_rows(raw / "node-label" / "paper" / "node-label.csv.gz", venue[:, None].tolist())
    offsets = {"paper": 0, "author": 60, "institution": 110, "field_of_study": 116}
    edges = np.concatenate([np.stack([e[:, 0] + offsets[s], e[:, 1] + offsets[t]], 1) for (s, _, t), e in rel.items()])
    return {"counts": counts, "offsets": offsets, "feats": feats, "venue": venue, "edges": edges}


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------
def test_normalize_edges_matches_store_csr():
    rng = np.random.default_rng(1)
    n = 300
    edges = rng.integers(0, n, size=(2000, 2))
    edges = np.concatenate([edges, [[4, 4], [4, 4], [8, 9], [9, 8], [8, 9]]])
    indptr, indices, report = prep.normalize_edges(edges[:, 0], edges[:, 1], n)
    ref_ptr, ref_idx = csr_from_edges(edges, n)
    assert np.array_equal(indptr, ref_ptr) and np.array_equal(indices, ref_idx) and indices.dtype == ref_idx.dtype
    loops = int((edges[:, 0] == edges[:, 1]).sum())
    m = len(undirected_set(edges))
    directed = len(set(map(tuple, edges[edges[:, 0] != edges[:, 1]].tolist())))
    assert report["self_loops_dropped"] == loops
    assert report["undirected_edges"] == m and report["adjacency_entries"] == 2 * m
    assert report["duplicates_merged"] == edges.shape[0] - loops - m
    assert report["exact_duplicates"] == edges.shape[0] - loops - directed
    assert report["reciprocal_pairs_merged"] == directed - m
    assert report["isolated_nodes"] == int((np.diff(ref_ptr) == 0).sum())
    with pytest.raises(ValueError, match="outside"):
        prep.normalize_edges(np.array([0]), np.array([n]), n)


# ----------------------------------------------------------------------------
# CoraFull end to end
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cora_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cora")
    data_root, raw = cora_root(tmp)
    store_root = tmp / "stores"
    report = prep.prepare("corafull", data_root=data_root, store_root=store_root, partitions=4, fine_per_block=3,
                          tool="pymetis")
    return data_root, store_root, raw, report


def test_corafull_store_is_block_contiguous_and_faithful(cora_run):
    data_root, store_root, raw, report = cora_run
    d = store_root / "corafull"
    s = GraphStore(store_root, "corafull")
    n = raw["n"]
    assert s.num_nodes == n and s.num_blocks == 4
    orig = np.load(d / "graph" / "orig_id.npy")
    assert np.array_equal(np.sort(orig), np.arange(n))
    block = np.asarray(s.part.node_block)
    fine = np.asarray(s.part.node_fine)
    assert np.all(np.diff(block) >= 0) and np.all(np.diff(fine) >= 0)
    assert np.array_equal(s.part.block_nodes, np.arange(n)) and np.array_equal(s.part.fine_nodes, np.arange(n))
    same = (np.diff(block) == 0) & (np.diff(fine) == 0)
    assert np.all(np.diff(orig)[same] > 0)  # source-id order inside every fine part
    assert np.array_equal(s.part.fine_block[fine], block)
    # edges and attributes survive the relabelling; loops, duplicates and zero entries are gone
    assert store_edges_in_source_ids(d) == undirected_set(raw["edges"])
    rows = np.repeat(np.arange(n), np.diff(s.indptr))
    assert np.all(rows != s.indices)
    for v in range(n):
        nb = s.indices[s.indptr[v]:s.indptr[v + 1]]
        assert np.all(np.diff(nb) > 0)
    assert s.indptr.dtype == np.int64 and s.indices.dtype == np.int32
    assert s.attrs.dtype == np.uint8 and np.array_equal(s.attrs, (raw["attrs"][orig] > 0).astype(np.uint8))
    assert np.array_equal(s.classes, raw["labels"][orig])
    assert np.all(s.node_type == 0)


def test_corafull_meta_report_and_manifest(cora_run):
    data_root, store_root, raw, report = cora_run
    d = store_root / "corafull"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for key in ("schema", "loader", "num_types", "type_names", "attr_dim", "attr_dtype", "has_class",
                "relation_pairs", "source_files", "normalization", "protocol_hash"):
        assert key in meta, key
    assert meta["schema"] == "corafull_bow_v1" and meta["loader"] == "pyg:CoraFull"
    assert meta["num_types"] == 1 and meta["type_names"] == ["paper"] and meta["has_class"] is True
    assert meta["attr_dim"] == 24 and meta["attr_dtype"] == "uint8"
    assert meta["relation_pairs"] == [[0, 0]]
    npz = data_root / "CoraFull" / "cora" / "raw" / "cora.npz"
    entry = meta["source_files"]["cora.npz"]
    assert entry["sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert entry["path"] == "CoraFull/cora/raw/cora.npz" and entry["bytes"] == npz.stat().st_size
    norm = meta["normalization"]
    loops = int((raw["edges"][:, 0] == raw["edges"][:, 1]).sum())
    assert loops >= 1 and norm["self_loops_dropped"] == loops
    assert norm["undirected_edges"] == len(undirected_set(raw["edges"]))
    assert norm["input_edges"] == raw["edges"].shape[0] and norm["duplicates_merged"] >= 2
    assert meta["loader_notes"]["adjacency_explicit_zeros"] == 1
    notes = meta["attribute_notes"]
    assert notes["attr_values_binary"] is False and notes["stored_dtype"] == "uint8" and "x > 0" in notes["attr_note"]
    for regime in protocol.regimes_for("corafull"):
        info = json.loads((d / "sigma" / regime / "sigma.json").read_text(encoding="utf-8"))
        assert info["num_codes"] >= 1
    assert report["nodes"] == raw["n"] and report["undirected_edges"] == norm["undirected_edges"]
    assert set(report["sigma"]) == set(protocol.regimes_for("corafull"))
    assert report["partitions"]["primary"]["num_blocks"] == 4 and report["partitions"]["primary"]["num_fine"] == 12
    assert report["partitions"]["primary"]["cut_edges"] == json.loads(
        (d / "part" / "primary" / "partition.json").read_text(encoding="utf-8"))["cut_edges"]
    assert set(report["stages"]) >= {"load", "partition", "fine", "relabel", "store", "sigma"}
    assert json.loads((d / prep.REPORT_NAME).read_text(encoding="utf-8")) == report
    pinfo = json.loads((d / "part" / "primary" / "partition.json").read_text(encoding="utf-8"))
    assert pinfo["tool"] == "pymetis" and pinfo["options"] == {"ptype": "kway", "objtype": "cut", "ufactor": 30,
                                                                "seed": 1}
    assert pinfo["contiguous"] is True and pinfo["partitions_override"] is True
    man = run_manifest(d)
    assert man["status"] == "complete" and man["experiment"] == "prepare_data"
    assert man["config"]["stages_run"] == ["load", "partition", "fine", "relabel", "store", "sigma", "report"]
    assert manifest.verify_manifest(d / prep.RUN_DIR) == []
    assert "graph/attrs.npy" in man["extra"]["store_files"]
    assert not (d / prep.WORK_DIR).exists()


def test_rerun_is_a_noop(cora_run):
    data_root, store_root, raw, report = cora_run
    d = store_root / "corafull"
    before = snapshot(d)
    again = prep.prepare("corafull", data_root=data_root, store_root=store_root, partitions=4, fine_per_block=3,
                         tool="pymetis")
    assert again == report
    assert snapshot(d) == before


def test_binary_attributes_are_stored_as_uint8(tmp_path):
    data_root, raw = cora_root(tmp_path, binary=True, n=80)
    report = prep.prepare("corafull", data_root=data_root, store_root=tmp_path / "s", partitions=2,
                          fine_per_block=2, skip_sigma=True, tool="pymetis")
    s = GraphStore(tmp_path / "s", "corafull")
    orig = np.load(tmp_path / "s" / "corafull" / "graph" / "orig_id.npy")
    assert s.attrs.dtype == np.uint8 and np.array_equal(s.attrs, raw["attrs"][orig].astype(np.uint8))
    assert report["attr_dtype"] == "uint8" and report["sigma"] == {}
    assert s.meta["attribute_notes"]["attr_values_binary"] is True and "attr_note" not in s.meta["attribute_notes"]


def test_restart_after_a_failed_stage(tmp_path, monkeypatch):
    data_root, raw = cora_root(tmp_path, n=80)
    kwargs = dict(data_root=data_root, store_root=tmp_path / "s", partitions=2, fine_per_block=2,
                  regimes=["R0", "R1"], tool="pymetis")

    def boom(ctx, rm):
        raise RuntimeError("injected sigma failure")

    monkeypatch.setitem(prep.STAGE_FUNCS, "sigma", boom)
    with pytest.raises(RuntimeError, match="injected"):
        prep.prepare("corafull", **kwargs)
    d = tmp_path / "s" / "corafull"
    man = run_manifest(d)
    assert man["status"] == "failed" and "injected" in man["extra"]["error"]
    for stage in ("load", "partition", "fine", "relabel", "store"):
        assert (d / prep.RUN_DIR / "stages" / f"{stage}.json").exists()
    assert not (d / prep.RUN_DIR / "stages" / "sigma.json").exists()
    monkeypatch.undo()
    report = prep.prepare("corafull", **kwargs)
    man = run_manifest(d)
    assert man["status"] == "complete" and man["config"]["stages_run"] == ["sigma", "report"]
    assert set(report["sigma"]) == {"R0", "R1"}
    attempts = list((d / prep.RUN_DIR / "attempts").glob("manifest-*.json"))
    assert len(attempts) == 1 and json.loads(attempts[0].read_text(encoding="utf-8"))["status"] == "failed"
    assert manifest.verify_manifest(d / prep.RUN_DIR) == []
    # a changed regime list reruns sigma (only) and leaves exactly the requested regimes
    kwargs["regimes"] = ["R2-32"]
    prep.prepare("corafull", **kwargs)
    assert run_manifest(d)["config"]["stages_run"] == ["sigma", "report"]
    assert sorted(p.name for p in (d / "sigma").iterdir()) == ["R2-32"]
    # --force redoes everything
    prep.prepare("corafull", force=True, **kwargs)
    assert run_manifest(d)["config"]["stages_run"] == ["load", "partition", "fine", "relabel", "store", "sigma",
                                                       "report"]


def test_import_partition(tmp_path):
    data_root, raw = cora_root(tmp_path, n=80)
    n, k = raw["n"], 4
    imported = np.arange(n) % k
    path = tmp_path / "blocks.npy"
    np.save(path, imported)
    prep.prepare("corafull", data_root=data_root, store_root=tmp_path / "s", partitions=k, fine_per_block=2,
                 import_partition=path, skip_sigma=True, tool="pymetis")
    d = tmp_path / "s" / "corafull"
    s = GraphStore(tmp_path / "s", "corafull")
    orig = np.load(d / "graph" / "orig_id.npy")
    assert np.array_equal(np.asarray(s.part.node_block), imported[orig])
    assert s.part.info["tool"] == "import" and s.part.info["source_sha256"] == part_mod.file_sha256(path)
    assert run_manifest(d)["inputs"]["import_partition"]["sha256"] == part_mod.file_sha256(path)
    with pytest.raises(ValueError, match="block ids for"):
        np.save(path, imported[:-1])
        prep.prepare("corafull", data_root=data_root, store_root=tmp_path / "s", partitions=k, fine_per_block=2,
                     import_partition=path, skip_sigma=True, tool="pymetis")


def test_empty_metis_part_is_filled(tmp_path, monkeypatch):
    """METIS may leave a part empty (ogbn-mag, P=2000); the primary partition still gets P blocks."""
    data_root, raw = cora_root(tmp_path, n=120)
    real = part_mod.run_pymetis_job

    def leaves_last_part_empty(indptr, indices, graph_ptr, nparts, **kwargs):
        labels, report = real(indptr, indices, graph_ptr, nparts, **kwargs)
        if len(nparts) == 1 and int(nparts[0]) > 2:  # the primary call, not the fine level or the split
            labels = np.where(labels == int(nparts[0]) - 1, int(nparts[0]) - 2, labels)
        return labels, report

    monkeypatch.setattr(part_mod, "run_pymetis_job", leaves_last_part_empty)
    report = prep.prepare("corafull", data_root=data_root, store_root=tmp_path / "s", partitions=4,
                          fine_per_block=2, skip_sigma=True, tool="pymetis")
    s = GraphStore(tmp_path / "s", "corafull")
    assert s.num_blocks == 4 and np.diff(s.part.block_ptr).min() > 0
    info = s.part.info
    assert info["metis_empty_parts"] == 1 and info["empty_parts_renumbered"] == 0
    (fill,) = info["empty_parts_filled"]
    assert fill["empty_part"] == 3 and fill["method"] == "metis_kway_2" and sum(fill["sizes_after"]) == fill["donor_size"]
    assert report["partitions"]["primary"]["metis_empty_parts"] == 1
    assert report["partitions"]["primary"]["empty_parts_filled"] == 1


def test_alternative_partitions(tmp_path):
    data_root, raw = cora_root(tmp_path, n=120)
    kwargs = dict(data_root=data_root, store_root=tmp_path / "s", partitions=4, fine_per_block=2,
                  alternatives=True, skip_sigma=True, tool="pymetis")
    report = prep.prepare("corafull", **kwargs)
    d = tmp_path / "s" / "corafull"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["partitions"] == ["primary", "bfs_random", "metis_x0.5", "metis_x2"]
    expected = {"bfs_random": 4, "metis_x0.5": 2, "metis_x2": 8}
    for name, blocks in expected.items():
        s = GraphStore(tmp_path / "s", "corafull", partition=name)
        assert s.num_blocks == blocks and not s.part.contiguous and s.part.has_fine
        assert np.array_equal(s.part.fine_block[np.asarray(s.part.node_fine)], np.asarray(s.part.node_block))
        assert report["partitions"][name]["num_blocks"] == blocks
    marker = json.loads((d / prep.RUN_DIR / "stages" / "alternatives.json").read_text(encoding="utf-8"))
    assert "kaffpa" in marker["info"]["results"]["kahip_fast"]["skipped"]
    assert not (d / "part" / "kahip_fast").exists()
    before = snapshot(d)
    prep.prepare("corafull", **kwargs)
    assert snapshot(d) == before


def test_missing_source_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"cora\.npz"):
        prep.prepare("corafull", data_root=tmp_path / "nowhere", store_root=tmp_path / "s")
    with pytest.raises(FileNotFoundError, match="num-node-dict"):
        prep.source_files("mag", tmp_path / "nowhere")
    assert not (tmp_path / "s").exists()


# ----------------------------------------------------------------------------
# MAG
# ----------------------------------------------------------------------------
def test_mag_end_to_end(tmp_path):
    raw = write_mag_raw(tmp_path / "data")
    report = prep.prepare("mag", data_root=tmp_path / "data", store_root=tmp_path / "s", partitions=3,
                          fine_per_block=2, tool="pymetis")
    d = tmp_path / "s" / "mag"
    s = GraphStore(tmp_path / "s", "mag")
    n = sum(raw["counts"].values())
    assert s.num_nodes == n and s.meta["type_names"] == MAG_TYPES
    assert s.meta["type_counts"] == [60, 50, 6, 12] and s.meta["type_offsets"] == [0, 60, 110, 116]
    orig = np.load(d / "graph" / "orig_id.npy")
    source_type = np.searchsorted(np.array([60, 110, 116]), orig, side="right")
    assert np.array_equal(s.node_type, source_type)
    papers = orig < 60
    assert np.array_equal(s.attrs[papers], raw["feats"].astype(np.float32)[orig[papers]])
    assert not s.attrs[~papers].any() and s.attrs.dtype == np.float32
    assert np.array_equal(s.classes[papers], raw["venue"][orig[papers]]) and np.all(s.classes[~papers] == -1)
    assert store_edges_in_source_ids(d) == undirected_set(raw["edges"])
    assert s.meta["normalization"]["self_loops_dropped"] == int((raw["edges"][:, 0] == raw["edges"][:, 1]).sum())
    pairs = {tuple(p) for p in s.meta["relation_pairs"]}
    assert pairs == {(0, 0), (0, 1), (1, 0), (0, 3), (3, 0), (1, 2), (2, 1)}
    assert s.meta["schema"] == "mag_intrinsic_v1" and s.meta["num_types"] == 4
    assert set(s.meta["source_files"]) >= {"triplet-type-list.csv.gz", "node-feat/paper/node-feat.csv.gz",
                                           "relations/paper___cites___paper/edge.csv.gz"}
    assert set(report["sigma"]) == set(protocol.regimes_for("mag"))
    assert report["type_counts"] == [60, 50, 6, 12] and report["labelled_nodes"] == 60
    assert manifest.verify_manifest(d / prep.RUN_DIR) == []


# ----------------------------------------------------------------------------
# products (loader only) and papers100M (out of core, end to end)
# ----------------------------------------------------------------------------
def write_ogb_csv_raw(raw: Path, n: int, edges: np.ndarray, feats: np.ndarray, labels: np.ndarray) -> None:
    write_gz(raw / "num-node-list.csv.gz", f"{n}\n")
    write_gz(raw / "num-edge-list.csv.gz", f"{edges.shape[0]}\n")
    write_rows(raw / "edge.csv.gz", edges.tolist())
    write_rows(raw / "node-feat.csv.gz", feats.tolist())
    write_gz(raw / "node-label.csv.gz", "".join("nan\n" if np.isnan(x) else f"{int(x)}\n" for x in labels))


@pytest.mark.parametrize("dataset", ["arxiv", "products"])
def test_ogb_homogeneous_loader(tmp_path, dataset):
    rng = np.random.default_rng(2)
    n = 40
    edges = planted_edges(rng, n, 2)
    feats = rng.normal(size=(n, 5))
    labels = rng.integers(0, 3, size=n).astype(np.float64)
    labels[3] = np.nan
    folder = f"ogbn_{dataset}"
    write_ogb_csv_raw(tmp_path / folder / folder / "raw", n, edges, feats, labels)
    loaded = prep.LOADERS[dataset](prep.source_files(dataset, tmp_path))
    assert loaded.num_nodes == n and np.array_equal(np.stack([loaded.src, loaded.dst], 1), edges)
    assert loaded.attrs.dtype == np.float32 and np.array_equal(loaded.attrs, feats.astype(np.float32))
    assert loaded.classes[3] == -1 and np.array_equal(np.delete(loaded.classes, 3), np.delete(labels, 3).astype(int))
    assert loaded.type_names == protocol.ATTRIBUTE_SCHEMAS[protocol.DATASETS[dataset]["schema"]]["node_types"]
    write_gz(tmp_path / folder / folder / "raw" / "num-edge-list.csv.gz", "999\n")
    with pytest.raises(ValueError, match="num-edge-list"):
        prep.LOADERS[dataset](prep.source_files(dataset, tmp_path))


def test_papers100m_out_of_core(tmp_path):
    rng = np.random.default_rng(4)
    n = 90
    edges = np.concatenate([planted_edges(rng, n, 3), [[2, 2], [6, 7], [7, 6]]])
    feats = rng.normal(size=(n, 6))  # float64 in the file, float32 in the store
    labels = np.full((n, 1), np.nan)
    labels[::4, 0] = rng.integers(0, 7, size=labels[::4].shape[0])
    raw = tmp_path / "data" / "ogbn_papers100M" / "ogbn_papers100M" / "raw"
    raw.mkdir(parents=True)
    np.savez_compressed(raw / "data.npz", edge_index=edges.T.astype(np.int64), num_nodes_list=np.array([n]),
                        num_edges_list=np.array([edges.shape[0]]), node_feat=feats,
                        node_year=np.zeros((n, 1), dtype=np.int64))
    np.savez_compressed(raw / "node-label.npz", node_label=labels)
    report = prep.prepare("papers100m", data_root=tmp_path / "data", store_root=tmp_path / "s", partitions=3,
                          fine_per_block=2, tool="pymetis")
    d = tmp_path / "s" / "papers100m"
    s = GraphStore(tmp_path / "s", "papers100m")
    orig = np.load(d / "graph" / "orig_id.npy")
    assert store_edges_in_source_ids(d) == undirected_set(edges)
    ref_ptr, ref_idx, ref_report = prep.normalize_edges(edges[:, 0], edges[:, 1], n)
    for key in ("input_edges", "self_loops_dropped", "duplicates_merged", "undirected_edges", "isolated_nodes"):
        assert s.meta["normalization"][key] == ref_report[key], key
    assert s.attrs.dtype == np.float32 and np.array_equal(s.attrs, feats.astype(np.float32)[orig])
    expected = np.where(np.isnan(labels[:, 0]), -1, np.nan_to_num(labels[:, 0])).astype(np.int32)
    assert np.array_equal(s.classes, expected[orig])
    assert np.all(np.diff(np.asarray(s.part.node_block)) >= 0) and np.all(np.diff(np.asarray(s.part.node_fine)) >= 0)
    assert s.meta["loader_notes"]["out_of_core"] is True and set(report["sigma"]) == {"R0"}
    assert manifest.verify_manifest(d / prep.RUN_DIR) == [] and not (d / prep.WORK_DIR).exists()


def test_cli_requires_a_store_root(capsys):
    with pytest.raises(SystemExit):
        prep.main(["--dataset", "corafull"])
    assert "--store-root" in capsys.readouterr().err
