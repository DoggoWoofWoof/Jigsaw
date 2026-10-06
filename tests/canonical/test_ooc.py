"""Out-of-core helpers: chunked CSR construction, permutation and store writing equal the in-RAM code."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.canonical import ooc
from scripts.canonical import store as store_mod
from scripts.canonical.prepare_data import normalize_edges

from conftest import synthetic_graph


def random_edges(n: int = 400, m: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, n, size=(m, 2))
    return np.concatenate([edges, [[5, 5], [5, 5], [1, 2], [2, 1], [1, 2]]])


@pytest.mark.parametrize("chunk_edges", [7, 64, 1 << 20])
def test_csr_from_edge_index_matches_store(tmp_path, chunk_edges):
    n = 400
    edges = random_edges(n)
    edges[:, 1][edges[:, 0] == 17] = 17  # node 17 only has self loops: isolated after normalisation
    edges = np.concatenate([edges, [[399, 398]]])
    indptr, indices, report = ooc.csr_from_edge_index(edges.T.copy(), n, tmp_path, chunk_edges=chunk_edges)
    ref_ptr, ref_idx = store_mod.csr_from_edges(edges, n)
    assert np.array_equal(indptr, ref_ptr) and np.array_equal(indices, ref_idx) and indices.dtype == ref_idx.dtype
    _, _, ref_report = normalize_edges(edges[:, 0], edges[:, 1], n)
    for key in ("input_edges", "self_loops_dropped", "duplicates_merged", "undirected_edges", "adjacency_entries",
                "isolated_nodes"):
        assert report[key] == ref_report[key], key
    assert np.array_equal(np.load(tmp_path / "indptr.npy"), ref_ptr)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["indices.npy", "indptr.npy"]
    ooc.close_memmap(indices)


def test_csr_from_edge_index_rejects_bad_ids(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        ooc.csr_from_edge_index(np.array([[0, 1], [1, 9]]), 5, tmp_path)


def test_permute_and_take_rows(tmp_path):
    n = 400
    indptr, indices = store_mod.csr_from_edges(random_edges(n), n)
    rng = np.random.default_rng(3)
    new_to_old = rng.permutation(n)
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(n)
    ref_ptr, ref_idx = store_mod.permute_csr(indptr, indices, new_to_old, old_to_new)
    ptr, idx = ooc.permute_csr_chunked(indptr, indices, new_to_old, tmp_path, chunk_entries=50)
    assert np.array_equal(ptr, ref_ptr) and np.array_equal(idx, ref_idx) and idx.dtype == ref_idx.dtype
    ooc.close_memmap(idx)
    attrs = rng.normal(size=(n, 3)).astype(np.float32)
    out = ooc.take_rows(attrs, new_to_old, tmp_path / "rows.npy", chunk_rows=33)
    assert np.array_equal(np.asarray(out), attrs[new_to_old])
    ooc.close_memmap(out)
    out = ooc.take_rows(attrs.astype(np.float64), new_to_old, tmp_path / "rows32.npy", dtype=np.float32, chunk_rows=7)
    assert out.dtype == np.float32 and np.array_equal(np.asarray(out), attrs[new_to_old])
    ooc.close_memmap(out)


def test_write_store_ooc_is_byte_identical(tmp_path):
    g = synthetic_graph()
    n = g["num_nodes"]
    common = dict(node_type=g["node_type"], attrs=g["attrs"], node_block=g["node_block"], node_fine=g["node_fine"],
                  classes=g["classes"], orig_id=np.arange(n) + 1000, meta={"schema": "synthetic"},
                  partition_info={"tool": "planted"})
    store_mod.write_store(tmp_path / "ram", "g", edges=g["edges"], num_nodes=n, **common)
    indptr, indices = store_mod.csr_from_edges(g["edges"], n)
    ooc.write_store_ooc(tmp_path / "ooc", "g", indptr=indptr, indices=indices, chunk_entries=97, chunk_rows=41,
                        **common)
    files = sorted(p.relative_to(tmp_path / "ram").as_posix() for p in (tmp_path / "ram").rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(tmp_path / "ooc").as_posix()
                           for p in (tmp_path / "ooc").rglob("*") if p.is_file())
    for rel in files:
        assert (tmp_path / "ram" / rel).read_bytes() == (tmp_path / "ooc" / rel).read_bytes(), rel
    rng = np.random.default_rng(5)
    block = rng.integers(0, 4, size=n)
    fine = block * 2 + rng.integers(0, 2, size=n)
    store_mod.add_partition(tmp_path / "ram" / "g", "alt", node_block=block, node_fine=fine, info={"tool": "x"})
    ooc.add_partition_ooc(tmp_path / "ooc" / "g", "alt", node_block=block, node_fine=fine, info={"tool": "x"},
                          chunk_entries=61)
    for rel in ["meta.json"] + [f"part/alt/{p.name}" for p in (tmp_path / "ram" / "g" / "part" / "alt").iterdir()]:
        assert (tmp_path / "ram" / "g" / rel).read_bytes() == (tmp_path / "ooc" / "g" / rel).read_bytes(), rel
    assert json.loads((tmp_path / "ooc" / "g" / "meta.json").read_text(encoding="utf-8"))["partitions"] == [
        "primary", "alt"]


def test_write_store_ooc_rejects_a_non_contiguous_order(tmp_path):
    g = synthetic_graph(num_nodes=60, num_blocks=3)
    indptr, indices = store_mod.csr_from_edges(g["edges"], 60)
    with pytest.raises(ValueError, match="block by block"):
        ooc.write_store_ooc(tmp_path, "g", indptr=indptr, indices=indices, node_type=g["node_type"],
                            attrs=g["attrs"], node_block=g["node_block"], node_fine=None,
                            new_to_old=np.arange(60))


@pytest.mark.parametrize("compressed", [False, True])
def test_npz_member_streaming(tmp_path, compressed):
    rng = np.random.default_rng(9)
    edge_index = rng.integers(0, 1000, size=(2, 777))
    feat = rng.normal(size=(123, 5))
    fortran = np.asfortranarray(rng.normal(size=(17, 3)))
    path = tmp_path / "data.npz"
    (np.savez_compressed if compressed else np.savez)(path, edge_index=edge_index, node_feat=feat, f=fortran)
    info = ooc.npz_member_info(path, "edge_index")
    assert info["shape"] == (2, 777) and info["stored"] is (not compressed)
    out = ooc.npz_member_to_npy(path, "edge_index", tmp_path / "e.npy", chunk_bytes=13)
    assert np.array_equal(np.asarray(out), edge_index)
    ooc.close_memmap(out)
    out = ooc.npz_member_to_npy(path, "node_feat.npy", tmp_path / "x.npy", dtype=np.float32, chunk_bytes=100)
    assert out.dtype == np.float32 and np.array_equal(np.asarray(out), feat.astype(np.float32))
    ooc.close_memmap(out)
    out = ooc.npz_member_to_npy(path, "f", tmp_path / "f.npy", chunk_bytes=24)
    assert np.array_equal(np.asarray(out), fortran)
    ooc.close_memmap(out)
    mm = ooc.npz_member_memmap(path, "node_feat")
    if compressed:
        assert mm is None
    else:
        assert np.array_equal(np.asarray(mm), feat)
        ooc.close_memmap(mm)
    with pytest.raises(KeyError):
        ooc.npz_member_info(path, "missing")
