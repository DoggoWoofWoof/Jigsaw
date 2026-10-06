"""Partitioners: METIS through pymetis, the fine level, BFS-random, imports and the canonical order."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import partition as part_mod
from scripts.canonical.store import csr_from_edges

pymetis = pytest.importorskip("pymetis")
FAKE = [sys.executable, str(Path(__file__).with_name("fake_partitioner.py"))]


def planted_csr(n: int = 3000, k: int = 6, seed: int = 3, degree: int = 8, cross: float = 0.05):
    """Symmetric CSR of a planted k-community graph (most edges inside a community)."""
    rng = np.random.default_rng(seed)
    community = rng.permutation(np.arange(n) % k)
    order = np.argsort(community, kind="stable")
    sizes = np.bincount(community, minlength=k)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    u = rng.integers(0, n, size=n * degree // 2)
    c = community[u]
    v = order[ptr[c] + (rng.random(u.shape[0]) * sizes[c]).astype(np.int64)]
    w = rng.integers(0, n, size=int(u.shape[0] * cross))
    x = rng.integers(0, n, size=w.shape[0])
    edges = np.concatenate([np.stack([u, v], 1), np.stack([w, x], 1)])
    indptr, indices = csr_from_edges(edges, n)
    return indptr, indices, community


@pytest.fixture(scope="module")
def planted():
    return planted_csr()


@pytest.fixture(scope="module")
def metis6(planted):
    indptr, indices, _ = planted
    return part_mod.metis_partition(indptr, indices, 6, tool="pymetis")


@pytest.fixture(autouse=True)
def _no_external_tools(monkeypatch):
    monkeypatch.delenv("GPMETIS_BIN", raising=False)
    monkeypatch.delenv("KAHIP_BIN", raising=False)


def _cut(indptr, indices, labels):
    rows = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
    return int(np.count_nonzero(labels[rows] != labels[indices])) // 2


def test_metis_kway_quality_balance_and_determinism(planted, metis6):
    indptr, indices, _ = planted
    n = indptr.shape[0] - 1
    blocks, info = metis6
    assert blocks.dtype == np.int32 and blocks.shape == (n,)
    sizes = np.bincount(blocks)
    assert sizes.shape[0] == 6 and sizes.min() > 0
    assert info["imbalance"] <= 1 + 30 / 1000 + 0.02
    assert info["edge_cut"] == _cut(indptr, indices, blocks)
    rng = np.random.default_rng(0)
    random_cut = np.mean([_cut(indptr, indices, rng.integers(0, 6, size=n)) for _ in range(3)])
    assert info["edge_cut"] < 0.25 * random_cut
    again, _ = part_mod.metis_partition(indptr, indices, 6, tool="pymetis")
    assert np.array_equal(blocks, again)
    assert info["tool"] == "pymetis" and info["version"]
    assert info["options"] == {"ptype": "kway", "objtype": "cut", "ufactor": 30, "seed": 1}


def test_metis_records_kway_for_small_nparts(planted):
    """pymetis would use recursive bisection for nparts <= 8; the canonical call is k-way."""
    indptr, indices, _ = planted
    blocks, info = part_mod.metis_partition(indptr, indices, 4, tool="pymetis")
    assert info["method"] == "kway" and info["recursive_bisection"] is False
    assert info["backend"]["function"] == "METIS_PartGraphKway" and info["backend"]["recursive"] is False
    options = pymetis.Options(seed=1, ufactor=30, objtype=int(pymetis.ObjType.CUT))
    _, direct = pymetis.part_graph(4, adjacency=pymetis.CSRAdjacency(indptr, indices.astype(np.int64)),
                                   recursive=False, options=options)
    expected, _ = part_mod.compact_labels(np.asarray(direct), 4)
    assert np.array_equal(blocks, expected)


def test_metis_single_part_and_bad_nparts(planted):
    indptr, indices, _ = planted
    blocks, info = part_mod.metis_partition(indptr, indices, 1, tool="pymetis")
    assert not blocks.any() and info["edge_cut"] == 0
    with pytest.raises(ValueError):
        part_mod.metis_partition(indptr, indices, 0, tool="pymetis")


def test_empty_metis_parts_are_renumbered():
    indptr, indices = csr_from_edges(np.array([[0, 1]]), 2)
    blocks, info = part_mod.metis_partition(indptr, indices, 2, tool="pymetis")
    assert blocks.max() + 1 == info["num_parts"] == 2 - info["empty_parts_renumbered"]
    assert np.bincount(blocks).min() > 0
    filled, info = part_mod.metis_partition(indptr, indices, 2, tool="pymetis", fill_empty=True)
    assert np.bincount(filled).tolist() == [1, 1] and info["empty_parts_renumbered"] == 0
    assert len(info["empty_parts_filled"]) == info["metis_empty_parts"]


def test_fill_empty_parts_splits_the_largest_parts(planted, metis6):
    indptr, indices, _ = planted
    blocks, _ = metis6
    labels = np.array([0, 1, 3, 4, 5, 6])[blocks]  # ids 2 and 7 of 8 are empty
    sizes = np.bincount(labels, minlength=8)
    donors = np.lexsort((np.arange(8), -sizes))[:2]
    filled, records = part_mod.fill_empty_parts(indptr, indices, labels, 8, tool="pymetis")
    assert np.bincount(filled, minlength=8).min() > 0
    assert [r["empty_part"] for r in records] == [2, 7]
    assert [r["donor_part"] for r in records] == donors.tolist()
    moved = filled != labels
    assert set(labels[moved].tolist()) == set(donors.tolist())
    for r in records:
        assert np.array_equal(moved & (labels == r["donor_part"]), filled == r["empty_part"])
        kept, gone = r["sizes_after"]
        assert kept + gone == r["donor_size"] == sizes[r["donor_part"]]
        assert r["method"] == "metis_kway_2" and abs(kept - gone) <= 0.1 * r["donor_size"]
    again, _ = part_mod.fill_empty_parts(indptr, indices, labels, 8, tool="pymetis")
    assert np.array_equal(filled, again)
    same, none = part_mod.fill_empty_parts(indptr, indices, blocks, 6, tool="pymetis")
    assert none == [] and np.array_equal(same, blocks)


def test_fill_empty_parts_fallback_and_errors(planted, monkeypatch):
    indptr, indices, _ = planted
    n = indptr.shape[0] - 1
    labels = np.arange(n) * 3 // n  # three equal parts, id 3 of 4 empty
    monkeypatch.setenv("FAKE_PARTITIONER_MODE", "one-sided")
    filled, (record,) = part_mod.fill_empty_parts(indptr, indices, labels, 4, binary=FAKE)
    assert record["donor_part"] == 0 and record["method"] == "node_id_halves" and record["tool"] == "gpmetis"
    members = np.flatnonzero(labels == 0)
    assert np.array_equal(np.flatnonzero(filled == 3), members[members.shape[0] // 2:])
    with pytest.raises(part_mod.PartitionError, match="outside"):
        part_mod.fill_empty_parts(indptr, indices, labels, 2, tool="pymetis")
    small_ptr, small_idx = csr_from_edges(np.array([[0, 1]]), 3)
    with pytest.raises(part_mod.PartitionError, match="too few"):
        part_mod.fill_empty_parts(small_ptr, small_idx, np.array([0, 1, 2]), 4, tool="pymetis")


def test_metis_fill_empty_through_gpmetis(planted, monkeypatch):
    indptr, indices, _ = planted
    monkeypatch.setenv("FAKE_PARTITIONER_MODE", "empty-last")
    renumbered, info = part_mod.metis_partition(indptr, indices, 3, binary=FAKE)
    assert info["num_parts"] == 2 and info["metis_empty_parts"] == 1 == info["empty_parts_renumbered"]
    assert info["empty_part_rule"] == "renumber" and info["empty_parts_filled"] == []
    blocks, info = part_mod.metis_partition(indptr, indices, 3, binary=FAKE, fill_empty=True)
    assert info["num_parts"] == 3 and info["metis_empty_parts"] == 1 and info["empty_parts_renumbered"] == 0
    assert info["empty_part_rule"] == part_mod.EMPTY_PART_RULE
    (record,) = info["empty_parts_filled"]
    assert (record["empty_part"], record["donor_part"], record["method"]) == (2, 1, "metis_kway_2")
    assert np.array_equal(blocks[blocks != 2], renumbered[blocks != 2])
    assert np.array_equal(np.flatnonzero(blocks == 2), np.flatnonzero(renumbered == 1)[record["sizes_after"][0]:])


def test_worker_failure_is_reported():
    indptr, indices = csr_from_edges(np.array([[0, 1], [1, 2]]), 3)
    with pytest.raises(part_mod.PartitionError, match="exit code"):
        part_mod.run_pymetis_job(indptr, indices, np.array([0, 3]), np.array([2, 2]), seed=1, ufactor=30)


def test_fine_partition_nests_and_covers(planted, metis6):
    indptr, indices, _ = planted
    blocks, _ = metis6
    info: dict = {}
    fine, fine_block = part_mod.fine_partition(indptr, indices, blocks, 5, tool="pymetis", info=info)
    assert fine.shape == blocks.shape and fine.min() == 0
    assert np.bincount(fine).min() > 0
    assert np.array_equal(fine_block[fine], blocks)
    assert np.all(np.diff(fine_block) >= 0)
    assert np.bincount(fine_block).max() <= 5
    assert info["num_fine"] == fine.max() + 1 == fine_block.shape[0]
    assert info["fine_parts_requested"] == 30
    # fine parts are METIS pieces: far fewer in-block edges cross them than random pieces
    rows = np.repeat(np.arange(blocks.shape[0]), np.diff(indptr))
    same_block = blocks[rows] == blocks[indices]
    random_fine = blocks * 5 + np.random.default_rng(0).integers(0, 5, size=blocks.shape[0])
    crossing = np.mean(fine[rows][same_block] != fine[indices][same_block])
    random_crossing = np.mean(random_fine[rows][same_block] != random_fine[indices][same_block])
    assert crossing < 0.75 * random_crossing
    batched, _ = part_mod.fine_partition(indptr, indices, blocks, 5, tool="pymetis",
                                         batch_entries=int(indptr[-1]) // 2)
    assert np.array_equal(batched, fine)


def test_fine_partition_small_blocks():
    indptr, indices = csr_from_edges(np.array([[0, 1], [2, 3], [3, 4], [4, 2], [5, 6]]), 7)
    blocks = np.array([0, 0, 1, 1, 1, 2, 3])
    info: dict = {}
    fine, fine_block = part_mod.fine_partition(indptr, indices, blocks, 5, tool="pymetis", info=info)
    assert np.array_equal(fine_block[fine], blocks)
    assert np.bincount(fine).min() > 0
    assert np.bincount(fine_block, minlength=4)[3] == 1
    assert info["fine_parts_requested"] == 2 + 3 + 1 + 1
    assert info["num_fine"] + info["empty_parts_renumbered"] == 7


def test_bfs_random_balanced_and_seeded(planted):
    indptr, indices, _ = planted
    n = indptr.shape[0] - 1
    blocks, info = part_mod.bfs_random_partition(indptr, indices, 7, seed=5)
    sizes = np.bincount(blocks)
    assert sizes.shape[0] == 7 and sizes.max() - sizes.min() <= 1 and sizes.sum() == n
    again, _ = part_mod.bfs_random_partition(indptr, indices, 7, seed=5)
    other, _ = part_mod.bfs_random_partition(indptr, indices, 7, seed=6)
    assert np.array_equal(blocks, again) and not np.array_equal(blocks, other)
    assert info["tool"] == "bfs_random" and info["imbalance"] <= 7 * (n // 7 + 1) / n + 1e-9
    # isolated nodes and several components
    indptr2, indices2 = csr_from_edges(np.array([[0, 1], [1, 2], [5, 6]]), 9)
    small, _ = part_mod.bfs_random_partition(indptr2, indices2, 4, seed=1)
    assert np.bincount(small).tolist() in ([3, 2, 2, 2],)


def test_scaled_nparts():
    assert part_mod.scaled_nparts(2000, 0.5) == 1000
    assert part_mod.scaled_nparts(20, 2.0) == 40
    assert part_mod.scaled_nparts(3, 0.1) == 1


def test_import_partition_validation(tmp_path):
    good = np.array([1, 0, 2, 2, 1], dtype=np.int64)
    np.save(tmp_path / "good.npy", good)
    blocks, info = part_mod.import_partition(tmp_path / "good.npy", 5, nparts=3)
    assert np.array_equal(blocks, good) and blocks.dtype == np.int32
    assert info["source_sha256"] == hashlib.sha256((tmp_path / "good.npy").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="block ids for"):
        part_mod.import_partition(tmp_path / "good.npy", 6)
    np.save(tmp_path / "range.npy", np.array([0, 1, 3]))
    with pytest.raises(ValueError, match="outside"):
        part_mod.import_partition(tmp_path / "range.npy", 3, nparts=3)
    np.save(tmp_path / "neg.npy", np.array([0, -1, 1]))
    with pytest.raises(ValueError, match="negative"):
        part_mod.import_partition(tmp_path / "neg.npy", 3)
    np.save(tmp_path / "empty.npy", np.array([0, 0, 2, 2]))
    with pytest.raises(ValueError, match="empty block"):
        part_mod.import_partition(tmp_path / "empty.npy", 4)
    (tmp_path / "graph.part.3").write_text("0\n2\n1\n1\n", encoding="ascii")
    text_blocks, _ = part_mod.import_partition(tmp_path / "graph.part.3", 4, nparts=3)
    assert text_blocks.tolist() == [0, 2, 1, 1]
    with pytest.raises(FileNotFoundError):
        part_mod.import_partition(tmp_path / "missing.npy", 4)


def test_import_partition_torch(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save({"node_to_coarse": torch.tensor([1, 0, 1, 0])}, tmp_path / "hier.pt")
    blocks, _ = part_mod.import_partition(tmp_path / "hier.pt", 4)
    assert blocks.tolist() == [1, 0, 1, 0]
    torch.save({"coarse_part_nodes_map": {0: [1, 3], 1: [0, 2]}}, tmp_path / "map.pt")
    blocks, _ = part_mod.import_partition(tmp_path / "map.pt", 4)
    assert blocks.tolist() == [1, 0, 1, 0]


def test_block_contiguous_order():
    node_block = np.array([2, 0, 1, 0, 2, 1, 0])
    node_fine = np.array([5, 1, 3, 0, 4, 3, 0])
    perm = part_mod.block_contiguous_order(node_block, node_fine)
    assert perm.tolist() == [3, 6, 1, 2, 5, 4, 0]
    assert part_mod.block_contiguous_order(node_block).tolist() == [1, 3, 6, 2, 5, 0, 4]
    assert np.all(np.diff(node_block[perm]) >= 0)


def test_metis_graph_file_round_trip(tmp_path):
    indptr, indices = csr_from_edges(np.array([[0, 1], [1, 2], [0, 2], [3, 4]]), 6)
    part_mod.write_metis_graph(tmp_path / "g.metis", indptr, indices, chunk_entries=3)
    lines = (tmp_path / "g.metis").read_text(encoding="ascii").split("\n")
    assert lines[0] == "6 4" and lines[1] == "2 3" and lines[6] == ""
    back_ptr, back_idx = part_mod.read_metis_graph(tmp_path / "g.metis")
    assert np.array_equal(back_ptr, indptr) and np.array_equal(back_idx, indices)


def test_gpmetis_command_and_parsing(planted, tmp_path, monkeypatch):
    indptr, indices, _ = planted
    n = indptr.shape[0] - 1
    record = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_PARTITIONER_RECORD", str(record))
    blocks, info = part_mod.metis_partition(indptr, indices, 3, binary=FAKE)
    assert info["tool"] == "gpmetis" and info["version"] == "5.1.0"
    assert info["backend"]["metis_idx_width_bits"] == "64" and info["backend"]["metis_edge_cut_total"] == 17
    assert np.array_equal(blocks, np.arange(n) * 3 // n)
    args = json.loads(record.read_text(encoding="utf-8"))
    assert args[:4] == ["-ptype=kway", "-objtype=cut", "-ufactor=30", "-seed=1"] and args[-1] == "3"
    fine, fine_block = part_mod.fine_partition(indptr, indices, blocks, 5, binary=FAKE)
    assert np.array_equal(fine_block[fine], blocks) and fine.max() + 1 == 15
    monkeypatch.setenv("FAKE_PARTITIONER_MODE", "crash")
    with pytest.raises(part_mod.PartitionError, match="139"):
        part_mod.metis_partition(indptr, indices, 3, binary=FAKE)


def test_kahip_command_and_unavailable(planted, tmp_path, monkeypatch):
    indptr, indices, _ = planted
    record = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_PARTITIONER_RECORD", str(record))
    blocks, info = part_mod.kahip_partition(indptr, indices, 4, binary=FAKE)
    assert np.bincount(blocks).shape[0] == 4 and info["tool"] == "kahip"
    args = json.loads(record.read_text(encoding="utf-8"))
    assert "--preconfiguration=fast" in args and "--k=4" in args and "--imbalance=3" in args
    monkeypatch.setattr(part_mod.shutil, "which", lambda name: None)
    with pytest.raises(part_mod.PartitionerUnavailable, match="KAHIP_BIN"):
        part_mod.kahip_partition(indptr, indices, 4)
