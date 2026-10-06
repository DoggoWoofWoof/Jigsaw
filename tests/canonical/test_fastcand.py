"""fastcand builds the graphs GraphStore.candidate and GraphStore.induced build, array for array."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.canonical import fastcand
from scripts.canonical import store as store_mod


@pytest.fixture(params=[False, True], ids=["resident", "mmap"])
def gstore(request, synthetic_store):
    root, _ = synthetic_store
    return store_mod.GraphStore(root, "synth", mmap=request.param)


def assert_same(a: store_mod.LocalGraph, b: store_mod.LocalGraph) -> None:
    assert a.global_ids.dtype == b.global_ids.dtype and np.array_equal(a.global_ids, b.global_ids)
    assert a.indptr.dtype == b.indptr.dtype and np.array_equal(a.indptr, b.indptr)
    assert a.indices.dtype == b.indices.dtype and np.array_equal(a.indices, b.indices)
    if b.labels is None:
        assert a.labels is None
    else:
        assert a.labels.dtype == b.labels.dtype and np.array_equal(a.labels, b.labels)


@pytest.mark.parametrize("halo,label_filter", [(True, True), (False, True), (True, False), (False, False),
                                               (2, True), (2, False), (3, True)])
@pytest.mark.parametrize("regime", ["R0", "R1"])
def test_candidate_equals_the_store_method(gstore, halo, label_filter, regime):
    rng = np.random.default_rng(11)
    codes = np.asarray(gstore.sigma(regime).codes)
    for trial in range(8):
        blocks = sorted(rng.choice(gstore.num_blocks, size=1 + trial % gstore.num_blocks, replace=False).tolist())
        qcodes = rng.choice(np.unique(codes), size=1 + trial % 5, replace=False)
        stats_old, stats_new = {}, {}
        old = gstore.candidate(blocks, qcodes, regime, stats=stats_old, halo=halo, label_filter=label_filter)
        new = fastcand.candidate(gstore, blocks, qcodes, regime, stats=stats_new, halo=halo,
                                 label_filter=label_filter)
        assert_same(new, old)
        assert stats_new == stats_old


def test_candidate_with_codes_the_data_does_not_carry(gstore):
    codes = np.asarray(gstore.sigma("R1").codes)
    table = gstore.sigma("R1")
    qcodes = np.array([-1, int(codes[0]), table.num_codes, table.num_codes + 5], dtype=np.int64)
    for blocks in ([0], [1, 2], list(range(gstore.num_blocks))):
        assert_same(fastcand.candidate(gstore, blocks, qcodes, "R1"), gstore.candidate(blocks, qcodes, "R1"))
    none = np.array([-1, table.num_codes + 1], dtype=np.int64)
    assert_same(fastcand.candidate(gstore, [0], none, "R1"), gstore.candidate([0], none, "R1"))


def test_candidate_of_no_blocks_is_empty(gstore):
    qcodes = np.asarray(gstore.sigma("R1").codes)[:3]
    assert_same(fastcand.candidate(gstore, [], qcodes, "R1"), gstore.candidate([], qcodes, "R1"))


def test_induced_equals_the_store_method(gstore):
    rng = np.random.default_rng(5)
    n = gstore.num_nodes
    for size in (0, 1, 7, n // 10, n // 2, n):
        nodes = np.sort(rng.choice(n, size=size, replace=False)).astype(np.int64)
        for regime in (None, "R1"):
            assert_same(fastcand.induced(gstore, nodes, regime), gstore.induced(nodes, regime))
    block = gstore.part.nodes_of_blocks([1])  # one contiguous run: the sliced read
    assert_same(fastcand.induced(gstore, block, "R0"), gstore.induced(block, "R0"))


@pytest.mark.parametrize("max_ranges", [0, 1, 4096])
def test_gather_rows_equals_the_store_function(gstore, max_ranges):
    rng = np.random.default_rng(2)
    n = gstore.num_nodes
    for size in (0, 3, n // 3):
        nodes = np.sort(rng.choice(n, size=size, replace=False)).astype(np.int64)
        lens_a, nbrs_a = fastcand.gather_rows(gstore.indptr, gstore.indices, nodes, max_ranges)
        lens_b, nbrs_b = store_mod.gather_rows(gstore.indptr, gstore.indices, nodes, max_ranges)
        assert np.array_equal(lens_a, lens_b) and np.array_equal(nbrs_a, nbrs_b) and nbrs_a.dtype == nbrs_b.dtype


def test_scratch_arrays_are_not_kept(gstore):
    """The bitmaps live for one call: nothing of size |V| stays referenced by the store."""
    before = {k: id(v) for k, v in vars(gstore).items()}
    fastcand.candidate(gstore, [0, 1], np.asarray(gstore.sigma("R1").codes)[:4], "R1")
    after = {k: id(v) for k, v in vars(gstore).items()}
    assert set(after) - set(before) <= {"_sigma"} and all(after[k] == before[k] for k in before if k in after)
