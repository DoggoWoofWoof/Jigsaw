from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.canonical import encoders, features, store as store_mod  # noqa: E402


def _random_graph(rng, n, p=0.2, num_types=1, d=12, rel_table=None):
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    g = store_mod.local_graph_from_edges(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    types = rng.integers(0, num_types, size=n)
    attrs = rng.normal(size=(n, d)).astype(np.float32)
    return features.graph_inputs("R0", g.indptr, g.indices, types, num_types, attrs=attrs, rel_table=rel_table), types, attrs, g


def _model(arch, in_dim, num_relations=1, **kw):
    torch.manual_seed(0)
    m = encoders.GraphEncoder(arch, in_dim, hidden=32, output=16, layers=3, num_relations=num_relations, **kw)
    return m.eval()


@pytest.mark.parametrize("arch", encoders.ARCHS)
def test_shapes_norms_and_batch_independence(arch):
    rng = np.random.default_rng(0)
    num_types = 3 if arch == "rgcn" else 1
    table = features.relation_table([(a, b) for a in range(3) for b in range(3)], 3) if arch == "rgcn" else None
    graphs = [_random_graph(rng, n, num_types=num_types, rel_table=table)[0] for n in (1, 5, 17, 9)]
    in_dim = graphs[0]["x"].shape[1]
    model = _model(arch, in_dim, num_relations=9 if arch == "rgcn" else 1)
    together = encoders.encode(model, graphs, batch_size=4)
    alone = np.concatenate([encoders.encode(model, [g]) for g in graphs])
    assert together.shape == (4, 16)
    assert np.allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-5)
    assert np.allclose(together, alone, atol=1e-5)


@pytest.mark.parametrize("arch", encoders.ARCHS)
def test_permutation_invariance(arch):
    rng = np.random.default_rng(1)
    num_types = 2 if arch == "rgcn" else 1
    table = features.relation_table([(0, 0), (0, 1), (1, 0), (1, 1)], 2) if arch == "rgcn" else None
    inputs, types, attrs, g = _random_graph(rng, 14, p=0.3, num_types=num_types, rel_table=table)
    perm = rng.permutation(14)
    inv = np.argsort(perm)
    edges = inv[g.edge_list()]
    g2 = store_mod.local_graph_from_edges(14, edges)
    inputs2 = features.graph_inputs("R0", g2.indptr, g2.indices, types[perm], num_types, attrs=attrs[perm], rel_table=table)
    model = _model(arch, inputs["x"].shape[1], num_relations=4 if arch == "rgcn" else 1)
    a, b = encoders.encode(model, [inputs]), encoders.encode(model, [inputs2])
    assert np.allclose(a, b, atol=1e-5)


def test_empty_graph_is_finite():
    model = _model("graphsage", 5)
    empty = {"x": np.zeros((0, 5), dtype=np.float32), "indptr": np.zeros(1, dtype=np.int64),
             "indices": np.zeros(0, dtype=np.int64)}
    one = {"x": np.ones((1, 5), dtype=np.float32), "indptr": np.zeros(2, dtype=np.int64),
           "indices": np.zeros(0, dtype=np.int64)}
    z = encoders.encode(model, [empty, one])
    assert np.isfinite(z).all()


def test_rgcn_uses_relations():
    rng = np.random.default_rng(2)
    table = features.relation_table([(0, 0), (0, 1), (1, 0), (1, 1)], 2)
    inputs, *_ = _random_graph(rng, 12, p=0.4, num_types=2, rel_table=table)
    model = _model("rgcn", inputs["x"].shape[1], num_relations=4)
    swapped = dict(inputs)
    swapped["rel"] = (inputs["rel"] + 1) % 4
    assert not np.allclose(encoders.encode(model, [inputs]), encoders.encode(model, [swapped]), atol=1e-4)


def test_state_roundtrip_and_no_residual_variant():
    model = _model("rgcn", 7, num_relations=5, residual=False, layer_norm=False)
    clone = encoders.build_encoder(model.config)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone.config == model.config
    state = encoders.export_state(model)
    assert all(v.dtype == np.float32 for v in state.values())
    assert "convs.0.bases" in state and "norms.0.weight" not in state


def test_encode_blocks_on_store(synthetic_store):
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth")
    rel_table = features.store_relation_table(gstore)
    num_rel = int(rel_table.max()) + 1
    for regime in ("R0", "R1"):
        in_dim = features.input_dim(regime, 8, 3)
        model = _model("rgcn", in_dim, num_relations=num_rel)
        emb = encoders.encode_blocks(model, gstore, regime, batch_size=4)
        assert emb.shape == (gstore.num_blocks, 16) and np.isfinite(emb).all()
        fine = encoders.encode_blocks(model, gstore, regime, fine=True, blocks=[0, 1])
        assert fine.shape == (2, 16)
