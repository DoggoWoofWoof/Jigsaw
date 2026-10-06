"""The numpy encoder against torch, its strict loading, its model files and its torch-free import."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import encoder_numpy, features, protocol, queries
from scripts.canonical import store as store_mod
from scripts.canonical.encoder_numpy import NumpyEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
TOL = 2e-5  # max abs difference between numpy and torch normalised embeddings
NUM_TYPES = 3
ATTR_DIM = 6
IN_DIM = features.input_dim("R0", ATTR_DIM, NUM_TYPES)
FULL_TABLE = features.relation_table([(a, b) for a in range(NUM_TYPES) for b in range(NUM_TYPES)], NUM_TYPES)
NUM_FULL_RELATIONS = int(FULL_TABLE.max()) + 1  # 9 relations, so rgcn uses 8 bases


@pytest.fixture(scope="module")
def torch_encoders():
    pytest.importorskip("torch")
    from scripts.canonical import encoders

    return encoders


def _graph(rng: np.random.Generator, n: int, *, rel_table: np.ndarray | None = None, p: float = 0.4,
           edgeless: bool = False) -> dict[str, np.ndarray]:
    """R0 inputs of a random typed graph; for n >= 5 two random nodes stay isolated."""
    k = 0 if edgeless else (n - 2 if n >= 5 else n)
    active = rng.permutation(n)[:k]
    edges = [(int(active[a]), int(active[b])) for a in range(k) for b in range(a + 1, k) if rng.random() < p]
    g = store_mod.local_graph_from_edges(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    types = rng.integers(0, NUM_TYPES, size=n)
    attrs = rng.normal(size=(n, ATTR_DIM)).astype(np.float32)
    return features.graph_inputs("R0", g.indptr, g.indices, types, NUM_TYPES, attrs=attrs, rel_table=rel_table)


def _graphs(seed: int, rel_table: np.ndarray | None = None) -> list[dict[str, np.ndarray]]:
    """Graphs with 0, 1, 5 and 17 nodes plus an edgeless 4-node graph."""
    rng = np.random.default_rng(seed)
    out = [_graph(rng, n, rel_table=rel_table) for n in (0, 1, 5, 17)]
    return out + [_graph(rng, 4, rel_table=rel_table, edgeless=True)]


def _torch_model(encoders, arch: str, *, in_dim: int = IN_DIM, hidden: int = 32, output: int = 16, layers: int = 3,
                 residual: bool = True, layer_norm: bool = True, num_relations: int = 1, **kw):
    import torch

    torch.manual_seed(0)
    model = encoders.GraphEncoder(arch, in_dim, hidden=hidden, output=output, layers=layers, residual=residual,
                                  layer_norm=layer_norm, num_relations=num_relations, **kw)
    with torch.no_grad():  # move zero biases and unit LayerNorm weights off their init
        for param in model.parameters():
            param.add_(0.1 * torch.randn_like(param))
    return model.eval()


def _numpy_model(arch: str, *, in_dim: int = IN_DIM, layers: int = 2, num_relations: int = 1,
                 seed: int = 0) -> tuple[dict, dict[str, np.ndarray]]:
    """A random torch-free config and state (no torch needed)."""
    config = encoder_numpy.normalize_config({"arch": arch, "in_dim": in_dim, "hidden": 16, "output": 8,
                                             "layers": layers, "num_relations": num_relations})
    rng = np.random.default_rng(seed)
    state = {k: (rng.normal(size=s) / np.sqrt(s[-1])).astype(np.float32)
             for k, s in encoder_numpy.state_shapes(config).items()}
    return config, state


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return float(np.abs(a - b).max()) if a.size else 0.0


# ----------------------------------------------------------------------------
# Equivalence with torch
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("arch", encoder_numpy.ARCHS)
def test_matches_torch(torch_encoders, arch, residual, layer_norm, layers):
    assert encoder_numpy.ARCHS == torch_encoders.ARCHS
    rgcn = arch == "rgcn"
    graphs = _graphs(seed=layers, rel_table=FULL_TABLE if rgcn else None)
    model = _torch_model(torch_encoders, arch, layers=layers, residual=residual, layer_norm=layer_norm,
                         num_relations=NUM_FULL_RELATIONS if rgcn else 1)
    state = torch_encoders.export_state(model)
    assert encoder_numpy.state_shapes(model.config) == {k: v.shape for k, v in state.items()}
    encoder = NumpyEncoder(model.config, state)
    assert encoder.config == model.config
    ref = torch_encoders.encode(model, graphs, batch_size=2)
    got = encoder.encode(graphs)
    assert got.shape == (len(graphs), 16) and got.dtype == np.float32
    assert _max_diff(got, ref) <= TOL
    assert _max_diff(encoder.encode(graphs, batch_size=1), ref) <= TOL


@pytest.mark.parametrize("variant", ["no_rel", "few_relations", "two_bases"])
def test_rgcn_relation_variants_match_torch(torch_encoders, variant):
    """Graphs without rel (all relation 0), relation ids >= num_relations (skipped), explicit num_bases."""
    with_rel = variant != "no_rel"
    graphs = _graphs(seed=7, rel_table=FULL_TABLE if with_rel else None)
    kw = {"num_relations": 4} if variant == "few_relations" else {"num_relations": NUM_FULL_RELATIONS}
    if variant == "two_bases":
        kw["num_bases"] = 2
    model = _torch_model(torch_encoders, "rgcn", **kw)
    got = NumpyEncoder(model.config, torch_encoders.export_state(model)).encode(graphs)
    assert _max_diff(got, torch_encoders.encode(model, graphs)) <= TOL


def test_gat_with_sharp_attention_matches_torch(torch_encoders):
    import torch

    graphs = _graphs(seed=11)
    model = _torch_model(torch_encoders, "gat")
    with torch.no_grad():
        for conv in model.convs:
            conv.att_src.mul_(50.0)
            conv.att_dst.mul_(50.0)
    got = NumpyEncoder(model.config, torch_encoders.export_state(model)).encode(graphs)
    assert np.isfinite(got).all()
    assert _max_diff(got, torch_encoders.encode(model, graphs)) <= TOL


@pytest.mark.parametrize("arch", sorted(protocol.TRAINING["architectures"]))
def test_matches_torch_at_the_deployed_size(torch_encoders, arch):
    spec = protocol.TRAINING["architectures"][arch]
    rgcn = arch == "rgcn"
    rng = np.random.default_rng(3)
    table = FULL_TABLE if rgcn else None
    graphs = [_graph(rng, n, rel_table=table, p=min(0.4, 6.0 / max(n, 1))) for n in (0, 1, 20, 50)]
    model = _torch_model(torch_encoders, arch, hidden=spec["hidden"], output=spec["output"], layers=spec["layers"],
                         residual=spec["residual"], layer_norm=spec["layer_norm"],
                         num_relations=NUM_FULL_RELATIONS if rgcn else 1)
    got = NumpyEncoder(model.config, torch_encoders.export_state(model)).encode(graphs)
    assert _max_diff(got, torch_encoders.encode(model, graphs)) <= TOL


def test_saved_torch_model_matches_torch(torch_encoders, tmp_path):
    model = _torch_model(torch_encoders, "rgcn", num_relations=NUM_FULL_RELATIONS)
    encoder_numpy.save_numpy_model(tmp_path / "m", model.config, torch_encoders.export_state(model),
                                   extra={"dataset": "synth", "regime": "R0", "seed": 0, "epoch": 1})
    loaded = NumpyEncoder.load(tmp_path / "m")
    graphs = _graphs(seed=4, rel_table=FULL_TABLE)
    assert loaded.config == model.config
    assert _max_diff(loaded.encode(graphs), torch_encoders.encode(model, graphs)) <= TOL


@pytest.mark.parametrize("regime", ["R0", "R1"])
@pytest.mark.parametrize("arch", encoder_numpy.ARCHS)
def test_blocks_match_torch(torch_encoders, synthetic_store, arch, regime):
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth")
    in_dim = features.input_dim(regime, int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    num_relations = int(features.store_relation_table(gstore).max()) + 1 if arch == "rgcn" else 1
    model = _torch_model(torch_encoders, arch, in_dim=in_dim, num_relations=num_relations)
    encoder = NumpyEncoder(model.config, torch_encoders.export_state(model))
    coarse = encoder_numpy.encode_blocks_numpy(encoder, gstore, regime, batch_size=4)
    assert coarse.shape == (gstore.num_blocks, 16)
    assert _max_diff(coarse, torch_encoders.encode_blocks(model, gstore, regime, batch_size=4)) <= TOL
    fine = encoder_numpy.encode_blocks_numpy(encoder, gstore, regime, fine=True, blocks=[0, 1])
    assert fine.shape == (2, 16)
    assert _max_diff(fine, torch_encoders.encode_blocks(model, gstore, regime, fine=True, blocks=[0, 1])) <= TOL


# ----------------------------------------------------------------------------
# numpy only
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("arch", encoder_numpy.ARCHS)
def test_batch_independence(arch):
    rgcn = arch == "rgcn"
    graphs = _graphs(seed=5, rel_table=FULL_TABLE if rgcn else None)
    encoder = NumpyEncoder(*_numpy_model(arch, num_relations=NUM_FULL_RELATIONS if rgcn else 1))
    together = encoder.encode(graphs)
    alone = np.stack([encoder.encode_one(g) for g in graphs])
    pairs = encoder.encode(graphs, batch_size=2)
    assert together.shape == (len(graphs), 8) and together.dtype == np.float32
    assert np.allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-5)
    assert _max_diff(together, alone) <= 1e-6 and _max_diff(together, pairs) <= 1e-6
    assert encoder.encode([]).shape == (0, 8)


def test_rel_is_ignored_by_other_architectures_and_required_consistently_by_rgcn():
    with_rel = _graphs(seed=6, rel_table=FULL_TABLE)
    without = [{k: v for k, v in g.items() if k != "rel"} for g in with_rel]
    sage = NumpyEncoder(*_numpy_model("graphsage"))
    assert np.array_equal(sage.encode(with_rel), sage.encode(without))
    assert np.array_equal(sage.encode(with_rel[:3] + without[3:]), sage.encode(without))
    rgcn = NumpyEncoder(*_numpy_model("rgcn", num_relations=NUM_FULL_RELATIONS))
    assert not np.allclose(rgcn.encode(with_rel), rgcn.encode(without), atol=1e-4)
    with pytest.raises(ValueError, match="rel"):
        rgcn.encode(with_rel[:3] + without[3:])


def test_rejects_malformed_inputs():
    encoder = NumpyEncoder(*_numpy_model("graphsage"))
    g = _graphs(seed=3)[3]
    with pytest.raises(ValueError, match="shape"):
        encoder.encode([dict(g, x=g["x"][:, :-1])])
    with pytest.raises(ValueError, match="neighbour"):
        encoder.encode([dict(g, indices=g["indices"] + 100)])
    with pytest.raises(ValueError, match="CSR"):
        encoder.encode([dict(g, indices=g["indices"][:-1])])
    with pytest.raises(ValueError, match="batch_size"):
        encoder.encode([g], batch_size=0)


def test_strict_state_checks():
    config, state = _numpy_model("graphsage")
    NumpyEncoder(config, state)
    missing = {k: v for k, v in state.items() if k != "norms.0.bias"}
    with pytest.raises(ValueError, match="missing"):
        NumpyEncoder(config, missing)
    extra = dict(state, **{"convs.0.lin_self.bias": np.zeros(16, dtype=np.float32)})
    with pytest.raises(ValueError, match="unexpected"):
        NumpyEncoder(config, extra)
    wrong = dict(state, **{"input_proj.weight": np.zeros((16, IN_DIM + 1), dtype=np.float32)})
    with pytest.raises(ValueError, match="shape"):
        NumpyEncoder(config, wrong)
    with pytest.raises(ValueError, match="unexpected"):  # LayerNorm weights in a config without LayerNorm
        NumpyEncoder(dict(config, layer_norm=False), state)
    rgcn_config, rgcn_state = _numpy_model("rgcn", num_relations=5)
    with pytest.raises(ValueError, match="shape"):  # 5 bases saved, 2 declared
        NumpyEncoder(dict(rgcn_config, num_bases=2), rgcn_state)
    with pytest.raises(ValueError, match="architecture"):
        NumpyEncoder(dict(config, arch="gcn"), state)


def test_save_load_roundtrip_and_tampering(tmp_path):
    config, state = _numpy_model("rgcn", num_relations=NUM_FULL_RELATIONS)
    model_dir = tmp_path / "model"
    extra = {"dataset": "synth", "regime": "R0", "seed": np.int64(3), "epoch": 5, "protocol_hash": "abc"}
    written = encoder_numpy.save_numpy_model(model_dir, config, state, extra=extra)
    assert set(written) == {"format", "config", "weights_sha256", "state_sha256", "extra"}
    assert written["state_sha256"] == encoder_numpy.state_sha256(state)
    assert written["format"] == encoder_numpy.FORMAT and written["extra"]["seed"] == 3
    assert sorted(p.name for p in model_dir.iterdir()) == ["model.json", "weights.npz"]
    with np.load(model_dir / "weights.npz", allow_pickle=False) as npz:
        assert sorted(npz.files) == sorted(state) and all(npz[k].dtype == np.float32 for k in npz.files)

    loaded_config, loaded_state, model_json = encoder_numpy.load_numpy_model(model_dir)
    assert loaded_config == config and model_json == written
    assert loaded_state.keys() == state.keys()
    assert all(np.array_equal(loaded_state[k], state[k]) for k in state)
    graphs = _graphs(seed=2, rel_table=FULL_TABLE)
    loaded = NumpyEncoder.load(model_dir)
    assert loaded.model_json == written
    assert np.array_equal(loaded.encode(graphs), NumpyEncoder(config, state).encode(graphs))

    weights = model_dir / "weights.npz"
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0x01
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        encoder_numpy.load_numpy_model(model_dir)
    with pytest.raises(ValueError, match="sha256"):
        NumpyEncoder.load(model_dir)


def test_save_refuses_bad_state_and_load_refuses_other_formats(tmp_path):
    config, state = _numpy_model("gin")
    incomplete = {k: v for k, v in state.items() if k != "input_proj.bias"}
    with pytest.raises(ValueError, match="missing"):
        encoder_numpy.save_numpy_model(tmp_path / "bad", config, incomplete)
    assert not (tmp_path / "bad").exists()
    encoder_numpy.save_numpy_model(tmp_path / "ok", config, state)
    meta_path = tmp_path / "ok" / "model.json"
    meta_path.write_text(meta_path.read_text(encoding="utf-8").replace(encoder_numpy.FORMAT, "other-format"),
                         encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        encoder_numpy.load_numpy_model(tmp_path / "ok")


@pytest.mark.parametrize("regime", ["R0", "R1"])
def test_query_of_a_whole_block_encodes_like_the_block(synthetic_store, regime):
    """Query-side inputs come from the payload only, yet match the block-side inputs of the same graph."""
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth")
    in_dim = features.input_dim(regime, int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    num_relations = int(features.store_relation_table(gstore).max()) + 1
    encoder = NumpyEncoder(*_numpy_model("rgcn", in_dim=in_dim, num_relations=num_relations))
    nodes = gstore.part.nodes_of_blocks([2])
    graph = gstore.induced(nodes)
    payload = queries.payload_for(gstore, nodes, regime)
    query = queries.Query(uid="block2", family="single", size=graph.n, kind="positive", regime=regime,
                          edges=graph.edge_list(), payload=payload, planted=nodes, seed=0, attempt=0)
    codes = gstore.sigma(regime).query_codes(payload, gstore)
    got = encoder_numpy.encode_query(encoder, gstore, query, regime, codes)
    want = encoder_numpy.encode_blocks_numpy(encoder, gstore, regime, blocks=[2])[0]
    assert got.shape == (8,)
    assert _max_diff(got, want) <= 1e-6


# ----------------------------------------------------------------------------
# Torch-free import and streaming use
# ----------------------------------------------------------------------------
def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def test_import_does_not_load_torch():
    code = ("import sys; import scripts.canonical.encoder_numpy; "
            "assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules")
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


def test_streaming_encode_does_not_load_torch(tmp_path, synthetic_store):
    """Load a saved model, open the store memory-mapped, encode blocks, fine parts and a query."""
    root, _ = synthetic_store
    gstore = store_mod.GraphStore(root, "synth")
    in_dim = features.input_dim("R1", int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    num_relations = int(features.store_relation_table(gstore).max()) + 1
    config, state = _numpy_model("rgcn", in_dim=in_dim, num_relations=num_relations)
    encoder_numpy.save_numpy_model(tmp_path / "model", config, state)
    num_fine = int(gstore.part.fine_ptr.shape[0]) - 1
    code = textwrap.dedent(f"""
        import sys
        from types import SimpleNamespace
        import numpy as np
        from scripts.canonical import encoder_numpy, features, store
        s = store.GraphStore({str(root)!r}, "synth", mmap=True)
        enc = encoder_numpy.NumpyEncoder.load({str(tmp_path / "model")!r})
        table = features.store_relation_table(s)
        coarse = encoder_numpy.encode_blocks_numpy(enc, s, "R1", batch_size=4, rel_table=table)
        fine = encoder_numpy.encode_blocks_numpy(enc, s, "R1", fine=True, rel_table=table)
        nodes = s.part.nodes_of_blocks([1])[:12]
        g = s.induced(nodes)
        payload = {{"types": np.asarray(s.node_type[nodes]), "classes": np.asarray(s.classes[nodes], dtype=np.int64)}}
        query = SimpleNamespace(indptr=g.indptr, indices=g.indices, payload=payload)
        z = encoder_numpy.encode_query(enc, s, query, "R1", s.sigma("R1").query_codes(payload, s), rel_table=table)
        assert coarse.shape == ({gstore.num_blocks}, 8) and fine.shape == ({num_fine}, 8) and z.shape == (8,)
        assert np.isfinite(coarse).all() and np.isfinite(fine).all() and np.isfinite(z).all()
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "torch_geometric"))
        assert not loaded, loaded
    """)
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


def test_state_sha256_names_the_weights_not_the_file(tmp_path):
    config, state = _numpy_model("graphsage")
    a = encoder_numpy.save_numpy_model(tmp_path / "a", config, state)
    b = encoder_numpy.save_numpy_model(tmp_path / "b", config, {k: v.copy() for k, v in reversed(list(state.items()))})
    assert a["state_sha256"] == b["state_sha256"] == encoder_numpy.state_sha256(state)
    # the file bytes depend on the weights alone: no zip timestamps, members in name order
    assert a["weights_sha256"] == b["weights_sha256"]
    assert (tmp_path / "a" / "weights.npz").read_bytes() == (tmp_path / "b" / "weights.npz").read_bytes()
    changed = dict(state, **{"input_proj.bias": state["input_proj.bias"] + np.float32(1e-3)})
    assert encoder_numpy.state_sha256(changed) != a["state_sha256"]
    meta_path = tmp_path / "a" / "model.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["state_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="state_sha256"):
        encoder_numpy.load_numpy_model(tmp_path / "a")
