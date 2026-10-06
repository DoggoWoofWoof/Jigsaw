"""Serving assembly: model checks, numpy block embeddings, staleness and the learned cascades."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from scripts.canonical import backends, cascade, encoder_numpy, features, protocol, queries, rankers, serving
from scripts.canonical import store as store_mod


@pytest.fixture(scope="module")
def sstore(clustered_store, tmp_path_factory):
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("serving")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    rankers.build_artifacts(gstore, ["R0", "R2-32"])
    for regime in ("R0", "R2-32"):
        cascade.build_label_index(gstore, regime)
    out = copy / "queries"
    queries.SplitBuilder(gstore, "synth", "dev", "20260607", sizes=[8], max_per_cell=1).build(
        ["R0", "R2-32"], out, master_tag="20260607")
    sets = {r: queries.QuerySet.load(out / f"{r}.npz") for r in ("R0", "R2-32")}
    return gstore, sets, copy


def _save_model(gstore, model_dir, *, arch="graphsage", regime="R0", trained=None, dataset="synth", seed=0,
                in_dim=None, num_relations=None):
    if in_dim is None:
        in_dim = features.input_dim(regime, int(gstore.meta["attr_dim"]), int(gstore.meta["num_types"]))
    config = {"arch": arch, "in_dim": in_dim, "hidden": 16, "output": 8, "layers": 2, "residual": True,
              "layer_norm": True, "dropout": 0.0}
    if arch == "rgcn":
        config["num_relations"] = num_relations or int(features.store_relation_table(gstore).max()) + 1
    rng = np.random.default_rng(seed)
    state = {k: (rng.normal(size=shape) / np.sqrt(shape[-1])).astype(np.float32)
             for k, shape in encoder_numpy.state_shapes(config).items()}
    extra = {"model_id": f"synth-{trained or regime}-{arch}-test", "dataset": dataset, "regime": trained or regime,
             "selected_epoch": 1}
    encoder_numpy.save_numpy_model(model_dir, config, state, extra=extra)
    return model_dir


def test_check_model_refuses_models_of_other_stores(sstore, tmp_path):
    gstore, _, _ = sstore
    ok = encoder_numpy.NumpyEncoder.load(_save_model(gstore, tmp_path / "ok"))
    assert serving.check_model(gstore, "R0", ok) is None
    shared = encoder_numpy.NumpyEncoder.load(_save_model(gstore, tmp_path / "r2", regime="R2-32", trained="R2-512"))
    assert serving.check_model(gstore, "R2-32", shared) is None  # every R2-* regime uses the R2-512 encoder
    cases = {
        "trained on": _save_model(gstore, tmp_path / "ds", dataset="arxiv"),
        "served by": _save_model(gstore, tmp_path / "rg", regime="R2-32", trained="R1"),
        "input width": _save_model(gstore, tmp_path / "dim", in_dim=5),
        "relations": _save_model(gstore, tmp_path / "rel", arch="rgcn", num_relations=2),
    }
    for match, model_dir in cases.items():
        regime = "R2-32" if match == "served by" else "R0"
        with pytest.raises(ValueError, match=match):
            serving.check_model(gstore, regime, encoder_numpy.NumpyEncoder.load(model_dir))


@pytest.mark.parametrize("arch", ["graphsage", "rgcn"])
def test_learned_cascade_ranks_by_numpy_cosine(sstore, tmp_path, arch):
    gstore, sets, _ = sstore
    model_dir = _save_model(gstore, tmp_path / arch, arch=arch)
    report = serving.build_block_embeddings(gstore, "R0", model_dir, batch_size=4)
    assert report["shape"] == [gstore.num_blocks, 8]
    encoder = encoder_numpy.NumpyEncoder.load(model_dir)
    table = features.store_relation_table(gstore) if arch == "rgcn" else None
    blocks = encoder_numpy.encode_blocks_numpy(encoder, gstore, "R0", rel_table=table, batch_size=4)
    assert np.array_equal(serving.load_block_embeddings(gstore, "R0", encoder, model_dir), blocks)
    other_batch = encoder_numpy.encode_blocks_numpy(encoder, gstore, "R0", rel_table=table, batch_size=64)
    assert np.allclose(other_batch, blocks, atol=1e-6)
    c = serving.make_cascade(gstore, "R0", "jigsaw_learned", backends.NetworkXBackend(), model_dir=model_dir,
                             budgets=[1, 2])
    assert (c.rel_table is not None) == (arch == "rgcn")
    for q in sets["R0"].queries[:6]:
        codes = gstore.sigma("R0").query_codes(q.payload, gstore)
        qi = c.query_inputs(q, codes)
        z = encoder_numpy.encode_query(encoder, gstore, q, "R0", codes, rel_table=table)
        assert np.allclose(qi.embedding, z)
        scores = (blocks / np.linalg.norm(blocks, axis=1, keepdims=True)) @ (z / np.linalg.norm(z))
        ranking, _ = c.bank.rank("learned", qi)
        assert np.allclose(scores[ranking], np.sort(scores)[::-1], atol=1e-6)
        rec = c.run(q, time_limit=10.0)
        assert rec["status"] in (backends.FOUND, backends.NONE)
        if q.kind == "positive":
            assert rec["status"] == backends.FOUND and rec["witness_check"] == "ok"


def test_rgcn_query_encoder_needs_relations(sstore, tmp_path):
    gstore, sets, _ = sstore
    encoder = encoder_numpy.NumpyEncoder.load(_save_model(gstore, tmp_path / "rgcn", arch="rgcn"))
    q = sets["R0"].queries[0]
    codes = gstore.sigma("R0").query_codes(q.payload, gstore)
    embed = serving.QueryEncoder(encoder, needs_rel=True)
    with pytest.raises(ValueError, match="relations"):
        embed(features.query_graph_inputs(gstore, q, "R0", codes))
    table = features.store_relation_table(gstore)
    assert embed(features.query_graph_inputs(gstore, q, "R0", codes, table)).shape == (8,)


def test_block_embeddings_of_other_weights_are_refused(sstore, tmp_path):
    gstore, _, _ = sstore
    model_dir = _save_model(gstore, tmp_path / "m", seed=1)
    serving.build_block_embeddings(gstore, "R0", model_dir)
    _save_model(gstore, model_dir, seed=2)  # retrained under the same model id
    encoder = encoder_numpy.NumpyEncoder.load(model_dir)
    with pytest.raises(ValueError, match="other weights"):
        serving.load_block_embeddings(gstore, "R0", encoder, model_dir)
    with pytest.raises(ValueError, match="other weights"):
        serving.make_cascade(gstore, "R0", "jigsaw_fused", backends.NetworkXBackend(), model_dir=model_dir)


def test_make_cascade_inputs_per_method(sstore, tmp_path):
    gstore, _, _ = sstore
    model_dir = _save_model(gstore, tmp_path / "m")
    serving.build_block_embeddings(gstore, "R0", model_dir)
    nx = backends.NetworkXBackend()
    kw = {"budgets": [1, 2]}  # the synthetic dataset has no protocol ladder
    for method in ("jigsaw_index", "mean_feature", "topo_feature", "random", "filterall", "direct"):
        c = serving.make_cascade(gstore, "R0", method, nx, **kw)
        assert c.embed is None
    for method in ("jigsaw_learned", "jigsaw_fused", "mean_rrf"):
        with pytest.raises(ValueError, match="model_dir"):
            serving.make_cascade(gstore, "R0", method, nx, **kw)
        assert serving.make_cascade(gstore, "R0", method, nx, model_dir=model_dir, **kw).embed is not None
    with pytest.raises(ValueError, match="router threshold"):
        serving.make_cascade(gstore, "R0", "jigsaw", nx, model_dir=model_dir, **kw)
    c = serving.make_cascade(gstore, "R0", "jigsaw", nx, model_dir=model_dir, router_tau=0.5, **kw)
    assert c.bank.router_tau == 0.5
    with pytest.raises(ValueError, match="not a cascade method"):
        serving.make_cascade(gstore, "R0", "classical_gql", nx, **kw)


def test_cli_embeds_every_served_regime(sstore, tmp_path, capsys):
    gstore, _, copy = sstore
    model_dir = _save_model(gstore, tmp_path / "r2", regime="R2-32", trained="R2-512")
    encoder = encoder_numpy.NumpyEncoder.load(model_dir)
    assert serving.served_regimes("arxiv", encoder) == [r for r in protocol.regimes_for("arxiv") if r.startswith("R2")]
    assert serving.main(["embed-blocks", "--root", str(copy), "--dataset", "synth", "--model-dir", str(model_dir),
                         "--regimes", "R2-32"]) == 0
    assert serving.main(["check", "--root", str(copy), "--dataset", "synth", "--model-dir", str(model_dir),
                         "--regimes", "R2-32"]) == 0
    out = capsys.readouterr().out
    assert '"R2-32"' in out and "skipped" in out
