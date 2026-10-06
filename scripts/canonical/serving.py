"""Serving assembly: the numpy query encoder, block embeddings and one method's Cascade.

    python -m scripts.canonical.serving embed-blocks --root STORES --dataset arxiv \\
        --model-dir MODELS/arxiv-R0-graphsage-seed7301-main
    python -m scripts.canonical.serving check --root STORES --dataset arxiv --regimes R0 \\
        --model-dir MODELS/arxiv-R0-graphsage-seed7301-main

Both sides of the learned cosine come from encoder_numpy: block embeddings are
built here, offline, and query embeddings inside the timed retrieval stage, so
serving never imports torch (protocol.SERVING["encoder"]). The trainer's torch
block_emb.npy only checks the numpy port; `check` recomputes that difference.
BLAS sums depend on the batch shape, so embeddings built with another batch
size differ in the last float32 bits; info.json records the batch size and the
canonical run reads the embedding files it built, by their SHA-256.

A model serves a regime when it was trained on the store's dataset for
protocol.TRAINED_REGIMES[regime] and its input width and relation count match
the store. Block embeddings record the weights' state_sha256, and loading
refuses embeddings built from other weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import cascade, encoder_numpy, features, protocol, rankers
from .store import GraphStore


def model_id_for(dataset: str, regime: str, arch: str, seed: int, arm: str = "main") -> str:
    """The trainer's model id (train_encoder.model_id_for), restated so serving never imports torch."""
    return f"{dataset}-{regime}-{arch}-seed{int(seed)}-{arm}"


def default_model_id(dataset: str, regime: str, seed: int) -> str:
    """The main-arm encoder of a training seed that serves `regime` on `dataset`."""
    return model_id_for(dataset, protocol.TRAINED_REGIMES[regime], protocol.DATASETS[dataset]["encoder"], seed)


class QueryEncoder:
    """A Cascade's embed callable: the encoder inputs of one query to its embedding."""

    def __init__(self, encoder: encoder_numpy.NumpyEncoder, *, needs_rel: bool) -> None:
        self.encoder = encoder
        self.needs_rel = needs_rel

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if self.needs_rel and "rel" not in inputs:
            raise ValueError("rgcn query inputs need edge relations (the Cascade's rel_table)")
        return self.encoder.encode_one(inputs)


def model_identity(encoder: encoder_numpy.NumpyEncoder) -> dict[str, Any]:
    model_json = encoder.model_json or {}
    extra = dict(model_json.get("extra") or {})
    return {"model_id": extra.get("model_id"), "dataset": extra.get("dataset"), "trained_regime": extra.get("regime"),
            "arch": encoder.arch, "weights_sha256": model_json.get("weights_sha256"),
            "state_sha256": model_json.get("state_sha256"), "selected_epoch": extra.get("selected_epoch")}


def check_model(store: GraphStore, regime: str, encoder: encoder_numpy.NumpyEncoder) -> np.ndarray | None:
    """The relation table the model reads (rgcn), else None; ValueError when it cannot serve the regime."""
    ident = model_identity(encoder)
    problems = []
    if ident["dataset"] is not None and ident["dataset"] != store.dataset:
        problems.append(f"trained on {ident['dataset']}, the store holds {store.dataset}")
    serving = protocol.TRAINED_REGIMES.get(regime)
    if serving is None:
        problems.append(f"unknown regime {regime!r}")
    elif ident["trained_regime"] is not None and ident["trained_regime"] != serving:
        problems.append(f"trained for {ident['trained_regime']}, {regime} is served by the {serving} encoder")
    in_dim = features.input_dim(regime, int(store.meta["attr_dim"]), int(store.meta["num_types"]))
    if encoder.in_dim != in_dim:
        problems.append(f"input width {encoder.in_dim}, {regime} inputs have {in_dim}")
    table = None
    if encoder.arch == "rgcn":
        table = features.store_relation_table(store)
        relations = int(table.max()) + 1
        if encoder.num_relations != relations:
            problems.append(f"{encoder.num_relations} relations, the store has {relations}")
    if problems:
        raise ValueError("the model cannot serve this store: " + "; ".join(problems))
    return table


def model_key(encoder: encoder_numpy.NumpyEncoder, model_dir: str | Path) -> str:
    return model_identity(encoder)["model_id"] or Path(model_dir).name


def build_block_embeddings(store: GraphStore, regime: str, model_dir: str | Path, *,
                           batch_size: int = 64) -> dict[str, Any]:
    """Encode every block with the numpy encoder and save it under the store's embeddings directory."""
    encoder = encoder_numpy.NumpyEncoder.load(model_dir)
    table = check_model(store, regime, encoder)
    ident = model_identity(encoder)
    t0 = time.perf_counter()
    emb = encoder_numpy.encode_blocks_numpy(encoder, store, regime, batch_size=batch_size, rel_table=table)
    seconds = time.perf_counter() - t0
    if not np.isfinite(emb).all():
        raise ValueError("non-finite block embeddings")
    key = model_key(encoder, model_dir)
    directory = rankers.save_block_embeddings(store, key, regime, emb, extra={
        "encoder": "encoder_numpy", "model_dir": str(model_dir), "arch": ident["arch"],
        "trained_regime": ident["trained_regime"], "weights_sha256": ident["weights_sha256"],
        "state_sha256": ident["state_sha256"], "selected_epoch": ident["selected_epoch"],
        "batch_size": int(batch_size), "seconds": seconds, "threads": _thread_info()})
    return {"model_id": key, "regime": regime, "dir": str(directory), "shape": list(emb.shape), "seconds": seconds}


def load_block_embeddings(store: GraphStore, regime: str, encoder: encoder_numpy.NumpyEncoder,
                          model_dir: str | Path) -> np.ndarray:
    ident = model_identity(encoder)
    emb, info = rankers.load_block_embeddings(store, model_key(encoder, model_dir), regime)
    for field in ("state_sha256", "weights_sha256"):
        if ident[field] is not None and info.get(field) != ident[field]:
            raise ValueError(f"block embeddings of {regime} were built from other weights ({field}); rebuild them")
    if emb.shape != (store.num_blocks, encoder.output):
        raise ValueError(f"block embeddings have shape {emb.shape}, expected {(store.num_blocks, encoder.output)}")
    return emb


def make_cascade(store: GraphStore, regime: str, method: str, backend: Any, *, model_dir: str | Path | None = None,
                 router_tau: float | None = None, **kwargs: Any) -> cascade.Cascade:
    """The Cascade of one method with everything its ranker reads.

    Methods whose ranker reads embeddings need model_dir. The router uses the
    frozen protocol threshold of the dataset unless router_tau is given.
    """
    spec = protocol.METHODS.get(method)
    if spec is None or "external" in spec:
        raise ValueError(f"{method!r} is not a cascade method")
    ranker = spec.get("ranker")
    bank = embed = table = None
    if ranker:
        block_emb = None
        if ranker in rankers.NEEDS_EMBEDDING:
            if model_dir is None:
                raise ValueError(f"{method} ranks with a trained encoder; pass model_dir")
            encoder = encoder_numpy.NumpyEncoder.load(model_dir)
            table = check_model(store, regime, encoder)
            block_emb = load_block_embeddings(store, regime, encoder, model_dir)
            embed = QueryEncoder(encoder, needs_rel=table is not None)
        if ranker == "router":
            router_tau = protocol.router_tau_for(store.dataset) if router_tau is None else float(router_tau)
            if router_tau is None:
                raise ValueError(f"no router threshold for {store.dataset}: fit it on validation master "
                                 "27182 (fit_router.py) or pass router_tau")
        bank = rankers.RankerBank(store, regime, block_emb=block_emb, router_tau=router_tau)
    return cascade.Cascade(store, regime, method, backend, bank=bank, embed=embed, rel_table=table, **kwargs)


def _thread_info() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return [{"threadpoolctl": "missing"}]
    return [{k: info.get(k) for k in ("internal_api", "num_threads", "version")} for info in threadpool_info()]


def parity(store: GraphStore, regime: str, model_dir: str | Path, *, blocks: int = 8) -> dict[str, Any]:
    """Largest difference between these numpy block embeddings and the trainer's torch ones (trained regime only)."""
    encoder = encoder_numpy.NumpyEncoder.load(model_dir)
    table = check_model(store, regime, encoder)
    torch_path = Path(model_dir) / "block_emb.npy"
    if not torch_path.exists() or model_identity(encoder)["trained_regime"] not in (None, regime):
        return {"skipped": "no torch embeddings of this regime"}
    reference = np.load(torch_path)
    ids = list(range(min(int(blocks), store.num_blocks)))
    emb = encoder_numpy.encode_blocks_numpy(encoder, store, regime, blocks=ids, rel_table=table)
    return {"blocks": len(ids), "max_abs_diff": float(np.abs(emb - reference[ids]).max()) if ids else 0.0}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def served_regimes(dataset: str, encoder: encoder_numpy.NumpyEncoder) -> list[str]:
    trained = model_identity(encoder)["trained_regime"]
    return [r for r in protocol.regimes_for(dataset) if trained is None or protocol.TRAINED_REGIMES[r] == trained]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("embed-blocks", "check"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--dataset", required=True)
        p.add_argument("--partition", default="primary")
        p.add_argument("--model-dir", required=True)
        p.add_argument("--regimes", nargs="*", default=None, help="default: every regime the model serves")
        p.add_argument("--batch-size", type=int, default=64)
        if name == "embed-blocks":
            p.add_argument("--run-dir", default=None, help="record the build in a manifest there (offline_build.py, E17)")
    args = parser.parse_args(argv)
    from . import offline_build

    store = GraphStore(args.root, args.dataset, partition=args.partition, mmap=True)
    encoder = encoder_numpy.NumpyEncoder.load(args.model_dir)
    regimes = args.regimes or served_regimes(args.dataset, encoder)
    report: dict[str, Any] = {"model": model_identity(encoder), "regimes": {}}
    inputs = {"store_meta": store.dir / "meta.json", "partition_info": Path(store.part.dir) / "partition.json",
              "model_dir": args.model_dir}
    with offline_build.build_run(getattr(args, "run_dir", None), stage="embeddings", store_dir=store.dir,
                                 dataset=args.dataset, config={"partition": args.partition, "model_dir": str(args.model_dir),
                                                               "regimes": regimes, "batch_size": args.batch_size},
                                 inputs=inputs, argv=argv) as written:
        for regime in regimes:
            if args.command == "embed-blocks":
                report["regimes"][regime] = build_block_embeddings(store, regime, args.model_dir,
                                                                   batch_size=args.batch_size)
                written.append(Path(report["regimes"][regime]["dir"]))
            else:
                emb = load_block_embeddings(store, regime, encoder, args.model_dir)
                report["regimes"][regime] = {"shape": list(emb.shape), "parity": parity(store, regime, args.model_dir)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
