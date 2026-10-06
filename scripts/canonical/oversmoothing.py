"""Layerwise encoder analysis (E6): how node and block representations evolve with depth.

    python -m scripts.canonical.oversmoothing run --store-root STORES --dataset arxiv --regime R0 \\
        --models-root MODELS --models arxiv-R0-graphsage-seed7301-arch-graphsage-L8-nores ... \\
        --out RUNS/E6/layerwise/arxiv-R0 [--partition primary] [--max-blocks 64] [--max-nodes 4096] \\
        [--knockout-residual]
    python -m scripts.canonical.oversmoothing summarize --inputs RUNS/E6/layerwise/arxiv-R0 ... \\
        --out RUNS/E6/layerwise/summary.csv

Design. Each trained encoder of the E6 architecture, depth and residual ablation
(protocol.ARCHITECTURE_ABLATION) is read from its export directory
MODELS_ROOT/<model_id>/ (train_encoder.export_model): torch_state.pt holds
{"config", "state_dict"}, loaded through encoders.build_encoder(config) and
load_state_dict(strict=True), in eval mode, under torch.no_grad() with one torch
thread and one BLAS thread, so the arithmetic is deterministic. The layers are
the model's own: GraphEncoder.forward(..., return_layers=True) gives h_1..h_L,
and h_0 = ReLU(input_proj(x)) is computed with the model's input_proj. Nothing
is reimplemented. Each analysed block of the store's partition is encoded as its
own graph, from the inputs the served block embeddings are built from: the
induced subgraph of the block's nodes, features.store_graph_inputs() and, for
rgcn, features.store_relation_table(). A block's readout z therefore equals its
row of serving.build_block_embeddings up to float32 rounding (the tests check
1e-5). Only the `run` path imports torch; the metrics, validate_record() and
summarize work without it.

Layers. "input" (x), "proj" (h_0), "1".."L" (h_1..h_L) and "readout" (z, block
metrics only).

Sample. All blocks when the partition has at most max_blocks; otherwise
max_blocks blocks drawn without replacement by np.random.default_rng(
protocol.derive_seed(seed, dataset, regime, "blocks")), sorted by id. Nodes are
stratified equally over the chosen blocks, up to max_nodes in total. Every block
gets min(size, q) for the largest q that fits, and the remainder goes one node
each to the first blocks (by id) that still have nodes left. A block's nodes are
drawn by default_rng(derive_seed(seed, dataset, regime, "nodes", block)) and
sorted. The sample depends only on the store, the regime and the settings, so
every model of a run and every training seed is measured on the same nodes.
seed, max_blocks and max_nodes come from sample_settings(), which reads
protocol.ARCHITECTURE_ABLATION["layerwise_sample"] when the protocol defines it
and falls back to DEFAULT_SAMPLE otherwise.

Definitions. All metric arithmetic is float64. Rows are normalised as
u_i = h_i / max(||h_i||, 1e-12), so an exactly-zero representation has cosine 0
with everything.
  mean_pairwise_cosine   the mean over ordered pairs i != j of the sampled nodes
                         of u_i . u_j. It is computed as
                         (||sum_i u_i||^2 - sum_i ||u_i||^2) / (n (n - 1)), with
                         no n x n matrix, and clipped to [-1, 1] against rounding.
  mean_average_distance  1 - mean_pairwise_cosine. This is the global MAD of
                         Chen et al. (AAAI 2020) without neighbourhood masks, the
                         legacy definition.
  effective_rank         exp(H(p)), where p = s / sum(s) and s holds the singular
                         values of the uncentred sampled node matrix (Roy and
                         Vetterli 2007). Singular values at or below the
                         np.linalg.matrix_rank tolerance
                         (s_max * max(rows, cols) * eps) count as zero and
                         contribute nothing. An all-zero matrix gives 0.
  zero_rows              the number of sampled nodes whose representation is
                         exactly zero
  block_effective_rank   effective_rank of the [blocks, dim] matrix of block
                         summaries
  block_separation       the mean over ordered pairs of blocks of
                         1 - cos(summary_a, summary_b)
A block summary is the mean representation over all nodes of the block (zero for
an empty block); for "readout" it is z. Pairwise metrics are null when there are
fewer than two rows. An all-zero layer has cosine 0 and MAD 1, and zero_rows
flags it.

Knockout. --knockout-residual also evaluates each residual model with
model.residual = False at inference: the legacy check on trained weights. The
result goes to <model_id>+knockout.json, with knockout_residual true and
model.residual still the trained value. Residual-off models get no knockout,
because it would equal the trained run; the manifest lists them. summarize keys
rows by knockout, so a knockout never pools with a trained residual-off model.

Rules. A model is refused, before OUT is touched, unless all of these hold:
  - the directory name, the model id in model.json and export.json, and
    serving.model_id_for(dataset, trained regime, arch, seed, arm) all agree;
  - model.json and export.json agree on dataset, regime, arch, seed and arm;
  - the torch weights and config equal the numpy serving copy (state_sha256);
  - export.json's torch_state_sha256 and config match torch_state.pt;
  - the model serves the analysed regime on this store (serving.check_model:
    dataset, protocol.TRAINED_REGIMES, input width, relation count).
OUT may hold only manifest.json and this run's files.

Output. OUT/manifest.json (manifest.RunManifest, experiment E6_layerwise, split
"none" because no query is read) and one JSON file per model and variant:
OUT/<model_id>.json, plus OUT/<model_id>+knockout.json with --knockout-residual.
Each file is self-describing: a reader never parses the file name or the model
id. Files are written with sort_keys and indent 2, and a rerun reproduces them
byte for byte apart from "seconds". Schema "jigsaw-layerwise-v1" (name
jigsaw-layerwise, version 1):
  schema             "jigsaw-layerwise-v1"
  model_id           "<dataset>-<trained_regime>-<arch>-seed<train_seed>-<arm>" as
                     recorded at export (serving.model_id_for); a join key only
  dataset            the store's dataset name, e.g. "arxiv"
  regime             the analysed input regime, e.g. "R0" or "R2-32"
  partition          the store partition, e.g. "primary"
  knockout_residual  bool; true means the trained weights evaluated without the
                     residual, not a model trained without it
  model              {arch: "graphsage" | "rgcn" | "gin" | "gat",
                      layers: int L (message-passing layers),
                      hidden: int,
                      residual: bool (as trained),
                      layer_norm: bool,
                      train_seed: int (training seed),
                      arm: str (e.g. "main" or "arch-graphsage-L8-nores"),
                      trained_regime: str (protocol.TRAINED_REGIMES[regime]),
                      torch_state_sha256: str}
  sample             {blocks: [int] sorted block ids,
                      nodes_per_block: [int] sampled nodes, same order,
                      block_sizes: [int] nodes per block, same order,
                      max_nodes: int, max_blocks: int, seed: int,
                      nodes_sha256: str, the SHA-256 of the sampled global node
                      ids as little-endian int64, in order}
  layers             a list in the order input, proj, 1..L, readout, each entry
                     {layer: str, dim: int, mean_average_distance: float,
                      mean_pairwise_cosine: float, effective_rank: float,
                      zero_rows: int, block_effective_rank: float,
                      block_separation: float}; the four node fields are null
                     for readout, and a pairwise metric is null over fewer than
                     two rows
  definitions        {field or metric: one-line definition}
  protocol_hash      protocol.protocol_hash() at analysis time
  seconds            wall-clock seconds for this model and variant (the only
                     field that differs between reruns)

summarize reads OUT directories whose manifest verifies: status complete, every
output unchanged, and nothing unrecorded. It writes one CSV row per (dataset,
regime, arch, layers, residual, knockout, layer) and arm. Each row holds n_seeds,
the seeds, seeds_complete (the seeds equal protocol.SEEDS["training"]) and, for
every metric, the mean and the sample SD over training seeds (ddof 1, empty for
a single seed). The records of one row must share hidden, layer_norm, partition,
trained regime, sample and protocol hash, and each training seed may appear only
once. The columns are dataset, regime, partition, arch, layers, hidden,
layer_norm, residual, knockout, arm, trained_regime, layer, dim, n_seeds, seeds
(";"-joined, ascending), seeds_complete, sample_blocks and sample_nodes, then
<metric>_mean and <metric>_sd for each metric in metric_names() order. Booleans
are written as true and false, and null as an empty cell. Rows are sorted by the
key, with layers in the order above.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from . import encoder_numpy, features, manifest, protocol, serving
from .store import PRIMARY, GraphStore

SCHEMA = "jigsaw-layerwise-v1"
EXPERIMENT = "E6_layerwise"
SPLIT = "none"  # the analysis reads no query set
TORCH_STATE = "torch_state.pt"
EXPORT_NAME = "export.json"  # train_encoder.EXPORT, restated so summarize never imports torch
KNOCKOUT_SUFFIX = "+knockout"
THREADS = 1  # one torch and one BLAS thread: deterministic CPU arithmetic
NORM_EPS = encoder_numpy.NORMALIZE_EPS  # 1e-12, the eps of F.normalize
DEFAULT_SAMPLE: dict[str, int] = {"seed": 0, "max_blocks": 64, "max_nodes": 4096}
NODE_METRICS = ("mean_average_distance", "mean_pairwise_cosine", "effective_rank", "zero_rows")
BLOCK_METRICS = ("block_effective_rank", "block_separation")
IDENTITY_FIELDS = ("model_id", "dataset", "regime", "arch", "seed", "arm")
RECORD_KEYS = ("schema", "model_id", "model", "dataset", "regime", "partition", "knockout_residual", "sample",
               "layers", "definitions", "protocol_hash", "seconds")
MODEL_KEYS = ("arch", "layers", "hidden", "residual", "layer_norm", "train_seed", "arm", "trained_regime",
              "torch_state_sha256")
BASE_COLUMNS = ("dataset", "regime", "partition", "arch", "layers", "hidden", "layer_norm", "residual", "knockout",
                "arm", "trained_regime", "layer", "dim", "n_seeds", "seeds", "seeds_complete", "sample_blocks",
                "sample_nodes")

DEFINITIONS: dict[str, str] = {
    "layer": "input: encoder input x; proj: h_0 = ReLU(input_proj(x)); k: h_k, the output of message-passing layer "
             "k after norm and ReLU; readout: z, the L2-normalised block embedding (block metrics only)",
    "dim": "width of the layer's representation",
    "mean_pairwise_cosine": f"mean over ordered pairs i != j of sampled nodes of cos(h_i, h_j), rows divided by "
                            f"max(norm, {NORM_EPS:g}) so a zero row has cosine 0",
    "mean_average_distance": "1 - mean_pairwise_cosine: the global MAD of Chen et al. (AAAI 2020) without "
                             "neighbourhood masks, the legacy definition",
    "effective_rank": "exp(entropy of s / sum(s)), s the singular values of the uncentred [sampled nodes, dim] "
                      "matrix (Roy and Vetterli 2007); values within the matrix_rank tolerance count as zero",
    "zero_rows": "number of sampled nodes whose representation is exactly zero",
    "block_effective_rank": "effective_rank of the [blocks, dim] matrix of block summaries",
    "block_separation": "mean over ordered pairs of analysed blocks of 1 - cos(summary_a, summary_b)",
    "block_summary": "mean representation over all nodes of a block (zero for an empty block); z for readout",
    "knockout_residual": "true: the trained weights evaluated with the residual removed at inference, not a model "
                         "trained without it",
    "sample": "blocks: analysed block ids; nodes_per_block: sampled nodes per block (stratified equally, max_nodes "
              "in total); nodes_sha256: SHA-256 of the sampled global node ids as little-endian int64, in order",
    "null": "a pairwise metric over fewer than two rows, and the node metrics of readout",
}


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
def sample_settings() -> dict[str, int]:
    """seed, max_blocks and max_nodes: protocol.ARCHITECTURE_ABLATION["layerwise_sample"] over DEFAULT_SAMPLE."""
    spec = dict(protocol.ARCHITECTURE_ABLATION.get("layerwise_sample") or {})
    unknown = sorted(set(spec) - set(DEFAULT_SAMPLE))
    if unknown:
        raise ValueError(f"unknown layerwise_sample settings {unknown}")
    return {key: int(spec.get(key, default)) for key, default in DEFAULT_SAMPLE.items()}


def metric_names() -> list[str]:
    """protocol.ARCHITECTURE_ABLATION["layerwise_metrics"], then the other metrics this module reports."""
    listed = [str(m) for m in protocol.ARCHITECTURE_ABLATION["layerwise_metrics"]]
    known = NODE_METRICS + BLOCK_METRICS
    unknown = [m for m in listed if m not in known]
    if unknown:
        raise ValueError(f"the protocol lists layerwise metrics this module does not compute: {unknown}")
    return listed + [m for m in known if m not in listed]


def layer_names(layers: int) -> list[str]:
    return ["input", "proj", *[str(i) for i in range(1, int(layers) + 1)], "readout"]


def output_name(model_id: str, knockout: bool) -> str:
    return f"{model_id}{KNOCKOUT_SUFFIX if knockout else ''}.json"


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _matrix(x: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 2:
        raise ValueError(f"expected a [rows, dim] matrix, got shape {x.shape}")
    return x


def unit_rows(x: np.ndarray) -> np.ndarray:
    """float64 rows divided by max(norm, 1e-12), as F.normalize: a zero row stays zero."""
    x = _matrix(x)
    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    return x / np.maximum(norms, NORM_EPS)[:, None]


def mean_pairwise_cosine(x: np.ndarray) -> float | None:
    """Mean of u_i . u_j over ordered pairs i != j, without the n x n matrix; None below two rows."""
    u = unit_rows(x)
    n = int(u.shape[0])
    if n < 2:
        return None
    total = u.sum(axis=0)
    value = (float(total @ total) - float(np.einsum("ij,ij->", u, u))) / (n * (n - 1))
    return min(1.0, max(-1.0, value))


def mean_average_distance(x: np.ndarray) -> float | None:
    """1 - mean_pairwise_cosine: global MAD (Chen et al., AAAI 2020) without neighbourhood masks."""
    cos = mean_pairwise_cosine(x)
    return None if cos is None else 1.0 - cos


def effective_rank(x: np.ndarray) -> float:
    """exp(entropy of s / sum(s)) over the singular values s above the matrix_rank tolerance (Roy and Vetterli 2007)."""
    x = _matrix(x)
    if x.size == 0:
        return 0.0
    s = np.linalg.svd(x, compute_uv=False)
    tolerance = float(s.max()) * max(x.shape) * float(np.finfo(np.float64).eps)
    s = s[s > tolerance]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def zero_rows(x: np.ndarray) -> int:
    x = _matrix(x, dtype=None)
    return int(x.shape[0] - np.count_nonzero(np.any(x != 0, axis=1)))


def block_separation(summaries: np.ndarray) -> float | None:
    """Mean over ordered pairs of blocks of 1 - cos(summary_a, summary_b)."""
    return mean_average_distance(summaries)


def node_metrics(rows: np.ndarray) -> dict[str, Any]:
    cos = mean_pairwise_cosine(rows)
    return {"mean_pairwise_cosine": cos, "mean_average_distance": None if cos is None else 1.0 - cos,
            "effective_rank": effective_rank(rows), "zero_rows": zero_rows(rows)}


def block_metrics(summaries: np.ndarray) -> dict[str, Any]:
    return {"block_effective_rank": effective_rank(summaries), "block_separation": block_separation(summaries)}


def block_summary(rep: np.ndarray) -> np.ndarray:
    """float64 mean over the rows of a block's representation; zero for an empty block."""
    rep = _matrix(rep, dtype=None)
    if rep.shape[0] == 0:
        return np.zeros(rep.shape[1], dtype=np.float64)
    return rep.mean(axis=0, dtype=np.float64)


# ----------------------------------------------------------------------------
# Sample
# ----------------------------------------------------------------------------
def choose_blocks(num_blocks: int, max_blocks: int, *, dataset: str, regime: str, seed: int) -> list[int]:
    """Every block when there are at most max_blocks, else a seeded draw without replacement, sorted."""
    num_blocks, max_blocks = int(num_blocks), int(max_blocks)
    if num_blocks <= max_blocks:
        return list(range(num_blocks))
    rng = np.random.default_rng(protocol.derive_seed(seed, dataset, regime, "blocks"))
    return sorted(int(b) for b in rng.choice(num_blocks, size=max_blocks, replace=False))


def node_quotas(sizes: Sequence[int], max_nodes: int) -> list[int]:
    """Equal stratification: min(size, q) for the largest q that fits, the remainder to the first open blocks."""
    sizes = [int(s) for s in sizes]
    max_nodes = int(max_nodes)
    if sum(sizes) <= max_nodes:
        return sizes
    lo, hi = 0, max(sizes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(s, mid) for s in sizes) <= max_nodes:
            lo = mid
        else:
            hi = mid - 1
    quotas = [min(s, lo) for s in sizes]
    left = max_nodes - sum(quotas)
    for i, size in enumerate(sizes):
        if left == 0:
            break
        if size > lo:
            quotas[i] += 1
            left -= 1
    return quotas


def sample_positions(size: int, quota: int, *, dataset: str, regime: str, seed: int, block: int) -> np.ndarray:
    """Sorted positions of the sampled nodes in the block's (sorted) node array."""
    size, quota = int(size), int(quota)
    if quota >= size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(protocol.derive_seed(seed, dataset, regime, "nodes", int(block)))
    return np.sort(rng.choice(size, size=quota, replace=False)).astype(np.int64)


@dataclass
class Sample:
    """The analysed blocks and nodes; the same for every model of a store, regime and settings."""

    blocks: list[int]
    block_sizes: list[int]
    positions: list[np.ndarray]  # per block: sorted positions in part.nodes_of_blocks([block])
    nodes: np.ndarray  # sampled global node ids, block by block
    max_blocks: int
    max_nodes: int
    seed: int

    def record(self) -> dict[str, Any]:
        digest = hashlib.sha256(np.ascontiguousarray(self.nodes, dtype="<i8").tobytes()).hexdigest()
        return {"blocks": list(self.blocks), "nodes_per_block": [int(p.shape[0]) for p in self.positions],
                "block_sizes": list(self.block_sizes), "max_blocks": int(self.max_blocks),
                "max_nodes": int(self.max_nodes), "seed": int(self.seed), "nodes_sha256": digest}


def draw_sample(store: GraphStore, regime: str, *, max_blocks: int, max_nodes: int, seed: int) -> Sample:
    if int(max_blocks) < 2 or int(max_nodes) < 2:
        raise ValueError("max_blocks and max_nodes must be at least 2 (the metrics are pairwise)")
    part = store.part
    blocks = choose_blocks(part.num_blocks, max_blocks, dataset=store.dataset, regime=regime, seed=seed)
    ptr = np.asarray(part.block_ptr, dtype=np.int64)
    sizes = [int(ptr[b + 1] - ptr[b]) for b in blocks]
    positions, nodes = [], []
    for block, size, quota in zip(blocks, sizes, node_quotas(sizes, max_nodes)):
        pos = sample_positions(size, quota, dataset=store.dataset, regime=regime, seed=seed, block=block)
        positions.append(pos)
        nodes.append(np.asarray(part.nodes_of_blocks([block]), dtype=np.int64)[pos])
    all_nodes = np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int64)
    return Sample(blocks, sizes, positions, all_nodes, int(max_blocks), int(max_nodes), int(seed))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
@dataclass
class LoadedModel:
    """An exported encoder, loaded strictly and cross-checked against its numpy copy (load_model)."""

    model_id: str
    model_dir: Path
    model: Any  # encoders.GraphEncoder in eval mode
    config: dict[str, Any]  # GraphEncoder.config
    identity: dict[str, Any]  # model_id, dataset, regime (trained), arch, seed, arm
    encoder: encoder_numpy.NumpyEncoder  # the serving copy, for serving.check_model
    torch_state_sha256: str

    def fields(self) -> dict[str, Any]:
        cfg = self.config
        return {"arch": str(cfg["arch"]), "layers": int(cfg["layers"]), "hidden": int(cfg["hidden"]),
                "residual": bool(cfg["residual"]), "layer_norm": bool(cfg["layer_norm"]),
                "train_seed": int(self.identity["seed"]), "arm": str(self.identity["arm"]),
                "trained_regime": str(self.identity["regime"]), "torch_state_sha256": self.torch_state_sha256}


def merged_identity(model_json: Mapping[str, Any],
                    export: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """IDENTITY_FIELDS from model.json "extra" and export.json; problems when they disagree or are missing."""
    extra = dict(model_json.get("extra") or {})
    identity: dict[str, Any] = {}
    problems: list[str] = []
    for name in IDENTITY_FIELDS:
        found = {}
        if extra.get(name) is not None:
            found["model.json"] = extra[name]
        if export is not None and export.get(name) is not None:
            found[EXPORT_NAME] = export[name]
        if not found:
            problems.append(f"neither model.json nor {EXPORT_NAME} records the {name}")
        elif len({str(v) for v in found.values()}) > 1:
            problems.append(f"the {name} differs: " + ", ".join(f"{k} {v!r}" for k, v in found.items()))
        else:
            identity[name] = next(iter(found.values()))
    seed = identity.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not str(seed).lstrip("-").isdigit():
            problems.append(f"the seed {seed!r} is not an integer")
            del identity["seed"]
        else:
            identity["seed"] = int(seed)
    return identity, problems


def load_model(model_dir: str | os.PathLike) -> LoadedModel:
    """The encoder of an export_model directory in eval mode; ValueError when any cross-check fails."""
    import torch

    from . import encoders

    model_dir = Path(model_dir)
    torch_path = model_dir / TORCH_STATE
    for path in (torch_path, model_dir / encoder_numpy.MODEL_NAME):
        if not path.is_file():
            raise ValueError(f"{model_dir}: no {path.name}; not an export_model directory")
    torch_sha = manifest.sha256_file(torch_path)["sha256"]
    payload = torch.load(torch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"config", "state_dict"} <= set(payload):
        raise ValueError(f"{torch_path}: expected a dict with 'config' and 'state_dict'")
    config = dict(payload["config"])
    model = encoders.build_encoder(config)
    if protocol.canonical_json(dict(model.config)) != protocol.canonical_json(config):
        raise ValueError(f"{torch_path}: build_encoder does not reproduce the saved config {config}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    np_config, np_state, model_json = encoder_numpy.load_numpy_model(model_dir)
    problems = []
    if encoder_numpy.normalize_config(np_config) != encoder_numpy.normalize_config(model.config):
        problems.append(f"model.json config {np_config} differs from {TORCH_STATE}")
    elif encoder_numpy.state_sha256(encoders.export_state(model)) != encoder_numpy.state_sha256(np_state):
        problems.append(f"{encoder_numpy.WEIGHTS_NAME} holds other weights than {TORCH_STATE}")
    export = None
    export_path = model_dir / EXPORT_NAME
    if export_path.is_file():
        export = json.loads(export_path.read_text(encoding="utf-8"))
        if export.get("torch_state_sha256") != torch_sha:
            problems.append(f"{EXPORT_NAME} records torch_state_sha256 {export.get('torch_state_sha256')}, "
                            f"{TORCH_STATE} has {torch_sha}")
        if "config" in export and protocol.canonical_json(dict(export["config"])) != \
                protocol.canonical_json(dict(model.config)):
            problems.append(f"{EXPORT_NAME} records another config than {TORCH_STATE}")
    identity, missing = merged_identity(model_json, export)
    problems += missing
    if "arch" in identity and identity["arch"] != model.config["arch"]:
        problems.append(f"the metadata names arch {identity['arch']}, the weights are {model.config['arch']}")
    if not missing:
        expected = serving.model_id_for(identity["dataset"], identity["regime"], identity["arch"], identity["seed"],
                                        identity["arm"])
        if identity["model_id"] != expected:
            problems.append(f"model id {identity['model_id']} does not match its identity fields ({expected})")
        if model_dir.name != identity["model_id"]:
            problems.append(f"the directory {model_dir.name} holds model {identity['model_id']}")
    if problems:
        raise ValueError(f"{model_dir}: " + "; ".join(problems))
    view = encoder_numpy.NumpyEncoder(np_config, np_state)
    view.model_json = {**model_json, "extra": {**dict(model_json.get("extra") or {}), **identity}}
    return LoadedModel(str(identity["model_id"]), model_dir, model, dict(model.config), identity, view, torch_sha)


def check_serving(store: GraphStore, regime: str, loaded: LoadedModel) -> np.ndarray | None:
    """The relation table the model reads (rgcn), else None; ValueError when it cannot serve `regime` here."""
    if regime not in protocol.TRAINED_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; known: {', '.join(protocol.TRAINED_REGIMES)}")
    try:
        return serving.check_model(store, regime, loaded.encoder)
    except ValueError as exc:
        raise ValueError(f"{loaded.model_id}: {exc}") from None


@contextmanager
def single_thread() -> Iterator[None]:
    """One torch thread and one BLAS thread inside the block; the previous limits come back after it."""
    import torch

    previous = torch.get_num_threads()
    limiter = None
    try:
        torch.set_num_threads(THREADS)
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # the torch limit still applies; numpy then runs with its default pool
            threadpool_limits = None
        if threadpool_limits is not None:
            limiter = threadpool_limits(limits=THREADS)
        yield
    finally:
        if limiter is not None:
            limiter.restore_original_limits()
        torch.set_num_threads(previous)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyse(store: GraphStore, regime: str, loaded: LoadedModel, sample: Sample, rel_table: np.ndarray | None, *,
            knockout: bool = False) -> tuple[dict[str, Any], np.ndarray]:
    """The record of one model and variant, and z [len(sample.blocks), output] float32 of the analysed blocks."""
    import torch
    import torch.nn.functional as F

    from . import encoders

    cfg = loaded.config
    if knockout and not cfg["residual"]:
        raise ValueError(f"{loaded.model_id} was trained without the residual; its knockout would equal it")
    names = layer_names(cfg["layers"])
    dims = [int(cfg["in_dim"])] + [int(cfg["hidden"])] * (int(cfg["layers"]) + 1) + [int(cfg["output"])]
    with_nodes = len(names) - 1  # every layer but readout
    rows: list[list[np.ndarray]] = [[] for _ in range(with_nodes)]
    summaries: list[list[np.ndarray]] = [[] for _ in names]
    zs: list[np.ndarray] = []
    model = loaded.model
    t0 = time.perf_counter()
    model.eval()
    model.residual = bool(cfg["residual"]) and not knockout
    try:
        with torch.no_grad():
            for block, positions in zip(sample.blocks, sample.positions):
                graph = store.induced(store.part.nodes_of_blocks([block]))
                batch = encoders.collate([features.store_graph_inputs(store, graph, regime, rel_table)])
                z, _, hidden = model(batch.x, batch.src, batch.dst, batch.batch, batch.num_graphs, batch.rel,
                                     return_layers=True)
                h0 = F.relu(model.input_proj(batch.x))
                reps = [rep.detach().cpu().numpy() for rep in (batch.x, h0, *hidden)] + \
                       [z.detach().cpu().numpy()]
                for i, array in enumerate(reps):
                    if array.ndim != 2 or array.shape[1] != dims[i]:
                        raise ValueError(f"{loaded.model_id}: layer {names[i]} has shape {array.shape}, "
                                         f"expected width {dims[i]}")
                    if not np.isfinite(array).all():
                        raise ValueError(f"{loaded.model_id}: non-finite {names[i]} representations in block {block}")
                    if i < with_nodes:
                        rows[i].append(array[positions].astype(np.float32, copy=False))
                        summaries[i].append(block_summary(array))
                    else:
                        summaries[i].append(array[0].astype(np.float64))
                        zs.append(array[0].astype(np.float32))
    finally:
        model.residual = bool(cfg["residual"])
    layers = []
    for i, name in enumerate(names):
        entry: dict[str, Any] = {"layer": name, "dim": dims[i]}
        stacked = np.stack(summaries[i]) if summaries[i] else np.zeros((0, dims[i]))
        entry.update(block_metrics(stacked))
        if i < with_nodes:
            node = np.concatenate(rows[i], axis=0) if rows[i] else np.zeros((0, dims[i]), dtype=np.float32)
            entry.update(node_metrics(node))
        else:
            entry.update({metric: None for metric in NODE_METRICS})
        layers.append(entry)
    record = {
        "schema": SCHEMA,
        "model_id": loaded.model_id,
        "model": loaded.fields(),
        "dataset": store.dataset,
        "regime": regime,
        "partition": store.partition_name,
        "knockout_residual": bool(knockout),
        "sample": sample.record(),
        "layers": layers,
        "definitions": dict(DEFINITIONS),
        "protocol_hash": protocol.protocol_hash(),
        "seconds": time.perf_counter() - t0,
    }
    z_all = np.stack(zs) if zs else np.zeros((0, int(cfg["output"])), dtype=np.float32)
    return record, z_all


def analyse_model(store: GraphStore, regime: str, model_dir: str | os.PathLike, *, max_blocks: int | None = None,
                  max_nodes: int | None = None,
                  knockout_residual: bool = False) -> list[tuple[dict[str, Any], np.ndarray]]:
    """(record, z) of one export directory: the trained variant, then the knockout when asked and residual."""
    settings = sample_settings()
    loaded = load_model(model_dir)
    table = check_serving(store, regime, loaded)
    sample = draw_sample(store, regime, max_blocks=settings["max_blocks"] if max_blocks is None else max_blocks,
                         max_nodes=settings["max_nodes"] if max_nodes is None else max_nodes, seed=settings["seed"])
    out = []
    with single_thread():
        out.append(analyse(store, regime, loaded, sample, table))
        if knockout_residual and loaded.config["residual"]:
            out.append(analyse(store, regime, loaded, sample, table, knockout=True))
    return out


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def write_json(path: Path, payload: Any) -> None:
    """Atomic, sorted, indent 2, LF line ends, no NaN: equal payloads give equal bytes on every OS."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _check_out(out: Path, names: Sequence[str]) -> None:
    if not out.exists():
        return
    if not out.is_dir():
        raise ValueError(f"{out} exists and is not a directory")
    allowed = {manifest.MANIFEST_NAME, *names}
    stray = sorted(str(p.relative_to(out)).replace(os.sep, "/") for p in out.rglob("*") if p.is_file())
    stray = [name for name in stray if name not in allowed]
    if stray:
        raise ValueError(f"{out} holds files this run would not write: {stray[:10]}; use an empty directory")


def run(*, store_root: str | os.PathLike, dataset: str, regime: str, models_root: str | os.PathLike,
        models: Sequence[str], out: str | os.PathLike, partition: str = PRIMARY, max_blocks: int | None = None,
        max_nodes: int | None = None, knockout_residual: bool = False,
        argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Analyse every model into OUT with a manifest. Everything is checked before OUT is created."""
    settings = sample_settings()
    max_blocks = settings["max_blocks"] if max_blocks is None else int(max_blocks)
    max_nodes = settings["max_nodes"] if max_nodes is None else int(max_nodes)
    models = [str(m) for m in models]
    if not models:
        raise ValueError("no models to analyse")
    repeated = sorted({m for m in models if models.count(m) > 1})
    if repeated:
        raise ValueError(f"models named more than once: {repeated}")
    store = GraphStore(store_root, dataset, partition=partition, mmap=True)
    loaded = [load_model(Path(models_root) / model_id) for model_id in models]
    tables = [check_serving(store, regime, lm) for lm in loaded]
    sample = draw_sample(store, regime, max_blocks=max_blocks, max_nodes=max_nodes, seed=settings["seed"])
    plan: list[tuple[LoadedModel, np.ndarray | None, bool]] = []
    skipped = []
    for lm, table in zip(loaded, tables):
        plan.append((lm, table, False))
        if knockout_residual:
            if lm.config["residual"]:
                plan.append((lm, table, True))
            else:
                skipped.append(lm.model_id)
    names = [output_name(lm.model_id, knockout) for lm, _, knockout in plan]
    out = Path(out)
    _check_out(out, names)
    for model_id in skipped:
        print(f"no knockout for {model_id}: trained without the residual", file=sys.stderr)
    config = {"dataset": dataset, "regime": regime, "partition": store.partition_name, "models": models,
              "models_root": str(models_root), "store_root": str(store_root), "max_blocks": max_blocks,
              "max_nodes": max_nodes, "sample_seed": settings["seed"], "knockout_residual": bool(knockout_residual),
              "threads": THREADS, "metrics": metric_names(), "schema": SCHEMA}
    inputs: dict[str, Any] = {"store_meta": store.dir / "meta.json",
                              "partition_json": store.part.dir / "partition.json"}
    sigma_dir = store.dir / "sigma" / regime
    if regime != "R0" and sigma_dir.is_dir():
        inputs[f"sigma_{regime}"] = sigma_dir
    for lm in loaded:
        for name in (TORCH_STATE, encoder_numpy.MODEL_NAME, encoder_numpy.WEIGHTS_NAME, EXPORT_NAME):
            if (lm.model_dir / name).is_file():
                inputs[f"{lm.model_id}/{name}"] = lm.model_dir / name
    seconds: dict[str, float] = {}
    with manifest.RunManifest(out, experiment=EXPERIMENT, config=config, inputs=inputs, split=SPLIT,
                              fast_input_hashes=True, argv=argv) as run_manifest:
        with single_thread():
            for (lm, table, knockout), name in zip(plan, names):
                record, _ = analyse(store, regime, lm, sample, table, knockout=knockout)
                write_json(out / name, record)
                seconds[name] = record["seconds"]
        run_manifest.note(outputs=names, knockout_skipped=skipped, sample=sample.record(), seconds=seconds)
    return {"out": str(out), "outputs": names, "knockout_skipped": skipped, "sample": sample.record()}


# ----------------------------------------------------------------------------
# Summarize
# ----------------------------------------------------------------------------
def validate_record(record: Any) -> None:
    """ValueError unless `record` is a complete jigsaw-layerwise-v1 record whose identity fields agree."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        found = record.get("schema") if isinstance(record, dict) else type(record).__name__
        raise ValueError(f"not a {SCHEMA} record (schema {found!r})")
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    missing = [k for k in RECORD_KEYS if k not in record] + [f"model.{k}" for k in MODEL_KEYS if k not in model]
    if missing:
        raise ValueError(f"record {record.get('model_id')} lacks {missing}")
    expected = serving.model_id_for(record["dataset"], model["trained_regime"], model["arch"], model["train_seed"],
                                    model["arm"])
    if record["model_id"] != expected:
        raise ValueError(f"record {record['model_id']} does not match its identity fields ({expected})")
    if protocol.TRAINED_REGIMES.get(record["regime"]) != model["trained_regime"]:
        raise ValueError(f"record {record['model_id']}: regime {record['regime']} is not served by a "
                         f"{model['trained_regime']} encoder")
    found = [entry.get("layer") for entry in record["layers"]]
    if found != layer_names(model["layers"]):
        raise ValueError(f"record {record['model_id']}: layers {found}, expected {layer_names(model['layers'])}")


def read_run(run_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """The records of a run directory whose manifest verifies (complete, unchanged, nothing unrecorded)."""
    run_dir = Path(run_dir)
    path = run_dir / manifest.MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"{run_dir}: no {manifest.MANIFEST_NAME}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("experiment") != EXPERIMENT:
        raise ValueError(f"{run_dir}: experiment {info.get('experiment')!r}, expected {EXPERIMENT}")
    problems = manifest.verify_manifest(run_dir)
    if problems:
        raise ValueError(f"{run_dir}: the run does not verify: {problems}")
    records = []
    for rel in sorted(info.get("outputs") or {}):
        record = json.loads((run_dir / rel).read_text(encoding="utf-8"))
        try:
            validate_record(record)
        except ValueError as exc:
            raise ValueError(f"{run_dir / rel}: {exc}") from None
        records.append(record)
    if not records:
        raise ValueError(f"{run_dir}: no layerwise records")
    return records


def _mean_sd(values: Sequence[Any], what: str) -> tuple[float | None, float | None]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None, None
    if len(present) != len(values):
        raise ValueError(f"{what} is null for some seeds only")
    n = len(present)
    mean = math.fsum(present) / n
    if n < 2:
        return mean, None
    return mean, math.sqrt(math.fsum((v - mean) ** 2 for v in present) / (n - 1))


def _same(records: Sequence[Mapping[str, Any]], values: Sequence[Any], what: str, label: str) -> Any:
    """The one value the records share (values[i] belongs to records[i]); ValueError when they differ."""
    if len({protocol.canonical_json(v) for v in values}) > 1:
        ids = ", ".join(str(r["model_id"]) for r in records)
        raise ValueError(f"{label}: the records differ in {what} ({ids}); analyse every seed with the same store, "
                         "settings and protocol")
    return values[0]


def summary_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dataset, regime, arch, layers, residual, knockout, arm) and layer, over training seeds."""
    for record in records:
        validate_record(record)
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for record in records:
        m = record["model"]
        key = (str(record["dataset"]), str(record["regime"]), str(m["arch"]), int(m["layers"]), bool(m["residual"]),
               bool(record["knockout_residual"]), str(m["arm"]))
        groups.setdefault(key, []).append(record)
    protocol_seeds = sorted(int(s) for s in protocol.SEEDS["training"])
    metrics = metric_names()
    rows = []
    for key in sorted(groups):
        dataset, regime, arch, layers, residual, knockout, arm = key
        label = (f"{dataset} {regime} {arch} L{layers} {'res' if residual else 'nores'}"
                 f"{' knockout' if knockout else ''} {arm}")
        recs = sorted(groups[key], key=lambda r: int(r["model"]["train_seed"]))
        seeds = [int(r["model"]["train_seed"]) for r in recs]
        repeated = sorted({s for s in seeds if seeds.count(s) > 1})
        if repeated:
            ids = sorted(str(r["model_id"]) for r in recs if int(r["model"]["train_seed"]) in repeated)
            raise ValueError(f"{label}: training seed(s) {repeated} appear more than once ({ids})")
        hidden = _same(recs, [r["model"]["hidden"] for r in recs], "hidden", label)
        layer_norm = _same(recs, [r["model"]["layer_norm"] for r in recs], "layer_norm", label)
        partition = _same(recs, [r["partition"] for r in recs], "partition", label)
        trained = _same(recs, [r["model"]["trained_regime"] for r in recs], "trained regime", label)
        sample = _same(recs, [r["sample"] for r in recs], "sample", label)
        _same(recs, [r["protocol_hash"] for r in recs], "protocol hash", label)
        for index, name in enumerate(layer_names(layers)):
            dim = _same(recs, [r["layers"][index]["dim"] for r in recs], f"the width of layer {name}", label)
            row: dict[str, Any] = {
                "dataset": dataset, "regime": regime, "partition": partition, "arch": arch, "layers": layers,
                "hidden": hidden, "layer_norm": layer_norm, "residual": residual, "knockout": knockout, "arm": arm,
                "trained_regime": trained, "layer": name, "dim": dim, "n_seeds": len(seeds),
                "seeds": ";".join(str(s) for s in seeds), "seeds_complete": sorted(seeds) == protocol_seeds,
                "sample_blocks": len(sample["blocks"]), "sample_nodes": int(sum(sample["nodes_per_block"])),
            }
            for metric in metrics:
                values = [r["layers"][index].get(metric) for r in recs]
                row[f"{metric}_mean"], row[f"{metric}_sd"] = _mean_sd(values, f"{label} layer {name} {metric}")
            rows.append(row)
    return rows


def csv_columns() -> list[str]:
    return [*BASE_COLUMNS, *[f"{metric}_{stat}" for metric in metric_names() for stat in ("mean", "sd")]]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_csv(path: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = csv_columns()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in columns])
    os.replace(tmp, path)


def summarize(inputs: Sequence[str | os.PathLike], out: str | os.PathLike) -> list[dict[str, Any]]:
    """Aggregate the records of verified run directories over training seeds into one CSV."""
    if not inputs:
        raise ValueError("no run directories to summarize")
    records = [record for run_dir in inputs for record in read_run(run_dir)]
    rows = summary_rows(records)
    write_csv(out, rows)
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = sample_settings()
    p = sub.add_parser("run", help="analyse exported encoders layer by layer into OUT")
    p.add_argument("--store-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default=PRIMARY)
    p.add_argument("--regime", required=True, help="an input regime the models were trained for (TRAINED_REGIMES)")
    p.add_argument("--models-root", required=True)
    p.add_argument("--models", nargs="+", required=True, help="model ids, the directory names under --models-root")
    p.add_argument("--out", required=True)
    p.add_argument("--max-blocks", type=int, default=defaults["max_blocks"])
    p.add_argument("--max-nodes", type=int, default=defaults["max_nodes"])
    p.add_argument("--knockout-residual", action="store_true",
                   help="also evaluate residual models with the residual removed at inference (legacy check)")
    s = sub.add_parser("summarize", help="mean and SD over training seeds of run directories, as CSV")
    s.add_argument("--inputs", nargs="+", required=True, help="OUT directories of `run`")
    s.add_argument("--out", required=True, help="CSV path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        rows = summarize(args.inputs, args.out)
        print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2, sort_keys=True))
        return 0
    result = run(store_root=args.store_root, dataset=args.dataset, regime=args.regime, models_root=args.models_root,
                 models=args.models, out=args.out, partition=args.partition, max_blocks=args.max_blocks,
                 max_nodes=args.max_nodes, knockout_residual=args.knockout_residual,
                 argv=None if argv is None else list(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
