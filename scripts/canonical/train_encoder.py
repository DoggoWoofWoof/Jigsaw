"""Canonical encoder trainer (protocol.TRAINING["trainer"]).

    python -m scripts.canonical.train_encoder --dataset arxiv --regime R0 --seed 7301 \\
        --store-root STORE --val-queries QUERIES/arxiv/validation/31415/R0.npz \\
        --run-dir RUNS/train/arxiv-R0-graphsage-seed7301-main --models-root MODELS

One run trains one encoder: a dataset, a trained regime, a training seed and an
arm (FACTORIAL or ARCHITECTURE_ABLATION overrides; "main" has none). The R2-*
regimes share the R2-512 encoder (protocol.TRAINED_REGIMES).

Objective (TRAINING["loss"]). A step encodes a batch of planted queries with the
live encoder (zq). The block keys are a bank of every block's embedding from the
momentum (key) encoder, refreshed every bank_refresh_steps steps, in which the
rows of the step's live set are replaced by live-encoder embeddings that carry
gradients. With logits l_b = <zq, key_b> / T and P the query's coverage targets,
the greedy one-hop cover (queries.greedy_cover):

  block_infonce  mean over p in P of -log(e^l_p / (e^l_p + sum over n not in P of e^l_n)),
                 T = temperature: every positive against every non-target block
  fine_nce       the same with the step's live fine parts as keys (in-batch
                 negatives); P = the fine parts that hold planted nodes
  node_nce       InfoNCE between each query node and the same data node inside
                 the live fine parts (legacy node alignment); weight beta
  coverage_loss  the legacy partition_coverage_loss (scripts/coverage_losses.py) at
                 T = coverage_temperature, over all blocks:
                   agg_P[ -log_softmax(l)_p + w_m softplus(max_{n not in P} l_n - l_p) ]
                   + w_b agg_P[ softplus(t + barrier_margin - l_p) ]
                 agg is "cvar" (mean of the ceil(rho |P|) largest terms of a query)
                 or "mean", then the mean over queries; t is the (k - |P| + 1)-th
                 largest non-target logit, k = K if |P| <= K, else |P| rounded up
                 to a multiple of topk_barrier_bucket
  total = alpha fine_nce + (1 - alpha - beta) block_infonce + beta node_nce + gamma coverage_loss

The live set holds live_positives blocks: the batch's targets by the number of
queries they serve, then block id (legacy), and the hardest negatives by bank
score in the remaining slots. After every optimizer step the key encoder follows
the live encoder by EMA (negative_encoder_momentum); momentum 0 makes it a copy,
the legacy no-momentum setting. Queries always use the live encoder.

Sampler. Draw i of step s in epoch e (epochs count from 1, steps and draws from
0) uses default_rng(derive_seed(training seed, dataset, regime, "train", e, s, i)):
a family by TRAINING["sampler"] probabilities, a base size, a jitter in
[-size_jitter, size_jitter], then QueryContext.sample. Failed draws are skipped
and counted, and the batch fills from the next draws. A draw whose node set is
held out (a query of any validation master: the --val-queries directory and
every --held-out directory) is skipped and counted the same way, so no
validation query is ever trained on. The node-set hash of every query drawn for
a step (a skipped step included) is appended to train_nodesets.txt, which the
sealed splits exclude when they are generated.

Schedule. AdamW (weight_decay from the protocol), linear warmup, then per-step
cosine decay to lr_final; gradient norm clipped at grad_clip_norm. Every validation_every_epochs
epochs and at the end, the live encoder (query and block side) is evaluated on
the checkpoint-selection validation queries (master 31415, positives): FullCov@K
of the hybrid policy for K in the ladder without its final step, averaged over K
and queries. Each evaluated state is saved as ckpt_epoch{E:03d}.pt and never
overwritten; the maximum metric, ties to the later epoch, is exported to
<models-root>/<model_id>/.

Run directory: manifest.json, ckpt_epoch*.pt, val_metrics.jsonl, train_log.jsonl,
train_nodesets.txt, selection.json, export.json. Model directory: block_emb.npy,
fine_emb.npy, torch_state.pt, weights.npz and model.json (encoder_numpy),
export.json. On CUDA, scatter-add kernels make runs close but not bitwise
repeatable; on CPU a run and its resumption are repeatable.

Each train_log.jsonl line is one epoch: its seconds, its end (utc), the CPU
seconds of this process and the mean busy CPUs of the whole machine
(host_busy_cpus). On a shared machine the difference between host_busy_cpus
and cpu_seconds / seconds is the load of other jobs, so epochs timed beside
other work can be told apart from quiet ones. Epoch lines and validation
records also carry the GPU peaks since the previous one
(gpu_peak_allocated_bytes, gpu_peak_reserved_bytes; None on the CPU) and the
process's peak memory so far (host_peak_bytes, which a resume starts again).
On CUDA the allocator's cache is held to the card's memory (cap_cuda_cache,
recorded as cuda_cache_cap in the run manifest).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import encoders, features, manifest, protocol, queries, rankers
from .store import GraphStore, csr_from_edges

MAIN_ARM = "main"
CHECKPOINT_RE = re.compile(r"ckpt_epoch(\d{3,})\.pt")
VAL_METRICS = "val_metrics.jsonl"
TRAIN_LOG = "train_log.jsonl"
TRAIN_NODESETS = "train_nodesets.txt"
SELECTION = "selection.json"
EXPORT = "export.json"
SELECTION_MASTER = next(m for m, role in protocol.SEEDS["validation_roles"].items() if role == "checkpoint_selection")

# Settings the protocol leaves open. Each is read from TRAINING first, so a
# protocol entry of the same name replaces the default.
LOSS_DEFAULTS: dict[str, Any] = {
    "coverage_temperature": None,  # None: TRAINING["loss"]["temperature"] (the protocol sets 0.05, the legacy value)
    "topk_barrier_bucket": 10,  # legacy coverage_topk_bucket_size
    "topk_barrier_margin": 0.0,  # legacy coverage_topk_margin
    "smoothmax_temperature": 0.1,  # legacy; used by aggregation "smoothmax" only
    "bank_refresh_steps": 10,  # legacy cache_refresh_steps
    "fine_positives_per_query": 8,  # live fine parts per query for fine_nce and node_nce
}
SCHEDULE_DEFAULTS: dict[str, Any] = {
    "optimizer": "adamw",
    "weight_decay": 0.0,  # the legacy trainer used Adam, which is AdamW without decay
    "grad_clip_norm": 1.0,  # legacy clip_grad_norm_(max_norm=1.0)
    "encode_batch_size": 32,  # graphs per forward pass for the bank, validation and export
}
MODEL_DEFAULTS: dict[str, Any] = {"layers": 6, "hidden": 256, "output": 128, "residual": True, "layer_norm": True,
                                  "dropout": 0.1}
SMOKE_KEYS = ("epochs", "steps_per_epoch", "batch_size", "validation_every_epochs", "layers", "hidden", "output")
FACTOR_LABELS = {"coverage_aggregation": "", "topk_barrier_weight": "barrier", "live_positives": "live"}


# ----------------------------------------------------------------------------
# Settings, arms and plans
# ----------------------------------------------------------------------------
def trained_regime(regime: str) -> str:
    """The regime whose encoder serves `regime` (every R2-* uses R2-512)."""
    if regime not in protocol.TRAINED_REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    return protocol.TRAINED_REGIMES[regime]


def trained_regimes_for(dataset: str) -> list[str]:
    """Regimes that get their own encoder on a dataset, in protocol order."""
    return list(dict.fromkeys(trained_regime(r) for r in protocol.regimes_for(dataset)))


def _per_dataset(key: str, value: Any, dataset: str) -> Any:
    if isinstance(value, Mapping):
        if dataset not in value:
            raise KeyError(f"TRAINING['loss'][{key!r}] has no entry for {dataset!r}")
        return value[dataset]
    return value


def loss_config(dataset: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Effective TRAINING["loss"] for a dataset: per-dataset entries resolved, overrides applied."""
    cfg = {key: _per_dataset(key, value, dataset) for key, value in protocol.TRAINING["loss"].items()}
    for key, default in LOSS_DEFAULTS.items():
        cfg.setdefault(key, default)
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f"unknown loss setting {key!r}")
        cfg[key] = _per_dataset(key, value, dataset)
    if cfg["coverage_temperature"] is None:
        cfg["coverage_temperature"] = cfg["temperature"]
    if not str(cfg["coverage_targets"]).startswith("overlap"):
        raise ValueError("the trainer implements overlap (greedy one-hop cover) coverage targets only")
    if cfg["coverage_aggregation"] not in ("mean", "cvar", "smoothmax"):
        raise ValueError(f"unknown coverage aggregation {cfg['coverage_aggregation']!r}")
    if not 0.0 <= float(cfg["negative_encoder_momentum"]) < 1.0:
        raise ValueError("negative_encoder_momentum must be in [0, 1)")
    return cfg


def model_settings(arch: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Encoder size: TRAINING["architectures"][arch], or its graphsage entry for gin and gat."""
    if arch not in encoders.ARCHS:
        raise ValueError(f"unknown architecture {arch!r}")
    archs = protocol.TRAINING["architectures"]
    spec = archs.get(arch, archs["graphsage"])
    out = {key: spec.get(key, default) for key, default in MODEL_DEFAULTS.items()}
    for key, value in (overrides or {}).items():
        if key not in out:
            raise KeyError(f"unknown encoder setting {key!r}")
        out[key] = value
    return out


def schedule_settings() -> dict[str, Any]:
    out = dict(protocol.TRAINING["schedule"])
    for key, default in SCHEDULE_DEFAULTS.items():
        out.setdefault(key, default)
    out["validation_every_epochs"] = int(protocol.TRAINING["validation_every_epochs"])
    if out.get("lr_schedule") != "cosine":
        raise ValueError(f"unsupported lr schedule {out.get('lr_schedule')!r}")
    if out.get("optimizer") != "adamw":
        raise ValueError(f"unsupported optimizer {out.get('optimizer')!r}")
    return out


def sampler_settings() -> dict[str, Any]:
    spec = copy.deepcopy(protocol.TRAINING["sampler"])
    unknown = set(spec["family_probabilities"]) - set(protocol.POSITIVE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown training families {sorted(unknown)}")
    return spec


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def grid_arm_name(assignment: Mapping[str, Any]) -> str:
    """Name of a factor-grid cell, e.g. grid-cvar-barrier0.35-live64."""
    return "grid-" + "-".join(f"{FACTOR_LABELS.get(k, k)}{_format_value(v)}" for k, v in assignment.items())


def arch_arm_name(variant: Mapping[str, Any]) -> str:
    """Name of an architecture-ablation variant, e.g. arch-graphsage-L6-res."""
    residual = "res" if variant.get("residual", True) else "nores"
    return f"arch-{variant['arch']}-L{int(variant['layers'])}-{residual}"


def arms(dataset: str) -> dict[str, dict[str, Any]]:
    """Every arm of a dataset: name -> {"kind", "loss", "model", "arch"} overrides."""
    out: dict[str, dict[str, Any]] = {MAIN_ARM: {"kind": "main", "loss": {}, "model": {}, "arch": None}}

    def add(name: str, entry: dict[str, Any]) -> None:
        if name in out:
            raise ValueError(f"duplicate arm name {name!r} for {dataset}")
        out[name] = entry

    spec = protocol.FACTORIAL.get(dataset) or {}
    factors = spec.get("factors") or {}
    keys = list(factors)
    if keys:
        for combo in itertools.product(*(factors[k] for k in keys)):
            assignment = dict(zip(keys, combo))
            add(grid_arm_name(assignment), {"kind": "factorial", "loss": assignment, "model": {}, "arch": None})
    for group in ("extra_arms", "arms"):
        for name, overrides in (spec.get(group) or {}).items():
            add(name, {"kind": "factorial", "loss": dict(overrides), "model": {}, "arch": None})
    variants = protocol.ARCHITECTURE_ABLATION.get(dataset)
    if isinstance(variants, list):
        for variant in variants:
            model = {k: v for k, v in variant.items() if k != "arch"}
            add(arch_arm_name(variant), {"kind": "architecture", "loss": {}, "model": model, "arch": variant["arch"]})
    return out


def model_id_for(dataset: str, regime: str, arch: str, seed: int, arm: str = MAIN_ARM) -> str:
    return f"{dataset}-{regime}-{arch}-seed{int(seed)}-{arm}"


def resolve_plan(dataset: str, regime: str, seed: int, *, arch: str | None = None, arm: str = MAIN_ARM,
                 smoke: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything that determines a run except paths and device, after arm and smoke overrides."""
    if dataset not in protocol.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    table = arms(dataset)
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r} for {dataset}; known: {', '.join(table)}")
    spec = table[arm]
    if arch is None:
        arch = spec["arch"] or protocol.DATASETS[dataset]["encoder"]
    elif spec["arch"] is not None and arch != spec["arch"]:
        raise ValueError(f"arm {arm} trains {spec['arch']}, not {arch}")
    regime_trained = trained_regime(regime)
    if regime_trained not in protocol.regimes_for(dataset):
        raise ValueError(f"{dataset} has no {regime_trained} regime")
    smoke = {k: v for k, v in (smoke or {}).items() if v is not None}
    unknown = set(smoke) - set(SMOKE_KEYS)
    if unknown:
        raise ValueError(f"unknown smoke settings {sorted(unknown)}")
    model = model_settings(arch, spec["model"])
    schedule = schedule_settings()
    for key in SMOKE_KEYS:
        if key in smoke:
            (model if key in ("layers", "hidden", "output") else schedule)[key] = int(smoke[key])
    if int(schedule["validation_every_epochs"]) < 1 or int(schedule["epochs"]) < 1:
        raise ValueError("epochs and validation_every_epochs must be at least 1")
    return {
        "dataset": dataset,
        "regime": regime_trained,
        "requested_regime": regime,
        "seed": int(seed),
        "seed_in_protocol": int(seed) in [int(s) for s in protocol.SEEDS["training"]],
        "arch": arch,
        "arm": arm,
        "arm_kind": spec["kind"],
        "overrides": {"loss": dict(spec["loss"]), "model": dict(spec["model"]), "arch": spec["arch"]},
        "loss": loss_config(dataset, spec["loss"]),
        "model": model,
        "schedule": schedule,
        "sampler": sampler_settings(),
        "smoke": smoke,
        "model_id": model_id_for(dataset, regime_trained, arch, seed, arm),
        "init_seed": protocol.derive_seed(seed, dataset, regime_trained, "init"),
        "protocol_hash": protocol.protocol_hash(),
    }


# ----------------------------------------------------------------------------
# Objective terms
# ----------------------------------------------------------------------------
def positive_mask(ids_per_query: Sequence[Iterable[int]], num_keys: int, device: Any = None) -> torch.Tensor:
    """[B, num_keys] bool mask of each query's positive key ids (ids out of range are ignored)."""
    mask = np.zeros((len(ids_per_query), num_keys), dtype=bool)
    for row, ids in enumerate(ids_per_query):
        ids = np.asarray(list(ids), dtype=np.int64)
        mask[row, ids[(ids >= 0) & (ids < num_keys)]] = True
    return torch.from_numpy(mask).to(device) if device is not None else torch.from_numpy(mask)


def multi_positive_nce(zq: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Mean over queries with a positive of the mean over positives p of
    softplus(logsumexp_{n not positive} l_n - l_p) = -log(e^l_p / (e^l_p + sum_n e^l_n)),
    l = zq keys^T / temperature. Other positives stay out of the denominator; a
    query without negatives contributes 0."""
    logits = zq @ keys.T / temperature
    valid = mask.any(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    logits, mask = logits[valid], mask[valid]
    has_neg = (~mask).any(dim=1, keepdim=True)
    neg = logits.masked_fill(mask, float("-inf"))
    neg_lse = torch.logsumexp(torch.where(has_neg, neg, torch.zeros_like(neg)), dim=1, keepdim=True)
    per = torch.where(has_neg & mask, F.softplus(neg_lse - logits), torch.zeros_like(logits))
    return (per.sum(dim=1) / mask.sum(dim=1)).mean()


def block_infonce(zq: torch.Tensor, block_keys: torch.Tensor, positives: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Query vs block embeddings; positives are the coverage targets (multi_positive_nce).

    Generalises the legacy coarse term (scripts/modal_train_graphsage.py:1710-1736,
    called at :1788), whose single positive was one graph of the query's blocks
    and whose negatives were the batch's other positives plus 8 cached neighbour
    blocks: here every target block is a positive and every other block a negative.
    """
    return multi_positive_nce(zq, block_keys, positives, temperature)


def fine_nce(zq: torch.Tensor, fine_keys: torch.Tensor, positives: torch.Tensor,
             temperature: float = 0.1) -> torch.Tensor:
    """Query vs the step's live fine parts; positives are the fine parts holding planted nodes.

    The legacy fine term (scripts/modal_train_graphsage.py:1787) contrasted each
    query with one positive graph and the batch's other positives; here the keys
    are the batch's fine parts, so the negatives stay in-batch.
    """
    return multi_positive_nce(zq, fine_keys, positives, temperature)


def node_nce(q_nodes: torch.Tensor, q_ids: np.ndarray, p_nodes: torch.Tensor, p_ids: np.ndarray,
             temperature: float = 0.1) -> torch.Tensor:
    """Legacy node_alignment_loss (scripts/modal_train_graphsage.py:1738-1783).

    q_ids and p_ids are the data-node ids of the rows of q_nodes (query nodes,
    query node j is planted[j]) and p_nodes (nodes of the live fine parts). The
    first row of every id is kept on each side; ids present on both sides give
    cross_entropy(q p^T / temperature, diagonal). Fewer than two shared ids give 0.
    """
    q_ids = np.asarray(q_ids, dtype=np.int64)
    p_ids = np.asarray(p_ids, dtype=np.int64)
    q_unique, q_first = np.unique(q_ids, return_index=True)
    p_unique, p_first = np.unique(p_ids, return_index=True)
    common, qi, pi = np.intersect1d(q_unique, p_unique, assume_unique=True, return_indices=True)
    if common.shape[0] < 2:
        return q_nodes.sum() * 0.0
    q_sel = q_nodes[torch.as_tensor(q_first[qi], device=q_nodes.device)]
    p_sel = p_nodes[torch.as_tensor(p_first[pi], device=p_nodes.device)]
    logits = q_sel @ p_sel.T / temperature
    return F.cross_entropy(logits, torch.arange(common.shape[0], device=logits.device))


def aggregate_positive_terms(per_positive_terms: torch.Tensor, positive_mask: torch.Tensor, mode: str = "mean",
                             cvar_fraction: float = 0.25, smoothmax_temperature: float = 0.1) -> torch.Tensor:
    """Per-query aggregate of the positive entries, then the mean over queries.

    mean: the mean; cvar: the mean of the ceil(n * cvar_fraction) largest terms;
    smoothmax: T logsumexp(terms / T) - T log n. Rows without positives are
    skipped. Port of _aggregate_positive_terms (scripts/coverage_losses.py:9-39).
    """
    aggregated = []
    for row_terms, row_mask in zip(per_positive_terms, positive_mask):
        values = row_terms[row_mask]
        if values.numel() == 0:
            continue
        if mode == "mean":
            aggregated.append(values.mean())
        elif mode == "cvar":
            count = max(1, math.ceil(values.numel() * float(cvar_fraction)))
            aggregated.append(torch.topk(values, count).values.mean())
        elif mode == "smoothmax":
            temperature = max(1e-6, float(smoothmax_temperature))
            aggregated.append(temperature * torch.logsumexp(values / temperature, dim=0)
                              - temperature * math.log(values.numel()))
        else:
            raise ValueError(f"Unknown positive aggregation {mode!r}; use mean, cvar, or smoothmax")
    if not aggregated:
        return per_positive_terms.new_tensor(0.0)
    return torch.stack(aggregated).mean()


def coverage_ce_terms(logits: torch.Tensor) -> torch.Tensor:
    """-log_softmax over all keys, per entry (scripts/coverage_losses.py:86-87)."""
    return -F.log_softmax(logits, dim=1)


def hardest_negative_margin_terms(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """softplus(hardest negative logit - l), per entry; 0 in rows without a negative
    (scripts/coverage_losses.py:89-97)."""
    negative_logits = logits.masked_fill(positive_mask, float("-inf"))
    hardest_negative = negative_logits.max(dim=1).values
    has_negative = torch.isfinite(hardest_negative)
    margin = F.softplus(hardest_negative.unsqueeze(1) - logits)
    return torch.where(has_negative.unsqueeze(1), margin, torch.zeros_like(margin))


def topk_barrier_terms(logits: torch.Tensor, positive_mask: torch.Tensor, target_topk: int,
                       bucket_size: int = 10, margin: float = 0.0) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Top-K barrier softplus(t + margin - l) per entry and the positive mask of its rows.

    For a query with c positives, k = target_topk if c <= target_topk, else c
    rounded up to a multiple of bucket_size; t is the (k - c + 1)-th largest
    negative logit (the last one when there are fewer). The threshold is not
    detached. Rows without positives or negatives get no barrier; (None, None)
    when no row has one. Port of scripts/coverage_losses.py:106-131.
    """
    barrier_rows = []
    barrier_masks = []
    for row_logits, row_positive_mask in zip(logits, positive_mask):
        positive_count = int(row_positive_mask.sum().item())
        if positive_count == 0:
            continue
        size = max(1, int(bucket_size))
        effective_topk = (int(target_topk) if positive_count <= target_topk
                          else ((positive_count + size - 1) // size) * size)
        negative_row = row_logits[~row_positive_mask]
        if negative_row.numel() == 0:
            continue
        kth_negative = min(max(1, effective_topk - positive_count + 1), negative_row.numel())
        threshold = torch.topk(negative_row, kth_negative).values[-1]
        barrier_rows.append(F.softplus(threshold + margin - row_logits))
        barrier_masks.append(row_positive_mask)
    if not barrier_rows:
        return None, None
    return torch.stack(barrier_rows), torch.stack(barrier_masks)


def coverage_loss(zq: torch.Tensor, keys: torch.Tensor, positive_mask: torch.Tensor, *, temperature: float,
                  margin_weight: float = 0.25, aggregation: str = "mean", cvar_fraction: float = 0.25,
                  smoothmax_temperature: float = 0.1, topk: int = 0, topk_weight: float = 0.0,
                  topk_bucket: int = 10, topk_margin: float = 0.0) -> torch.Tensor:
    """Multi-label coverage loss aligned with FullCov@K (scripts/coverage_losses.py:42-142).

    l = zq keys^T / temperature over all keys; queries without positives are
    dropped; loss = agg(coverage_ce_terms + margin_weight * hardest_negative_margin_terms)
    + topk_weight * agg(topk_barrier_terms) with agg = aggregate_positive_terms.
    The legacy function merged live rows into the bank itself; here `keys` is the
    merged matrix (merge_live). margin_weight is the legacy constant 0.25.
    """
    logits = torch.matmul(zq, keys.T) / temperature
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return zq.new_tensor(0.0)
    logits = logits[valid]
    positive_mask = positive_mask[valid]
    loss = aggregate_positive_terms(
        coverage_ce_terms(logits) + margin_weight * hardest_negative_margin_terms(logits, positive_mask),
        positive_mask, mode=aggregation, cvar_fraction=cvar_fraction, smoothmax_temperature=smoothmax_temperature)
    if topk > 0 and topk_weight > 0.0:
        terms, masks = topk_barrier_terms(logits, positive_mask, topk, topk_bucket, topk_margin)
        if terms is not None:
            barrier = aggregate_positive_terms(terms, masks, mode=aggregation, cvar_fraction=cvar_fraction,
                                               smoothmax_temperature=smoothmax_temperature)
            loss = loss + topk_weight * barrier
    return loss


def combine_objective(terms: Mapping[str, Any], cfg: Mapping[str, Any]) -> Any:
    """alpha fine_nce + (1 - alpha - beta) block_infonce + beta node_nce + gamma coverage.

    The legacy hierarchical combination (scripts/modal_train_graphsage.py:1793)
    plus GAMMA_PARTITION times the coverage loss (:3016).
    """
    alpha = float(cfg["fine_nce_weight_alpha"])
    beta = float(cfg["node_nce_weight_beta"])
    gamma = float(cfg["coverage_weight_gamma"])
    return (alpha * terms["fine_nce"] + (1.0 - alpha - beta) * terms["block_infonce"] + beta * terms["node_nce"]
            + gamma * terms["coverage"])


# ----------------------------------------------------------------------------
# Live set, bank and momentum encoder
# ----------------------------------------------------------------------------
def select_live_blocks(positives: np.ndarray, bank_scores: np.ndarray | None, limit: int) -> np.ndarray:
    """Blocks re-encoded by the live encoder in a step.

    First the batch's targets by the number of queries they serve, then block id
    (legacy encode_live_positive_partitions, scripts/modal_train_graphsage.py:2593-2635),
    up to `limit`; free slots go to the hardest negatives: blocks that are no
    query's target, by the highest bank score any query gives them, then block id.
    """
    positives = np.asarray(positives, dtype=bool)
    if limit <= 0 or positives.ndim != 2:
        return np.zeros(0, dtype=np.int64)
    num_blocks = positives.shape[1]
    ids = np.arange(num_blocks, dtype=np.int64)
    counts = positives.sum(axis=0)
    order = np.lexsort((ids, -counts))
    chosen = [int(b) for b in order if counts[b] > 0][:limit]
    if len(chosen) < limit and bank_scores is not None:
        scores = np.where(positives, -np.inf, np.asarray(bank_scores, dtype=np.float64)).max(axis=0)
        cand = np.flatnonzero((counts == 0) & np.isfinite(scores))
        cand = cand[np.lexsort((cand, -scores[cand]))]
        chosen += [int(b) for b in cand[:limit - len(chosen)]]
    return np.asarray(chosen, dtype=np.int64)


def select_fine_parts(items: Sequence["TrainQuery"], per_query: int) -> np.ndarray:
    """The step's live fine parts: per query, the parts with the most planted nodes
    (ties to the lower id), up to per_query each, in first-seen order."""
    chosen: dict[int, None] = {}
    if per_query <= 0:
        return np.zeros(0, dtype=np.int64)
    for item in items:
        if item.fines.shape[0] == 0:
            continue
        order = np.lexsort((item.fines, -item.fine_counts))[:per_query]
        for fine in item.fines[order].tolist():
            chosen.setdefault(int(fine), None)
    return np.asarray(list(chosen), dtype=np.int64)


def merge_live(bank: torch.Tensor, live_index: torch.Tensor, live_embeddings: torch.Tensor) -> torch.Tensor:
    """The bank with the live rows replaced (scripts/coverage_losses.py:64-66)."""
    keys = bank.clone()
    keys[live_index] = live_embeddings
    return keys


@torch.no_grad()
def ema_update(key_model: torch.nn.Module, model: torch.nn.Module, momentum: float) -> None:
    """key = momentum * key + (1 - momentum) * live (scripts/modal_train_graphsage.py:2467-2474)."""
    for key_param, param in zip(key_model.parameters(), model.parameters()):
        key_param.mul_(momentum).add_(param.detach(), alpha=1.0 - momentum)
    for key_buffer, buffer in zip(key_model.buffers(), model.buffers()):
        key_buffer.copy_(buffer)


def lr_at(step: int, total_steps: int, *, lr: float, lr_final: float, warmup: int) -> float:
    """Learning rate of global step `step` (from 0): lr * (step + 1) / warmup during the
    warmup, then cosine decay from lr to lr_final at the last step."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    span = max(1, int(total_steps) - int(warmup) - 1)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------------
# Sampler and inputs
# ----------------------------------------------------------------------------
@dataclass
class TrainQuery:
    """A training query: the planted structure, its coverage targets and fine parts."""

    index: int
    family: str
    size: int
    planted: np.ndarray
    edges: np.ndarray
    targets: list[int]
    fines: np.ndarray
    fine_counts: np.ndarray
    nodeset: str


class TrainSampler:
    """The deterministic training-query stream (see the module docstring)."""

    def __init__(self, store: GraphStore, dataset: str, regime: str, training_seed: int,
                 settings: Mapping[str, Any] | None = None, *, max_draws_per_query: int = 20,
                 held_out: Iterable[str] = ()) -> None:
        settings = dict(settings or sampler_settings())
        self.held_out = frozenset(held_out)
        probabilities = settings["family_probabilities"]
        self.families = list(probabilities)
        self.probs = np.asarray([float(probabilities[f]) for f in self.families], dtype=np.float64)
        self.sizes = [int(s) for s in settings["sizes"]]
        self.jitter = int(settings["size_jitter"])
        self.store = store
        self.dataset = dataset
        self.regime = regime
        self.seed = int(training_seed)
        self.ctx = queries.QueryContext(store)
        self.max_draws_per_query = int(max_draws_per_query)
        part = store.part
        self.node_fine = part.node_fine if part.has_fine else None  # indexed per query, never copied whole
        self.stats: dict[str, Any] = {
            "draws": 0, "failed": 0, "held_out": 0, "short_batches": 0,
            "failed_by_family": {f: 0 for f in self.families},
            "accepted_by_family": {f: 0 for f in self.families},
        }

    def rng(self, epoch: int, step: int, index: int) -> np.random.Generator:
        return np.random.default_rng(protocol.derive_seed(self.seed, self.dataset, self.regime, "train",
                                                          int(epoch), int(step), int(index)))

    def draw(self, epoch: int, step: int, index: int) -> tuple[str, int, queries.PlantedQuery | None]:
        """One draw: (family, size, planted query or None when the draw fails)."""
        rng = self.rng(epoch, step, index)
        family = self.families[int(rng.choice(len(self.families), p=self.probs))]
        base = self.sizes[int(rng.integers(len(self.sizes)))]
        size = max(5, int(base) + int(rng.integers(-self.jitter, self.jitter + 1)))
        return family, size, self.ctx.sample(family, size, rng)

    def make_item(self, index: int, planted_query: queries.PlantedQuery) -> TrainQuery:
        planted = np.asarray(planted_query.planted, dtype=np.int64)
        if self.node_fine is not None:
            fines, counts = np.unique(np.asarray(self.node_fine[planted], dtype=np.int64), return_counts=True)
        else:
            fines, counts = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return TrainQuery(index=int(index), family=planted_query.family, size=int(planted_query.size),
                          planted=planted, edges=np.asarray(planted_query.edges, dtype=np.int64),
                          targets=[int(b) for b in queries.greedy_cover(self.store, planted)],
                          fines=fines.astype(np.int64), fine_counts=counts.astype(np.int64),
                          nodeset=queries.nodeset_hash(planted))

    def batch(self, epoch: int, step: int, batch_size: int) -> list[TrainQuery]:
        """Up to batch_size queries from draws 0, 1, ... of the step (at most max_draws_per_query each)."""
        items: list[TrainQuery] = []
        index = 0
        limit = int(batch_size) * self.max_draws_per_query
        while len(items) < batch_size and index < limit:
            family, _, planted = self.draw(epoch, step, index)
            self.stats["draws"] += 1
            if planted is None:
                self.stats["failed"] += 1
                self.stats["failed_by_family"][family] += 1
            elif self.held_out and queries.nodeset_hash(np.asarray(planted.planted, dtype=np.int64)) in self.held_out:
                self.stats["held_out"] += 1
            else:
                self.stats["accepted_by_family"][family] += 1
                items.append(self.make_item(index, planted))
            index += 1
        if len(items) < batch_size:
            self.stats["short_batches"] += 1
        return items


class InputBuilder:
    """Encoder inputs of training queries, validation queries, blocks and fine parts.

    Block and fine-part inputs are built exactly as encoders.encode_blocks builds
    them and cached until the cache holds max_cache_bytes (cache=False: never).
    """

    def __init__(self, store: GraphStore, regime: str, arch: str, *, cache: bool = True,
                 max_cache_bytes: int = 4 << 30) -> None:
        self.store = store
        self.regime = regime
        self.rel_table = features.store_relation_table(store) if arch == "rgcn" else None
        self.cache = cache
        self.max_cache_bytes = int(max_cache_bytes)
        self.cached_bytes = 0
        self._blocks: dict[int, dict[str, np.ndarray]] = {}
        self._fine: dict[int, tuple[dict[str, np.ndarray], np.ndarray]] = {}

    def _admit(self, arrays: Iterable[np.ndarray]) -> bool:
        if not self.cache:
            return False
        size = sum(int(a.nbytes) for a in arrays)
        if self.cached_bytes + size > self.max_cache_bytes:
            return False
        self.cached_bytes += size
        return True

    def query(self, planted: np.ndarray, edges: np.ndarray) -> dict[str, np.ndarray]:
        """Inputs of a planted query from its payload (query node j is planted[j])."""
        planted = np.asarray(planted, dtype=np.int64)
        payload = queries.payload_for(self.store, planted, self.regime)
        codes = self.store.sigma(self.regime).query_codes(payload, self.store)
        indptr, indices = csr_from_edges(edges, int(planted.shape[0]))
        query = SimpleNamespace(indptr=indptr, indices=indices, payload=payload)
        return features.query_graph_inputs(self.store, query, self.regime, codes, self.rel_table)

    def query_record(self, query: queries.Query) -> dict[str, np.ndarray]:
        """Inputs of a stored query (a QuerySet entry) from its payload."""
        codes = self.store.sigma(self.regime).query_codes(query.payload, self.store)
        return features.query_graph_inputs(self.store, query, self.regime, codes, self.rel_table)

    def block(self, block: int) -> dict[str, np.ndarray]:
        block = int(block)
        hit = self._blocks.get(block)
        if hit is not None:
            return hit
        graph = self.store.induced(self.store.part.nodes_of_blocks([block]))
        inputs = features.store_graph_inputs(self.store, graph, self.regime, self.rel_table)
        if self._admit(inputs.values()):
            self._blocks[block] = inputs
        return inputs

    def fine(self, fine: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """(inputs, data-node ids of the rows) of a fine part."""
        fine = int(fine)
        hit = self._fine.get(fine)
        if hit is not None:
            return hit
        graph = self.store.induced(self.store.part.fine_nodes_of(fine))
        out = (features.store_graph_inputs(self.store, graph, self.regime, self.rel_table),
               np.asarray(graph.global_ids, dtype=np.int64))
        if self._admit([*out[0].values(), out[1]]):
            self._fine[fine] = out
        return out


# ----------------------------------------------------------------------------
# Validation, checkpoints and export
# ----------------------------------------------------------------------------
def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def is_smoke_query_set(meta: Mapping[str, Any]) -> bool:
    """True for a query set built with smoke caps (max_per_cell or other sizes)."""
    sizes = meta.get("sizes")
    return meta.get("max_per_cell") is not None or (sizes is not None and list(sizes) != list(protocol.QUERY_SIZES))


def load_validation_set(path: str | os.PathLike, dataset: str, regime: str) -> tuple[list[queries.Query], dict]:
    """(positive queries, meta) of the checkpoint-selection validation set; refuses any other set."""
    path = Path(path)
    if path.is_dir():
        path = path / f"{regime}.npz"
    query_set = queries.QuerySet.load(path)
    meta = query_set.meta
    problems = []
    if meta.get("dataset") != dataset:
        problems.append(f"dataset {meta.get('dataset')!r} is not {dataset!r}")
    if query_set.regime != regime:
        problems.append(f"regime {query_set.regime} is not the trained regime {regime}")
    if meta.get("split") != "validation" or str(meta.get("master")) != SELECTION_MASTER:
        problems.append(f"split {meta.get('split')}/{meta.get('master')} is not validation/{SELECTION_MASTER} "
                        "(the checkpoint-selection master)")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    positives = query_set.select(kind="positive")
    if not positives:
        raise ValueError(f"{path}: no positive queries")
    return positives, dict(meta)


def load_validation_queries(path: str | os.PathLike, dataset: str, regime: str) -> list[queries.Query]:
    """Positive queries of the checkpoint-selection validation set; refuses any other set."""
    return load_validation_set(path, dataset, regime)[0]


def held_out_nodesets(val_path: str | os.PathLike, val_set: Sequence[queries.Query],
                      extra: Sequence[str | os.PathLike] = ()) -> set[str]:
    """Node-set hashes training never draws: the selection queries, their directory's nodesets.txt
    (every query of that master, negatives and other regimes included) and every `extra` path."""
    held = {queries.nodeset_hash(np.asarray(q.planted, dtype=np.int64)) for q in val_set}
    own = Path(val_path).parent / "nodesets.txt"
    paths = ([own] if own.exists() else []) + [Path(p) for p in extra]
    missing = [str(p) for p in paths if not (p / "nodesets.txt" if p.is_dir() else p).exists()]
    if missing:
        raise FileNotFoundError(f"held-out node sets not found: {missing}")
    return held | queries.read_exclusions(paths)


def validate(model: encoders.GraphEncoder, store: GraphStore, regime: str, val_queries: Sequence[queries.Query],
             val_inputs: Sequence[dict[str, np.ndarray]], ladder: Sequence[int], *, device: Any = "cpu",
             batch_size: int = 32) -> dict[str, Any]:
    """Hybrid FullCov@K of the live encoder, K over `ladder`; metric = mean over K and queries."""
    blocks = encoders.encode_blocks(model, store, regime, device=device, batch_size=batch_size)
    zq = encoders.encode(model, list(val_inputs), device=device, batch_size=batch_size)
    scores = _unit_rows(zq) @ _unit_rows(blocks).T
    hits = np.zeros((len(val_queries), len(ladder)), dtype=bool)
    for i, query in enumerate(val_queries):
        selector = rankers.HybridSelector(rankers.ranking_from_scores(scores[i]), store.part)
        for j, budget in enumerate(ladder):
            hits[i, j] = queries.covers(store, query.planted, selector.prefix(int(budget)))
    return {
        "metric": float(hits.mean()) if hits.size else 0.0,
        "fullcov_at": {str(int(k)): float(hits[:, j].mean()) for j, k in enumerate(ladder)},
        "queries": len(val_queries),
        "ladder": [int(k) for k in ladder],
    }


def checkpoint_path(run_dir: str | os.PathLike, epoch: int) -> Path:
    return Path(run_dir) / f"ckpt_epoch{int(epoch):03d}.pt"


def list_checkpoints(run_dir: str | os.PathLike) -> list[tuple[int, Path]]:
    """(epoch, path) of the run's checkpoints, by epoch."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    out = []
    for path in run_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            out.append((int(match.group(1)), path))
    return sorted(out)


def save_checkpoint(path: str | os.PathLike, payload: Mapping[str, Any]) -> None:
    """Write a checkpoint through a temporary file; an existing checkpoint is never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; checkpoints are never overwritten")
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def read_checkpoint(path: str | os.PathLike) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def _same_config(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return protocol.canonical_json(dict(a)) == protocol.canonical_json(dict(b))


def load_checkpoint(path: str | os.PathLike, *, expect_config: Mapping[str, Any] | None = None,
                    device: Any = "cpu") -> tuple[encoders.GraphEncoder, dict[str, Any]]:
    """The live encoder of a checkpoint, loaded strictly. A config other than expect_config raises."""
    ckpt = read_checkpoint(path)
    config = ckpt["config"]
    if expect_config is not None and not _same_config(config, expect_config):
        raise ValueError(f"{path}: encoder config {config} differs from the expected {dict(expect_config)}")
    model = encoders.build_encoder(config)
    if not _same_config(model.config, config):
        raise ValueError(f"{path}: build_encoder does not reproduce the saved config {config}")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def select_checkpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The record with the maximum metric; ties go to the later epoch (TRAINING["checkpoint_rule"])."""
    if not records:
        raise ValueError("no validation records to select from")
    return dict(max(records, key=lambda r: (float(r["metric"]), int(r["epoch"]))))


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    os.replace(tmp, path)


def _torch_save(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: str | os.PathLike, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _scalar(value: Any) -> float:
    return float(value.detach()) if torch.is_tensor(value) else float(value)


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _numpy_export(model: encoders.GraphEncoder, model_dir: Path, extra: Mapping[str, Any]) -> Any:
    """weights.npz and model.json through encoder_numpy.save_numpy_model, or "pending" without it."""
    try:
        from . import encoder_numpy
    except ImportError:
        return "pending"
    save = getattr(encoder_numpy, "save_numpy_model", None)
    if save is None:
        return "pending"
    try:
        model_json = save(model_dir, dict(model.config), encoders.export_state(model), extra=dict(extra))
    except Exception as exc:  # the torch export stands; the numpy files can be rebuilt from torch_state.pt
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "written", "files": ["weights.npz", "model.json"],
            "weights_sha256": model_json.get("weights_sha256")}


def _numpy_parity(model_dir: Path, store: GraphStore, regime: str, block_emb: np.ndarray, count: int = 4) -> dict:
    """Largest difference between numpy and torch embeddings of the first blocks."""
    try:
        from . import encoder_numpy

        encoder = encoder_numpy.NumpyEncoder.load(model_dir)
        ids = list(range(min(int(count), int(block_emb.shape[0]))))
        emb = encoder_numpy.encode_blocks_numpy(encoder, store, regime, blocks=ids)
        return {"blocks": len(ids), "max_abs_diff": float(np.abs(emb - block_emb[ids]).max()) if ids else 0.0}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def export_model(model: encoders.GraphEncoder, store: GraphStore, plan: Mapping[str, Any],
                 model_dir: str | os.PathLike, *, selection: Mapping[str, Any], device: Any = "cpu",
                 batch_size: int = 32, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the selected encoder's embeddings and weights to model_dir; returns export.json."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    regime = plan["regime"]
    model.eval()
    block_emb = encoders.encode_blocks(model, store, regime, device=device, batch_size=batch_size)
    _save_npy(model_dir / "block_emb.npy", block_emb.astype(np.float32))
    fine_info = None
    if store.part.has_fine:
        fine_emb = encoders.encode_blocks(model, store, regime, device=device, batch_size=batch_size, fine=True)
        _save_npy(model_dir / "fine_emb.npy", fine_emb.astype(np.float32))
        fine_info = {"file": "fine_emb.npy", "shape": list(fine_emb.shape),
                     "sha256": manifest.sha256_file(model_dir / "fine_emb.npy")["sha256"]}
    _torch_save(model_dir / "torch_state.pt", {"config": dict(model.config), "state_dict": _cpu_state(model)})
    identity = {"model_id": plan["model_id"], "dataset": plan["dataset"], "regime": plan["regime"],
                "arch": plan["arch"], "seed": plan["seed"], "arm": plan["arm"]}
    numpy_export = _numpy_export(model, model_dir, {**identity, "selected_epoch": int(selection["epoch"])})
    parity = None
    if isinstance(numpy_export, dict) and numpy_export.get("status") == "written":
        parity = _numpy_parity(model_dir, store, regime, block_emb)
    export = {
        **identity,
        "arm_kind": plan["arm_kind"],
        "requested_regime": plan["requested_regime"],
        "selected_epoch": int(selection["epoch"]),
        "metric": float(selection["metric"]),
        "fullcov_at": dict(selection.get("fullcov_at", {})),
        "checkpoint": selection["checkpoint"],
        "checkpoint_sha256": selection.get("checkpoint_sha256"),
        "protocol_hash": protocol.protocol_hash(),
        "store_meta_sha256": manifest.sha256_file(store.dir / "meta.json")["sha256"],
        "partition": store.partition_name,
        "num_blocks": int(store.num_blocks),
        "config": dict(model.config),
        "block_emb": {"file": "block_emb.npy", "shape": list(block_emb.shape),
                      "sha256": manifest.sha256_file(model_dir / "block_emb.npy")["sha256"]},
        "fine_emb": fine_info,
        "torch_state_sha256": manifest.sha256_file(model_dir / "torch_state.pt")["sha256"],
        "numpy_export": numpy_export,
        "numpy_parity": parity,
        "smoke": dict(plan["smoke"]),
        "created_utc": manifest.utc_now(),
        **dict(extra or {}),
    }
    _write_json(model_dir / EXPORT, export)
    return export


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    return torch.device(name)


def cap_cuda_cache(dev: torch.device) -> dict[str, Any] | None:
    """Hold PyTorch's CUDA cache to the card's own memory (None off the GPU).

    On Windows the NVIDIA driver backs allocations past the card's memory with host RAM
    (sysmem fallback), so cudaMalloc never fails and the caching allocator never frees its
    cache: a MAG run reserved 107 GB for at most 21 GB of live tensors, the excess in host RAM.
    At memory fraction 1.0 the allocator frees cached blocks before it reserves past the card,
    as on Linux, where cudaMalloc fails there. A step whose live tensors exceed the card now
    fails as on Linux and is counted in oom_steps.
    """
    if dev.type != "cuda":
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(1.0, index)
    return {"memory_fraction": 1.0, "device_total_bytes": int(torch.cuda.get_device_properties(index).total_memory)}


class Trainer:
    """Live encoder, key encoder, optimizer, block bank and sampler of one run."""

    def __init__(self, plan: Mapping[str, Any], store: GraphStore, *, device: Any = "cpu",
                 cache_inputs: bool = True, held_out: Iterable[str] = ()) -> None:
        self.plan = plan
        self.store = store
        self.device = torch.device(device)
        self.loss = dict(plan["loss"])
        self.schedule = dict(plan["schedule"])
        regime = plan["regime"]
        self.inputs = InputBuilder(store, regime, plan["arch"], cache=cache_inputs)
        self.sampler = TrainSampler(store, plan["dataset"], regime, plan["seed"], plan["sampler"], held_out=held_out)
        uses_fine = self.loss["fine_nce_weight_alpha"] > 0 or self.loss["node_nce_weight_beta"] > 0
        if uses_fine and not store.part.has_fine:
            raise ValueError("fine_nce and node_nce need a fine level in the partition")
        in_dim = features.input_dim(regime, int(store.meta["attr_dim"]), int(store.meta["num_types"]))
        probe = int(self.inputs.block(0)["x"].shape[1])
        if probe != in_dim:
            raise ValueError(f"block inputs have {probe} columns, features.input_dim gives {in_dim}")
        table = self.inputs.rel_table
        num_relations = int(table.max()) + 1 if table is not None else 1
        m = plan["model"]
        torch.manual_seed(int(plan["init_seed"]))
        self.model = encoders.GraphEncoder(plan["arch"], in_dim, hidden=int(m["hidden"]), output=int(m["output"]),
                                           layers=int(m["layers"]), residual=bool(m["residual"]),
                                           layer_norm=bool(m["layer_norm"]), num_relations=num_relations,
                                           dropout=float(m["dropout"])).to(self.device)
        self.key_model = copy.deepcopy(self.model).eval()
        for param in self.key_model.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.schedule["lr"]),
                                           weight_decay=float(self.schedule["weight_decay"]))
        self.total_steps = int(self.schedule["epochs"]) * int(self.schedule["steps_per_epoch"])
        self.global_step = 0
        self.bank: torch.Tensor | None = None
        self.bank_refreshes = 0

    @torch.no_grad()
    def refresh_bank(self) -> None:
        """Every block's embedding from the key encoder (legacy refresh_embedding_cache, :2515-2581)."""
        graphs = [self.inputs.block(b) for b in range(self.store.num_blocks)]
        emb = encoders.encode(self.key_model, graphs, device=self.device,
                              batch_size=int(self.schedule["encode_batch_size"]))
        self.bank = torch.from_numpy(emb).to(self.device)
        self.bank_refreshes += 1

    def _forward(self, graphs: Sequence[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        b = encoders.collate(graphs).to(self.device)
        return self.model(b.x, b.src, b.dst, b.batch, b.num_graphs, b.rel)

    def compute_terms(self, items: Sequence[TrainQuery]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The objective terms of a batch (floats 0.0 for terms that are off) and step statistics."""
        cfg = self.loss
        alpha, beta, gamma = (float(cfg["fine_nce_weight_alpha"]), float(cfg["node_nce_weight_beta"]),
                              float(cfg["coverage_weight_gamma"]))
        temperature = float(cfg["temperature"])
        zq, zq_nodes = self._forward([self.inputs.query(it.planted, it.edges) for it in items])
        assert self.bank is not None
        num_blocks = int(self.bank.shape[0])
        pos_np = np.zeros((len(items), num_blocks), dtype=bool)
        for i, item in enumerate(items):
            pos_np[i, item.targets] = True
        positives = torch.from_numpy(pos_np).to(self.device)
        keys = self.bank
        live = np.zeros(0, dtype=np.int64)
        limit = int(cfg["live_positives"])
        if limit > 0:
            with torch.no_grad():
                bank_scores = (zq.detach() @ self.bank.T).float().cpu().numpy()
            live = select_live_blocks(pos_np, bank_scores, limit)
            if live.shape[0]:
                z_live, _ = self._forward([self.inputs.block(int(b)) for b in live])
                keys = merge_live(self.bank, torch.from_numpy(live).to(self.device), z_live)
        terms: dict[str, Any] = {"block_infonce": block_infonce(zq, keys, positives, temperature),
                                 "fine_nce": 0.0, "node_nce": 0.0, "coverage": 0.0}
        fine_ids = np.zeros(0, dtype=np.int64)
        if alpha > 0 or beta > 0:
            fine_ids = select_fine_parts(items, int(cfg["fine_positives_per_query"]))
            if fine_ids.shape[0]:
                built = [self.inputs.fine(int(f)) for f in fine_ids]
                z_fine, fine_nodes = self._forward([g for g, _ in built])
                fine_mask = torch.from_numpy(np.stack([np.isin(fine_ids, it.fines) for it in items])).to(self.device)
                if alpha > 0:
                    terms["fine_nce"] = fine_nce(zq, z_fine, fine_mask, temperature)
                if beta > 0:
                    terms["node_nce"] = node_nce(zq_nodes, np.concatenate([it.planted for it in items]), fine_nodes,
                                                 np.concatenate([ids for _, ids in built]), temperature)
        if gamma > 0:
            terms["coverage"] = coverage_loss(
                zq, keys, positives, temperature=float(cfg["coverage_temperature"]),
                margin_weight=float(cfg["hardest_negative_margin_weight"]),
                aggregation=str(cfg["coverage_aggregation"]), cvar_fraction=float(cfg["cvar_rho"]),
                smoothmax_temperature=float(cfg["smoothmax_temperature"]),
                topk=int(cfg["topk_barrier_K"]), topk_weight=float(cfg["topk_barrier_weight"]),
                topk_bucket=int(cfg["topk_barrier_bucket"]), topk_margin=float(cfg["topk_barrier_margin"]))
        stats = {"live_blocks": int(live.shape[0]), "fine_parts": int(fine_ids.shape[0]),
                 "targets_mean": float(np.mean([len(it.targets) for it in items]))}
        return terms, stats

    def train_step(self, epoch: int, step: int) -> dict[str, Any]:
        """One optimizer step on the batch of (epoch, step); returns its record.

        The record lists the node-set hashes of every query drawn for the step. A
        step without queries or gradient is skipped; so is a CUDA out-of-memory
        step (as the legacy trainer did), which the record marks with "oom".
        """
        t = self.global_step
        refresh = int(self.loss["bank_refresh_steps"])
        if self.bank is None or (refresh > 0 and t % refresh == 0):
            self.refresh_bank()
        lr = lr_at(t, self.total_steps, lr=float(self.schedule["lr"]), lr_final=float(self.schedule["lr_final"]),
                   warmup=int(self.schedule["warmup_steps"]))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        items = self.sampler.batch(epoch, step, int(self.schedule["batch_size"]))
        record: dict[str, Any] = {"lr": lr, "queries": len(items), "nodesets": [it.nodeset for it in items],
                                  "skipped": not items}
        self.global_step += 1
        if not items:
            return record
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            terms, stats = self.compute_terms(items)
            total = combine_objective(terms, self.loss)
            if torch.is_tensor(total) and total.requires_grad:
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=float(self.schedule["grad_clip_norm"]))
                self.optimizer.step()
                ema_update(self.key_model, self.model, float(self.loss["negative_encoder_momentum"]))
            else:
                record["skipped"] = True
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda":
                raise
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            record.update(skipped=True, oom=True)
            return record
        record.update(stats)
        record["terms"] = {k: _scalar(v) for k, v in terms.items()}
        record["total"] = _scalar(total)
        return record

    def state_payload(self, epoch: int, val_record: Mapping[str, Any], nodeset_lines: int,
                      train_args: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "config": dict(self.model.config),
            "state_dict": _cpu_state(self.model),
            "momentum_state_dict": _cpu_state(self.key_model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "metric": float(val_record["metric"]),
            "protocol_hash": protocol.protocol_hash(),
            "train_args": dict(train_args),
            # Exact resumption.
            "global_step": int(self.global_step),
            "bank": None if self.bank is None else self.bank.detach().cpu().clone(),
            "bank_refreshes": int(self.bank_refreshes),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "sampler_stats": copy.deepcopy(self.sampler.stats),
            "train_nodesets_lines": int(nodeset_lines),
            "val_record": dict(val_record),
        }

    def load_state(self, ckpt: Mapping[str, Any]) -> None:
        """Restore a checkpoint strictly (resumption)."""
        if not _same_config(ckpt["config"], self.model.config):
            raise ValueError(f"checkpoint config {ckpt['config']} differs from {self.model.config}")
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.key_model.load_state_dict(ckpt["momentum_state_dict"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt["global_step"])
        bank = ckpt.get("bank")
        self.bank = None if bank is None else bank.to(self.device)
        self.bank_refreshes = int(ckpt.get("bank_refreshes", 0))
        torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        self.sampler.stats = copy.deepcopy(ckpt["sampler_stats"])


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
RESUME_IGNORED = ("paths", "device", "resume", "cache_inputs", "requested_regime")


def _resume_mismatch(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    keys = sorted((set(saved) | set(current)) - set(RESUME_IGNORED))
    return [k for k in keys if protocol.canonical_json(saved.get(k)) != protocol.canonical_json(current.get(k))]


def _truncate_lines(path: Path, count: int) -> None:
    lines = path.read_text(encoding="ascii").splitlines() if path.exists() else []
    if len(lines) < count:
        raise ValueError(f"{path} has {len(lines)} lines, the checkpoint recorded {count}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines[:count]), encoding="ascii")
    os.replace(tmp, path)


def _clear_stale_logs(run_dir: Path) -> list[str]:
    """Logs of an attempt that left no checkpoint; a fresh start replaces them."""
    cleared = []
    for name in (TRAIN_NODESETS, VAL_METRICS, TRAIN_LOG, SELECTION, EXPORT):
        path = run_dir / name
        if path.exists():
            path.unlink()
            cleared.append(name)
    return cleared


def _archive_manifest(run_dir: Path) -> str | None:
    current = run_dir / manifest.MANIFEST_NAME
    if not current.exists():
        return None
    for k in itertools.count(1):
        target = run_dir / f"manifest.previous-{k}.json"
        if not target.exists():
            os.replace(current, target)
            return target.name
    return None


def host_busy_seconds() -> float | None:
    """The busy CPU-seconds of every logical CPU of the machine since boot (psutil's busy time), or None."""
    try:
        import psutil
    except ImportError:
        return None
    times = psutil.cpu_times()
    return float(sum(times) - sum(getattr(times, name, 0.0) for name in ("idle", "iowait", "guest", "guest_nice")))


def epoch_memory(dev: torch.device) -> dict[str, int | None]:
    """This process's peak memory so far (a resume starts it again) and the GPU peaks since the last call."""
    from .partition import self_peak_memory

    host = self_peak_memory()
    out: dict[str, int | None] = {"host_peak_bytes": max(host.values()) if host else None,
                                  "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    if dev.type == "cuda":
        out["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(dev))
        out["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
        torch.cuda.reset_peak_memory_stats(dev)
    return out


def _epoch_summary(epoch: int, records: Sequence[Mapping[str, Any]], trainer: Trainer, seconds: float,
                   failed_before: int, cpu_seconds: float | None = None, host_busy: float | None = None,
                   memory: Mapping[str, int | None] | None = None) -> dict[str, Any]:
    done = [r for r in records if not r.get("skipped") and not r.get("oom")]
    summary: dict[str, Any] = {
        "epoch": int(epoch), "global_step": int(trainer.global_step), "steps": len(records),
        "trained_steps": len(done), "skipped_steps": sum(1 for r in records if r.get("skipped")),
        "oom_steps": sum(1 for r in records if r.get("oom")),
        "queries": int(sum(int(r.get("queries", 0)) for r in records)),
        "failed_draws": int(trainer.sampler.stats["failed"] - failed_before),
        "lr_last": float(records[-1]["lr"]) if records and "lr" in records[-1] else None,
        "bank_refreshes": int(trainer.bank_refreshes), "seconds": round(seconds, 3), "utc": manifest.utc_now(),
        # the machine's load beside this run: busy CPUs of the whole machine minus cpu_seconds / seconds
        "cpu_seconds": None if cpu_seconds is None else round(cpu_seconds, 3),
        "host_busy_cpus": None if host_busy is None or seconds <= 0 else round(host_busy / seconds, 3),
        **(memory or {}),
    }
    if done:
        summary["loss"] = float(np.mean([r["total"] for r in done]))
        summary["terms"] = {k: float(np.mean([r["terms"][k] for r in done])) for k in done[0]["terms"]}
        for key in ("live_blocks", "fine_parts", "targets_mean"):
            summary[f"{key}_mean"] = float(np.mean([r[key] for r in done]))
    return summary


def run(plan: Mapping[str, Any], *, store_root: str | os.PathLike, val_queries: str | os.PathLike,
        run_dir: str | os.PathLike, models_root: str | os.PathLike, device: str = "auto", resume: bool = False,
        partition: str = "primary", cache_inputs: bool = True, argv: Sequence[str] | None = None,
        held_out: Sequence[str | os.PathLike] = ()) -> dict[str, Any]:
    """Train, validate, select and export one encoder (see the module docstring)."""
    plan = dict(plan)
    run_dir = Path(run_dir)
    models_root = Path(models_root)
    existing = list_checkpoints(run_dir)
    if existing and not resume:
        raise FileExistsError(f"{run_dir} already holds checkpoints; pass --resume to continue that run")
    dev = resolve_device(device)
    cuda_cache_cap = cap_cuda_cache(dev)
    dataset, regime = plan["dataset"], plan["regime"]
    store = GraphStore(store_root, dataset, mmap=True, partition=partition)
    if partition == "primary" and store.num_blocks != int(protocol.DATASETS[dataset]["partitions"]):
        raise ValueError(f"{dataset} store has {store.num_blocks} blocks; the protocol fixes "
                         f"{protocol.DATASETS[dataset]['partitions']}")
    val_path = Path(val_queries)
    if val_path.is_dir():
        val_path = val_path / f"{regime}.npz"
    val_set, val_meta = load_validation_set(val_path, dataset, regime)
    if is_smoke_query_set(val_meta) and not plan["smoke"]:
        raise ValueError(f"{val_path} is a smoke query set (max_per_cell {val_meta.get('max_per_cell')}, sizes "
                         f"{val_meta.get('sizes')}); only smoke runs may select checkpoints on it")
    held = held_out_nodesets(val_path, val_set, held_out)
    ladder = protocol.ladder_for(dataset)[:-1]
    schedule = plan["schedule"]
    epochs, steps_per_epoch = int(schedule["epochs"]), int(schedule["steps_per_epoch"])
    val_every = int(schedule["validation_every_epochs"])
    batch = int(schedule["encode_batch_size"])
    train_args = {**plan, "partition": partition, "device": str(dev), "cache_inputs": bool(cache_inputs),
                  "paths": {"store_root": str(store_root), "val_queries": str(val_path), "run_dir": str(run_dir),
                            "models_root": str(models_root), "held_out": [str(p) for p in held_out]},
                  # resume compares the node sets themselves, so a moved query root still resumes
                  "held_out": {"nodesets": len(held),
                               "sha256": hashlib.sha256("\n".join(sorted(held)).encode("ascii")).hexdigest()}}
    trainer = Trainer(plan, store, device=dev, cache_inputs=cache_inputs, held_out=held)
    val_inputs = [trainer.inputs.query_record(q) for q in val_set]
    run_dir.mkdir(parents=True, exist_ok=True)
    nodeset_path = run_dir / TRAIN_NODESETS
    start_epoch, resumed_from, archived, cleared = 1, None, None, []
    if not existing:
        cleared = _clear_stale_logs(run_dir)
        if cleared:
            archived = _archive_manifest(run_dir)
    else:
        last_epoch, last_path = existing[-1]
        ckpt = read_checkpoint(last_path)
        if ckpt.get("protocol_hash") != protocol.protocol_hash():
            raise ValueError(f"{last_path} was trained under protocol {ckpt.get('protocol_hash')}, "
                             f"not {protocol.protocol_hash()}")
        mismatch = _resume_mismatch(ckpt["train_args"], train_args)
        if mismatch:
            raise ValueError(f"cannot resume {run_dir}: settings differ in {mismatch}")
        trainer.load_state(ckpt)
        _truncate_lines(nodeset_path, int(ckpt["train_nodesets_lines"]))
        kept = [r for r in read_jsonl(run_dir / VAL_METRICS) if int(r["epoch"]) <= last_epoch]
        if not any(int(r["epoch"]) == last_epoch for r in kept):
            kept.append({**ckpt["val_record"], "checkpoint_sha256": manifest.sha256_file(last_path)["sha256"]})
        _rewrite_jsonl(run_dir / VAL_METRICS, kept)
        logged = read_jsonl(run_dir / TRAIN_LOG)
        _rewrite_jsonl(run_dir / TRAIN_LOG, [r for r in logged if int(r["epoch"]) <= last_epoch])
        start_epoch = last_epoch + 1
        resumed_from = {"epoch": last_epoch, "checkpoint": last_path.name}
        archived = _archive_manifest(run_dir)
    nodeset_lines = len(nodeset_path.read_text(encoding="ascii").splitlines()) if nodeset_path.exists() else 0
    config = {**train_args, "num_blocks": int(store.num_blocks), "ladder": ladder,
              "validation": {"path": str(val_path), "positives": len(val_set), "master": SELECTION_MASTER,
                             "smoke_query_set": is_smoke_query_set(val_meta),
                             "query_set_protocol_hash": val_meta.get("protocol_hash")},
              "resumed_from": resumed_from, "archived_manifest": archived, "cleared_stale_logs": cleared,
              "cuda_cache_cap": cuda_cache_cap}
    inputs = {"store_meta": store.dir / "meta.json", "partition_info": store.part.dir / "partition.json",
              "sigma": store.dir / "sigma" / regime, "val_queries": val_path}
    argv = list(argv) if argv is not None else list(sys.argv)
    with manifest.RunManifest(run_dir, experiment="train_encoder", config=config, inputs=inputs,
                              split="validation", fast_input_hashes=True, argv=argv) as run_manifest:
        with open(nodeset_path, "a", encoding="ascii") as nodeset_log:
            epoch_memory(dev)  # the GPU peaks of each epoch and validation start here
            for epoch in range(start_epoch, epochs + 1):
                t0, c0, h0 = time.perf_counter(), time.process_time(), host_busy_seconds()
                failed_before = int(trainer.sampler.stats["failed"])
                records = []
                for step in range(steps_per_epoch):
                    record = trainer.train_step(epoch, step)
                    hashes = record.pop("nodesets")
                    nodeset_log.write("".join(h + "\n" for h in hashes))
                    nodeset_lines += len(hashes)
                    records.append(record)
                nodeset_log.flush()
                seconds, h1 = time.perf_counter() - t0, host_busy_seconds()
                append_jsonl(run_dir / TRAIN_LOG, _epoch_summary(
                    epoch, records, trainer, seconds, failed_before, cpu_seconds=time.process_time() - c0,
                    host_busy=None if h0 is None or h1 is None else h1 - h0, memory=epoch_memory(dev)))
                if epoch % val_every == 0 or epoch == epochs:
                    os.fsync(nodeset_log.fileno())
                    t1 = time.perf_counter()
                    val = validate(trainer.model, store, regime, val_set, val_inputs, ladder, device=dev,
                                   batch_size=batch)
                    path = checkpoint_path(run_dir, epoch)
                    record = {"epoch": int(epoch), "global_step": int(trainer.global_step), **val,
                              "checkpoint": path.name, "utc": manifest.utc_now(),
                              "seconds": round(time.perf_counter() - t1, 3), **epoch_memory(dev)}
                    save_checkpoint(path, trainer.state_payload(epoch, record, nodeset_lines, train_args))
                    digest = manifest.sha256_file(path)["sha256"]
                    append_jsonl(run_dir / VAL_METRICS, {**record, "checkpoint_sha256": digest})
        records = read_jsonl(run_dir / VAL_METRICS)
        chosen = select_checkpoint(records)
        selection = {
            "rule": protocol.TRAINING["checkpoint_rule"], "model_id": plan["model_id"], "epoch": int(chosen["epoch"]),
            "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen.get("checkpoint_sha256"),
            "metric": float(chosen["metric"]), "fullcov_at": chosen.get("fullcov_at", {}),
            "candidates": [{"epoch": int(r["epoch"]), "metric": float(r["metric"])} for r in records],
            "utc": manifest.utc_now(),
        }
        _write_json(run_dir / SELECTION, selection)
        model, _ = load_checkpoint(run_dir / selection["checkpoint"], expect_config=trainer.model.config, device=dev)
        train_hashes = set(nodeset_path.read_text(encoding="ascii").split()) if nodeset_path.exists() else set()
        overlap = len(train_hashes & held)  # 0 by construction; the export records the audit
        extra = {"run_dir": str(run_dir), "sampler_stats": trainer.sampler.stats, "train_queries": nodeset_lines,
                 "train_val_nodeset_overlap": overlap, "held_out_nodesets": len(held)}
        export = export_model(model, store, plan, models_root / plan["model_id"], selection=selection, device=dev,
                              batch_size=batch, extra=extra)
        _write_json(run_dir / EXPORT, export)
        run_manifest.note(selected_epoch=selection["epoch"], metric=selection["metric"],
                          model_dir=str(models_root / plan["model_id"]), export_sha256=manifest.sha256_file(
                              models_root / plan["model_id"] / EXPORT)["sha256"],
                          sampler_stats=trainer.sampler.stats, train_queries=nodeset_lines,
                          train_val_nodeset_overlap=overlap, bank_refreshes=trainer.bank_refreshes,
                          numpy_export=export["numpy_export"], resumed_from=resumed_from)
    return {"plan": plan, "selection": selection, "export": export, "run_dir": str(run_dir),
            "model_dir": str(models_root / plan["model_id"])}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(protocol.DATASETS))
    parser.add_argument("--regime", required=True, choices=list(protocol.TRAINED_REGIMES),
                        help="R2-* train the shared R2-512 encoder")
    parser.add_argument("--seed", required=True, type=int, help="training seed (protocol SEEDS['training'])")
    parser.add_argument("--arch", default=None, choices=list(encoders.ARCHS),
                        help="default: the arm's architecture, else the dataset's encoder")
    parser.add_argument("--arm", default=MAIN_ARM, help="FACTORIAL or ARCHITECTURE_ABLATION arm (default: main)")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--val-queries", required=True,
                        help="checkpoint-selection QuerySet .npz (validation master 31415), or its directory")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--held-out", nargs="*", default=[],
                        help="query-set directories (or nodesets.txt files) whose node sets training never draws; "
                             "the --val-queries directory is always held out")
    parser.add_argument("--partition", default="primary")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="continue from the run directory's last checkpoint")
    parser.add_argument("--no-input-cache", action="store_true", help="rebuild block and fine inputs when needed")
    smoke = parser.add_argument_group("smoke runs only (recorded in the manifest and export.json)")
    smoke.add_argument("--epochs", type=int, default=None)
    smoke.add_argument("--steps-per-epoch", type=int, default=None)
    smoke.add_argument("--batch-size", type=int, default=None)
    smoke.add_argument("--val-every", type=int, default=None, help="validation interval in epochs")
    smoke.add_argument("--layers", type=int, default=None)
    smoke.add_argument("--hidden", type=int, default=None)
    smoke.add_argument("--output", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smoke = {"epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
             "validation_every_epochs": args.val_every, "layers": args.layers, "hidden": args.hidden,
             "output": args.output}
    plan = resolve_plan(args.dataset, args.regime, args.seed, arch=args.arch, arm=args.arm, smoke=smoke)
    if plan["requested_regime"] != plan["regime"]:
        print(f"note: {plan['requested_regime']} uses the {plan['regime']} encoder", file=sys.stderr)
    if plan["smoke"]:
        print(f"note: smoke run with {plan['smoke']}; not a canonical model", file=sys.stderr)
    if not plan["seed_in_protocol"]:
        print(f"note: seed {plan['seed']} is not in SEEDS['training']", file=sys.stderr)
    recorded = ["-m", "scripts.canonical.train_encoder", *argv] if argv is not None else list(sys.argv)
    result = run(plan, store_root=args.store_root, val_queries=args.val_queries, run_dir=args.run_dir,
                 models_root=args.models_root, device=args.device, resume=args.resume, partition=args.partition,
                 cache_inputs=not args.no_input_cache, argv=recorded, held_out=args.held_out)
    print(json.dumps({"model_id": plan["model_id"], "selected_epoch": result["selection"]["epoch"],
                      "metric": result["selection"]["metric"], "model_dir": result["model_dir"],
                      "numpy_export": result["export"]["numpy_export"]}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
