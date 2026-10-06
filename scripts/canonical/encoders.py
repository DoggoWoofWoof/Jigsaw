"""Canonical graph encoders in plain torch (no torch_geometric).

Every encoder the canonical runs train or ablate is defined here, with the
state-dict names that encoder_numpy.py mirrors for torch-free serving:

  input_proj   Linear(d_in, H); h_0 = ReLU(input_proj(x))
  convs.{i}    message passing layer i (below)
  norms.{i}    LayerNorm(H, eps 1e-5), or identity when layer_norm is off
  layer i      h_i = ReLU(norm_i(conv_i(h_{i-1}) + h_{i-1}))   with the residual
               h_i = ReLU(norm_i(conv_i(h_{i-1})))             without it
  readout      r = concat over layers i of [mean, max, sum] of h_i over the graph
               z = normalize(readout_proj(r) + readout_skip(r))
  readout_proj Linear(3LH, 2H) ReLU Dropout Linear(2H, H) ReLU Dropout Linear(H, out)

Convolutions. Messages run from src to dst, and every undirected edge gives two.
  graphsage  conv(h)_v = lin_nbr(mean_{u->v} h_u) + lin_self(h_v)          (lin_self has no bias)
  rgcn       conv(h)_v = sum_r mean_{u -r-> v} h_u W_r + h_v W_root + b,
             W_r = sum_b comp[r, b] bases[b], num_bases = min(8, R)         (x @ W, [in, out])
  gin        conv(h)_v = mlp(h_v + sum_{u->v} h_u), mlp = Linear ReLU Dropout Linear
  gat        conv(h)_v = sum_{u in N(v) + v} alpha_uv W h_u + b, one head,
             alpha_uv = softmax_u LeakyReLU_0.2(a_src . W h_u + a_dst . W h_v)
A node without incoming messages aggregates zeros; pools of an empty graph are
zeros; normalize divides by max(norm, 1e-12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("graphsage", "rgcn", "gin", "gat")


# ----------------------------------------------------------------------------
# Scatter helpers
# ----------------------------------------------------------------------------
def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_zeros((size,) + tuple(values.shape[1:]))
    return out.index_add_(0, index, values)


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    total = scatter_sum(values, index, size)
    count = torch.bincount(index, minlength=size).clamp_min(1).to(values.dtype)
    return total / count.view(-1, *([1] * (values.dim() - 1)))


def scatter_max(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_full((size,) + tuple(values.shape[1:]), float("-inf"))
    expanded = index.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    out = out.scatter_reduce(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


# ----------------------------------------------------------------------------
# Convolutions
# ----------------------------------------------------------------------------
class SAGEConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_nbr = nn.Linear(dim, dim)
        self.lin_self = nn.Linear(dim, dim, bias=False)

    def forward(self, h, src, dst, rel=None):
        return self.lin_nbr(scatter_mean(h[src], dst, h.shape[0])) + self.lin_self(h)


class RGCNConv(nn.Module):
    def __init__(self, dim: int, num_relations: int, num_bases: int | None = None) -> None:
        super().__init__()
        self.num_relations = max(1, int(num_relations))
        self.num_bases = int(num_bases or min(8, self.num_relations))
        self.bases = nn.Parameter(torch.empty(self.num_bases, dim, dim))
        self.comp = nn.Parameter(torch.empty(self.num_relations, self.num_bases))
        self.root = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        for p in (self.bases, self.comp, self.root):
            bound = math.sqrt(6.0 / (p.shape[-2] + p.shape[-1]))
            nn.init.uniform_(p, -bound, bound)

    def relation_weights(self) -> torch.Tensor:
        return torch.einsum("rb,bio->rio", self.comp, self.bases)

    def forward(self, h, src, dst, rel=None):
        if rel is None:
            rel = torch.zeros_like(src)
        weights = self.relation_weights()
        out = h @ self.root + self.bias
        for r in range(self.num_relations):
            mask = rel == r
            if bool(mask.any()):
                out = out + scatter_mean(h[src[mask]], dst[mask], h.shape[0]) @ weights[r]
        return out


class GINConv(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, h, src, dst, rel=None):
        return self.mlp(h + scatter_sum(h[src], dst, h.shape[0]))


class GATConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(dim))
        self.att_dst = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        bound = math.sqrt(6.0 / (1 + dim))
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)

    def forward(self, h, src, dst, rel=None):
        n = h.shape[0]
        loop = torch.arange(n, device=h.device)
        s = torch.cat([src, loop])
        d = torch.cat([dst, loop])
        wh = self.lin(h)
        score = F.leaky_relu((wh * self.att_src).sum(-1)[s] + (wh * self.att_dst).sum(-1)[d], 0.2)
        top = score.new_full((n,), float("-inf")).scatter_reduce(0, d, score, reduce="amax", include_self=True)
        w = torch.exp(score - top[d])
        alpha = w / scatter_sum(w, d, n)[d]
        return scatter_sum(alpha.unsqueeze(1) * wh[s], d, n) + self.bias


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, arch: str, in_dim: int, *, hidden: int = 256, output: int = 128, layers: int = 6,
                 residual: bool = True, layer_norm: bool = True, num_relations: int = 1,
                 num_bases: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture {arch!r}")
        self.config = {
            "arch": arch, "in_dim": int(in_dim), "hidden": int(hidden), "output": int(output),
            "layers": int(layers), "residual": bool(residual), "layer_norm": bool(layer_norm),
            "num_relations": int(max(1, num_relations)), "num_bases": num_bases, "dropout": float(dropout),
        }
        self.residual = bool(residual)
        self.input_proj = nn.Linear(in_dim, hidden)
        convs = []
        for _ in range(layers):
            if arch == "graphsage":
                convs.append(SAGEConv(hidden))
            elif arch == "rgcn":
                conv = RGCNConv(hidden, num_relations, num_bases)
                self.config["num_bases"] = conv.num_bases
                convs.append(conv)
            elif arch == "gin":
                convs.append(GINConv(hidden, dropout))
            else:
                convs.append(GATConv(hidden))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) if layer_norm else nn.Identity() for _ in range(layers)])
        readout_dim = hidden * layers * 3
        self.readout_proj = nn.Sequential(
            nn.Linear(readout_dim, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, output),
        )
        self.readout_skip = nn.Linear(readout_dim, output)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, batch: torch.Tensor,
                num_graphs: int, rel: torch.Tensor | None = None, return_layers: bool = False):
        h = F.relu(self.input_proj(x))
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, src, dst, rel)
            h = F.relu(norm(m + h if self.residual else m))
            layer_outputs.append(h)
        pooled = []
        for hl in layer_outputs:
            pooled += [scatter_mean(hl, batch, num_graphs), scatter_max(hl, batch, num_graphs),
                       scatter_sum(hl, batch, num_graphs)]
        r = torch.cat(pooled, dim=1)
        z = F.normalize(self.readout_proj(r) + self.readout_skip(r), dim=1)
        node = F.normalize(h, dim=1)
        if return_layers:
            return z, node, layer_outputs
        return z, node


def build_encoder(config: dict[str, Any]) -> GraphEncoder:
    kwargs = {k: config[k] for k in ("hidden", "output", "layers", "residual", "layer_norm", "num_relations",
                                     "num_bases", "dropout") if k in config}
    return GraphEncoder(config["arch"], config["in_dim"], **kwargs)


# ----------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------
@dataclass
class GraphBatch:
    x: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    batch: torch.Tensor
    num_graphs: int
    rel: torch.Tensor | None = None

    def to(self, device: Any) -> "GraphBatch":
        return GraphBatch(self.x.to(device), self.src.to(device), self.dst.to(device), self.batch.to(device),
                          self.num_graphs, None if self.rel is None else self.rel.to(device))


def collate(graphs: Sequence[dict[str, np.ndarray]]) -> GraphBatch:
    """Disjoint union of graphs given as features.graph_inputs() dicts."""
    xs, srcs, dsts, rels, owners = [], [], [], [], []
    offset = 0
    with_rel = bool(graphs) and all("rel" in g for g in graphs)
    for gi, g in enumerate(graphs):
        n = int(g["x"].shape[0])
        indptr = np.asarray(g["indptr"], dtype=np.int64)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        xs.append(np.asarray(g["x"], dtype=np.float32))
        srcs.append(np.asarray(g["indices"], dtype=np.int64) + offset)
        dsts.append(rows + offset)
        owners.append(np.full(n, gi, dtype=np.int64))
        if with_rel:
            rels.append(np.asarray(g["rel"], dtype=np.int64))
        offset += n
    cat = lambda parts, dtype: torch.from_numpy(np.concatenate(parts) if parts else np.zeros(0, dtype=dtype))
    x = torch.from_numpy(np.concatenate(xs, axis=0)) if xs else torch.zeros((0, 0))
    return GraphBatch(x=x, src=cat(srcs, np.int64), dst=cat(dsts, np.int64), batch=cat(owners, np.int64),
                      num_graphs=len(graphs), rel=cat(rels, np.int64) if with_rel else None)


@torch.no_grad()
def encode(model: GraphEncoder, graphs: Sequence[dict[str, np.ndarray]], *, device: Any = "cpu",
           batch_size: int = 16) -> np.ndarray:
    """Eval-mode graph embeddings [len(graphs), out] as float32."""
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(graphs), batch_size):
        b = collate(graphs[start:start + batch_size]).to(device)
        z, _ = model(b.x, b.src, b.dst, b.batch, b.num_graphs, b.rel)
        out.append(z.float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.config["output"]), dtype=np.float32)


def encode_blocks(model: GraphEncoder, store: Any, regime: str, *, device: Any = "cpu", batch_size: int = 16,
                  blocks: Iterable[int] | None = None, fine: bool = False) -> np.ndarray:
    """Embeddings of every block (or fine part) of the store's partition, in id order."""
    from . import features

    rel_table = features.store_relation_table(store) if model.config["arch"] == "rgcn" else None
    part = store.part
    ids = list(blocks) if blocks is not None else list(range(int(part.fine_ptr.shape[0]) - 1 if fine else part.num_blocks))
    out = []
    for start in range(0, len(ids), batch_size):
        chunk = []
        for b in ids[start:start + batch_size]:
            nodes = part.fine_nodes_of(b) if fine else part.nodes_of_blocks([b])
            chunk.append(features.store_graph_inputs(store, store.induced(nodes), regime, rel_table))
        out.append(encode(model, chunk, device=device, batch_size=batch_size))
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.config["output"]), dtype=np.float32)


def export_state(model: GraphEncoder) -> dict[str, np.ndarray]:
    """float32 numpy copy of the state dict, the input of encoder_numpy.NumpyEncoder."""
    return {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
