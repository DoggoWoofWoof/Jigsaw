"""Torch-free graph encoder: encoders.GraphEncoder.forward in numpy, eval mode.

Serving encodes queries and blocks with this module and never imports torch
(protocol.SERVING["encoder"]; block embeddings are built offline by serving.py).
NumpyEncoder runs the state dict of a trained encoders.GraphEncoder
(encoders.export_state) through the same float32 operations, so its embeddings
match torch's (the trainer records the difference as numpy_parity):

  h_0 = ReLU(input_proj(x))
  h_i = ReLU(norm_i(conv_i(h_{i-1}) + h_{i-1}))   with the residual
  h_i = ReLU(norm_i(conv_i(h_{i-1})))             without it; norm_i is LayerNorm(eps 1e-5) or identity
  r   = concat over i = 1..L of [mean, max, sum] of h_i over the graph   (h_0 is not pooled)
  z   = normalize(readout_proj(r) + readout_skip(r)), dividing by max(norm, 1e-12)

Dropout is the identity (eval mode). A batch is the disjoint union built by
encoders.collate: the CSR entry (row v, column u) is the message u -> v, so the
messages of a batch are grouped by destination and every scatter is a reduction
over consecutive segments. Empty segments give zeros, as encoders.scatter_mean
and scatter_max do. Only summation order differs from torch (BLAS matmuls,
pairwise instead of sequential sums, LayerNorm moments): pure float32 keeps the
normalised outputs within 1e-6 of torch, at the deployed size too, against the
2e-5 the tests allow.

Model directory (save_numpy_model, load_numpy_model):
  weights.npz  float32 arrays under the torch state-dict names (the np.savez layout
               with fixed zip metadata, no pickle)
  model.json   {"format", "config", "weights_sha256", "state_sha256", "extra"}
weights_sha256 hashes the file; state_sha256 hashes the arrays (names, shapes,
float32 bytes), so it names the weights whatever container holds them.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import features

FORMAT = "jigsaw-numpy-encoder-v1"
WEIGHTS_NAME = "weights.npz"
MODEL_NAME = "model.json"
ARCHS = ("graphsage", "rgcn", "gin", "gat")  # encoders.ARCHS, restated to stay torch-free
LAYER_NORM_EPS = 1e-5  # nn.LayerNorm default
NORMALIZE_EPS = 1e-12  # F.normalize default
GAT_NEGATIVE_SLOPE = 0.2
# GraphEncoder.__init__ defaults, used like encoders.build_encoder for keys a config leaves out.
CONFIG_DEFAULTS: dict[str, Any] = {"hidden": 256, "output": 128, "layers": 6, "residual": True, "layer_norm": True,
                                   "num_relations": 1, "num_bases": None, "dropout": 0.1}


# ----------------------------------------------------------------------------
# Config and state checks
# ----------------------------------------------------------------------------
def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The GraphEncoder.config that encoders.build_encoder(config) would produce."""
    if "arch" not in config or "in_dim" not in config:
        raise ValueError("an encoder config needs 'arch' and 'in_dim'")
    arch = str(config["arch"])
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture {arch!r}")
    given = {**CONFIG_DEFAULTS, **{k: config[k] for k in CONFIG_DEFAULTS if k in config}}
    out = {
        "arch": arch, "in_dim": int(config["in_dim"]), "hidden": int(given["hidden"]), "output": int(given["output"]),
        "layers": int(given["layers"]), "residual": bool(given["residual"]), "layer_norm": bool(given["layer_norm"]),
        "num_relations": int(max(1, int(given["num_relations"]))), "num_bases": given["num_bases"],
        "dropout": float(given["dropout"]),
    }
    if out["layers"] < 1:
        # GraphEncoder.forward concatenates the pools of at least one layer.
        raise ValueError("an encoder needs at least one layer")
    if arch == "rgcn":
        out["num_bases"] = int(out["num_bases"] or min(8, out["num_relations"]))  # as RGCNConv.__init__
    return out


def state_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Every state-dict key of GraphEncoder(config) with its shape."""
    cfg = normalize_config(config)
    arch, hidden, layers, output = cfg["arch"], cfg["hidden"], cfg["layers"], cfg["output"]
    square, vector = (hidden, hidden), (hidden,)
    readout = 3 * layers * hidden
    shapes: dict[str, tuple[int, ...]] = {"input_proj.weight": (hidden, cfg["in_dim"]), "input_proj.bias": vector}
    for i in range(layers):
        p = f"convs.{i}."
        if arch == "graphsage":
            shapes.update({p + "lin_nbr.weight": square, p + "lin_nbr.bias": vector, p + "lin_self.weight": square})
        elif arch == "rgcn":
            bases, relations = cfg["num_bases"], cfg["num_relations"]
            shapes.update({p + "bases": (bases, hidden, hidden), p + "comp": (relations, bases), p + "root": square,
                           p + "bias": vector})
        elif arch == "gin":
            shapes.update({p + "mlp.0.weight": square, p + "mlp.0.bias": vector, p + "mlp.3.weight": square,
                           p + "mlp.3.bias": vector})
        else:
            shapes.update({p + "lin.weight": square, p + "att_src": vector, p + "att_dst": vector, p + "bias": vector})
        if cfg["layer_norm"]:
            shapes.update({f"norms.{i}.weight": vector, f"norms.{i}.bias": vector})
    shapes.update({
        "readout_proj.0.weight": (2 * hidden, readout), "readout_proj.0.bias": (2 * hidden,),
        "readout_proj.3.weight": (hidden, 2 * hidden), "readout_proj.3.bias": vector,
        "readout_proj.6.weight": (output, hidden), "readout_proj.6.bias": (output,),
        "readout_skip.weight": (output, readout), "readout_skip.bias": (output,),
    })
    return shapes


def check_state(config: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """float32 arrays of `state` after the checks of load_state_dict(strict=True).

    Raises ValueError on a missing key, an unexpected key or a shape mismatch.
    """
    shapes = state_shapes(config)
    arrays = {k: np.asarray(state[k], dtype=np.float32) for k in shapes if k in state}
    problems = []
    missing = sorted(set(shapes) - set(state))
    unexpected = sorted(set(state) - set(shapes))
    if missing:
        problems.append(f"missing keys {missing}")
    if unexpected:
        problems.append(f"unexpected keys {unexpected}")
    wrong = [f"{k} {tuple(a.shape)} != {shapes[k]}" for k, a in arrays.items() if tuple(a.shape) != shapes[k]]
    if wrong:
        problems.append(f"shape mismatches {wrong}")
    if problems:
        raise ValueError("encoder state does not match its config: " + "; ".join(problems))
    return arrays


# ----------------------------------------------------------------------------
# Segment reductions (the scatters of encoders.py over a sorted index)
# ----------------------------------------------------------------------------
def _segment_starts(counts: np.ndarray) -> np.ndarray:
    return np.cumsum(counts) - counts


def segment_sum(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Sums over consecutive segments of `values` with lengths `counts`; zeros for empty segments."""
    counts = np.asarray(counts, dtype=np.int64)
    rows = values.reshape(values.shape[0], math.prod(values.shape[1:]))
    out = np.zeros((counts.shape[0], rows.shape[1]), dtype=values.dtype)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.add.reduceat(rows, _segment_starts(counts)[nonempty], axis=0)
    return out.reshape((counts.shape[0],) + values.shape[1:])


def segment_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """segment_sum divided by max(count, 1), as encoders.scatter_mean."""
    counts = np.asarray(counts, dtype=np.int64)
    denom = np.maximum(counts, 1).astype(values.dtype)
    return segment_sum(values, counts) / denom.reshape((-1,) + (1,) * (values.ndim - 1))


def segment_amax(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Maxima over consecutive segments; -inf for empty segments (scatter_reduce amax on -inf)."""
    counts = np.asarray(counts, dtype=np.int64)
    out = np.full((counts.shape[0],) + values.shape[1:], -np.inf, dtype=values.dtype)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.maximum.reduceat(values, _segment_starts(counts)[nonempty], axis=0)
    return out


def segment_max(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """segment_amax with every infinite entry set to 0, as encoders.scatter_max."""
    out = segment_amax(values, counts)
    out[np.isinf(out)] = 0
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, np.float32(0))


def _leaky_relu(x: np.ndarray, slope: float = GAT_NEGATIVE_SLOPE) -> np.ndarray:
    return np.where(x > 0, x, x * np.float32(slope))


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = LAYER_NORM_EPS) -> np.ndarray:
    """nn.LayerNorm over the last axis: biased variance, eps inside the square root."""
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=1, keepdims=True)
    return centered * (np.float32(1) / np.sqrt(var + np.float32(eps))) * weight + bias


def _l2_normalize(z: np.ndarray, eps: float = NORMALIZE_EPS) -> np.ndarray:
    """F.normalize(z, dim=1): rows divided by max(L2 norm, eps)."""
    norm = np.sqrt((z * z).sum(axis=1, keepdims=True))
    return z / np.maximum(norm, np.float32(eps))


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
@dataclass
class _Batch:
    """Disjoint union of graphs, as encoders.collate builds it."""

    x: np.ndarray  # float32 [N, d_in]
    src: np.ndarray  # int64 [E] source of every message (the CSR column plus the graph offset)
    dst: np.ndarray  # int64 [E] destination (the CSR row plus the offset), non-decreasing
    counts: np.ndarray  # int64 [N] messages received by every node
    rel: np.ndarray | None  # int64 [E] relation of every message (rgcn only)
    sizes: np.ndarray  # int64 [G] nodes per graph


class NumpyEncoder:
    """encoders.GraphEncoder in eval mode, evaluated with numpy (see the module docstring).

    `config` is GraphEncoder.config and `state` is encoders.export_state(model);
    the state is checked like load_state_dict(strict=True).
    """

    def __init__(self, config: dict, state: dict[str, np.ndarray]) -> None:
        cfg = normalize_config(config)
        weights = check_state(cfg, state)
        self.config = cfg
        self.arch = cfg["arch"]
        self.in_dim = cfg["in_dim"]
        self.hidden = cfg["hidden"]
        self.output = cfg["output"]
        self.layers = cfg["layers"]
        self.residual = cfg["residual"]
        self.layer_norm = cfg["layer_norm"]
        self.num_relations = cfg["num_relations"]
        self.model_json: dict | None = None  # set by load()

        def linear(name: str) -> tuple[np.ndarray, np.ndarray | None]:
            # nn.Linear computes x @ weight.T + bias; keep weight.T contiguous for the matmul.
            bias = weights.get(name + ".bias")
            return np.ascontiguousarray(weights[name + ".weight"].T), None if bias is None else bias.copy()

        self._input = linear("input_proj")
        self._convs = []
        for i in range(self.layers):
            p = f"convs.{i}."
            if self.arch == "graphsage":
                conv = {"nbr": linear(p + "lin_nbr"), "self": linear(p + "lin_self")[0]}
            elif self.arch == "rgcn":
                bases, comp = weights[p + "bases"], weights[p + "comp"]
                # relation_weights(): einsum("rb,bio->rio", comp, bases), each [in, out]
                rel_w = (comp @ bases.reshape(bases.shape[0], -1)).reshape(comp.shape[0], self.hidden, self.hidden)
                conv = {"rel": np.ascontiguousarray(rel_w), "root": weights[p + "root"].copy(),
                        "bias": weights[p + "bias"].copy()}
            elif self.arch == "gin":
                conv = {"mlp0": linear(p + "mlp.0"), "mlp3": linear(p + "mlp.3")}
            else:
                conv = {"lin": linear(p + "lin")[0], "att_src": weights[p + "att_src"].copy(),
                        "att_dst": weights[p + "att_dst"].copy(), "bias": weights[p + "bias"].copy()}
            self._convs.append(conv)
        self._norms = [(weights[f"norms.{i}.weight"].copy(), weights[f"norms.{i}.bias"].copy())
                       if self.layer_norm else None for i in range(self.layers)]
        self._readout = [linear(f"readout_proj.{j}") for j in (0, 3, 6)]
        self._skip = linear("readout_skip")

    @classmethod
    def load(cls, model_dir: str | os.PathLike) -> "NumpyEncoder":
        """Encoder from a directory written by save_numpy_model (SHA-256 checked)."""
        config, state, model_json = load_numpy_model(model_dir)
        encoder = cls(config, state)
        encoder.model_json = model_json
        return encoder

    # ------------------------------------------------------------------ public
    def encode(self, graphs: Sequence[dict[str, np.ndarray]], batch_size: int = 64) -> np.ndarray:
        """float32 embeddings [len(graphs), output] of features.graph_inputs() dicts."""
        graphs = list(graphs)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not graphs:
            return np.zeros((0, self.output), dtype=np.float32)
        with_rel = False
        if self.arch == "rgcn":
            present = {"rel" in g for g in graphs}
            if len(present) > 1:
                # encoders.collate drops every relation of a batch in which one graph lacks them,
                # so torch's output would depend on the batch boundaries.
                raise ValueError("rgcn needs 'rel' on every graph or on none")
            with_rel = present.pop()
        out = [self._forward(self._collate(graphs[start:start + batch_size], with_rel))
               for start in range(0, len(graphs), batch_size)]
        return np.concatenate(out, axis=0)

    def encode_one(self, graph: dict[str, np.ndarray]) -> np.ndarray:
        """float32 embedding [output] of one graph."""
        return self.encode([graph])[0]

    # ------------------------------------------------------------------ internals
    def _collate(self, graphs: Sequence[dict[str, np.ndarray]], with_rel: bool) -> _Batch:
        xs, srcs, counts, rels, sizes = [], [], [], [], []
        offset = 0
        for gi, g in enumerate(graphs):
            x = np.asarray(g["x"], dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != self.in_dim:
                raise ValueError(f"graph {gi}: x has shape {x.shape}, the encoder takes [n, {self.in_dim}]")
            n = int(x.shape[0])
            indptr = np.asarray(g["indptr"], dtype=np.int64)
            indices = np.asarray(g["indices"], dtype=np.int64)
            if indptr.shape != (n + 1,) or indices.ndim != 1:
                raise ValueError(f"graph {gi}: indptr must be [n+1] and indices 1-D for n = {n}")
            lens = np.diff(indptr)
            if (lens < 0).any() or int(lens.sum()) != indices.shape[0]:
                raise ValueError(f"graph {gi}: indptr does not describe {indices.shape[0]} CSR entries")
            if indices.shape[0] and (int(indices.min()) < 0 or int(indices.max()) >= n):
                raise ValueError(f"graph {gi}: a neighbour id is outside 0..{n - 1}")
            if with_rel:
                rel = np.asarray(g["rel"], dtype=np.int64)
                if rel.shape != indices.shape:
                    raise ValueError(f"graph {gi}: rel must have one entry per CSR entry")
                rels.append(rel)
            xs.append(x)
            srcs.append(indices + offset)
            counts.append(lens)
            sizes.append(n)
            offset += n
        counts_all = np.concatenate(counts)
        return _Batch(
            x=np.concatenate(xs, axis=0),
            src=np.concatenate(srcs),
            dst=np.repeat(np.arange(offset, dtype=np.int64), counts_all),
            counts=counts_all,
            rel=np.concatenate(rels) if with_rel else None,
            sizes=np.asarray(sizes, dtype=np.int64),
        )

    def _relation_groups(self, b: _Batch) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """(relation, sources, receiving nodes, messages per receiving node) for every relation r < R with messages.

        Without relations every message has relation 0; ids outside 0..R-1 are
        skipped, as RGCNConv.forward only visits range(num_relations).
        """
        rel = b.rel if b.rel is not None else np.zeros_like(b.src)
        n = int(b.x.shape[0])
        groups = []
        for r in range(self.num_relations):
            mask = rel == r
            if mask.any():
                per_node = np.bincount(b.dst[mask], minlength=n)
                rows = np.flatnonzero(per_node)
                groups.append((r, b.src[mask], rows, per_node[rows].astype(np.int64)))
        return groups

    def _conv(self, conv: dict[str, Any], h: np.ndarray, b: _Batch, groups: list) -> np.ndarray:
        if self.arch == "graphsage":
            w_nbr, b_nbr = conv["nbr"]
            return (segment_mean(h[b.src], b.counts) @ w_nbr + b_nbr) + h @ conv["self"]
        if self.arch == "rgcn":
            out = h @ conv["root"] + conv["bias"]
            for r, src, rows, counts in groups:
                # torch adds scatter_mean(...) @ W_r to every node; nodes without messages of r get exact zeros.
                out[rows] = out[rows] + segment_mean(h[src], counts) @ conv["rel"][r]
            return out
        if self.arch == "gin":
            (w0, b0), (w3, b3) = conv["mlp0"], conv["mlp3"]
            return _relu((h + segment_sum(h[b.src], b.counts)) @ w0 + b0) @ w3 + b3
        return self._gat(conv, h, b)

    def _gat(self, conv: dict[str, Any], h: np.ndarray, b: _Batch) -> np.ndarray:
        """GATConv: one head over the messages plus a self loop per node, which torch appends last."""
        wh = h @ conv["lin"]
        a_src = (wh * conv["att_src"]).sum(axis=1)
        a_dst = (wh * conv["att_dst"]).sum(axis=1)
        score = _leaky_relu(a_src[b.src] + a_dst[b.dst])
        score_loop = _leaky_relu(a_src + a_dst)
        top = np.maximum(segment_amax(score, b.counts), score_loop)
        w = np.exp(score - top[b.dst])
        w_loop = np.exp(score_loop - top)
        denom = segment_sum(w, b.counts) + w_loop
        alpha = w / denom[b.dst]
        alpha_loop = w_loop / denom
        return (segment_sum(alpha[:, None] * wh[b.src], b.counts) + alpha_loop[:, None] * wh) + conv["bias"]

    def _forward(self, b: _Batch) -> np.ndarray:
        w_in, b_in = self._input
        h = _relu(b.x @ w_in + b_in)
        groups = self._relation_groups(b) if self.arch == "rgcn" else []
        pooled = []
        for conv, norm in zip(self._convs, self._norms):
            m = self._conv(conv, h, b, groups)
            pre = m + h if self.residual else m
            h = _relu(pre if norm is None else _layer_norm(pre, *norm))
            pooled += [segment_mean(h, b.sizes), segment_max(h, b.sizes), segment_sum(h, b.sizes)]
        r = np.concatenate(pooled, axis=1)
        (w0, b0), (w3, b3), (w6, b6) = self._readout
        w_skip, b_skip = self._skip
        proj = _relu(_relu(r @ w0 + b0) @ w3 + b3) @ w6 + b6
        return _l2_normalize(proj + (r @ w_skip + b_skip)).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Queries and blocks
# ----------------------------------------------------------------------------
def _relation_table_for(encoder: NumpyEncoder, store: Any, rel_table: np.ndarray | None) -> np.ndarray | None:
    """The store's relation table for rgcn (as encoders.encode_blocks); None for other architectures."""
    if encoder.arch != "rgcn":
        return None
    return features.store_relation_table(store) if rel_table is None else np.asarray(rel_table, dtype=np.int64)


def encode_query(encoder: NumpyEncoder, store: Any, query: Any, regime: str, codes: np.ndarray,
                 rel_table: np.ndarray | None = None) -> np.ndarray:
    """float32 embedding [output] of a query from its payload (features.query_graph_inputs).

    `codes` come from SigmaTable.query_codes(). For rgcn, pass the store's
    relation table (features.store_relation_table) when encoding many queries;
    it is rebuilt from the store otherwise. Other architectures ignore it.
    """
    table = _relation_table_for(encoder, store, rel_table)
    return encoder.encode_one(features.query_graph_inputs(store, query, regime, codes, table))


def encode_blocks_numpy(encoder: NumpyEncoder, store: Any, regime: str, *, blocks: Iterable[int] | None = None,
                        fine: bool = False, batch_size: int = 64, rel_table: np.ndarray | None = None) -> np.ndarray:
    """Embeddings of every block (or fine part) of the store's partition, in id order.

    Builds the graphs and inputs of encoders.encode_blocks: the induced subgraph
    of the block's nodes and features.store_graph_inputs(). rel_table is used by
    rgcn only and defaults to features.store_relation_table(store).
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    table = _relation_table_for(encoder, store, rel_table)
    part = store.part
    if fine and not getattr(part, "has_fine", False):
        raise ValueError("the partition has no fine level")
    ids = list(blocks) if blocks is not None else list(range(int(part.fine_ptr.shape[0]) - 1 if fine
                                                             else part.num_blocks))
    out = []
    for start in range(0, len(ids), batch_size):
        chunk = []
        for b in ids[start:start + batch_size]:
            nodes = part.fine_nodes_of(b) if fine else part.nodes_of_blocks([b])
            chunk.append(features.store_graph_inputs(store, store.induced(nodes), regime, table))
        out.append(encoder.encode(chunk, batch_size=batch_size))
    return np.concatenate(out, axis=0) if out else np.zeros((0, encoder.output), dtype=np.float32)


# ----------------------------------------------------------------------------
# Model files
# ----------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def state_sha256(state: Mapping[str, Any]) -> str:
    """SHA-256 of the weights: every array in name order with its name, shape and float32 bytes."""
    h = hashlib.sha256()
    for name in sorted(state):
        a = np.ascontiguousarray(np.asarray(state[name], dtype=np.float32))
        h.update(name.encode("utf-8") + b"\0" + json.dumps(list(a.shape)).encode("ascii") + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """The np.savez layout with sorted members and fixed zip metadata, so equal weights give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def save_numpy_model(model_dir: str | os.PathLike, config: dict, state: dict[str, np.ndarray],
                     extra: dict | None = None) -> dict:
    """Write weights.npz, then model.json, each atomically; returns the model.json dict.

    The state is checked against the config first, so every saved model loads.
    weights_sha256 is the SHA-256 of the weights.npz bytes, so a reader that
    lands between the two writes gets a ValueError, never a silent mix. The
    bytes depend on the weights alone (npz_bytes), so re-exporting a model
    reproduces its weights_sha256.
    """
    arrays = check_state(config, state)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = npz_bytes(arrays)
    model_json = {"format": FORMAT, "config": dict(config), "weights_sha256": hashlib.sha256(data).hexdigest(),
                  "state_sha256": state_sha256(arrays), "extra": dict(extra or {})}
    text = json.dumps(model_json, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic_write_bytes(model_dir / WEIGHTS_NAME, data)
    _atomic_write_bytes(model_dir / MODEL_NAME, text.encode("utf-8"))
    return json.loads(text)


def load_numpy_model(model_dir: str | os.PathLike) -> tuple[dict, dict[str, np.ndarray], dict]:
    """(config, state, model.json) of a saved model.

    Raises ValueError when model.json has another format or weights.npz does not
    match its recorded SHA-256. The arrays are parsed from the bytes that were hashed.
    """
    model_dir = Path(model_dir)
    model_json = json.loads((model_dir / MODEL_NAME).read_text(encoding="utf-8"))
    found = model_json.get("format") if isinstance(model_json, dict) else None
    if found != FORMAT:
        raise ValueError(f"{model_dir / MODEL_NAME}: format {found!r}, expected {FORMAT!r}")
    if not isinstance(model_json.get("config"), dict) or not isinstance(model_json.get("weights_sha256"), str):
        raise ValueError(f"{model_dir / MODEL_NAME}: needs 'config' and 'weights_sha256'")
    data = (model_dir / WEIGHTS_NAME).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != model_json["weights_sha256"]:
        raise ValueError(f"{model_dir / WEIGHTS_NAME}: sha256 {digest} does not match model.json "
                         f"({model_json['weights_sha256']})")
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        state = {k: np.asarray(npz[k]) for k in npz.files}
    recorded = model_json.get("state_sha256")
    if recorded is not None and state_sha256(state) != recorded:
        raise ValueError(f"{model_dir / WEIGHTS_NAME}: state_sha256 does not match model.json")
    return dict(model_json["config"]), state, model_json
