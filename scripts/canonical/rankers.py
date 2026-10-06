"""Block rankers, selection policies and retrieval metrics.

A ranker scores every block of the primary partition for one query:

  learned              cosine between the query embedding and each block embedding
  index                sum over the distinct query labels a block holds of
                       idf(l) = log(1 + B / bdf(l)), with B blocks and bdf(l) the
                       blocks holding a node labelled l; ties to the lower block id
  index_count          the same with weight 1 per label (the legacy FeatureIndex)
  mean_feature         cosine between the mean encoder inputs of the query and of the block
  topo_feature         minus the Euclidean distance between z-scored size-free
                       statistics of the query and of the block
  fused_learned_index  reciprocal rank fusion of learned and index
  fused_learned_mean   reciprocal rank fusion of learned and mean_feature
  router               index with ties broken by the learned score when
                       selectivity(Q) <= tau, else fused_learned_index
  random               a uniform permutation seeded by the query uid

Every ranking is a full permutation of the blocks. The per-block inputs
(postings, mean inputs, statistics, block embeddings) are built offline and
stored next to the partition; a query reads only the rows it needs.

A policy turns the ranking into a block order whose prefixes are the ladder's candidates:

  topk    the ranking itself
  hybrid  the top seed_count blocks of the ranking, then greedy growth along the
          block graph: each step adds the frontier block with the best mix of
          rank score and boundary support (log1p cut-edge weight and the number
          of selected neighbours), and every teleport_every-th step adds the
          best-ranked unselected block instead. Ported from
          hybrid_boundary_expand in scripts/retrieval_strategies.py.

Both policies are prefix-consistent: the order for budget K is the first K blocks
of the order for any larger budget, so the ladder's candidates are nested.
HybridSelector extends the order lazily, so a query solved at an early ladder
step never pays for the greedy growth of later steps.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import protocol

ARTIFACT_VERSION = "rankers-v1"
RANKER_NAMES = ("learned", "index", "index_count", "mean_feature", "topo_feature", "fused_learned_index",
                "fused_learned_mean", "router", "random")
NEEDS_EMBEDDING = {"learned", "fused_learned_index", "fused_learned_mean", "router"}


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
class TopKSelector:
    name = "topk"

    def __init__(self, ranking: np.ndarray, part: Any = None) -> None:
        self.ranking = np.asarray(ranking, dtype=np.int64)

    def prefix(self, budget: int) -> np.ndarray:
        return self.ranking[:max(0, int(budget))]


class HybridSelector:
    """Lazy hybrid order over the blocks of a partition (see the module docstring)."""

    name = "hybrid"

    def __init__(self, ranking: np.ndarray, part: Any, *, seed_count: int | None = None,
                 model_weight: float | None = None, boundary_weight_mix: Sequence[float] | None = None,
                 teleport_every: int | None = None) -> None:
        spec = protocol.SELECTION["hybrid"]
        self.seed_count = max(1, int(spec["seed_count"] if seed_count is None else seed_count))
        self.model_weight = min(1.0, max(0.0, float(spec["model_weight"] if model_weight is None else model_weight)))
        mix = spec["boundary_weight_mix"] if boundary_weight_mix is None else boundary_weight_mix
        self.mix = (float(mix[0]), float(mix[1]))
        self.teleport_every = max(0, int(spec["teleport_every"] if teleport_every is None else teleport_every))
        self.part = part
        self.num_blocks = int(part.num_blocks)
        ranking = np.asarray(ranking, dtype=np.int64)
        _, first = np.unique(ranking, return_index=True)
        self.ranking = ranking[np.sort(first)]  # first occurrence of each block, in order
        n_ranked = int(self.ranking.shape[0])
        # Rank score 1 - rank / (n - 1); an unranked block scores 0 and sorts after every ranked one.
        self.rank_pos = np.full(self.num_blocks, n_ranked + 1, dtype=np.int64)
        self.rank_pos[self.ranking] = np.arange(n_ranked, dtype=np.int64)
        self.model = np.zeros(self.num_blocks, dtype=np.float64)
        self.model[self.ranking] = 1.0 - np.arange(n_ranked, dtype=np.float64) / max(1, n_ranked - 1)
        self.selected = np.zeros(self.num_blocks, dtype=bool)
        self.frontier = np.zeros(self.num_blocks, dtype=bool)
        self.weight = np.zeros(self.num_blocks, dtype=np.float64)
        self.support = np.zeros(self.num_blocks, dtype=np.int64)
        self.order: list[int] = []
        self._next_ranked = 0
        self._additions = 0
        self._exhausted = False

    def _add(self, block: int) -> None:
        self.selected[block] = True
        self.order.append(int(block))
        self.frontier[block] = False
        self.weight[block] = 0.0
        self.support[block] = 0
        nbrs, w = self.part.block_neighbors(int(block))
        nbrs = np.asarray(nbrs, dtype=np.int64)
        open_ = ~self.selected[nbrs]
        nbrs = nbrs[open_]
        self.weight[nbrs] += np.log1p(np.maximum(0.0, np.asarray(w, dtype=np.float64)[open_]))
        self.support[nbrs] += 1
        self.frontier[nbrs] = True

    def _best_ranked(self) -> int | None:
        while self._next_ranked < self.ranking.shape[0] and self.selected[self.ranking[self._next_ranked]]:
            self._next_ranked += 1
        if self._next_ranked < self.ranking.shape[0]:
            return int(self.ranking[self._next_ranked])
        return None

    def _step(self) -> bool:
        self._additions += 1
        best_ranked = self._best_ranked()
        if self.teleport_every and self._additions % self.teleport_every == 0 and best_ranked is not None:
            self._add(best_ranked)
            return True
        cand = np.flatnonzero(self.frontier)
        if cand.shape[0] == 0:
            if best_ranked is None:
                return False
            self._add(best_ranked)
            return True
        weight = self.weight[cand]
        support = self.support[cand].astype(np.float64)
        boundary = self.mix[0] * (weight / max(1e-12, float(weight.max()))) + \
            self.mix[1] * (support / max(1.0, float(support.max())))
        combined = self.model_weight * self.model[cand] + (1.0 - self.model_weight) * boundary
        best = cand[np.lexsort((cand, self.rank_pos[cand], -combined))[0]]
        self._add(int(best))
        return True

    def prefix(self, budget: int) -> np.ndarray:
        budget = min(max(0, int(budget)), self.num_blocks)
        seeds = self.ranking[:min(budget, self.seed_count)]
        if len(self.order) < seeds.shape[0]:
            # Greedy steps start only after the whole seed set, so a larger budget
            # first completes the seeds of a smaller one.
            for block in seeds.tolist():
                if not self.selected[block]:
                    self._add(block)
        while len(self.order) < budget and not self._exhausted:
            if not self._step():
                self._exhausted = True
        return np.asarray(self.order[:budget], dtype=np.int64)


def make_selector(policy: str, ranking: np.ndarray, part: Any, **params: Any):
    if policy == "hybrid":
        return HybridSelector(ranking, part, **params)
    if policy == "topk":
        return TopKSelector(ranking, part)
    raise ValueError(f"unknown selection policy {policy!r}")


def hybrid_order(ranking: np.ndarray, part: Any, budget: int, **params: Any) -> np.ndarray:
    """The first `budget` blocks of the hybrid order."""
    return HybridSelector(ranking, part, **params).prefix(budget)


# ----------------------------------------------------------------------------
# Rank utilities
# ----------------------------------------------------------------------------
def ranking_from_scores(scores: np.ndarray) -> np.ndarray:
    """Blocks by descending score; ties to the lower block id."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(scores.shape[0]), -scores)).astype(np.int64)


def rrf_scores(rankings: Sequence[np.ndarray], num_blocks: int, k: int | None = None) -> np.ndarray:
    """Reciprocal rank fusion, sum over rankings of 1 / (k + rank) with rank from 1."""
    k = int(protocol.RANKERS["rrf_k"] if k is None else k)
    out = np.zeros(num_blocks, dtype=np.float64)
    for ranking in rankings:
        ranking = np.asarray(ranking, dtype=np.int64)
        out[ranking] += 1.0 / (k + np.arange(1, ranking.shape[0] + 1, dtype=np.float64))
    return out


# ----------------------------------------------------------------------------
# Offline ranker inputs
# ----------------------------------------------------------------------------
TOPO_STATS = ("mean_degree", "degree_cv", "leaf_fraction", "transitivity")  # then one fraction per node type


def artifact_dir(store: Any, name: str) -> Path:
    """Ranker inputs on the store's current partition: a regime name, or "topology"."""
    return Path(store.part.dir) / "rankers" / name


def embeddings_dir(store: Any, model_id: str, regime: str) -> Path:
    return Path(store.part.dir) / "embeddings" / model_id / regime


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _block_chunks(part: Any, chunk_nodes: int):
    """(first block, end block, nodes) over consecutive block ranges of about chunk_nodes nodes."""
    ptr = np.asarray(part.block_ptr, dtype=np.int64)
    num_blocks = int(ptr.shape[0]) - 1
    a = 0
    while a < num_blocks:
        b = int(np.searchsorted(ptr, ptr[a] + chunk_nodes, side="right")) - 1
        b = min(max(b, a + 1), num_blocks)
        yield a, b, np.asarray(part.block_nodes[int(ptr[a]):int(ptr[b])], dtype=np.int64)
        a = b


def _store_check(store: Any, regime: str | None) -> dict[str, Any]:
    out = {"version": ARTIFACT_VERSION, "dataset": store.dataset, "partition": store.partition_name,
           "num_nodes": int(store.num_nodes), "num_blocks": int(store.num_blocks)}
    if regime is not None:
        out.update({"regime": regime, "num_codes": int(store.sigma(regime).num_codes)})
    return out


def _load_checked(directory: Path, store: Any, regime: str | None) -> dict[str, Any]:
    path = directory / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"{directory} has no ranker inputs; run python -m scripts.canonical.rankers build")
    info = json.loads(path.read_text(encoding="utf-8"))
    want = _store_check(store, regime)
    stale = {k: (info.get(k), v) for k, v in want.items() if info.get(k) != v}
    if stale:
        raise ValueError(f"stale ranker inputs in {directory}: {stale}")
    return info


@dataclass
class Postings:
    """Blocks holding each sigma code: blocks[ptr[c]:ptr[c + 1]], ascending."""

    ptr: np.ndarray  # int64 [num_codes + 1]
    blocks: np.ndarray  # int32
    num_blocks: int

    @property
    def num_codes(self) -> int:
        return int(self.ptr.shape[0]) - 1

    def block_df(self, codes: np.ndarray) -> np.ndarray:
        """Blocks holding each code; 0 for fresh codes."""
        codes = np.asarray(codes, dtype=np.int64)
        known = (codes >= 0) & (codes < self.num_codes)
        out = np.zeros(codes.shape[0], dtype=np.int64)
        c = codes[known]
        out[known] = np.asarray(self.ptr[c + 1], dtype=np.int64) - np.asarray(self.ptr[c], dtype=np.int64)
        return out

    def idf(self, codes: np.ndarray) -> np.ndarray:
        bdf = self.block_df(codes)
        return np.where(bdf > 0, np.log1p(self.num_blocks / np.maximum(bdf, 1)), 0.0)

    def scores(self, codes: np.ndarray, *, weighting: str = "idf") -> np.ndarray:
        """Per block, the summed weight of the distinct query codes it holds.

        Weights are added in code order, so two blocks holding the same codes get
        bitwise equal scores and tie exactly.
        """
        labels = np.unique(np.asarray(codes, dtype=np.int64))
        labels = labels[(labels >= 0) & (labels < self.num_codes)]
        starts = np.asarray(self.ptr[labels], dtype=np.int64)
        lens = np.asarray(self.ptr[labels + 1], dtype=np.int64) - starts
        if weighting == "idf":
            weight = np.log1p(self.num_blocks / np.maximum(lens, 1))
        elif weighting == "count":
            weight = np.ones(labels.shape[0], dtype=np.float64)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
        total = int(lens.sum())
        pos = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
        blocks = np.asarray(self.blocks[pos], dtype=np.int64)
        return np.bincount(blocks, weights=np.repeat(weight, lens), minlength=self.num_blocks).astype(np.float64)

    @classmethod
    def load(cls, directory: Path, num_blocks: int, *, mmap: bool) -> "Postings":
        mode = "r" if mmap else None
        return cls(np.load(directory / "postings_ptr.npy", mmap_mode=mode),
                   np.load(directory / "postings_blocks.npy", mmap_mode=mode), int(num_blocks))


def build_postings(store: Any, regime: str, *, chunk_nodes: int = 1 << 24) -> Postings:
    """Distinct (code, block) pairs of the store, grouped by code."""
    table = store.sigma(regime)
    part = store.part
    num_blocks = int(part.num_blocks)
    keys = []
    for _, _, nodes in _block_chunks(part, chunk_nodes):
        blk = np.asarray(part.node_block[nodes], dtype=np.int64)
        keys.append(np.unique(np.asarray(table.codes[nodes], dtype=np.int64) * num_blocks + blk))
    # Chunks hold disjoint blocks, so their key sets are disjoint.
    keys = np.sort(np.concatenate(keys)) if keys else np.zeros(0, dtype=np.int64)
    ptr = np.zeros(table.num_codes + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys // num_blocks, minlength=table.num_codes), out=ptr[1:])
    return Postings(ptr, (keys % num_blocks).astype(np.int32), num_blocks)


def build_mean_inputs(store: Any, regime: str) -> np.ndarray:
    """float32 [B, d_in]: the mean encoder input of each block's induced subgraph."""
    from . import features

    rows = []
    for b in range(store.num_blocks):
        x = features.store_graph_inputs(store, store.block(b), regime)["x"]
        rows.append(x.astype(np.float64).mean(axis=0) if x.shape[0] else np.zeros(x.shape[1]))
    return np.stack(rows).astype(np.float32)


def topo_stats(indptr: np.ndarray, indices: np.ndarray, types: np.ndarray, num_types: int) -> np.ndarray:
    """Size-free statistics of one graph: TOPO_STATS, then the fraction of nodes of each type.

    transitivity = 3 triangles / connected triples = sum over edges uv of common(u, v) / sum_v d(d - 1).
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    n = int(indptr.shape[0]) - 1
    out = np.zeros(len(TOPO_STATS) + num_types, dtype=np.float64)
    if n <= 0:
        return out
    deg = np.diff(indptr).astype(np.float64)
    mean = float(deg.mean())
    out[0] = mean
    out[1] = float(deg.std()) / mean if mean > 0 else 0.0
    out[2] = float((deg <= 1).mean())
    triples = float((deg * (deg - 1.0)).sum())
    if triples > 0:
        from scipy import sparse

        a = sparse.csr_matrix((np.ones(np.asarray(indices).shape[0]), np.asarray(indices, dtype=np.int64), indptr),
                              shape=(n, n))
        out[3] = float((a @ a).multiply(a).sum()) / triples
    out[len(TOPO_STATS):] = np.bincount(np.asarray(types, dtype=np.int64), minlength=num_types)[:num_types] / n
    return out


def build_topo_stats(store: Any) -> np.ndarray:
    num_types = int(store.meta["num_types"])
    rows = []
    for b in range(store.num_blocks):
        g = store.block(b)
        rows.append(topo_stats(g.indptr, g.indices, np.asarray(store.node_type[g.global_ids]), num_types))
    return np.stack(rows)


def build_artifacts(store: Any, regimes: Sequence[str], *, chunk_nodes: int = 1 << 24) -> dict[str, Any]:
    """Write postings and mean inputs per regime, and the topology statistics, next to the partition."""
    report: dict[str, Any] = {}
    for regime in regimes:
        directory = artifact_dir(store, regime)
        post = build_postings(store, regime, chunk_nodes=chunk_nodes)
        _save(directory / "postings_ptr.npy", post.ptr)
        _save(directory / "postings_blocks.npy", post.blocks)
        mean = build_mean_inputs(store, regime)
        _save(directory / "mean_inputs.npy", mean)
        info = _store_check(store, regime)
        info.update({"pairs": int(post.blocks.shape[0]), "input_dim": int(mean.shape[1]),
                     "idf": "log(1 + B / bdf)"})
        _write_json(directory / "info.json", info)
        report[regime] = info
    directory = artifact_dir(store, "topology")
    stats = build_topo_stats(store)
    _save(directory / "topo_stats.npy", stats)
    info = _store_check(store, None)
    info.update({"stats": list(TOPO_STATS) + [f"type_fraction_{t}" for t in range(stats.shape[1] - len(TOPO_STATS))],
                 "mean": stats.mean(axis=0).tolist(), "std": stats.std(axis=0).tolist()})
    _write_json(directory / "info.json", info)
    report["topology"] = {k: info[k] for k in ("num_blocks", "stats")}
    return report


def save_block_embeddings(store: Any, model_id: str, regime: str, embeddings: np.ndarray,
                          extra: dict[str, Any] | None = None) -> Path:
    directory = embeddings_dir(store, model_id, regime)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != store.num_blocks:
        raise ValueError(f"{embeddings.shape[0]} embeddings for {store.num_blocks} blocks")
    _save(directory / "block_emb.npy", embeddings)
    info = _store_check(store, regime)
    info.update({"model_id": model_id, "dim": int(embeddings.shape[1])})
    info.update(dict(extra or {}))
    _write_json(directory / "info.json", info)
    return directory


def load_block_embeddings(store: Any, model_id: str, regime: str) -> tuple[np.ndarray, dict[str, Any]]:
    directory = embeddings_dir(store, model_id, regime)
    info = _load_checked(directory, store, regime)
    return np.load(directory / "block_emb.npy"), info


# ----------------------------------------------------------------------------
# Query-time rankers
# ----------------------------------------------------------------------------
@dataclass
class QueryInputs:
    """What the rankers read about one query; the cascade builds it from the payload."""

    uid: str
    codes: np.ndarray  # sigma codes of the query nodes
    types: np.ndarray  # node types of the query nodes
    indptr: np.ndarray  # query CSR
    indices: np.ndarray
    x: np.ndarray | None = None  # encoder inputs [n, d_in]
    embedding: np.ndarray | None = None  # query embedding
    extra: dict[str, Any] = field(default_factory=dict)


def _unit_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def random_ranking(num_blocks: int, uid: str) -> np.ndarray:
    return np.random.default_rng(protocol.derive_seed("random_ranker", uid)).permutation(num_blocks).astype(np.int64)


class RankerBank:
    """Every ranker over one store partition and regime. Inputs load on first use."""

    def __init__(self, store: Any, regime: str, *, block_emb: np.ndarray | None = None,
                 router_tau: float | None = None, mmap: bool | None = None) -> None:
        self.store = store
        self.regime = regime
        self.num_blocks = int(store.num_blocks)
        self.mmap = bool(store.mmap if mmap is None else mmap)
        self.block_emb = None if block_emb is None else _unit_rows(block_emb)
        self.router_tau = router_tau
        self._postings: Postings | None = None
        self._mean: np.ndarray | None = None
        self._topo: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._rrf_k = int(protocol.RANKERS["rrf_k"])

    # -------------------------------------------------------------- inputs
    @property
    def postings(self) -> Postings:
        if self._postings is None:
            directory = artifact_dir(self.store, self.regime)
            _load_checked(directory, self.store, self.regime)
            self._postings = Postings.load(directory, self.num_blocks, mmap=self.mmap)
        return self._postings

    @property
    def mean_inputs(self) -> np.ndarray:
        if self._mean is None:
            directory = artifact_dir(self.store, self.regime)
            _load_checked(directory, self.store, self.regime)
            self._mean = _unit_rows(np.load(directory / "mean_inputs.npy"))
        return self._mean

    @property
    def topo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z-scored block statistics, mean, std) with std 0 replaced by 1."""
        if self._topo is None:
            directory = artifact_dir(self.store, "topology")
            info = _load_checked(directory, self.store, None)
            stats = np.load(directory / "topo_stats.npy")
            mean = np.asarray(info["mean"], dtype=np.float64)
            std = np.asarray(info["std"], dtype=np.float64)
            std = np.where(std > 0, std, 1.0)
            self._topo = ((stats - mean) / std, mean, std)
        return self._topo

    # -------------------------------------------------------------- scores
    def selectivity(self, codes: np.ndarray) -> float:
        """Mean over query nodes of log10 of the node frequency of the code (fresh codes count as 1)."""
        df = self.store.sigma(self.regime).label_frequency(np.asarray(codes, dtype=np.int64))
        return float(np.mean(np.log10(np.maximum(df, 1)))) if df.shape[0] else 0.0

    def scores(self, name: str, q: QueryInputs) -> np.ndarray:
        if name == "learned":
            if self.block_emb is None or q.embedding is None:
                raise ValueError("the learned ranker needs block embeddings and a query embedding")
            return self.block_emb @ _unit_rows(np.asarray(q.embedding).reshape(-1))
        if name == "index":
            return self.postings.scores(q.codes, weighting="idf")
        if name == "index_count":
            return self.postings.scores(q.codes, weighting="count")
        if name == "mean_feature":
            if q.x is None:
                raise ValueError("mean_feature needs the query's encoder inputs")
            x = np.asarray(q.x, dtype=np.float64)
            return self.mean_inputs @ _unit_rows(x.mean(axis=0) if x.shape[0] else np.zeros(x.shape[1]))
        if name == "topo_feature":
            z, mean, std = self.topo
            num_types = z.shape[1] - len(TOPO_STATS)
            zq = (topo_stats(q.indptr, q.indices, q.types, num_types) - mean) / std
            return -np.linalg.norm(z - zq, axis=1)
        raise ValueError(f"{name!r} has no score vector")

    def rank(self, name: str, q: QueryInputs) -> tuple[np.ndarray, dict[str, Any]]:
        """A full block ranking and what the ranker decided on the way."""
        info: dict[str, Any] = {"ranker": name}
        if name == "random":
            return random_ranking(self.num_blocks, q.uid), info
        if name in ("learned", "index", "index_count", "mean_feature", "topo_feature"):
            return ranking_from_scores(self.scores(name, q)), info
        if name == "fused_learned_index":
            parts = [ranking_from_scores(self.scores("learned", q)), ranking_from_scores(self.scores("index", q))]
            return ranking_from_scores(rrf_scores(parts, self.num_blocks, self._rrf_k)), info
        if name == "fused_learned_mean":
            parts = [ranking_from_scores(self.scores("learned", q)), ranking_from_scores(self.scores("mean_feature", q))]
            return ranking_from_scores(rrf_scores(parts, self.num_blocks, self._rrf_k)), info
        if name == "router":
            if self.router_tau is None:
                raise ValueError("the router needs a fitted tau")
            sel = self.selectivity(q.codes)
            info.update({"selectivity": sel, "tau": float(self.router_tau)})
            if sel <= float(self.router_tau):
                info["route"] = "index"
                index, learned = self.scores("index", q), self.scores("learned", q)
                return np.lexsort((np.arange(self.num_blocks), -learned, -index)).astype(np.int64), info
            info["route"] = "fused_learned_index"
            return self.rank("fused_learned_index", q)[0], info
        raise ValueError(f"unknown ranker {name!r}")


def fit_router_tau(selectivity: Sequence[float], cost_index: Sequence[float],
                   cost_fused: Sequence[float]) -> tuple[float, dict[str, Any]]:
    """tau minimising the summed cost when queries with selectivity <= tau go to index.

    Candidates are the midpoints between consecutive distinct selectivities plus one
    value below and one above all of them; ties go to the median optimal candidate.
    """
    sel = np.asarray(selectivity, dtype=np.float64)
    ci = np.asarray(cost_index, dtype=np.float64)
    cf = np.asarray(cost_fused, dtype=np.float64)
    if sel.shape[0] == 0:
        raise ValueError("no validation queries")
    u = np.unique(sel)
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1.0]])
    costs = np.asarray([ci[sel <= t].sum() + cf[sel > t].sum() for t in cands])
    best = float(costs.min())
    optimal = np.flatnonzero(np.isclose(costs, best, rtol=0.0, atol=1e-9))
    pick = int(optimal[optimal.shape[0] // 2])
    return float(cands[pick]), {
        "queries": int(sel.shape[0]), "candidates": int(cands.shape[0]), "cost": best,
        "cost_all_index": float(ci.sum()), "cost_all_fused": float(cf.sum()),
        "optimal_interval": [float(cands[optimal[0]]), float(cands[optimal[-1]])],
        "routed_to_index": int((sel <= cands[pick]).sum()),
    }


# ----------------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------------
def coverage_cost(selector: Any, idx: np.ndarray, blk: np.ndarray, num_planted: int, num_blocks: int) -> int | None:
    """Smallest K whose prefix covers every planted node (FullCov), None if no prefix does.

    (idx, blk) is queries.cover_incidence(): block blk's candidate holds planted[idx].
    The prefix grows by doubling, so a lazy selector computes at most twice the order it needs.
    """
    idx = np.asarray(idx, dtype=np.int64)
    blk = np.asarray(blk, dtype=np.int64)
    never = np.iinfo(np.int64).max
    k = 1
    while True:
        order = np.asarray(selector.prefix(k), dtype=np.int64)
        pos = np.full(num_blocks, never, dtype=np.int64)
        pos[order] = np.arange(order.shape[0], dtype=np.int64)
        earliest = np.full(num_planted, never, dtype=np.int64)
        np.minimum.at(earliest, idx, pos[blk])
        if num_planted == 0 or (earliest < never).all():
            return int(earliest.max(initial=-1)) + 1
        if order.shape[0] < k or k >= num_blocks:
            return None
        k = min(2 * k, num_blocks)


def fullcov_at(cost: int | None, budgets: Sequence[int]) -> list[bool]:
    return [cost is not None and cost <= int(b) for b in budgets]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .store import GraphStore

    parser = argparse.ArgumentParser(description="Offline ranker inputs.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the offline ranker inputs of a store partition")
    b.add_argument("--root", required=True)
    b.add_argument("--dataset", required=True)
    b.add_argument("--partition", default=None)
    b.add_argument("--regimes", nargs="*", default=None)
    b.add_argument("--chunk-nodes", type=int, default=1 << 24)
    b.add_argument("--run-dir", default=None, help="record the build in a manifest there (offline_build.py, E17)")
    args = parser.parse_args(argv)
    if args.command == "build":
        from . import offline_build

        kwargs = {} if args.partition is None else {"partition": args.partition}
        store = GraphStore(args.root, args.dataset, mmap=True, **kwargs)
        regimes = args.regimes or [r for r in protocol.regimes_for(args.dataset)
                                   if (store.dir / "sigma" / r).exists()]
        inputs = {"store_meta": store.dir / "meta.json", "partition_info": Path(store.part.dir) / "partition.json",
                  **{f"sigma/{r}": store.dir / "sigma" / r for r in regimes}}
        with offline_build.build_run(args.run_dir, stage="rankers", store_dir=store.dir, dataset=args.dataset,
                                     config={"partition": Path(store.part.dir).name, "regimes": regimes,
                                             "chunk_nodes": args.chunk_nodes}, inputs=inputs, argv=argv) as written:
            report = build_artifacts(store, regimes, chunk_nodes=args.chunk_nodes)
            written.extend(artifact_dir(store, name) for name in (*regimes, "topology"))
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
