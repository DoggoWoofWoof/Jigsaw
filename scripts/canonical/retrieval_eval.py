"""Retrieval evaluation: blocks to coverage and FullCov@K per ranker, policy and encoder (E4, E5, E6).

    python -m scripts.canonical.retrieval_eval run --experiment E4 --store-root STORES --dataset arxiv \\
        --queries QUERIES/arxiv/dev/20260607 --models-root MODELS --out RUNS/E4/arxiv-dev-20260607
    python -m scripts.canonical.retrieval_eval run --experiment E5 --matrix MATRIX.json ...
    python -m scripts.canonical.retrieval_eval run --rankers index learned --regimes R0 --limit 50 ...
    python -m scripts.canonical.retrieval_eval summarize --out RUNS/E4/arxiv-dev-20260607

Measures. For every positive query of one split master (the sm_* families too,
marked by their family) and every arm (regime, ranker, policy, encoder):

  cost        smallest K in 1..P whose candidate, the first K blocks of the
              policy's order plus their one-hop halo, holds every planted node:
              blocks to FullCov (P blocks in the partition)
  home_cost   smallest K whose first K blocks hold the home block of every
              planted node (no halo), so home_cost >= cost
  fullcov_at  cost <= K for each K of the ladder as Cascade builds it
              (ladder_budgets: protocol.ladder_for(dataset), rescaled on a
              sensitivity partition, clipped to P, P last; config "budgets")

Retrieval is the cascade's own: the same sigma codes, QueryInputs
(build_query_inputs is Cascade.query_inputs), RankerBank, block embeddings and
selector, so a ranking here is bit for bit the one the cascade solves on (the
tests check it). query_retrieval() is shared with fit_router.py, so the router
threshold is fit on the numbers E4 reports. Every order ends with every block,
so a cost of None cannot happen; if it does, the record carries an error and is
never dropped.

Plans (experiment_plan). E4: every ranker (rankers.RANKER_NAMES), policies
hybrid and topk, the main-arm encoder of each training seed for the rankers
that read one, regimes R0, R1 and R2-512 where the dataset has them. E5 / E6:
the learned ranker under both policies in R0 for the factorial / architecture
rows of a train_matrix JSON (--matrix) of the dataset, plus the main-arm
encoders of the same seeds as the reference. A row that duplicates an earlier
one (duplicate_of) is evaluated on its own export when that exists, else on
the model it duplicates, and its records name the weights used. Without
--experiment a dev run takes any --rankers and --regimes; with one, --regimes,
--rankers, --policies, --seeds and --models narrow the plan.

Sealed query sets (meta split test or reserve_*) run only as a whole named plan
(an E5/E6 matrix complete for every training seed), on the primary partition,
with the protocol's router threshold and no --limit. These checks run before
seal.authorize_sealed_run, so a refused run writes and logs nothing. Models,
block embeddings (--build-embeddings builds missing ones), banks and the router
threshold are assembled before the manifest, so a missing input writes nothing.
Running the same command again resumes: complete lines are kept, recorded
queries skipped, and a run with other settings is refused.

Files (a reducer reads them with load_records(); it needs no torch):
  OUT/manifest.json            manifest.RunManifest: "experiment" (E4, E5, E6, or "retrieval" for a
                               free-form run), "split", "protocol_hash", "config" (run_config(): plan,
                               models with identities and block-embedding hashes, budgets, router_tau,
                               query-set hashes, limit), "extra" (per-arm counts, errors)
  OUT/manifest.attempt<n>.json the manifests of earlier attempts of a resumed run
  OUT/records/<regime>/<arm>.jsonl  one JSON line per positive query, keys sorted
An arm is RANKER+POLICY for the rankers without an encoder (index+hybrid,
random+topk) and RANKER+POLICY+model=MODEL_ID for learned, fused_learned_index,
fused_learned_mean and router (learned+topk+model=arxiv-R0-graphsage-seed7301-main);
the file stem is the arm with any of \\/:*?"<>| replaced by "_" (model ids hold none).

Record schema RECORD_SCHEMA = "jigsaw-retrieval-record-v1"; null is JSON null:
  schema              str        RECORD_SCHEMA
  experiment          str|null   E4, E5 or E6; null for a free-form run
  dataset             str        protocol.DATASETS key
  partition           str        the store partition; "primary" (store.PRIMARY) for every E4/E5/E6 number
  split               str        the query set's meta split: dev, validation, test, reserve_a, reserve_b
  master              str        the query set's master tag
  regime              str        label regime of the queries and of retrieval: R0 R1 R2-512 R2-128 R2-32
  uid                 str        query uid {dataset}/{split}/{family}/{size}/{index:04d}; pairs the records
                                 of one query across arms, policies and training seeds
  family              str        positive family (protocol.POSITIVE_FAMILIES) or sm_any, sm_dense, sm_sparse
  size                int        nominal query size
  kind                str        "positive"
  query_index         int        position of the query in the run's selection (set order)
  arm                 str        the arm, as above
  ranker              str        one of rankers.RANKER_NAMES
  policy              str        "hybrid" (protocol.SELECTION["policy"]) or "topk" (its ablation)
  model_id            str|null   the encoder arm: {dataset}-{train_regime}-{train_arch}-seed{train_seed}-
                                 {train_arm} (train_encoder.model_id_for); null without an encoder
  evaluated_model_id  str|null   the weights loaded: model_id, or duplicate_of for a matrix row without
                                 its own export
  duplicate_of        str|null   the train_matrix row this model's row duplicates
  model_kind          str|null   main, factorial, architecture (train_matrix kinds) or given (--models)
  train_seed          int|null   training seed (protocol.SEEDS["training"])
  train_arm           str|null   "main", a factorial arm (grid_arm_name(), infonce_only, no_cvar, ...) or
                                 an architecture arm (arch_arm_name(), e.g. arch-graphsage-L6-res)
  train_regime        str|null   the regime the encoder was trained for (R2-* are served by R2-512)
  train_arch          str|null   encoder architecture
  num_blocks          int        P
  planted             int        planted nodes, one per query node
  cost                int|null   blocks to FullCov, 1..P; null only with error
  home_cost           int|null   blocks to home coverage, cost..P; null only with error
  fullcov_at          obj|null   {"K": cost <= K} for each K of config "budgets", keys decimal strings
  selectivity         float|null mean over query nodes of log10(max(1, df(sigma code))): the router's input
  route               str|null   router only: "index" (selectivity <= tau) or "fused_learned_index"
  tau                 float|null router only: the threshold used
  seconds             obj|null   {inputs, embed, rank, select}: single-thread wall clock of the codes and
                                 encoder inputs, the query encoder, the ranking, and the selector up to
                                 the last non-final budget; information only (latency is E2's)
  attempt             int        attempt of the run that wrote the record
  error               str        only on failure: an exception, or no prefix covering the witness
  traceback           str        only with an exception
A reducer pairs records by uid within a regime: PE5 compares rankers (seeds
averaged per uid, protocol.STATISTICS) under the hybrid policy
(protocol.PRIMARY_ENDPOINTS); PE6 compares train_arm values of one model_kind
against the main reference, per seed. Records with an error count against
their arm: they are uncovered at every K and stay in the FullCov denominator,
and they leave the cost statistics, where they are counted (reduce_retrieval
and summarize() alike).
"""

from __future__ import annotations

# run_experiment first: it sets the BLAS thread variables before numpy loads.
from . import run_experiment as rx  # isort: skip

import argparse
import json
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import encoder_numpy, features, manifest, protocol, queries, rankers, seal, serving
from .store import PRIMARY, GraphStore

RECORD_SCHEMA = "jigsaw-retrieval-record-v1"
EXPERIMENTS = ("E4", "E5", "E6")
E4_REGIMES = ("R0", "R1", "R2-512")
MATRIX_KINDS = {"E5": "factorial", "E6": "architecture"}
ABLATION_REGIME = "R0"  # the regime the factorial and architecture rows train and are evaluated in
MATRIX_KEYS = ("model_id", "kind", "dataset", "regime", "arch", "arm", "seed", "duplicate_of")
# train_encoder's names, restated because that module imports torch (the tests compare them)
EXPORT = "export.json"
MAIN_ARM = "main"


def policies() -> list[str]:
    """The selection policy and its ablation (protocol.SELECTION): hybrid, topk."""
    return [str(protocol.SELECTION["policy"]), str(protocol.SELECTION["ablation_policy"])]


def ladder_budgets(dataset: str, num_blocks: int, partition: str = PRIMARY) -> list[int]:
    """The ladder as Cascade.__post_init__ builds it: on a sensitivity partition of B blocks a step of K
    takes ceil(K * B / P) blocks (protocol.LADDER); clipped to B, with B last."""
    num_blocks = int(num_blocks)
    budgets = [int(k) for k in protocol.ladder_for(dataset)]
    primary = int(protocol.DATASETS[dataset]["partitions"])
    if partition != PRIMARY and num_blocks != primary:
        budgets = sorted({max(1, math.ceil(k * num_blocks / primary)) for k in budgets})
    budgets = [min(k, num_blocks) for k in budgets]
    if budgets[-1] != num_blocks:
        budgets.append(num_blocks)
    return budgets


# ----------------------------------------------------------------------------
# One query
# ----------------------------------------------------------------------------
def build_query_inputs(store: GraphStore, regime: str, q: queries.Query, codes: np.ndarray, *, ranker: str,
                       embed: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
                       rel_table: np.ndarray | None = None) -> tuple[rankers.QueryInputs, dict[str, float]]:
    """Cascade.query_inputs statement for statement, with the encoder call timed apart."""
    t0 = time.perf_counter()
    embed_seconds = 0.0
    qi = rankers.QueryInputs(q.uid, codes, np.asarray(q.payload["types"], dtype=np.int64), q.indptr, q.indices)
    if ranker in rankers.NEEDS_EMBEDDING or ranker == "mean_feature":
        inputs = features.query_graph_inputs(store, q, regime, codes, rel_table)
        qi.x = inputs["x"]
        if ranker in rankers.NEEDS_EMBEDDING:
            if embed is None:
                raise ValueError(f"{ranker} ranks with a query encoder; none was given")
            t1 = time.perf_counter()
            qi.embedding = np.asarray(embed(inputs), dtype=np.float64).reshape(-1)
            embed_seconds = time.perf_counter() - t1
    return qi, {"inputs": time.perf_counter() - t0 - embed_seconds, "embed": embed_seconds}


@dataclass
class Retrieval:
    codes: np.ndarray
    inputs: rankers.QueryInputs
    ranking: np.ndarray
    info: dict[str, Any]
    selector: Any
    seconds: dict[str, float]


def retrieve(store: GraphStore, regime: str, q: queries.Query, *, bank: rankers.RankerBank, ranker: str, policy: str,
             embed: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None, rel_table: np.ndarray | None = None,
             budgets: Sequence[int] = (), policy_params: Mapping[str, Any] | None = None) -> Retrieval:
    """The cascade's retrieval for one query: codes, inputs, ranking, and the order up to every
    non-final budget (the selection work Cascade.run does before its final step)."""
    t0 = time.perf_counter()
    codes = store.sigma(regime).query_codes(q.payload, store)
    code_seconds = time.perf_counter() - t0
    qi, seconds = build_query_inputs(store, regime, q, codes, ranker=ranker, embed=embed, rel_table=rel_table)
    t1 = time.perf_counter()
    ranking, info = bank.rank(ranker, qi)
    t2 = time.perf_counter()
    selector = rankers.make_selector(policy, ranking, store.part, **dict(policy_params or {}))
    num_blocks = int(store.num_blocks)
    for k in budgets:
        if int(k) < num_blocks:
            selector.prefix(int(k))
    t3 = time.perf_counter()
    return Retrieval(codes, qi, ranking, info, selector, {"inputs": code_seconds + seconds["inputs"],
                                                          "embed": seconds["embed"], "rank": t2 - t1,
                                                          "select": t3 - t2})


def query_retrieval(store: GraphStore, regime: str, q: queries.Query, *, bank: rankers.RankerBank, ranker: str,
                    policy: str, embed: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
                    rel_table: np.ndarray | None = None, budgets: Sequence[int],
                    policy_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Cost to coverage, home cost, FullCov@K and the ranker's decisions for one positive query."""
    if q.kind != "positive":
        raise ValueError(f"{q.uid} is a {q.kind} query; only a positive has a planted witness")
    budgets = [int(k) for k in budgets]
    r = retrieve(store, regime, q, bank=bank, ranker=ranker, policy=policy, embed=embed, rel_table=rel_table,
                 budgets=budgets, policy_params=policy_params)
    planted = np.asarray(q.planted, dtype=np.int64)  # query node j maps to planted[j]
    n = int(planted.shape[0])
    num_blocks = int(store.num_blocks)
    idx, blk = queries.cover_incidence(store, planted)
    cost = rankers.coverage_cost(r.selector, idx, blk, n, num_blocks)
    home = np.asarray(store.part.node_block[planted], dtype=np.int64)
    home_cost = rankers.coverage_cost(r.selector, np.arange(n, dtype=np.int64), home, n, num_blocks)
    rec: dict[str, Any] = {
        "uid": q.uid, "family": q.family, "size": int(q.size), "kind": q.kind, "regime": regime,
        "dataset": store.dataset, "ranker": ranker, "policy": policy, "num_blocks": num_blocks, "planted": n,
        "cost": cost, "home_cost": home_cost,
        "fullcov_at": {str(k): ok for k, ok in zip(budgets, rankers.fullcov_at(cost, budgets))},
        "selectivity": float(bank.selectivity(r.codes)), "route": r.info.get("route"), "tau": r.info.get("tau"),
        "seconds": r.seconds,
    }
    if cost is None or home_cost is None:
        rec["error"] = "no prefix of the order covers the planted witness"
    return rec


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def parse_model_id(model_id: str, dataset: str) -> dict[str, Any] | None:
    """{regime, arch, seed, arm} of `{dataset}-{regime}-{arch}-seed{seed}-{arm}` (serving.model_id_for), or None.

    Arms may hold hyphens (grid-cvar-barrier0.35-live64); architecture names hold none.
    """
    regimes = "|".join(re.escape(r) for r in sorted(protocol.LABEL_REGIMES, key=len, reverse=True))
    match = re.fullmatch(rf"{re.escape(dataset)}-({regimes})-([A-Za-z0-9_]+)-seed(\d+)-(.+)", str(model_id))
    if match is None:
        return None
    return {"regime": match.group(1), "arch": match.group(2), "seed": int(match.group(3)), "arm": match.group(4)}


@dataclass(frozen=True)
class ModelSpec:
    """An encoder under evaluation: model_id names the arm, load_id the weights evaluated."""

    model_id: str
    load_id: str
    train_seed: int | None
    train_arm: str | None
    kind: str  # main, factorial, architecture, or given (a --models id)
    duplicate_of: str | None = None
    train_regime: str | None = None
    train_arch: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "evaluated_model_id": self.load_id, "duplicate_of": self.duplicate_of,
                "model_kind": self.kind, "train_seed": self.train_seed, "train_arm": self.train_arm,
                "train_regime": self.train_regime, "train_arch": self.train_arch}


NO_MODEL = {"model_id": None, "evaluated_model_id": None, "duplicate_of": None, "model_kind": None,
            "train_seed": None, "train_arm": None, "train_regime": None, "train_arch": None}


def main_model(dataset: str, regime: str, seed: int) -> ModelSpec:
    model_id = serving.default_model_id(dataset, regime, int(seed))
    return ModelSpec(model_id, model_id, int(seed), MAIN_ARM, "main", None, protocol.TRAINED_REGIMES[regime],
                     str(protocol.DATASETS[dataset]["encoder"]))


def given_model(dataset: str, model_id: str) -> ModelSpec:
    parsed = parse_model_id(model_id, dataset) or {}
    return ModelSpec(model_id, model_id, parsed.get("seed"), parsed.get("arm"), "given", None, parsed.get("regime"),
                     parsed.get("arch"))


def matrix_model(row: Mapping[str, Any], models_root: str | Path) -> ModelSpec:
    """A train_matrix row; a duplicate is evaluated on the model it duplicates when its own export is missing."""
    model_id = str(row["model_id"])
    duplicate = row.get("duplicate_of") or None
    own = (Path(models_root) / model_id / EXPORT).exists()
    load_id = model_id if duplicate is None or own else str(duplicate)
    return ModelSpec(model_id, load_id, int(row["seed"]), str(row["arm"]), str(row["kind"]),
                     None if duplicate is None else str(duplicate), str(row["regime"]), str(row["arch"]))


def load_matrix(path: str | Path) -> list[dict[str, Any]]:
    """The rows of a train_matrix JSON (scripts/canonical/launch/train_matrix.py), checked."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit(f"{path} is not a train_matrix JSON list of rows")
    for row in rows:
        missing = [k for k in MATRIX_KEYS if k not in row]
        if missing:
            raise SystemExit(f"{path}: row {row.get('model_id')} lacks {missing}")
        parsed = parse_model_id(row["model_id"], str(row["dataset"]))
        if parsed is None or (parsed["regime"], parsed["arch"], parsed["seed"], parsed["arm"]) != (
                str(row["regime"]), str(row["arch"]), int(row["seed"]), str(row["arm"])):
            raise SystemExit(f"{path}: model id {row['model_id']} does not name the row's regime, arch, seed and arm")
    return rows


@dataclass
class LoadedModel:
    load_id: str
    model_dir: Path
    encoder: encoder_numpy.NumpyEncoder
    embed: serving.QueryEncoder
    rel_table: np.ndarray | None
    bank: rankers.RankerBank
    identity: dict[str, Any]
    embeddings_dir: Path
    embeddings_sha256: str
    built: bool = False


def load_model(store: GraphStore, regime: str, models_root: str | Path, load_id: str, *,
               router_tau: float | None = None, build: bool = False) -> LoadedModel:
    """Encoder, query encoder, relation table and ranker bank of one model, as serving.make_cascade builds them."""
    model_dir = Path(models_root) / load_id
    if not (model_dir / encoder_numpy.MODEL_NAME).exists():
        raise FileNotFoundError(f"no model {load_id}: {model_dir / encoder_numpy.MODEL_NAME} is missing")
    encoder = encoder_numpy.NumpyEncoder.load(model_dir)
    identity = serving.model_identity(encoder)
    if identity["model_id"] not in (None, load_id):
        raise ValueError(f"{model_dir} holds model {identity['model_id']}, not {load_id}")
    table = serving.check_model(store, regime, encoder)
    emb_dir = rankers.embeddings_dir(store, serving.model_key(encoder, model_dir), regime)
    built = False
    if not (emb_dir / "info.json").exists():
        if not build:
            raise FileNotFoundError(f"no block embeddings of {load_id} for {regime} in {emb_dir}; pass "
                                    "--build-embeddings or run python -m scripts.canonical.serving embed-blocks")
        serving.build_block_embeddings(store, regime, model_dir)
        built = True
    block_emb = serving.load_block_embeddings(store, regime, encoder, model_dir)
    bank = rankers.RankerBank(store, regime, block_emb=block_emb, router_tau=router_tau)
    return LoadedModel(load_id, model_dir, encoder, serving.QueryEncoder(encoder, needs_rel=table is not None), table,
                       bank, identity, emb_dir, manifest.sha256_file(emb_dir / "block_emb.npy")["sha256"], built)


# ----------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    ranker: str
    policy: str
    model: ModelSpec | None = None

    @property
    def name(self) -> str:
        text = f"{self.ranker}+{self.policy}"
        return text if self.model is None else f"{text}+model={self.model.model_id}"


@dataclass
class Plan:
    experiment: str | None
    regimes: list[str]
    rankers: list[str]
    policies: list[str]
    models: dict[str, list[ModelSpec]]  # per regime; read by the rankers in NEEDS_EMBEDDING

    @property
    def learned(self) -> bool:
        return any(r in rankers.NEEDS_EMBEDDING for r in self.rankers)

    @property
    def seeds(self) -> list[int]:
        return sorted({m.train_seed for specs in self.models.values() for m in specs if m.train_seed is not None})

    def arms(self, regime: str) -> list[Arm]:
        out = []
        for ranker in self.rankers:
            for policy in self.policies:
                if ranker in rankers.NEEDS_EMBEDDING:
                    out += [Arm(ranker, policy, m) for m in self.models.get(regime, [])]
                else:
                    out.append(Arm(ranker, policy))
        return out


def experiment_plan(name: str, dataset: str, *, matrix: Sequence[Mapping[str, Any]] | None = None,
                    models_root: str | Path | None = None) -> Plan:
    """Regimes, rankers, policies and models of a named experiment (E4, E5, E6)."""
    seeds = [int(s) for s in protocol.SEEDS["training"]]
    if name == "E4":
        regimes = [r for r in E4_REGIMES if r in protocol.regimes_for(dataset)]
        return Plan(name, regimes, list(rankers.RANKER_NAMES), policies(),
                    {r: [main_model(dataset, r, s) for s in seeds] for r in regimes})
    if name in MATRIX_KINDS:
        kind = MATRIX_KINDS[name]
        if matrix is None or models_root is None:
            raise SystemExit(f"{name} evaluates the {kind} rows of a training matrix: pass --matrix and --models-root")
        rows = [row for row in matrix if row.get("kind") == kind and row.get("dataset") == dataset]
        if not rows:
            raise SystemExit(f"the matrix has no {kind} rows for {dataset}")
        trained = protocol.TRAINED_REGIMES[ABLATION_REGIME]
        other = sorted({str(row["regime"]) for row in rows} - {trained})
        if other:
            raise SystemExit(f"{name} evaluates {trained} encoders; the matrix rows train {other}")
        ids = [str(row["model_id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit("a model id is listed twice in the matrix")
        reference = [main_model(dataset, ABLATION_REGIME, s) for s in sorted({int(row["seed"]) for row in rows})]
        clash = sorted({m.model_id for m in reference} & set(ids))
        if clash:
            raise SystemExit(f"matrix rows {clash} carry the id of a main-arm reference model")
        return Plan(name, [ABLATION_REGIME], ["learned"], policies(),
                    {ABLATION_REGIME: reference + [matrix_model(row, models_root) for row in rows]})
    raise ValueError(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS)}")


def protocol_arm_count(dataset: str, kind: str) -> int:
    """Arms per training seed that the protocol registers for E5 (factorial) or E6 (architecture)."""
    if kind == "factorial":
        spec = protocol.FACTORIAL.get(dataset) or {}
        factors = spec.get("factors") or {}
        grid = math.prod(len(v) for v in factors.values()) if factors else 0
        return grid + len(spec.get("extra_arms") or {}) + len(spec.get("arms") or {})
    variants = protocol.ARCHITECTURE_ABLATION.get(dataset)
    return len(variants) if isinstance(variants, list) else 0


def matrix_gaps(matrix: Sequence[Mapping[str, Any]], dataset: str, kind: str) -> list[str]:
    """What a matrix lacks against the protocol: every arm of the kind for every training seed."""
    expected = protocol_arm_count(dataset, kind)
    if expected == 0:
        return [f"the protocol registers no {kind} arms for {dataset}"]
    arms: dict[int, set[str]] = {int(s): set() for s in protocol.SEEDS["training"]}
    strangers = set()
    for row in matrix:
        if row.get("kind") == kind and row.get("dataset") == dataset:
            if int(row["seed"]) in arms:
                arms[int(row["seed"])].add(str(row["arm"]))
            else:
                strangers.add(int(row["seed"]))
    problems = [f"seeds {sorted(strangers)} are not training seeds"] if strangers else []
    found = {s: len(a) for s, a in arms.items()}
    if any(n != expected for n in found.values()):
        problems.append(f"an incomplete {kind} matrix: {expected} arms per training seed, found {found}")
    return problems


def _narrow(values: Sequence[Any], wanted: Iterable[Any] | None, what: str) -> list[Any]:
    if wanted is None:
        return list(values)
    wanted = list(dict.fromkeys(wanted))
    unknown = [w for w in wanted if w not in values]
    if unknown:
        raise SystemExit(f"{what} outside the plan: {unknown}")
    return [v for v in values if v in wanted]


def resolve_plan(args: argparse.Namespace) -> Plan:
    dataset = args.dataset
    if dataset not in protocol.DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}")
    seeds = None if args.seeds is None else [int(s) for s in args.seeds]
    if args.experiment:
        matrix = load_matrix(args.matrix) if args.matrix else None
        if matrix is not None and args.experiment not in MATRIX_KINDS:
            raise SystemExit("--matrix belongs to E5 and E6")
        full = experiment_plan(args.experiment, dataset, matrix=matrix, models_root=args.models_root)
        regimes = _narrow(full.regimes, args.regimes, "regimes")
        names = _narrow(full.rankers, args.rankers, "rankers")
        chosen = _narrow(full.policies, args.policies, "policies")
        _narrow(full.seeds, seeds, "seeds")
        _narrow(sorted({m.model_id for specs in full.models.values() for m in specs}), args.models, "models")
        models = {r: [m for m in full.models[r] if (seeds is None or m.train_seed in seeds)
                      and (args.models is None or m.model_id in args.models)] for r in regimes}
        plan = Plan(full.experiment, regimes, names, chosen, models)
    else:
        if args.matrix:
            raise SystemExit("--matrix belongs to E5 and E6: name the --experiment")
        if not args.rankers or not args.regimes:
            raise SystemExit("without --experiment give --rankers and --regimes")
        regimes = list(dict.fromkeys(args.regimes))
        unknown = [r for r in regimes if r not in protocol.regimes_for(dataset)]
        if unknown:
            raise SystemExit(f"{dataset} has no regimes {unknown}")
        names = list(dict.fromkeys(args.rankers))
        chosen = list(dict.fromkeys(args.policies or policies()))
        learned = any(r in rankers.NEEDS_EMBEDDING for r in names)
        if not learned and (args.models or seeds):
            raise SystemExit("--models and --seeds choose encoders, and none of the rankers reads one")
        if args.models and seeds:
            raise SystemExit("--models names the encoders; drop --seeds")
        if args.models:
            specs = [given_model(dataset, m) for m in dict.fromkeys(args.models)]
            models = {r: list(specs) for r in regimes}
        else:
            train = list(dict.fromkeys(seeds or [int(s) for s in protocol.SEEDS["training"]]))
            models = {r: [main_model(dataset, r, s) for s in train] for r in regimes}
        plan = Plan(None, regimes, names, chosen, models)
    if not plan.learned:
        plan.models = {r: [] for r in plan.regimes}
    elif not any(plan.models.values()):
        raise SystemExit("the filters leave no encoder for the learned rankers")
    return plan


def check_sealed_plan(args: argparse.Namespace, plan: Plan) -> None:
    """A sealed run is a whole pre-registered experiment: its regimes, rankers, policies, seeds and models,
    the protocol's router threshold, the primary partition and every positive."""
    problems = [flag for flag, value in (("--limit", args.limit), ("--router-tau", args.router_tau))
                if value is not None]
    if args.partition != PRIMARY:
        problems.append(f"--partition {args.partition}")
    if not plan.experiment:
        problems.append("a free-form plan (name an --experiment)")
    else:
        matrix = load_matrix(args.matrix) if args.matrix else None
        full = experiment_plan(plan.experiment, args.dataset, matrix=matrix, models_root=args.models_root)
        for what in ("regimes", "rankers", "policies", "seeds"):
            left = [v for v in getattr(full, what) if v not in getattr(plan, what)]
            if left:
                problems.append(f"{what} other than the plan's ({left} left out)")
        ids = {r: sorted(m.model_id for m in full.models[r]) for r in full.regimes}
        if {r: sorted(m.model_id for m in plan.models.get(r, [])) for r in full.regimes} != ids:
            problems.append("models other than the plan's")
        if plan.experiment in MATRIX_KINDS and matrix is not None:
            problems += matrix_gaps(matrix, args.dataset, MATRIX_KINDS[plan.experiment])
    if problems:
        raise SystemExit(f"a sealed run is a pre-registered experiment; not allowed: {'; '.join(problems)}")


def router_threshold(args: argparse.Namespace, plan: Plan) -> float | None:
    """The router's tau: the frozen protocol value, or --router-tau on a dev run; None without the router."""
    if "router" not in plan.rankers:
        return None
    tau = protocol.router_tau_for(args.dataset) if args.router_tau is None else float(args.router_tau)
    if tau is None:
        raise SystemExit(f"no router threshold for {args.dataset}: fit it with python -m scripts.canonical.fit_router "
                         "(the validation master of role router_tau_and_rank_fusion) or pass --router-tau on a dev run")
    return tau


# ----------------------------------------------------------------------------
# Assembly, records and the run
# ----------------------------------------------------------------------------
@dataclass
class Assembly:
    base: dict[str, rankers.RankerBank]  # regime -> bank of the rankers without an encoder
    models: dict[tuple[str, str], LoadedModel]  # (regime, load_id) -> model

    def serving(self, regime: str, arm: Arm) -> tuple[rankers.RankerBank, Any, np.ndarray | None]:
        if arm.model is None:
            return self.base[regime], None, None
        model = self.models[(regime, arm.model.load_id)]
        return model.bank, model.embed, model.rel_table


def assemble(store: GraphStore, plan: Plan, models_root: str | Path, *, router_tau: float | None,
             build: bool = False) -> Assembly:
    """Every bank and model of the plan, with the inputs each ranker reads loaded before the first query."""
    base: dict[str, rankers.RankerBank] = {}
    models: dict[tuple[str, str], LoadedModel] = {}
    for regime in plan.regimes:
        arms = plan.arms(regime)
        if any(a.model is None for a in arms):
            base[regime] = rankers.RankerBank(store, regime)
        for arm in arms:
            if arm.model is not None and (regime, arm.model.load_id) not in models:
                models[(regime, arm.model.load_id)] = load_model(store, regime, models_root, arm.model.load_id,
                                                                 router_tau=router_tau, build=build)
    assembly = Assembly(base, models)
    for regime in plan.regimes:
        for arm in plan.arms(regime):
            bank = assembly.serving(regime, arm)[0]
            for name in rx.RANKER_INPUTS.get(arm.ranker, ()):
                getattr(bank, name)  # loads, and checks against the store, the offline ranker inputs
    return assembly


def record_name(arm: Arm) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", arm.name) + ".jsonl"


def evaluate(store: GraphStore, regime: str, q: queries.Query, arm: Arm, assembly: Assembly,
             budgets: Sequence[int]) -> dict[str, Any]:
    """The record of one (query, arm) without the run's fields; an exception becomes an error record."""
    bank, embed, table = assembly.serving(regime, arm)
    try:
        rec = query_retrieval(store, regime, q, bank=bank, ranker=arm.ranker, policy=arm.policy, embed=embed,
                              rel_table=table, budgets=budgets)
    except Exception as exc:  # a bug: recorded and counted, the run goes on
        rec = {"uid": q.uid, "family": q.family, "size": int(q.size), "kind": q.kind, "regime": regime,
               "dataset": store.dataset, "ranker": arm.ranker, "policy": arm.policy,
               "num_blocks": int(store.num_blocks), "planted": int(np.asarray(q.planted).shape[0]), "cost": None,
               "home_cost": None, "fullcov_at": None, "selectivity": None, "route": None, "tau": None,
               "seconds": None, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
    rec.update(arm.model.describe() if arm.model is not None else NO_MODEL)
    rec["arm"] = arm.name
    return rec


def run_config(args: argparse.Namespace, plan: Plan, sets: dict[str, queries.QuerySet],
               selected: dict[str, list[queries.Query]], assembly: Assembly, budgets: list[int],
               router_tau: float | None) -> dict[str, Any]:
    """Every setting that changes the records; a resumed run must match it exactly."""
    models = {}
    for regime in plan.regimes:
        models[regime] = [{**m.describe(), "identity": assembly.models[(regime, m.load_id)].identity,
                           "embeddings_sha256": assembly.models[(regime, m.load_id)].embeddings_sha256}
                          for m in plan.models[regime]]
    return {
        "schema": RECORD_SCHEMA, "experiment": plan.experiment, "dataset": args.dataset, "partition": args.partition,
        "regimes": plan.regimes, "rankers": plan.rankers, "policies": plan.policies, "seeds": plan.seeds,
        "arms": {r: [a.name for a in plan.arms(r)] for r in plan.regimes},
        "models": models, "router_tau": router_tau, "budgets": budgets, "limit": args.limit,
        "query_sets": {r: {"path": str(Path(sets[r].path).resolve()),
                           "sha256": manifest.sha256_file(sets[r].path)["sha256"],
                           "split": sets[r].meta.get("split"), "master": sets[r].meta.get("master"),
                           "positives": len(selected[r])} for r in plan.regimes},
        "threads": int(protocol.SERVING["threads"]),
    }


def run_inputs(args: argparse.Namespace, plan: Plan, store: GraphStore, sets: dict[str, queries.QuerySet],
               assembly: Assembly) -> dict[str, Path]:
    inputs: dict[str, Path] = {"store_meta": store.dir / "meta.json",
                               "partition": Path(store.part.dir) / "partition.json",
                               "rankers/topology": rankers.artifact_dir(store, "topology") / "info.json"}
    if args.matrix:
        inputs["matrix"] = Path(args.matrix)
    for regime in plan.regimes:
        inputs[f"queries/{regime}"] = Path(sets[regime].path)
        inputs[f"rankers/{regime}"] = rankers.artifact_dir(store, regime) / "info.json"
    for (regime, load_id), model in assembly.models.items():
        inputs[f"model/{load_id}"] = model.model_dir / encoder_numpy.MODEL_NAME
        inputs[f"model/{load_id}/export"] = model.model_dir / EXPORT
        inputs[f"embeddings/{load_id}/{regime}"] = model.embeddings_dir / "block_emb.npy"
        inputs[f"embeddings/{load_id}/{regime}/info"] = model.embeddings_dir / "info.json"
    return {k: v for k, v in inputs.items() if Path(v).exists()}


def run_regime(store: GraphStore, regime: str, plan: Plan, assembly: Assembly, qs: list[queries.Query],
               budgets: list[int], out: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    arms = plan.arms(regime)
    files = [rx.RecordFile(out / "records" / regime / record_name(arm)) for arm in arms]
    counts = {arm.name: {"run": 0, "recorded_before": len(f.done), "errors": 0} for arm, f in zip(arms, files)}
    try:
        for position, q in enumerate(qs):
            # arms in an order rotated by the query's position, as run_experiment does
            for j in rx.arm_order(len(arms), position):
                arm, f = arms[j], files[j]
                if q.uid in f.done:
                    continue
                rec = evaluate(store, regime, q, arm, assembly, budgets)
                rec.update({"schema": RECORD_SCHEMA, "query_index": position, **context})
                f.write(rec)
                counts[arm.name]["run"] += 1
                counts[arm.name]["errors"] += int("error" in rec)
    finally:
        for f in files:
            f.close()
    return {"positives": len(qs), "arms": counts,
            "recovered_bytes": {arm.name: f.recovered_bytes for arm, f in zip(arms, files) if f.recovered_bytes}}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit needs N >= 1")
    plan = resolve_plan(args)
    sets = rx.load_sets(args, plan)
    split = str(next(iter(sets.values())).meta.get("split", "dev"))
    masters = {str(qs.meta.get("master")) for qs in sets.values()}
    if len(masters) > 1:
        raise SystemExit(f"the query sets come from different masters: {sorted(masters)}")
    sealed = split in queries.TEST_SPLITS
    if sealed:
        check_sealed_plan(args, plan)  # before the gate, so a refused run leaves no unseal entry
    router_tau = router_threshold(args, plan)
    if sealed:
        seal.authorize_sealed_run(unseal=args.unseal, reason=args.reason or f"retrieval {plan.experiment}")
    selected = {r: rx.select_queries(sets[r].queries, kinds=["positive"], limit=args.limit) for r in plan.regimes}

    threads, limiter = rx.pin_threads(int(protocol.SERVING["threads"]))
    try:
        store = GraphStore(args.store_root, args.dataset, mmap=True, partition=args.partition)
        budgets = ladder_budgets(args.dataset, store.num_blocks, store.partition_name)
        # everything is assembled before the manifest, so a missing model or embedding writes nothing
        assembly = assemble(store, plan, args.models_root, router_tau=router_tau, build=args.build_embeddings)
        out = Path(args.out)
        config = run_config(args, plan, sets, selected, assembly, budgets, router_tau)
        attempt = rx.check_resume(out, config)
        context = {"experiment": plan.experiment, "split": split, "master": masters.pop(), "attempt": attempt,
                   "partition": store.partition_name}
        summary: dict[str, Any] = {"attempt": attempt, "regimes": {}}
        with manifest.RunManifest(out, experiment=plan.experiment or "retrieval", config=config, split=split,
                                  inputs=run_inputs(args, plan, store, sets, assembly)) as run:
            run.note(threads=threads, attempt=attempt,
                     embeddings_built=sorted(f"{k[1]}/{k[0]}" for k, m in assembly.models.items() if m.built))
            for regime in plan.regimes:
                summary["regimes"][regime] = run_regime(store, regime, plan, assembly, selected[regime], budgets,
                                                        out, context)
            summary["torch_imported"] = "torch" in sys.modules
            run.note(**summary)
        return summary
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


# ----------------------------------------------------------------------------
# Reading a run
# ----------------------------------------------------------------------------
RecordKey = tuple[str, str, str, str | None]  # (regime, ranker, policy, model_id)


def load_records(out: str | Path) -> dict[RecordKey, dict[str, dict[str, Any]]]:
    """Records of a run by (regime, ranker, policy, model_id): {uid: record}. The reducer's entry point."""
    groups: dict[RecordKey, dict[str, dict[str, Any]]] = {}
    for path in sorted((Path(out) / "records").glob("*/*.jsonl")):
        for rec in rx.read_records(path):
            if rec.get("schema") != RECORD_SCHEMA:
                raise ValueError(f"{path}: a record of schema {rec.get('schema')!r}, not {RECORD_SCHEMA}")
            key = (rec["regime"], rec["ranker"], rec["policy"], rec.get("model_id"))
            group = groups.setdefault(key, {})
            if rec["uid"] in group:
                raise ValueError(f"{path}: {rec['uid']} is recorded twice for {key}")
            group[rec["uid"]] = rec
    return groups


def _stat(fn: Callable[[np.ndarray], Any], values: np.ndarray) -> float | None:
    return float(fn(values)) if values.shape[0] else None


def summarize(out: str | Path) -> dict[str, dict[str, Any]]:
    """Per REGIME/ARM: n, errors, mean and median cost and home cost over the records without error,
    and FullCov@K rates over every record (an error is uncovered). Tests and intervals are the reducer's."""
    result = {}
    for (regime, ranker, policy, model_id), recs in load_records(out).items():
        ok = [r for r in recs.values() if r.get("error") is None]
        cost = np.asarray([r["cost"] for r in ok], dtype=np.float64)
        home = np.asarray([r["home_cost"] for r in ok], dtype=np.float64)
        budgets = sorted({k for r in ok for k in r["fullcov_at"]}, key=int)
        name = f"{regime}/{ranker}+{policy}" + ("" if model_id is None else f"+model={model_id}")
        result[name] = {
            "regime": regime, "ranker": ranker, "policy": policy, "model_id": model_id,
            "n": len(recs), "errors": len(recs) - len(ok),
            "mean_cost": _stat(np.mean, cost), "median_cost": _stat(np.median, cost),
            "mean_home_cost": _stat(np.mean, home), "median_home_cost": _stat(np.median, home),
            "fullcov_rate": {k: sum(bool(r["fullcov_at"][k]) for r in ok) / len(recs) for k in budgets},
        }
    return result


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="evaluate retrieval on the positives of one split master")
    p.add_argument("--store-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default=PRIMARY)
    p.add_argument("--queries", required=True, help="directory of <regime>.npz query sets of one split master")
    p.add_argument("--models-root", default="models/canonical")
    p.add_argument("--out", required=True)
    p.add_argument("--experiment", choices=list(EXPERIMENTS))
    p.add_argument("--regimes", nargs="+", choices=list(protocol.LABEL_REGIMES))
    p.add_argument("--rankers", nargs="+", choices=list(rankers.RANKER_NAMES))
    p.add_argument("--policies", nargs="+", choices=policies())
    p.add_argument("--seeds", nargs="+", type=int, help="training seeds of the encoders")
    p.add_argument("--models", nargs="+", help="model ids: the encoders of a dev run, or a subset of a plan's")
    p.add_argument("--matrix", default=None, help="train_matrix JSON (E5, E6)")
    p.add_argument("--router-tau", type=float, default=None, help="dev runs before the freeze: the router threshold")
    p.add_argument("--build-embeddings", action="store_true", help="build missing block embeddings (serving.py)")
    p.add_argument("--limit", type=int, default=None, help="dev runs only: the first N positives of each regime")
    p.add_argument("--unseal", action="store_true")
    p.add_argument("--reason", default="")
    s = sub.add_parser("summarize", help="n, mean and median cost and FullCov@K rates of a run's records")
    s.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        print(json.dumps(summarize(args.out), indent=2, sort_keys=True))
        return 0
    summary = execute(args)
    print(json.dumps(summary, indent=2, sort_keys=True, default=rx._json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
