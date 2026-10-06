"""Fit the router threshold tau of a dataset on its validation master (protocol.RANKERS["router_tau"]).

    python -m scripts.canonical.fit_router fit --store-root STORES --dataset arxiv --queries-root QUERIES \\
        --models-root MODELS --out RUNS/router_fit/arxiv/router_fit.json
    python -m scripts.canonical.fit_router merge --inputs RUNS/router_fit/*/router_fit.json --out router_tau.json

The router (rankers.RankerBank.rank) sends a query to the index ranker when its
selectivity is at most tau and to fused_learned_index otherwise. The rule
(protocol.RANKERS["router_tau_fit"], implemented by rankers.fit_router_tau):
minimise the summed hybrid cost to coverage over the validation positives of
every regime of the dataset; candidates are the midpoints between consecutive
distinct selectivities and one value beyond each end; ties go to the median
optimal candidate. Per positive:

  selectivity  RankerBank.selectivity of the query's sigma codes
  cost_index   hybrid cost to coverage (blocks to FullCov) of the index ranker
  cost_fused   mean over the training seeds of the hybrid cost of fused_learned_index
               with the seed's main-arm encoder of the regime (R2-* regimes are
               served by the R2-512 encoder, protocol.TRAINED_REGIMES)

Costs come from retrieval_eval.query_retrieval, the function E4 reports with.
The router's index route breaks index ties by the learned score where
cost_index breaks them by block id; cost_router_realized reruns the router with
the fitted tau, so any difference is on record.

Inputs. QUERIES/<dataset>/validation/<master>/<regime>.npz for every regime of
protocol.regimes_for(dataset), where <master> is the validation master whose
role in protocol.SEEDS["validation_roles"] is "router_tau_and_rank_fusion"
(router_master()). A set of another split, master, dataset or regime is
refused. A smoke fit (a set built with max_per_cell or other sizes, other seeds,
another partition) is refused without --allow-smoke and is marked smoke; merge
refuses smoke fits without --allow-smoke.

Outputs. FILE (--out) in a directory that holds this fit only (other files, or
the manifest of a fit with other settings, are refused before anything is written):
  manifest.json          manifest.RunManifest, experiment "router_fit", split "validation"
  <stem>.queries.jsonl   one line per validation positive (QUERY_SCHEMA): uid, regime, family, size,
                         selectivity, cost_index, cost_fused (seed mean), cost_fused_by_seed {seed: int},
                         route ("index" or "fused_learned_index" at the fitted tau), cost_router (the
                         fit's modelled cost), cost_router_realized (seed mean) and _by_seed
  FILE, FIT_SCHEMA = "jigsaw-router-fit-v1":
    schema, dataset, split, master, tau (float), policy (protocol.SELECTION["policy"])
    fit           rankers.fit_router_tau info: queries, candidates, cost (the minimum), cost_all_index,
                  cost_all_fused, optimal_interval [lo, hi], routed_to_index
    per_regime    {regime: {queries, routed_to_index, cost_index, cost_fused, cost_router,
                  cost_router_realized}}, costs summed over the regime's positives
    seeds         the training seeds averaged in cost_fused
    models        {model_id: serving.model_identity()}
    embeddings    {model_id: {regime: sha256 of block_emb.npy}}
    query_sets    {regime: {path, sha256, master, positives}}
    budgets, partition, protocol_hash, smoke (bool), smoke_reasons [str]
merge writes {dataset: tau} (sorted) and prints the literal for
protocol.RANKERS["router_tau"]. Writing it into protocol.py changes
protocol_hash(), so merge compares the fits with each other, not with the
current protocol.
"""

from __future__ import annotations

# run_experiment first: it sets the BLAS thread variables before numpy loads.
from . import run_experiment as rx  # isort: skip

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_numpy, manifest, protocol, queries, rankers, retrieval_eval, serving
from .store import PRIMARY, GraphStore

FIT_SCHEMA = "jigsaw-router-fit-v1"
QUERY_SCHEMA = "jigsaw-router-fit-query-v1"
ROUTER_ROLE = "router_tau_and_rank_fusion"
SPLIT = "validation"
INDEX_ROUTE = "index"
FUSED_ROUTE = "fused_learned_index"


def router_master() -> str:
    """The validation master that fits the router (protocol.SEEDS["validation_roles"])."""
    masters = [m for m, role in protocol.SEEDS["validation_roles"].items() if role == ROUTER_ROLE]
    if len(masters) != 1:
        raise ValueError(f"protocol.SEEDS['validation_roles'] gives {len(masters)} masters the role {ROUTER_ROLE}")
    return str(masters[0])


def is_smoke_set(meta: Mapping[str, Any]) -> bool:
    """train_encoder.is_smoke_query_set, restated because train_encoder imports torch."""
    sizes = meta.get("sizes")
    return meta.get("max_per_cell") is not None or (sizes is not None and list(sizes) != list(protocol.QUERY_SIZES))


def query_set_path(root: str | Path, dataset: str, master: str, regime: str) -> Path:
    return Path(root) / dataset / SPLIT / master / f"{regime}.npz"


def load_router_sets(root: str | Path, dataset: str, master: str) -> dict[str, queries.QuerySet]:
    """The router master's validation set of every regime of the dataset; any other set is refused."""
    sets = {}
    for regime in protocol.regimes_for(dataset):
        path = query_set_path(root, dataset, master, regime)
        if not path.exists():
            raise SystemExit(f"no router validation set {path}")
        qs = queries.QuerySet.load(path)
        found = {"split": qs.meta.get("split"), "master": qs.meta.get("master"),
                 "dataset": qs.meta.get("dataset", dataset), "regime": qs.regime}
        wanted = {"split": SPLIT, "master": master, "dataset": dataset, "regime": regime}
        wrong = [f"{k} {found[k]!r}" for k in wanted if str(found[k]) != wanted[k]]
        if wrong:
            raise SystemExit(f"{path} is not the router validation set ({SPLIT} master {master} of "
                             f"{dataset}/{regime}): {', '.join(wrong)}")
        sets[regime] = qs
    return sets


def sidecar_path(out: str | Path) -> Path:
    out = Path(out)
    return out.with_name(out.stem + ".queries.jsonl")


def check_fit_dir(out: str | Path) -> None:
    """The fit's directory holds this fit only: its manifest hashes every file in it."""
    out = Path(out)
    own = {out.name, sidecar_path(out).name, manifest.MANIFEST_NAME}
    if out.parent.exists():
        strangers = sorted(p.name for p in out.parent.iterdir() if p.name not in own and not p.name.endswith(".tmp")
                           and re.fullmatch(r"manifest\.attempt\d+\.json", p.name) is None)
        if strangers:
            raise SystemExit(f"{out.parent} holds other files ({', '.join(strangers[:5])}); give the fit a "
                             "directory of its own")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def fit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset
    if dataset not in protocol.DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}")
    master = router_master()
    regimes = protocol.regimes_for(dataset)
    protocol_seeds = [int(s) for s in protocol.SEEDS["training"]]
    seeds = [int(s) for s in (args.seeds or protocol_seeds)]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("a seed is listed twice")
    sets = load_router_sets(args.queries_root, dataset, master)
    smoke = [f"{r}: a smoke query set (max_per_cell {qs.meta.get('max_per_cell')}, sizes {qs.meta.get('sizes')})"
             for r, qs in sets.items() if is_smoke_set(qs.meta)]
    if seeds != protocol_seeds:
        smoke.append(f"seeds {seeds}, not {protocol_seeds}")
    if args.partition != PRIMARY:
        smoke.append(f"partition {args.partition}")
    if smoke and not args.allow_smoke:
        raise SystemExit("a smoke fit needs --allow-smoke: " + "; ".join(smoke))
    positives = {r: rx.select_queries(sets[r].queries, kinds=["positive"]) for r in regimes}
    if not any(positives.values()):
        raise SystemExit("the router validation sets hold no positives")
    out = Path(args.out)
    check_fit_dir(out)
    policy = str(protocol.SELECTION["policy"])

    threads, limiter = rx.pin_threads(int(protocol.SERVING["threads"]))
    try:
        store = GraphStore(args.store_root, dataset, mmap=True, partition=args.partition)
        budgets = retrieval_eval.ladder_budgets(dataset, store.num_blocks, store.partition_name)
        base = {r: rankers.RankerBank(store, r) for r in regimes}
        models = {(r, s): retrieval_eval.load_model(store, r, args.models_root, serving.default_model_id(dataset, r, s),
                                                    build=args.build_embeddings)
                  for r in regimes for s in seeds}
        for bank in [*base.values(), *(m.bank for m in models.values())]:
            bank.postings  # the index and fused rankers read them; loaded and checked before any output
        identities = {m.load_id: m.identity for m in models.values()}
        embeddings: dict[str, dict[str, str]] = {}
        for (r, _), m in models.items():
            embeddings.setdefault(m.load_id, {})[r] = m.embeddings_sha256
        query_sets = {r: {"path": str(Path(sets[r].path).resolve()), "sha256": manifest.sha256_file(sets[r].path)["sha256"],
                          "master": str(sets[r].meta.get("master")), "positives": len(positives[r])} for r in regimes}
        config = {"schema": FIT_SCHEMA, "dataset": dataset, "master": master, "partition": args.partition,
                  "regimes": regimes, "seeds": seeds, "policy": policy, "budgets": budgets, "models": identities,
                  "embeddings": embeddings, "query_sets": query_sets, "smoke_reasons": smoke,
                  "threads": int(protocol.SERVING["threads"])}
        attempt = rx.check_resume(out.parent, config)
        inputs: dict[str, Path] = {"store_meta": store.dir / "meta.json",
                                   "partition": Path(store.part.dir) / "partition.json"}
        for r in regimes:
            inputs[f"queries/{r}"] = Path(sets[r].path)
            inputs[f"rankers/{r}"] = rankers.artifact_dir(store, r) / "info.json"
        for (r, _), m in models.items():
            inputs[f"model/{m.load_id}"] = m.model_dir / encoder_numpy.MODEL_NAME
            inputs[f"model/{m.load_id}/export"] = m.model_dir / retrieval_eval.EXPORT
            inputs[f"embeddings/{m.load_id}/{r}"] = m.embeddings_dir / "block_emb.npy"
        with manifest.RunManifest(out.parent, experiment="router_fit", config=config, split=SPLIT,
                                  inputs={k: v for k, v in inputs.items() if v.exists()}) as run:
            run.note(threads=threads, attempt=attempt)
            result = _fit(store, dataset, regimes, seeds, positives, base, models, budgets, policy)
            result.update({"schema": FIT_SCHEMA, "dataset": dataset, "split": SPLIT, "master": master,
                           "policy": policy, "seeds": seeds, "models": identities, "embeddings": embeddings,
                           "query_sets": {r: {k: v[k] for k in ("path", "sha256", "master", "positives")}
                                          for r, v in query_sets.items()},
                           "budgets": budgets, "partition": args.partition, "protocol_hash": protocol.protocol_hash(),
                           "smoke": bool(smoke), "smoke_reasons": smoke})
            rows = result.pop("rows")
            _atomic_write(sidecar_path(out), "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            _atomic_write(out, json.dumps(result, indent=2, sort_keys=True) + "\n")
            run.note(tau=result["tau"], queries=len(rows), fit_sha256=manifest.sha256_file(out)["sha256"])
        return result
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def _fit(store: GraphStore, dataset: str, regimes: Sequence[str], seeds: Sequence[int],
         positives: Mapping[str, Sequence[queries.Query]], base: Mapping[str, rankers.RankerBank],
         models: Mapping[tuple[str, int], retrieval_eval.LoadedModel], budgets: Sequence[int],
         policy: str) -> dict[str, Any]:
    def cost(regime: str, q: queries.Query, ranker: str, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is None:
            rec = retrieval_eval.query_retrieval(store, regime, q, bank=base[regime], ranker=ranker, policy=policy,
                                                 budgets=budgets)
        else:
            m = models[(regime, seed)]
            rec = retrieval_eval.query_retrieval(store, regime, q, bank=m.bank, ranker=ranker, policy=policy,
                                                 embed=m.embed, rel_table=m.rel_table, budgets=budgets)
        if rec.get("error") is not None:
            raise RuntimeError(f"{q.uid} ({ranker}, seed {seed}): {rec['error']}")
        return int(rec["cost"]), rec

    rows: list[dict[str, Any]] = []
    todo: list[tuple[str, queries.Query]] = []
    for regime in regimes:
        for q in positives[regime]:
            cost_index, rec = cost(regime, q, "index")
            fused = {s: cost(regime, q, FUSED_ROUTE, s)[0] for s in seeds}
            rows.append({"schema": QUERY_SCHEMA, "uid": q.uid, "regime": regime, "family": q.family,
                         "size": int(q.size), "selectivity": float(rec["selectivity"]), "cost_index": cost_index,
                         "cost_fused_by_seed": {str(s): c for s, c in fused.items()},
                         "cost_fused": float(np.mean(list(fused.values())))})
            todo.append((regime, q))
    sel = [row["selectivity"] for row in rows]
    tau, info = rankers.fit_router_tau(sel, [row["cost_index"] for row in rows], [row["cost_fused"] for row in rows])

    for m in models.values():
        m.bank.router_tau = tau
    per_regime = {r: {"queries": 0, "routed_to_index": 0, "cost_index": 0.0, "cost_fused": 0.0, "cost_router": 0.0,
                      "cost_router_realized": 0.0} for r in regimes}
    for row, (regime, q) in zip(rows, todo):
        route = INDEX_ROUTE if row["selectivity"] <= tau else FUSED_ROUTE
        realized = {}
        for s in seeds:
            realized[s], rec = cost(regime, q, "router", s)
            if rec["route"] != route:
                raise RuntimeError(f"{q.uid}: the router took {rec['route']}, the fit models {route}")
        row.update({"route": route, "cost_router": float(row["cost_index"] if route == INDEX_ROUTE else row["cost_fused"]),
                    "cost_router_realized_by_seed": {str(s): c for s, c in realized.items()},
                    "cost_router_realized": float(np.mean(list(realized.values())))})
        agg = per_regime[regime]
        agg["queries"] += 1
        agg["routed_to_index"] += int(route == INDEX_ROUTE)
        for key in ("cost_index", "cost_fused", "cost_router", "cost_router_realized"):
            agg[key] += float(row[key])
    return {"tau": float(tau), "fit": info, "per_regime": per_regime, "rows": rows}


# ----------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------
def merge(paths: Sequence[str | Path], *, allow_smoke: bool = False) -> dict[str, float]:
    """{dataset: tau} of per-dataset fits made under one protocol."""
    fits = []
    for path in paths:
        fitted = json.loads(Path(path).read_text(encoding="utf-8"))
        if fitted.get("schema") != FIT_SCHEMA:
            raise SystemExit(f"{path} is not a router fit ({FIT_SCHEMA})")
        fits.append((str(path), fitted))
    if not fits:
        raise SystemExit("no fits to merge")
    datasets = [f["dataset"] for _, f in fits]
    twice = sorted({d for d in datasets if datasets.count(d) > 1})
    if twice:
        raise SystemExit(f"datasets fit more than once: {twice}")
    hashes = sorted({str(f.get("protocol_hash")) for _, f in fits})
    if len(hashes) > 1:
        raise SystemExit(f"the fits were made under different protocols: {hashes}")
    master = router_master()
    other = [p for p, f in fits if f.get("master") != master or f.get("split") != SPLIT]
    if other:
        raise SystemExit(f"fits on another master than {SPLIT} {master}: {other}")
    smoke = [p for p, f in fits if f.get("smoke")]
    if smoke and not allow_smoke:
        raise SystemExit(f"smoke fits need --allow-smoke: {smoke}")
    return {f["dataset"]: float(f["tau"]) for _, f in sorted(fits, key=lambda item: item[1]["dataset"])}


def literal(taus: Mapping[str, float]) -> str:
    """The protocol.py line of the merged thresholds (repr keeps every float bit)."""
    body = ", ".join(f"{json.dumps(d)}: {float(t)!r}" for d, t in sorted(taus.items()))
    return f'RANKERS["router_tau"] = {{{body}}}'


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="fit tau of one dataset on the router validation master")
    f.add_argument("--store-root", required=True)
    f.add_argument("--dataset", required=True)
    f.add_argument("--partition", default=PRIMARY)
    f.add_argument("--queries-root", required=True, help="root of <dataset>/validation/<master>/<regime>.npz")
    f.add_argument("--models-root", default="models/canonical")
    f.add_argument("--out", required=True, help="the fit JSON; its directory holds this fit only")
    f.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the protocol's training seeds")
    f.add_argument("--build-embeddings", action="store_true", help="build missing block embeddings (serving.py)")
    f.add_argument("--allow-smoke", action="store_true", help="accept smoke query sets, seeds or partitions")
    m = sub.add_parser("merge", help="merge per-dataset fits into RANKERS['router_tau']")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--allow-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        result = fit(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    taus = merge(args.inputs, allow_smoke=args.allow_smoke)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, json.dumps(taus, indent=2, sort_keys=True) + "\n")
    fitted = json.loads(Path(args.inputs[0]).read_text(encoding="utf-8"))["protocol_hash"]
    if fitted != protocol.protocol_hash():
        print(f"note: fit under protocol {fitted}, current {protocol.protocol_hash()}", file=sys.stderr)
    print(literal(taus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
