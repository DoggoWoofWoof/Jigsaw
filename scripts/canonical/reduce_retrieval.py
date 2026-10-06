"""Retrieval reducer: E4, E5 and E6 retrieval records and the E6 layerwise analysis into numbers.json.

    python -m scripts.canonical.reduce_retrieval --e4 RUNS/E4/arxiv RUNS/E4/mag --e5 RUNS/E5/arxiv \\
        --e6 RUNS/E6/arxiv --layerwise RUNS/E6/layerwise/arxiv-R0 --out NUMBERS/retrieval \\
        [--allow-dev] [--by-family]

Reads run directories and writes OUT/numbers.json and OUT/figures/<FIGURE_ID>.csv through
numbers_json.Numbers, for build_paper.py. Imports no torch. Prints a JSON summary (out, split,
frozen, protocol_hash, sources, counts of numbers, tables and figures, warnings) on stdout.

Inputs
  --e4, --e5, --e6 DIR ...  retrieval_eval.py run directories of that experiment, read with
                            retrieval_eval.load_records. Directories of one experiment, dataset and
                            partition (one per regime, say) are merged; an arm in two of them is refused.
  --layerwise DIR ...       oversmoothing.py run directories (experiment E6_layerwise), read with
                            oversmoothing.read_run and summarised with oversmoothing.summary_rows.
  --allow-dev               accept the dev and validation splits (build_paper still marks them DEV).
  --by-family               E4 also reports each query family on its own (SUB family-FAMILY).
  --pattern-audit DIR ...   pattern_audit.py output directories (E20): E4 also reports the planted
                            positives that some main encoder of the regime drew in training (SUB seen) and
                            the others (SUB novel), descriptive keys, tables and figures only.
A flag may be repeated; its directories add up.

Refused before anything is written (SystemExit):
  - a manifest whose experiment is not its flag's (E4, E5, E6, E6_layerwise; a free-form "retrieval"
    run is never reduced), whose status is not "complete", or whose outputs do not verify
    (manifest.verify_manifest: a changed, missing or unrecorded file);
  - a record of another schema or a uid recorded twice for one arm (load_records); a record whose
    experiment, split, dataset, partition or arm is not its run's, that is not a positive, whose
    family is neither planted nor sm (below), or whose num_blocks is not the last budget; a record
    without an error whose cost is not in 1..P or whose fullcov_at does not hold exactly the run's
    budgets (config "budgets") with fullcov_at[K] = (cost <= K); records that do not match the arms
    the manifest planned (config "arms");
  - retrieval inputs of more than one split, and inputs (layerwise included) of more than one
    protocol hash;
  - the dev and validation splits without --allow-dev, and any split but dev, validation, test,
    reserve_a and reserve_b (a layerwise run has split "none" and joins any);
  - arms of one regime evaluated on different queries (uid sets);
  - arms paired or pooled over different training-seed sets: the encoder rankers of one E4 regime
    and policy, and every E5/E6 arm against main, must have the same seeds (a missing seed breaks the
    pairing);
  - an E4 encoder ranker whose model is not the main arm; an E5/E6 record of another ranker than
    learned, or of another model_kind than main and the experiment's (factorial, architecture); a
    run without the main arm; a duplicate row whose duplicated model the run lacks, or whose records
    differ from that model's although evaluated on its weights; two layerwise models with one label;
  - --out inside an input directory, and a directory given twice;
  - a pattern audit whose manifest is not a complete, verified pattern_audit run, whose file is not a
    complete jigsaw-pattern-audit-v1 audit of a known dataset, of another split or protocol hash than
    the runs, or that audits a dataset and regime twice; on a test split, an audit of other encoders
    than the main encoders of every training seed, or of a smoke run (reducer.py's rule); a seen query
    outside the audited set or seen by an encoder the regime does not list; an E4 regime it audits
    whose runs used another query set (the manifest's sha256 of input queries/<regime>), hold a query
    outside the audited set, or rank with an encoder the audit did not regenerate;
  - protocol.STATISTICS other than STATISTICS_IMPLEMENTED, the rules below.

Output. numbers.json has split = the retrieval inputs' split ("none" when only layerwise runs are
given), protocol_hash = the inputs' hash (refused when it is not the code's), frozen = that hash
equals protocol.FROZEN_PROTOCOL_HASH and every input's manifest has protocol_frozen true (it was run
under the frozen protocol; a warning names an input that was not). Every input directory is a
source RUN_ID = EXPERIMENT.SCOPE, or EXPERIMENT.SCOPE.run<i> (i = 1, 2, ... in the sorted order of
the paths as given) when several share it; its path is the directory as given, manifest_sha256 the
SHA-256 of its manifest.json. Each number lists the sources whose records it reads. A figure CSV
already in OUT/figures that this reduction does not write is left alone and named in a warning.

Segments of keys and ids:
  SCOPE  the dataset (arxiv); DATASET-part-PARTITION for a partition other than "primary"
  SUB    planted: queries of protocol.POSITIVE_FAMILIES (the pre-registered positives, the population of
         the power analysis); sm: the standard-SM suite (families sm_any, sm_dense, sm_sparse), which
         is never pooled with planted and is no pre-registered endpoint. With --by-family, E4 also
         reports family-FAMILY for each family present (descriptive keys, tables and figures only).
         With --pattern-audit, E4 also reports seen and novel: the planted positives isomorphic to a
         training query of some audited main encoder of the regime, and the others (descriptive; every
         number of these subsets also lists the audit, source E20.<dataset>).
  REG    label regime (R0, R1, R2-512)
  POL    policy: hybrid or topk
  RNK    ranker (rankers.RANKER_NAMES)
  ARM    a train_arm (main, grid-mean-barrier0.35-live64, no_cvar, arch-gin-L6-res, ...)
  K      a budget of the run's ladder (config "budgets"; the last is P, every block)
  MODEL  a layerwise model: its train_arm, plus "+knockout" for a residual knockout
Every character outside [A-Za-z0-9_-] of a segment becomes "_" (grid-mean-barrier0_35-live64,
main_knockout); two names that clash that way are refused.

Statistics (stats.py; protocol.STATISTICS must equal STATISTICS_IMPLEMENTED):
  unit          the query (uid). An encoder ranker (rankers.NEEDS_EMBEDDING) and every E5/E6 arm is
                evaluated once per training seed; each outcome is averaged per uid over the seeds
                (stats.seed_average) before anything else, so n counts queries, never seeds. The
                spread over seeds is reported apart (.seed<S> keys, .seeds tables).
  covered       FullCov at K of a record: cost <= K. A record with an error counts as not covered
                and stays in n.
  cost          blocks to FullCov, 1..P. A record with an error has none: its query leaves the cost
                statistics of that arm (of every seed, for a seed-averaged arm) and is counted.
  rate CI       without an encoder: Wilson 95% (Numbers.rate). Seed-averaged: 95% percentile
                interval of stats.hierarchical_bootstrap_diff against an all-zero table (queries and
                seeds resampled, N_HIER resamples); its point is the mean of the seed averages.
  mean CI       without an encoder, and one seed alone: 95% percentile bootstrap of the mean over
                queries (stats.paired_bootstrap_ci of the values, N_BOOT); seed-averaged: hierarchical.
  median CI     stats.quantile_with_ci of the (seed-averaged) values, queries resampled, N_HIER.
  paired        both sides averaged per uid over their seeds, then paired by uid.
  cost ratio    stats.paired_log_ratio of the paired costs: exp(mean over queries of
                log(cost_A / cost_B)), the geometric-mean ratio; 95% percentile bootstrap interval
                (N_BOOT); two-sided sign-flip randomization p (100,000 flips, inside stats). Queries
                with an error on either side are left out and counted (.cost.excluded).
  FullCov diff  stats.paired_compare of the seed-averaged covered indicators: 100 x mean(A - B) in pp,
                95% percentile bootstrap interval (N_BOOT), sign-flip p (N_PERM flips).
  Holm          stats.holm over the p values of one table (its "p" column; a duplicate row has none).
  resampling    N_BOOT = 20,000, N_PERM = 100,000, N_HIER = 5,000 (the stats.py defaults), seed 0.
Values are stored unrounded; "digits" only sets the printed precision. Every number but the text key
carries n.

Keys (unit in brackets):

E4 (--e4). For each SCOPE, SUB, REG, POL, RNK; P4 = E4.SCOPE.SUB.REG.POL.RNK:
  P4.n                          [count]  queries of the subset (n = the same)
  P4.errors                     [count]  records with an error, over every seed; n = records
  P4.seeds                      [count]  training seeds averaged (encoder rankers only)
  P4.fullcov.K<K>               [pct]    share of queries covered at K (seed-averaged), rate CI,
                                         n = queries; every K of the ladder, P included
  P4.cost.mean                  [number] mean over queries of the (seed-averaged) cost, mean CI,
                                         n = queries with a cost
  P4.cost.median                [number] median of the same values, median CI, n
  P4.cost.gm_ratio_p1_vs_index  [ratio]  exp(mean over paired queries of log((cost + 1) / (cost_index + 1)))
                                         against the index ranker of the same SCOPE, SUB, REG and POL,
                                         cost-ratio CI, n = paired queries; not for index itself; its p
                                         is not reported (PE5 holds the tests)
  P4.seed<S>.fullcov.K<K>       [pct]    training seed S alone: share covered, Wilson CI, n = queries
  P4.seed<S>.cost.mean          [number] training seed S alone: mean cost, mean CI, n = queries without
                                         an error in seed S (encoder rankers only, as the line above)

PE5 (--e4, policy hybrid, SUB planted and sm). For each SCOPE, SUB, REG and comparison A_vs_B, A one
of learned, fused_learned_index, fused_learned_mean, router and B one of index, fused_learned_mean
(Mean-RRF), A != B (seven comparisons); Q5 = PE5.SCOPE.SUB.REG.A_vs_B:
  Q5.n                          [count]  queries compared
  Q5.cost.gm_ratio              [ratio]  cost ratio A / B, CI, n = paired queries
  Q5.cost.excluded              [count]  queries left out of the ratio (an error on either side)
  Q5.cost.p, Q5.cost.p_holm     [pvalue] its sign-flip p; Holm within table PE5.SCOPE.SUB.cost
  Q5.fullcov.K<K1>.diff         [pp]     FullCov diff A - B at the first budget K1, CI, n
  Q5.fullcov.K<K1>.p, .p_holm   [pvalue] its sign-flip p; Holm within table PE5.SCOPE.SUB.fullcov
A comparison whose ranker a run lacks is left out with a warning.

PE6 (--e5, SUB planted and sm): the main arm against the control, the factorial arm whose resolved
loss is the main recipe with coverage_aggregation "mean" and nothing else changed (protocol.FACTORIAL
and TRAINING["loss"], as train_encoder resolves them: arxiv grid-mean-barrier0.35-live64, mag
no_cvar). For each SCOPE, SUB, REG and POL; Q6 = PE6.SCOPE.SUB.REG.POL:
  PE6.SCOPE.control_arm         [text]   the control's train_arm
  Q6.n                          [count]  queries compared
  Q6.cost.gm_ratio              [ratio]  cost ratio main / control, CI, n
  Q6.cost.excluded              [count]  as Q5
  Q6.cost.p, Q6.cost.p_holm     [pvalue] Holm within table PE6.SCOPE.SUB.cost
  Q6.fullcov.K<K>.diff          [pp]     FullCov diff main - control at every K but P, CI, n
  Q6.fullcov.K<K>.p, .p_holm    [pvalue] Holm within table PE6.SCOPE.SUB.fullcov

E5 (--e5) and E6 (--e6), EX = E5 or E6, SUB planted and sm. For each SCOPE, SUB, REG, POL and ARM
(main included):
  EX.SCOPE.SUB.REG.POL.arm.ARM.*   the P4 keys but gm_ratio_p1_vs_index (.n, .errors, .seeds,
                                   .fullcov.K<K>, .cost.mean, .cost.median, .seed<S>.*). A duplicate row
                                   (duplicate_of set) evaluated on the weights it duplicates has none,
                                   its records being that model's; one evaluated on its own export has
                                   them, while its table rows still report the model it duplicates.
and, for each ARM but main and the duplicate rows, ARM against main at the same seeds;
QX = EX.SCOPE.SUB.REG.POL.vs_main.ARM:
  QX.n                          [count]  queries compared
  QX.cost.gm_ratio              [ratio]  cost ratio ARM / main, CI, n
  QX.cost.excluded              [count]  as Q5
  QX.cost.p, QX.cost.p_holm     [pvalue] Holm within table EX.SCOPE.SUB.cost
  QX.fullcov.K<K>.diff          [pp]     FullCov diff ARM - main at every K but P, CI, n
  QX.fullcov.K<K>.p, .p_holm    [pvalue] Holm within table EX.SCOPE.SUB.fullcov
E5 arms are the factorial rows (grid-*, infonce_only, full, no_*); E6 arms are the architecture,
depth and residual rows (arch-ARCH-L<layers>-res|nores), each against the main architecture.

E6_layerwise (--layerwise). For each SCOPE, REG (the analysed regime) and MODEL;
QL = E6_layerwise.SCOPE.REG.MODEL:
  QL.seeds                          [count]  training seeds of the model (n = the same)
  QL.layer-<LAYER>.<METRIC>.mean    [number] mean over training seeds, n = seeds
  QL.layer-<LAYER>.<METRIC>.sd      [number] sample SD over seeds (ddof 1), n = seeds; two seeds or more
LAYER is input, proj, 1..L or readout; METRIC each of oversmoothing.metric_names(). A null metric
(node metrics at readout, a pairwise metric over one row) has no key.

Tables. Cells refer to the keys above; "n/a" marks a missing key, "-" a cell that does not apply.
  E4.SCOPE.SUB.REG.POL.fullcov  Ranker | n | Seeds | Errors | K=<K> ...    FullCov@K with CI
  E4.SCOPE.SUB.REG.POL.cost     Ranker | n | Errors | Mean cost | Median cost | GM ratio (cost+1) vs index
  E4.SCOPE.SUB.REG.POL.seeds    Ranker | Seed | K=<K> ... | Mean cost    encoder rankers, one row per seed
  PE5.SCOPE.SUB.cost            Regime | Comparison | n | Mean cost A | Mean cost B | GM cost ratio A/B | p |
                                p (Holm) | FullCov@K1 A-B (pp)
  PE5.SCOPE.SUB.fullcov         Regime | Comparison | n | FullCov@K1 A-B (pp) | p | p (Holm)
  PE6.SCOPE.SUB.cost            Regime | Policy | n | Mean cost main | Mean cost control |
                                GM cost ratio main/control | p | p (Holm)
  PE6.SCOPE.SUB.fullcov         Regime | Policy | K | n | FullCov main | FullCov control |
                                main - control (pp) | p | p (Holm)
  EX.SCOPE.SUB.REG.POL.arms     Arm | n | Seeds | Errors | K=<K> ... | Mean cost | Median cost
  EX.SCOPE.SUB.REG.POL.seeds    Arm | Seed | K=<K> ... | Mean cost    one row per arm and seed
  EX.SCOPE.SUB.cost             Regime | Policy | Arm | n | Mean cost | Mean cost main |
                                GM cost ratio arm/main | p | p (Holm)
  EX.SCOPE.SUB.fullcov          Regime | Policy | Arm | K | n | FullCov | FullCov main | arm - main (pp) |
                                p | p (Holm)
  E6_layerwise.SCOPE.REG.MODEL  Layer | Width | <METRIC> mean | <METRIC> SD ...
Rows follow rankers.RANKER_NAMES, the protocol's arm order (main first) and the ladder. A duplicate
row is labelled "ARM (duplicate of TARGET)", its cells refer to TARGET's keys, against main it reads
"same model", and it is not in a .seeds table; the notes name the weights its records came from.

Figures (curve: series, x, y, lo, hi):
  E4.SCOPE.SUB.REG.POL.fullcov_curve   series RNK, x K (log axis), y FullCov@K in %, lo/hi its CI
  EX.SCOPE.SUB.REG.POL.fullcov_curve   series ARM (duplicate rows left out), as above
  E6_layerwise.SCOPE.REG.METRIC        series MODEL, x the layer (input -1, proj 0, message-passing
                                       layer k at k, readout L+1), y the mean over seeds, lo/hi
                                       empty; null values left out
"""

from __future__ import annotations

# retrieval_eval first: through run_experiment it sets the BLAS thread variables before numpy loads.
from . import retrieval_eval as rev  # isort: skip

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import numpy as np

from . import manifest, oversmoothing, protocol, queries, rankers, stats
from . import numbers_json as nj
from .build_paper import series_key
from .store import PRIMARY

N_BOOT = 20_000  # percentile bootstrap of means and paired differences (STATISTICS paired_interval)
N_PERM = 100_000  # sign flips of paired_compare and paired_log_ratio (STATISTICS paired_test)
N_HIER = 5_000  # hierarchical bootstrap and median resamples: the stats.py defaults
SEED = 0  # STATISTICS bootstrap_seed

# The rules this module implements; a protocol that states others is refused.
STATISTICS_IMPLEMENTED: dict[str, Any] = {
    "unit": "query",
    "replicates": "training seeds, averaged per query before tests",
    "rate_interval": "wilson_95",
    "paired_test": "sign_flip_randomization_100k",
    "paired_interval": "percentile_bootstrap_20k",
    "multiplicity": "holm_within_table",
    "noninferiority_margin_pp": 1.0,
    "cost_summary": "geometric_mean_ratio_with_bootstrap_ci",
    "alpha": 0.05,
    "bootstrap_seed": 0,
}

RETRIEVAL_EXPERIMENTS = ("E4", "E5", "E6")
LAYERWISE = oversmoothing.EXPERIMENT  # "E6_layerwise"
DEV_SPLITS = ("dev", "validation")
PLANTED, SM = "planted", "sm"
SEEN, NOVEL = "seen", "novel"
COMPARED_SUBSETS = (PLANTED, SM)
FAMILY_PREFIX = "family-"
SM_FAMILIES = tuple(sorted(queries.SM_FAMILIES.values()))
PE5_RANKERS = ("learned", "fused_learned_index", "fused_learned_mean", "router")
PE5_REFERENCES = ("index", "fused_learned_mean")
PE6_CONTROL = {"coverage_aggregation": "mean"}
ALL = "all"  # the one replicate of a ranker without an encoder
NA, DASH, SAME = "n/a", "-", "same model"
MODEL_FIELDS = ("model_id", "evaluated_model_id", "duplicate_of", "model_kind", "train_seed", "train_arm",
                "train_regime", "train_arch")
LAYERWISE_DIGITS = {"mean_average_distance": 3, "mean_pairwise_cosine": 3, "block_separation": 3,
                    "effective_rank": 1, "block_effective_rank": 1, "zero_rows": 1}

# pattern_audit.py's names, restated because it imports torch through train_encoder (the tests compare them)
AUDIT_SCHEMA = "jigsaw-pattern-audit-v1"
AUDIT_EXPERIMENT = "pattern_audit"
AUDIT_FILE = "pattern_audit.json"

# train_encoder's arm names, restated because that module imports torch (the tests compare them)
FACTOR_LABELS = {"coverage_aggregation": "", "topk_barrier_weight": "barrier", "live_positives": "live"}


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


# ----------------------------------------------------------------------------
# Protocol arms
# ----------------------------------------------------------------------------
def factorial_arms(dataset: str) -> dict[str, dict[str, Any]]:
    """The E5 arms of a dataset in train_encoder.arms() order: name -> loss overrides."""
    spec = protocol.FACTORIAL.get(dataset) or {}
    factors = spec.get("factors") or {}
    out: dict[str, dict[str, Any]] = {}
    keys = list(factors)
    if keys:
        for combo in itertools.product(*(factors[k] for k in keys)):
            assignment = dict(zip(keys, combo))
            out[grid_arm_name(assignment)] = assignment
    for group in ("extra_arms", "arms"):
        for name, overrides in (spec.get(group) or {}).items():
            out[name] = dict(overrides)
    return out


def architecture_arms(dataset: str) -> list[str]:
    variants = protocol.ARCHITECTURE_ABLATION.get(dataset)
    return [arch_arm_name(v) for v in variants] if isinstance(variants, list) else []


def protocol_arm_order(dataset: str, kind: str) -> list[str]:
    return list(factorial_arms(dataset)) if kind == "factorial" else architecture_arms(dataset)


def _per_dataset(value: Any, dataset: str) -> Any:
    return value.get(dataset) if isinstance(value, Mapping) else value


def effective_loss(dataset: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """TRAINING["loss"] with per-dataset entries resolved and an arm's overrides applied (loss_config)."""
    out = {k: _per_dataset(v, dataset) for k, v in protocol.TRAINING["loss"].items()}
    out.update({k: _per_dataset(v, dataset) for k, v in overrides.items()})
    return out


def pe6_control(dataset: str) -> str | None:
    """The factorial arm that is the main recipe with mean coverage aggregation and nothing else changed."""
    main = effective_loss(dataset, {})
    found = []
    for name, overrides in factorial_arms(dataset).items():
        loss = effective_loss(dataset, overrides)
        changed = {k for k in set(main) | set(loss) if main.get(k) != loss.get(k)}
        if changed == set(PE6_CONTROL) and all(loss[k] == v for k, v in PE6_CONTROL.items()):
            found.append(name)
    if len(found) > 1:
        refuse(f"{dataset}: several factorial arms qualify as the PE6 control: {found}")
    return found[0] if found else None


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def refuse(message: str) -> NoReturn:
    raise SystemExit(message)


def segment(text: Any) -> str:
    """A key segment: every character outside [A-Za-z0-9_-] becomes _."""
    out = _UNSAFE.sub("_", str(text))
    if not out:
        refuse(f"an empty key segment for {text!r}")
    return out


def scope_name(dataset: str, partition: str) -> str:
    return segment(dataset) if partition == PRIMARY else f"{segment(dataset)}-part-{segment(partition)}"


def subset_of(family: str) -> str | None:
    if family in protocol.POSITIVE_FAMILIES:
        return PLANTED
    if family in SM_FAMILIES:
        return SM
    return None


def ordered(values: Iterable[str], order: Sequence[str]) -> list[str]:
    """values in the given order, then the others sorted."""
    values = set(values)
    return [v for v in order if v in values] + sorted(values - set(order))


def pct(x: float) -> float:
    return min(100.0, max(0.0, 100.0 * float(x)))


def pp(x: float) -> float:
    return min(100.0, max(-100.0, 100.0 * float(x)))


def union(*groups: Iterable[str]) -> list[str]:
    return sorted({s for g in groups for s in g})


def ref(nums: nj.Numbers, key: str, *, ci: bool = False) -> Any:
    """A table cell for `key`: its value (with its interval), or n/a when there is no such number."""
    entry = nums.numbers.get(key)
    if entry is None:
        return NA
    return {"key": key, "ci": True} if ci and entry["ci"] is not None else {"key": key}


@dataclass(frozen=True)
class Later:
    """A cell whose key is written after the row is built (the p values of a table)."""

    key: str
    ci: bool = False


def resolve(nums: nj.Numbers, rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[ref(nums, c.key, ci=c.ci) if isinstance(c, Later) else c for c in row] for row in rows]


def check_statistics() -> None:
    if protocol.STATISTICS != STATISTICS_IMPLEMENTED:
        changed = sorted(k for k in set(protocol.STATISTICS) | set(STATISTICS_IMPLEMENTED)
                         if protocol.STATISTICS.get(k) != STATISTICS_IMPLEMENTED.get(k))
        refuse(f"protocol.STATISTICS states other rules than this reducer implements ({changed}); update "
               "reduce_retrieval.py first")


def arm_name(key: tuple) -> str:
    """retrieval_eval's arm string of a load_records key (regime, ranker, policy, model_id)."""
    _, ranker, policy, model_id = key
    return f"{ranker}+{policy}" + ("" if model_id is None else f"+model={model_id}")


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
@dataclass
class RetrievalRun:
    given: str
    experiment: str
    split: str
    protocol_hash: str
    dataset: str
    partition: str
    budgets: list[int]
    manifest_sha256: str
    protocol_frozen: bool
    groups: dict[tuple, dict[str, dict[str, Any]]]
    run_id: str = ""
    query_sha: dict[str, str | None] = field(default_factory=dict)  # regime -> sha256 of input queries/<regime>

    @property
    def scope(self) -> str:
        return scope_name(self.dataset, self.partition)


@dataclass
class AuditRun:
    """A pattern_audit.py output: per regime the audited query set, its uids, the seen uids and the encoders."""

    given: str
    dataset: str
    split: str
    protocol_hash: str
    manifest_sha256: str
    protocol_frozen: bool
    regimes: dict[str, dict[str, Any]]
    experiment: str = "E20"
    run_id: str = ""

    @property
    def scope(self) -> str:
        return segment(self.dataset)


@dataclass
class LayerwiseRun:
    given: str
    split: str
    protocol_hash: str
    dataset: str
    partition: str
    manifest_sha256: str
    protocol_frozen: bool
    records: list[dict[str, Any]]
    experiment: str = LAYERWISE
    run_id: str = ""

    @property
    def scope(self) -> str:
        return scope_name(self.dataset, self.partition)


def read_manifest(given: str) -> tuple[dict[str, Any], str]:
    path = Path(given) / manifest.MANIFEST_NAME
    if not path.is_file():
        refuse(f"{given}: no {manifest.MANIFEST_NAME}")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        refuse(f"{given}: unreadable {manifest.MANIFEST_NAME} ({exc})")
    if not isinstance(info, dict):
        refuse(f"{given}: {manifest.MANIFEST_NAME} is not an object")
    return info, nj.file_sha256(path)


def check_manifest(given: str, info: Mapping[str, Any], experiment: str) -> None:
    if info.get("experiment") != experiment:
        refuse(f"{given}: a run of experiment {info.get('experiment')!r}, given as {experiment}")
    if info.get("status") != "complete":
        refuse(f"{given}: manifest status {info.get('status')!r}; only a complete run is reduced")
    problems = manifest.verify_manifest(given)
    if problems:
        refuse(f"{given}: the run does not verify against its manifest: {problems[:5]}")
    if not isinstance(info.get("protocol_hash"), str) or not info["protocol_hash"]:
        refuse(f"{given}: the manifest has no protocol_hash")
    if not isinstance(info.get("split"), str) or not info["split"]:
        refuse(f"{given}: the manifest has no split")


def read_retrieval_run(given: str, experiment: str) -> RetrievalRun:
    info, sha = read_manifest(given)
    check_manifest(given, info, experiment)
    try:
        groups = rev.load_records(given)
    except (ValueError, KeyError, TypeError) as exc:
        refuse(f"{given}: {exc}")
    if not groups:
        refuse(f"{given}: no records")
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    try:
        budgets = [int(k) for k in config.get("budgets") or []]
    except (TypeError, ValueError):
        refuse(f"{given}: config budgets {config.get('budgets')!r} are not integers")
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] < 1:
        refuse(f"{given}: config budgets {config.get('budgets')!r} are not an increasing ladder")
    first = next(iter(next(iter(groups.values())).values()))
    dataset, partition = str(first.get("dataset")), str(first.get("partition"))
    if config.get("dataset") not in (None, dataset):
        refuse(f"{given}: the records are of {dataset}, the manifest names {config.get('dataset')}")
    hashed = {k.split("/", 1)[1]: v for k, v in (info.get("inputs") or {}).items() if k.startswith("queries/")}
    run = RetrievalRun(given=str(given), experiment=experiment, split=str(info["split"]),
                       protocol_hash=str(info["protocol_hash"]), dataset=dataset, partition=partition,
                       budgets=budgets, manifest_sha256=sha, protocol_frozen=info.get("protocol_frozen") is True,
                       groups=groups, query_sha={r: v.get("sha256") if isinstance(v, dict) else None
                                                 for r, v in hashed.items()})
    check_records(run, config)
    return run


def check_records(run: RetrievalRun, config: Mapping[str, Any]) -> None:
    budgets = run.budgets
    last = budgets[-1]
    names = {str(k) for k in budgets}
    planned = config.get("arms")
    if isinstance(planned, Mapping):
        want = {str(r): {str(a) for a in arms} for r, arms in planned.items()}
        found: dict[str, set[str]] = {}
        for key in run.groups:
            found.setdefault(str(key[0]), set()).add(arm_name(key))
        if found != want:
            missing = sorted(f"{r}/{a}" for r in want for a in want[r] - found.get(r, set()))
            extra = sorted(f"{r}/{a}" for r in found for a in found[r] - want.get(r, set()))
            refuse(f"{run.given}: the records do not match the planned arms (missing {missing[:5]}, "
                   f"unplanned {extra[:5]})")
    for key, recs in run.groups.items():
        where = f"{run.given}: {key[0]}/{arm_name(key)}"
        for uid, rec in recs.items():
            for name, want_value in (("experiment", run.experiment), ("split", run.split),
                                     ("dataset", run.dataset), ("partition", run.partition),
                                     ("kind", "positive"), ("arm", arm_name(key))):
                if rec.get(name) != want_value:
                    refuse(f"{where}: {uid} has {name} {rec.get(name)!r}, expected {want_value!r}")
            if subset_of(str(rec.get("family"))) is None:
                refuse(f"{where}: {uid} has family {rec.get('family')!r}, neither a planted positive family "
                       f"{protocol.POSITIVE_FAMILIES} nor a standard-SM family {list(SM_FAMILIES)}")
            if rec.get("num_blocks") != last:
                refuse(f"{where}: {uid} has {rec.get('num_blocks')} blocks, but the ladder ends at {last}")
            if rec.get("error") is not None:
                continue  # a failure whatever else it holds: not covered, no cost
            cost = rec.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, int) or not 1 <= cost <= last:
                refuse(f"{where}: {uid} has cost {cost!r} without an error; a cost is 1..{last}")
            fullcov = rec.get("fullcov_at")
            if not isinstance(fullcov, dict) or set(fullcov) != names:
                refuse(f"{where}: {uid} has fullcov_at budgets {sorted(fullcov or {}, key=str)}, the run's are "
                       f"{budgets}")
            if any(bool(fullcov[str(k)]) != (cost <= k) for k in budgets):
                refuse(f"{where}: {uid} has fullcov_at {fullcov}, which disagrees with cost {cost}")


def read_layerwise_run(given: str) -> LayerwiseRun:
    info, sha = read_manifest(given)
    check_manifest(given, info, LAYERWISE)
    try:
        records = oversmoothing.read_run(given)
    except (ValueError, KeyError, TypeError) as exc:
        refuse(f"{given}: {exc}")
    places = sorted({(str(r["dataset"]), str(r["partition"])) for r in records})
    if len(places) != 1:
        refuse(f"{given}: records of several datasets or partitions {places}")
    stale = sorted({str(r["model_id"]) for r in records if r["protocol_hash"] != info["protocol_hash"]})
    if stale:
        refuse(f"{given}: records {stale[:3]} were analysed under another protocol hash than the manifest's")
    return LayerwiseRun(given=str(given), split=str(info["split"]), protocol_hash=str(info["protocol_hash"]),
                        dataset=places[0][0], partition=places[0][1], manifest_sha256=sha,
                        protocol_frozen=info.get("protocol_frozen") is True, records=records)


def read_audit(given: str) -> AuditRun:
    info, sha = read_manifest(given)
    check_manifest(given, info, AUDIT_EXPERIMENT)
    path = Path(given) / AUDIT_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        refuse(f"{given}: unreadable {AUDIT_FILE} ({exc})")
    if not isinstance(doc, dict) or doc.get("schema") != AUDIT_SCHEMA or doc.get("complete") is not True:
        refuse(f"{given}: {AUDIT_FILE} is not a complete {AUDIT_SCHEMA} audit")
    if doc.get("split") != info["split"] or doc.get("protocol_hash") != info["protocol_hash"]:
        refuse(f"{given}: {AUDIT_FILE} and its manifest name different splits or protocol hashes")
    dataset = doc.get("dataset")
    if dataset not in protocol.DATASETS:
        refuse(f"{given}: audits unknown dataset {dataset!r}")
    described = doc.get("encoders") or {}
    regimes: dict[str, dict[str, Any]] = {}
    for regime, entry in sorted((doc.get("regimes") or {}).items()):
        where = f"{given} {regime}"
        encoders = set(entry.get("encoders") or [])
        if encoders - set(described):
            refuse(f"{where}: an encoder of the regime has no entry in encoders")
        if info["split"] in queries.TEST_SPLITS:  # as reducer.py, so both reducers split the same queries
            want = {rev.main_model(dataset, regime, s).model_id for s in protocol.SEEDS["training"]}
            if encoders != want or any(described[m].get("smoke") for m in encoders):
                refuse(f"{where}: audits encoders {sorted(encoders)}; a test split needs the main encoders of "
                       f"every training seed, not smoke runs ({sorted(want)})")
        uids, seen = set(entry.get("uids") or []), dict(entry.get("seen") or {})
        for uid, per in seen.items():
            if uid not in uids or not per or set(per) - encoders:
                refuse(f"{where}: seen query {uid} is not in the set or names another encoder")
        regimes[str(regime)] = {"sha256": (entry.get("query_set") or {}).get("sha256"), "uids": uids,
                                "seen": set(seen), "encoders": encoders}
    return AuditRun(str(given), str(dataset), str(info["split"]), str(info["protocol_hash"]), sha,
                    info.get("protocol_frozen") is True, regimes)


def audits_by_regime(audits: Sequence[AuditRun], split: str) -> dict[tuple[str, str], AuditRun]:
    out: dict[tuple[str, str], AuditRun] = {}
    for audit in audits:
        if audit.split != split:
            refuse(f"{audit.given}: an audit of split {audit.split!r}, the retrieval inputs' is {split!r}")
        for regime in audit.regimes:
            if (audit.dataset, regime) in out:
                refuse(f"{audit.given}: {audit.dataset} {regime} is also audited by "
                       f"{out[(audit.dataset, regime)].given}")
            out[(audit.dataset, regime)] = audit
    return out


def check_out(out: str, inputs: Sequence[str]) -> None:
    target = Path(out).resolve()
    seen: dict[Path, str] = {}
    for given in inputs:
        d = Path(given).resolve()
        if d in seen:
            refuse(f"{given} is given twice (also as {seen[d]})")
        seen[d] = given
        if target == d or d in target.parents:
            refuse(f"--out {out} lies inside the input {given}; the run would no longer verify")


def check_split(retrieval: Sequence[RetrievalRun], layerwise: Sequence[LayerwiseRun], allow_dev: bool) -> str:
    for run in layerwise:
        if run.split != oversmoothing.SPLIT:
            refuse(f"{run.given}: a layerwise run of split {run.split!r}, expected {oversmoothing.SPLIT!r}")
    splits = sorted({run.split for run in retrieval})
    if len(splits) > 1:
        refuse(f"the retrieval inputs come from different splits {splits}; reduce each split on its own")
    if not splits:
        return oversmoothing.SPLIT
    split = splits[0]
    if split not in (*DEV_SPLITS, *queries.TEST_SPLITS):
        refuse(f"unknown split {split!r}")
    if split in DEV_SPLITS and not allow_dev:
        refuse(f"split {split!r} holds development numbers; pass --allow-dev to reduce it")
    return split


def check_hash(runs: Sequence[Any]) -> str:
    hashes = sorted({run.protocol_hash for run in runs})
    if len(hashes) > 1:
        refuse(f"the inputs ran under different protocol hashes {[h[:12] for h in hashes]}")
    return hashes[0]


def assign_run_ids(runs: Sequence[Any]) -> None:
    by: dict[tuple[str, str], list[Any]] = {}
    for run in runs:
        by.setdefault((run.experiment, run.scope), []).append(run)
    for (experiment, scope), members in by.items():
        members.sort(key=lambda r: r.given)
        for i, run in enumerate(members, start=1):
            run.run_id = f"{experiment}.{scope}" + (f".run{i}" if len(members) > 1 else "")


# ----------------------------------------------------------------------------
# Scopes and arms
# ----------------------------------------------------------------------------
@dataclass
class Scope:
    """The merged records of one experiment on one dataset and partition."""

    experiment: str
    name: str
    dataset: str
    partition: str
    budgets: list[int]
    groups: dict[tuple, dict[str, dict[str, Any]]]  # load_records keys (regime, ranker, policy, model_id)
    source_of: dict[tuple, str]
    query_sha: dict[str, dict[str, str | None]] = field(default_factory=dict)  # regime -> run -> query set

    def regimes(self) -> list[str]:
        return ordered({k[0] for k in self.groups}, list(protocol.LABEL_REGIMES))

    def policies(self, regime: str) -> list[str]:
        return ordered({k[2] for k in self.groups if k[0] == regime}, rev.policies())

    def subsets(self, regime: str, by_family: bool) -> dict[str, list[str]]:
        """SUB -> the sorted uids of the regime in it; empty subsets left out."""
        recs = next(recs for k, recs in self.groups.items() if k[0] == regime)
        families = {uid: str(rec["family"]) for uid, rec in recs.items()}
        out: dict[str, list[str]] = {PLANTED: [], SM: []}
        for uid in sorted(families):
            out[subset_of(families[uid]) or PLANTED].append(uid)  # every family was checked
        if by_family:
            for family in ordered(set(families.values()), [*protocol.POSITIVE_FAMILIES, *SM_FAMILIES]):
                out[FAMILY_PREFIX + segment(family)] = sorted(u for u, f in families.items() if f == family)
        return {sub: uids for sub, uids in out.items() if uids}


def merge_scopes(runs: Sequence[RetrievalRun]) -> list[Scope]:
    by: dict[tuple[str, str], list[RetrievalRun]] = {}
    for run in runs:
        by.setdefault((run.experiment, run.scope), []).append(run)
    scopes = []
    for (experiment, name) in sorted(by, key=lambda k: (RETRIEVAL_EXPERIMENTS.index(k[0]), k[1])):
        members = sorted(by[(experiment, name)], key=lambda r: r.given)
        ladders = sorted({tuple(r.budgets) for r in members})
        if len(ladders) > 1:
            refuse(f"{experiment} {name}: the runs have different ladders {ladders}")
        groups: dict[tuple, dict[str, dict[str, Any]]] = {}
        source_of: dict[tuple, str] = {}
        given_of: dict[tuple, str] = {}
        for run in members:
            for key, recs in run.groups.items():
                if key in groups:
                    refuse(f"{experiment} {name}: arm {key[0]}/{arm_name(key)} is in both {given_of[key]} and "
                           f"{run.given}")
                groups[key], source_of[key], given_of[key] = recs, run.run_id, run.given
        scope = Scope(experiment, name, members[0].dataset, members[0].partition, list(members[0].budgets),
                      groups, source_of)
        for run in members:
            for regime in sorted({k[0] for k in run.groups}):
                scope.query_sha.setdefault(regime, {})[run.given] = run.query_sha.get(regime)
        for regime in scope.regimes():
            sets = {frozenset(recs) for key, recs in groups.items() if key[0] == regime}
            if len(sets) > 1:
                refuse(f"{experiment} {name} {regime}: the arms were evaluated on different queries (uid sets of "
                       f"sizes {sorted(len(s) for s in sets)}); pairing needs one query set")
        scopes.append(scope)
    return scopes


@dataclass
class ArmData:
    """One arm within a subset: covered and cost per replicate (a training seed, or ALL) and query."""

    label: str
    learned: bool
    seeds: tuple[int, ...]
    uids: tuple[str, ...]
    budgets: tuple[int, ...]
    covered: dict[int, dict[str, dict[str, float]]]  # K -> replicate -> uid -> 0 or 1
    cost: dict[str, dict[str, int | None]]  # replicate -> uid -> cost, None with an error
    errors: int
    records: int
    sources: tuple[str, ...]
    _cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    def cost_table(self) -> dict[str, dict[str, float]]:
        """replicate -> uid -> cost over the queries without an error in any replicate."""
        if "cost_table" not in self._cache:
            bad = {u for per in self.cost.values() for u, c in per.items() if c is None}
            self._cache["cost_table"] = {rep: {u: float(c) for u, c in per.items() if u not in bad}
                                         for rep, per in self.cost.items()}
        return self._cache["cost_table"]

    def cost_avg(self) -> dict[str, float]:
        if "cost_avg" not in self._cache:
            self._cache["cost_avg"] = seed_mean(self.cost_table(), self.label)
        return self._cache["cost_avg"]

    def covered_avg(self, k: int) -> dict[str, float]:
        if ("covered", k) not in self._cache:
            self._cache[("covered", k)] = seed_mean(self.covered[k], self.label)
        return self._cache[("covered", k)]


def seed_mean(table: Mapping[str, Mapping[str, float]], label: str) -> dict[str, float]:
    """uid -> mean over the replicates (stats.seed_average, which refuses unequal replicate sets)."""
    rows = [{"arm": label, "query_uid": uid, "train_seed": rep, "solved": value}
            for rep, per in table.items() for uid, value in per.items()]
    return stats.seed_average(rows)[label] if rows else {}


def model_identity(recs: Mapping[str, Mapping[str, Any]], where: str) -> dict[str, Any]:
    first = next(iter(recs.values()))
    ident = {f: first.get(f) for f in MODEL_FIELDS}
    for uid, rec in recs.items():
        if any(rec.get(f) != ident[f] for f in MODEL_FIELDS):
            refuse(f"{where}: the records of one arm name different models ({uid})")
    return ident


def build_arm(label: str, learned: bool, parts: Sequence[tuple[int | None, Mapping[str, Any], str]],
              uids: Sequence[str], budgets: Sequence[int], where: str) -> ArmData:
    """parts: (training seed, or None without an encoder; {uid: record}; source run id)."""
    covered: dict[int, dict[str, dict[str, float]]] = {int(k): {} for k in budgets}
    cost: dict[str, dict[str, int | None]] = {}
    seeds, sources = [], set()
    errors = 0
    for seed, recs, source in parts:
        rep = ALL if seed is None else str(seed)
        if rep in cost:
            refuse(f"{where}: training seed {seed} appears twice")
        sources.add(source)
        cost[rep] = {}
        for k in budgets:
            covered[int(k)][rep] = {}
        for uid in uids:
            rec = recs[uid]
            failed = rec.get("error") is not None
            errors += int(failed)
            value = None if failed else int(rec["cost"])
            cost[rep][uid] = value
            for k in budgets:
                covered[int(k)][rep][uid] = 0.0 if value is None else float(value <= int(k))
        if seed is not None:
            seeds.append(int(seed))
    return ArmData(label, learned, tuple(sorted(seeds)), tuple(uids), tuple(int(k) for k in budgets), covered, cost,
                   errors, len(parts) * len(uids), tuple(sorted(sources)))


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def hierarchical(table: Mapping[str, Mapping[str, float]]) -> tuple[float, float, float]:
    """(mean of the seed averages, lo, hi): queries and seeds resampled (hierarchical bootstrap)."""
    units = sorted(next(iter(table.values())))
    zero = {"zero": dict.fromkeys(units, 0.0)}
    return stats.hierarchical_bootstrap_diff(table, zero, n_boot=N_HIER, seed=SEED)


def check_pairing(a: ArmData, b: ArmData, where: str) -> None:
    if a.uids != b.uids:
        refuse(f"{where}: {a.label} and {b.label} hold different queries")
    if a.learned and b.learned and a.seeds != b.seeds:
        refuse(f"{where}: {a.label} has training seeds {list(a.seeds)} and {b.label} {list(b.seeds)}; the "
               "pairing needs the same seeds (a missing seed breaks it)")


@dataclass(frozen=True)
class Ratio:
    n: int
    gm: float
    lo: float
    hi: float
    p: float
    excluded: int


@dataclass(frozen=True)
class Diff:
    n: int
    diff: float
    lo: float
    hi: float
    p: float


def cost_ratio(a: ArmData, b: ArmData, where: str, *, plus_one: bool = False) -> Ratio | None:
    """Geometric-mean ratio of the seed-averaged costs a / b (each plus one with plus_one), paired by uid."""
    check_pairing(a, b, where)
    ca, cb = a.cost_avg(), b.cost_avg()
    common = sorted(set(ca) & set(cb))
    if not common:
        return None
    shift = 1.0 if plus_one else 0.0
    r = stats.paired_log_ratio({u: ca[u] + shift for u in common}, {u: cb[u] + shift for u in common},
                               n_boot=N_BOOT, n_perm=N_PERM, seed=SEED)
    return Ratio(int(r["n_units"]), float(r["gm_ratio"]), float(r["lo"]), float(r["hi"]), float(r["p_value"]),
                 len(a.uids) - len(common))


def fullcov_diff(a: ArmData, b: ArmData, k: int, where: str) -> Diff:
    check_pairing(a, b, where)
    r = stats.paired_compare(a.covered_avg(k), b.covered_avg(k), n_boot=N_BOOT, n_perm=N_PERM, seed=SEED)
    return Diff(r.n_units, r.diff, r.lo, r.hi, r.p_value)


@dataclass
class Test:
    prefix: str  # the keys are PREFIX.p and PREFIX.p_holm
    p: float
    n: int
    sources: list[str]


def write_tests(nums: nj.Numbers, tests: Sequence[Test]) -> None:
    """The p values of one table and their Holm adjustment within it."""
    if not tests:
        return
    adjusted = stats.holm({t.prefix: t.p for t in tests})
    for t in tests:
        nums.number(f"{t.prefix}.p", t.p, "pvalue", n=t.n, source=t.sources)
        nums.number(f"{t.prefix}.p_holm", adjusted[t.prefix], "pvalue", n=t.n, source=t.sources)


def compare(nums: nj.Numbers, q: str, a: ArmData, b: ArmData, fullcov_budgets: Sequence[int], where: str,
            cost_tests: list[Test], cov_tests: list[Test]) -> None:
    """Q.n, Q.cost.* and Q.fullcov.K<K>.diff of A against B; the tests join their tables' families."""
    src = union(a.sources, b.sources)
    n = len(a.uids)
    nums.number(f"{q}.n", n, "count", n=n, source=src)
    ratio = cost_ratio(a, b, where)
    if ratio is not None:
        nums.number(f"{q}.cost.gm_ratio", ratio.gm, "ratio", ci=(ratio.lo, ratio.hi), n=ratio.n, digits=2,
                    source=src)
        nums.number(f"{q}.cost.excluded", ratio.excluded, "count", n=n, source=src)
        cost_tests.append(Test(f"{q}.cost", ratio.p, ratio.n, src))
    for k in fullcov_budgets:
        d = fullcov_diff(a, b, k, where)
        nums.number(f"{q}.fullcov.K{k}.diff", pp(d.diff), "pp", ci=(pp(d.lo), pp(d.hi)), n=d.n, digits=1,
                    source=src)
        cov_tests.append(Test(f"{q}.fullcov.K{k}", d.p, d.n, src))


# ----------------------------------------------------------------------------
# Descriptive keys, tables and curves
# ----------------------------------------------------------------------------
def describe(nums: nj.Numbers, prefix: str, arm: ArmData, *, index: ArmData | None = None, where: str = "") -> None:
    """The descriptive keys of one arm (P4 in the module docstring)."""
    src = list(arm.sources)
    n = len(arm.uids)
    nums.number(f"{prefix}.n", n, "count", n=n, source=src)
    nums.number(f"{prefix}.errors", arm.errors, "count", n=arm.records, source=src)
    if arm.learned:
        nums.number(f"{prefix}.seeds", len(arm.seeds), "count", n=len(arm.seeds), source=src)
    for k in arm.budgets:
        key = f"{prefix}.fullcov.K{k}"
        if arm.learned:
            value, lo, hi = hierarchical(arm.covered[k])
            nums.number(key, pct(value), "pct", ci=(pct(lo), pct(hi)), n=n, digits=1, source=src)
        else:
            nums.rate(key, int(round(sum(arm.covered[k][ALL].values()))), n, source=src)
    avg = arm.cost_avg()
    if avg:
        values = np.array([avg[u] for u in sorted(avg)], dtype=float)
        if arm.learned:
            mean, lo, hi = hierarchical(arm.cost_table())
        else:
            mean = float(values.mean())
            lo, hi = stats.paired_bootstrap_ci(values, n_boot=N_BOOT, seed=SEED)
        nums.number(f"{prefix}.cost.mean", mean, "number", ci=(lo, hi), n=int(values.size), source=src)
        med, mlo, mhi = stats.quantile_with_ci(values, 0.5, n_boot=N_HIER, seed=SEED)
        nums.number(f"{prefix}.cost.median", med, "number", ci=(mlo, mhi), n=int(values.size), source=src)
    if index is not None and index is not arm:
        ratio = cost_ratio(arm, index, where, plus_one=True)
        if ratio is not None:
            nums.number(f"{prefix}.cost.gm_ratio_p1_vs_index", ratio.gm, "ratio", ci=(ratio.lo, ratio.hi),
                        n=ratio.n, digits=2, source=union(src, index.sources))
    if arm.learned:
        for seed in arm.seeds:
            rep = str(seed)
            for k in arm.budgets:
                nums.rate(f"{prefix}.seed{seed}.fullcov.K{k}", int(round(sum(arm.covered[k][rep].values()))), n,
                          source=src)
            costs = np.array([c for c in arm.cost[rep].values() if c is not None], dtype=float)
            if costs.size:
                lo, hi = stats.paired_bootstrap_ci(costs, n_boot=N_BOOT, seed=SEED)
                nums.number(f"{prefix}.seed{seed}.cost.mean", float(costs.mean()), "number", ci=(lo, hi),
                            n=int(costs.size), source=src)


@dataclass(frozen=True)
class Row:
    label: str
    prefix: str  # the key prefix its cells read
    arm: ArmData
    duplicate: bool = False


def curve(nums: nj.Numbers, fig_id: str, series: Sequence[tuple[str, str]], budgets: Sequence[int],
          notes: str) -> None:
    """FullCov@K against K, one series per (name, key prefix), read from the fullcov keys."""
    styles: dict[str, str] = {}
    for name, _ in series:
        if styles.setdefault(series_key(name), name) != name:
            refuse(f"figure {fig_id}: series {styles[series_key(name)]!r} and {name!r} would share one style")
    rows = []
    for name, prefix in series:
        for k in budgets:
            entry = nums.numbers.get(f"{prefix}.fullcov.K{k}")
            if entry is not None:
                rows.append({"series": name, "x": k, "y": entry["value"], "lo": entry["ci"][0],
                             "hi": entry["ci"][1]})
    if rows:
        nums.figure(fig_id, "curve", rows, x="K (blocks retrieved)", y="FullCov@K (%)", log_x=True, notes=notes)


FULLCOV_NOTE = ("FullCov@K: share of queries whose first K blocks, with their one-hop halo, hold every planted "
                "node, in %, with a 95% interval: Wilson without an encoder; for an encoder ranker or arm, "
                "averaged per query over the training seeds, hierarchical bootstrap over queries and seeds. "
                "A record with an error counts as not covered and stays in n. Errors: records with an error, "
                "over every seed.")
COST_NOTE = ("Cost: blocks to FullCov (1..P), mean and median over queries of the seed-averaged cost, 95% "
             "bootstrap intervals; a query with an error in any seed leaves the cost and is counted in Errors.")
SEEDS_NOTE = ("One training seed at a time: FullCov@K in % with Wilson intervals over queries; mean cost with "
              "a bootstrap interval over the queries without an error in that seed. The spread over seeds, "
              "reported apart from the seed-averaged n.")


def descriptive_tables(nums: nj.Numbers, base: str, rows: Sequence[Row], budgets: Sequence[int], *, first: str,
                       e4: bool, notes: str) -> None:
    """The per-(SCOPE, SUB, REG, POL) tables: E4 fullcov and cost, or E5/E6 arms; and the seeds table."""
    ks = [f"K={k}" for k in budgets]
    fullcov = [[r.label, ref(nums, f"{r.prefix}.n"), ref(nums, f"{r.prefix}.seeds") if r.arm.learned else DASH,
                ref(nums, f"{r.prefix}.errors"), *[ref(nums, f"{r.prefix}.fullcov.K{k}", ci=True) for k in budgets]]
               for r in rows]
    if e4:
        nums.table(f"{base}.fullcov", [first, "n", "Seeds", "Errors", *ks], fullcov, notes=f"{FULLCOV_NOTE} {notes}")
        cost = [[r.label, ref(nums, f"{r.prefix}.n"), ref(nums, f"{r.prefix}.errors"),
                 ref(nums, f"{r.prefix}.cost.mean", ci=True), ref(nums, f"{r.prefix}.cost.median", ci=True),
                 DASH if r.arm.label == "index" else ref(nums, f"{r.prefix}.cost.gm_ratio_p1_vs_index", ci=True)]
                for r in rows]
        nums.table(f"{base}.cost", [first, "n", "Errors", "Mean cost", "Median cost", "GM ratio (cost+1) vs index"],
                   cost, notes=f"{COST_NOTE} GM ratio: geometric mean over paired queries of (cost+1)/(cost_index+1), "
                               f"95% bootstrap interval; no test in this table (PE5 holds the tests). {notes}")
    else:
        arms_rows = [row + [ref(nums, f"{r.prefix}.cost.mean", ci=True), ref(nums, f"{r.prefix}.cost.median", ci=True)]
                     for row, r in zip(fullcov, rows)]
        nums.table(f"{base}.arms", [first, "n", "Seeds", "Errors", *ks, "Mean cost", "Median cost"], arms_rows,
                   notes=f"{FULLCOV_NOTE} {COST_NOTE} {notes}")
    seeds = [[r.label, str(seed), *[ref(nums, f"{r.prefix}.seed{seed}.fullcov.K{k}", ci=True) for k in budgets],
              ref(nums, f"{r.prefix}.seed{seed}.cost.mean", ci=True)]
             for r in rows if r.arm.learned and not r.duplicate for seed in r.arm.seeds]
    if seeds:
        nums.table(f"{base}.seeds", [first, "Seed", *ks, "Mean cost"], seeds, notes=f"{SEEDS_NOTE} {notes}")


def subset_note(sub: str) -> str:
    if sub == PLANTED:
        return f"the planted positive families ({', '.join(protocol.POSITIVE_FAMILIES)})"
    if sub == SM:
        return f"the standard-SM suite ({', '.join(SM_FAMILIES)}), not a pre-registered endpoint"
    if sub == SEEN:
        return ("the planted positives isomorphic to a training query of some main encoder of the regime "
                "(pattern audit)")
    if sub == NOVEL:
        return "the planted positives no main encoder of the regime drew in training (pattern audit)"
    return f"family {sub[len(FAMILY_PREFIX):]} alone"


# ----------------------------------------------------------------------------
# E4 and PE5
# ----------------------------------------------------------------------------
def e4_arms(scope: Scope, regime: str, policy: str, uids: Sequence[str]) -> dict[str, ArmData]:
    members: dict[str, list[tuple[tuple, dict[str, Any]]]] = {}
    for key, recs in scope.groups.items():
        if key[0] == regime and key[2] == policy:
            members.setdefault(key[1], []).append((key, recs))
    arms: dict[str, ArmData] = {}
    for ranker in ordered(members, rankers.RANKER_NAMES):
        learned = ranker in rankers.NEEDS_EMBEDDING
        where = f"E4 {scope.name} {regime} {ranker}+{policy}"
        parts = []
        for key, recs in sorted(members[ranker], key=lambda kr: str(kr[0][3])):
            ident = model_identity(recs, where)
            if learned:
                if ident["model_kind"] != "main" or ident["train_arm"] != rev.MAIN_ARM:
                    refuse(f"{where}: model {key[3]} is a {ident['model_kind']} {ident['train_arm']} model; E4 "
                           "evaluates the main-arm encoders")
                parts.append((int(ident["train_seed"]), recs, scope.source_of[key]))
            else:
                if key[3] is not None:
                    refuse(f"{where}: a ranker without an encoder names model {key[3]}")
                parts.append((None, recs, scope.source_of[key]))
        arms[ranker] = build_arm(ranker, learned, parts, uids, scope.budgets, where)
    seed_sets = {arm.seeds for arm in arms.values() if arm.learned}
    if len(seed_sets) > 1:
        refuse(f"E4 {scope.name} {regime} {policy}: the encoder rankers have different training seeds "
               f"{sorted(seed_sets)}; the pairing needs the same seeds (a missing seed breaks it)")
    return arms


def audited_subsets(scope: Scope, regime: str, audit: AuditRun, planted: Sequence[str]) -> dict[str, list[str]]:
    """SUB seen and novel of the planted positives, once the audit is shown to cover the regime's runs."""
    entry, where = audit.regimes[regime], f"E4 {scope.name} {regime}"
    for given, sha in sorted(scope.query_sha.get(regime, {}).items()):
        if sha is None:
            refuse(f"{where}: {given} records no sha256 of its query set to match {audit.given} against")
        if sha != entry["sha256"]:
            refuse(f"{where}: {given} ran query set {sha[:12]}..., {audit.given} audited "
                   f"{str(entry['sha256'])[:12]}...")
    outside = sorted({u for k, recs in scope.groups.items() if k[0] == regime for u in recs} - entry["uids"])
    if outside:
        refuse(f"{where}: {len(outside)} queries are not in the set {audit.given} audited, e.g. {outside[0]}")
    for key in sorted(scope.groups, key=str):
        if key[0] == regime and key[1] in rankers.NEEDS_EMBEDDING and key[3] not in entry["encoders"]:
            refuse(f"{where}: {key[1]} ranks with {key[3]}, which {audit.given} did not audit "
                   f"({sorted(entry['encoders'])})")
    return {SEEN: [u for u in planted if u in entry["seen"]], NOVEL: [u for u in planted if u not in entry["seen"]]}


def reduce_e4(nums: nj.Numbers, scope: Scope, by_family: bool, warnings: list[str],
              audits: Mapping[tuple[str, str], AuditRun] | None = None) -> None:
    audits = audits or {}
    subsets = {regime: scope.subsets(regime, by_family) for regime in scope.regimes()}
    for regime, subs in subsets.items():
        audit = audits.get((scope.dataset, regime))
        if audit is not None:
            subs.update({s: u for s, u in audited_subsets(scope, regime, audit, subs.get(PLANTED, [])).items() if u})
    order = [PLANTED, SM, NOVEL, SEEN,
             *[FAMILY_PREFIX + segment(f) for f in (*protocol.POSITIVE_FAMILIES, *SM_FAMILIES)]]
    hybrid = rev.policies()[0]
    for sub in ordered({s for subs in subsets.values() for s in subs}, order):
        hybrid_arms: dict[str, dict[str, ArmData]] = {}
        for regime, subs in subsets.items():
            if sub not in subs:
                continue
            for policy in scope.policies(regime):
                arms = e4_arms(scope, regime, policy, subs[sub])
                if sub in (SEEN, NOVEL):
                    for arm in arms.values():  # the audit chose the queries
                        arm.sources = tuple(sorted({*arm.sources, audits[(scope.dataset, regime)].run_id}))
                base = f"E4.{scope.name}.{sub}.{segment(regime)}.{policy}"
                for ranker, arm in arms.items():
                    describe(nums, f"{base}.{ranker}", arm, index=arms.get("index"), where=f"{base}.{ranker}")
                rows = [Row(ranker, f"{base}.{ranker}", arm) for ranker, arm in arms.items()]
                note = f"{scope.dataset} {regime}, policy {policy}, queries: {subset_note(sub)}."
                descriptive_tables(nums, base, rows, scope.budgets, first="Ranker", e4=True, notes=note)
                curve(nums, f"{base}.fullcov_curve", [(r.label, r.prefix) for r in rows], scope.budgets,
                      notes=f"{FULLCOV_NOTE} {note}")
                if policy == hybrid:
                    hybrid_arms[regime] = arms
        if sub in COMPARED_SUBSETS and hybrid_arms:
            reduce_pe5(nums, scope, sub, hybrid_arms, warnings)


def reduce_pe5(nums: nj.Numbers, scope: Scope, sub: str, arms_by_regime: Mapping[str, Mapping[str, ArmData]],
               warnings: list[str]) -> None:
    k1 = scope.budgets[0] if len(scope.budgets) > 1 else None
    hybrid = rev.policies()[0]
    cost_tests: list[Test] = []
    cov_tests: list[Test] = []
    done = []
    for regime, arms in arms_by_regime.items():
        for a in PE5_RANKERS:
            for b in PE5_REFERENCES:
                if a == b:
                    continue
                if a not in arms or b not in arms:
                    warnings.append(f"PE5 {scope.name} {sub} {regime}: no {a if a not in arms else b} records; "
                                    f"{a} vs {b} left out")
                    continue
                q = f"PE5.{scope.name}.{sub}.{segment(regime)}.{a}_vs_{b}"
                compare(nums, q, arms[a], arms[b], [k1] if k1 else [], f"PE5 {scope.name} {sub} {regime}",
                        cost_tests, cov_tests)
                done.append((regime, a, b, q))
    write_tests(nums, cost_tests)
    write_tests(nums, cov_tests)
    if not done:
        return
    e4 = f"E4.{scope.name}.{sub}"
    dk = f"FullCov@{k1} A-B (pp)" if k1 else "FullCov A-B (pp)"
    pre = "PE5" if sub == PLANTED else "Not a pre-registered endpoint (PE5 is the planted positives)"
    cost_rows = [[regime, f"{a} vs {b}", ref(nums, f"{q}.n"),
                  ref(nums, f"{e4}.{segment(regime)}.{hybrid}.{a}.cost.mean", ci=True),
                  ref(nums, f"{e4}.{segment(regime)}.{hybrid}.{b}.cost.mean", ci=True),
                  ref(nums, f"{q}.cost.gm_ratio", ci=True), ref(nums, f"{q}.cost.p"), ref(nums, f"{q}.cost.p_holm"),
                  ref(nums, f"{q}.fullcov.K{k1}.diff", ci=True) if k1 else NA] for regime, a, b, q in done]
    nums.table(f"PE5.{scope.name}.{sub}.cost",
               ["Regime", "Comparison", "n", "Mean cost A", "Mean cost B", "GM cost ratio A/B", "p", "p (Holm)", dk],
               cost_rows,
               notes=f"{pre}: {scope.dataset}, policy {hybrid}, queries: {subset_note(sub)}. Encoder rankers are "
                     "averaged per query over the training seeds, then paired by query. GM cost ratio: geometric "
                     "mean over queries of cost A / cost B, 95% bootstrap interval (20,000), two-sided sign-flip p "
                     "(100,000), Holm over every p of this table; queries with an error on either side are left "
                     "out. fused_learned_mean is Mean-RRF. The FullCov column is the paired difference at the first "
                     f"budget; its tests are in table PE5.{scope.name}.{sub}.fullcov.")
    if k1:
        nums.table(f"PE5.{scope.name}.{sub}.fullcov", ["Regime", "Comparison", "n", dk, "p", "p (Holm)"],
                   [[regime, f"{a} vs {b}", ref(nums, f"{q}.n"), ref(nums, f"{q}.fullcov.K{k1}.diff", ci=True),
                     ref(nums, f"{q}.fullcov.K{k1}.p"), ref(nums, f"{q}.fullcov.K{k1}.p_holm")]
                    for regime, a, b, q in done],
                   notes=f"{pre}: FullCov@{k1} of A minus B in pp over queries (seed-averaged; an error counts as "
                         "not covered), 95% bootstrap interval (20,000), sign-flip p (100,000), Holm over every p "
                         "of this table.")


# ----------------------------------------------------------------------------
# E5, E6 and PE6
# ----------------------------------------------------------------------------
@dataclass
class MatrixArm:
    arm: str
    data: ArmData
    duplicate_of: str | None  # the train_arm this row duplicates
    own_weights: bool  # a duplicate evaluated on its own export (for some seed)


def matrix_arms(scope: Scope, regime: str, policy: str, uids: Sequence[str], kind: str) -> dict[str, MatrixArm]:
    where = f"{scope.experiment} {scope.name} {regime} {policy}"
    members: dict[str, list[tuple[int, dict[str, Any], str, dict[str, Any]]]] = {}
    for key, recs in scope.groups.items():
        if key[0] != regime or key[2] != policy:
            continue
        if key[1] != "learned":
            refuse(f"{where}: {scope.experiment} evaluates the learned ranker; the run holds {key[1]}")
        ident = model_identity(recs, f"{where} model {key[3]}")
        model_kind, arm = ident["model_kind"], str(ident["train_arm"])
        if model_kind not in ("main", kind) or (model_kind == "main") != (arm == rev.MAIN_ARM):
            refuse(f"{where}: model {key[3]} is a {model_kind} {arm} model; {scope.experiment} holds the main "
                   f"reference and {kind} rows")
        members.setdefault(arm, []).append((int(ident["train_seed"]), recs, scope.source_of[key], ident))
    if rev.MAIN_ARM not in members:
        refuse(f"{where}: no main-arm reference models")
    out: dict[str, MatrixArm] = {}
    for arm, parts in members.items():
        data = build_arm(arm, True, [(s, r, src) for s, r, src, _ in parts], uids, scope.budgets, f"{where} {arm}")
        duplicates = {ident["duplicate_of"] for *_, ident in parts}
        target, own = None, False
        if duplicates != {None}:
            if None in duplicates:
                refuse(f"{where}: {arm} is a duplicate row for some seeds only")
            targets = set()
            for seed, _, _, ident in parts:
                parsed = rev.parse_model_id(str(ident["duplicate_of"]), scope.dataset)
                if parsed is None or parsed["seed"] != seed:
                    refuse(f"{where}: {arm} of seed {seed} duplicates {ident['duplicate_of']}, not a model of that "
                           "seed")
                targets.add(parsed["arm"])
            if len(targets) != 1:
                refuse(f"{where}: {arm} duplicates several arms {sorted(targets)}")
            target = targets.pop()
            own = any(ident["evaluated_model_id"] != ident["duplicate_of"] for *_, ident in parts)
        out[arm] = MatrixArm(arm, data, target, own)
    main = out[rev.MAIN_ARM]
    for m in out.values():
        if m.data.seeds != main.data.seeds:
            refuse(f"{where}: {m.arm} has training seeds {list(m.data.seeds)}, main {list(main.data.seeds)}; the "
                   "comparison at the same seed needs every seed (a missing seed breaks the pairing)")
        if m.duplicate_of is None:
            continue
        target = out.get(m.duplicate_of)
        if target is None:
            refuse(f"{where}: {m.arm} duplicates {m.duplicate_of}, which the run lacks")
        if target.duplicate_of is not None:
            refuse(f"{where}: {m.arm} duplicates {m.duplicate_of}, itself a duplicate row")
        if not m.own_weights and m.data.cost != target.data.cost:
            refuse(f"{where}: {m.arm} was evaluated on the weights of {m.duplicate_of}, but its records differ")
    return out


def duplicate_note(arms: Mapping[str, MatrixArm], names: Sequence[str], base: str, segs: Mapping[str, str]) -> str:
    parts = []
    for name in names:
        m = arms[name]
        if m.duplicate_of is None:
            continue
        if m.own_weights:
            parts.append(f"{name} has the settings of {m.duplicate_of} and was evaluated on its own export; its "
                         f"rows report {m.duplicate_of}, its own numbers are the keys {base}.arm.{segs[name]}.*")
        else:
            parts.append(f"{name} has the settings of {m.duplicate_of} and was evaluated on its weights")
    return ("Duplicate rows (train_matrix duplicate_of): " + "; ".join(parts) + ".") if parts else ""


def reduce_matrix(nums: nj.Numbers, scope: Scope) -> dict[tuple[str, str, str], dict[str, MatrixArm]]:
    """E5 or E6 keys and tables of one scope; returns the arms by (SUB, REG, POL) for PE6."""
    exp = scope.experiment
    kind = rev.MATRIX_KINDS[exp]
    order = [rev.MAIN_ARM, *protocol_arm_order(scope.dataset, kind)]
    assembled: dict[tuple[str, str, str], dict[str, MatrixArm]] = {}
    for sub in COMPARED_SUBSETS:
        cost_tests: list[Test] = []
        cov_tests: list[Test] = []
        cost_rows: list[list[Any]] = []
        cov_rows: list[list[Any]] = []
        for regime in scope.regimes():
            uids = scope.subsets(regime, False).get(sub)
            if not uids:
                continue
            for policy in scope.policies(regime):
                arms = matrix_arms(scope, regime, policy, uids, kind)
                assembled[(sub, regime, policy)] = arms
                names = ordered(arms, order)
                segs = {name: segment(name) for name in names}
                if len(set(segs.values())) != len(segs):
                    refuse(f"{exp} {scope.name}: arm names {names} clash as key segments")
                base = f"{exp}.{scope.name}.{sub}.{segment(regime)}.{policy}"
                target_of = {name: arms[name].duplicate_of or name for name in names}
                label_of = {name: name if target_of[name] == name else f"{name} (duplicate of {target_of[name]})"
                            for name in names}
                for name in names:
                    if arms[name].duplicate_of is None or arms[name].own_weights:
                        describe(nums, f"{base}.arm.{segs[name]}", arms[name].data)
                rows = [Row(label_of[name], f"{base}.arm.{segs[target_of[name]]}", arms[target_of[name]].data,
                            duplicate=arms[name].duplicate_of is not None) for name in names]
                note = (f"{scope.dataset} {regime}, policy {policy}, learned ranker, queries: {subset_note(sub)}. "
                        f"{duplicate_note(arms, names, base, segs)}").strip()
                descriptive_tables(nums, base, rows, scope.budgets, first="Arm", e4=False, notes=note)
                curve(nums, f"{base}.fullcov_curve",
                      [(name, f"{base}.arm.{segs[name]}") for name in names if arms[name].duplicate_of is None],
                      scope.budgets, notes=f"{FULLCOV_NOTE} {note} Duplicate rows are left out.")
                main = arms[rev.MAIN_ARM].data
                main_p = f"{base}.arm.{segs[rev.MAIN_ARM]}"
                for name in names:
                    if name == rev.MAIN_ARM:
                        continue
                    target = target_of[name]
                    arm_p = f"{base}.arm.{segs[target]}"
                    head = [regime, policy, label_of[name]]
                    if target == rev.MAIN_ARM:
                        cost_rows.append([*head, ref(nums, f"{arm_p}.n"), ref(nums, f"{arm_p}.cost.mean", ci=True),
                                          ref(nums, f"{main_p}.cost.mean", ci=True), SAME, DASH, DASH])
                        for k in scope.budgets[:-1]:
                            cov_rows.append([*head, str(k), ref(nums, f"{arm_p}.n"),
                                             ref(nums, f"{arm_p}.fullcov.K{k}", ci=True),
                                             ref(nums, f"{main_p}.fullcov.K{k}", ci=True), SAME, DASH, DASH])
                        continue
                    q = f"{base}.vs_main.{segs[target]}"
                    if arms[name].duplicate_of is None:
                        compare(nums, q, arms[name].data, main, scope.budgets[:-1],
                                f"{exp} {scope.name} {sub} {regime} {policy} {name}", cost_tests, cov_tests)
                    cost_rows.append([*head, Later(f"{q}.n"), ref(nums, f"{arm_p}.cost.mean", ci=True),
                                      ref(nums, f"{main_p}.cost.mean", ci=True), Later(f"{q}.cost.gm_ratio", True),
                                      Later(f"{q}.cost.p"), Later(f"{q}.cost.p_holm")])
                    for k in scope.budgets[:-1]:
                        cov_rows.append([*head, str(k), Later(f"{q}.n"), ref(nums, f"{arm_p}.fullcov.K{k}", ci=True),
                                         ref(nums, f"{main_p}.fullcov.K{k}", ci=True),
                                         Later(f"{q}.fullcov.K{k}.diff", True), Later(f"{q}.fullcov.K{k}.p"),
                                         Later(f"{q}.fullcov.K{k}.p_holm")])
        write_tests(nums, cost_tests)
        write_tests(nums, cov_tests)
        what = "factorial arm" if kind == "factorial" else "architecture, depth or residual variant"
        family = (f"{exp}, {scope.dataset}: every {what} against the main arm at the same training seeds, learned "
                  f"ranker, queries: {subset_note(sub)}; both sides averaged per query over the seeds, then paired "
                  "by query. A duplicate row reports the model it duplicates and has no test of its own.")
        if cost_rows:
            nums.table(f"{exp}.{scope.name}.{sub}.cost",
                       ["Regime", "Policy", "Arm", "n", "Mean cost", "Mean cost main", "GM cost ratio arm/main", "p",
                        "p (Holm)"], resolve(nums, cost_rows),
                       notes=f"{family} GM cost ratio: geometric mean over queries of cost arm / cost main, 95% "
                             "bootstrap interval (20,000), sign-flip p (100,000), Holm over every p of this table; "
                             "queries with an error on either side are left out.")
        if cov_rows:
            nums.table(f"{exp}.{scope.name}.{sub}.fullcov",
                       ["Regime", "Policy", "Arm", "K", "n", "FullCov", "FullCov main", "arm - main (pp)", "p",
                        "p (Holm)"], resolve(nums, cov_rows),
                       notes=f"{family} FullCov@K in % and the paired difference in pp (an error counts as not "
                             "covered), 95% bootstrap interval (20,000), sign-flip p (100,000), Holm over every p of "
                             "this table; every K but P.")
    return assembled


def reduce_pe6(nums: nj.Numbers, scope: Scope, assembled: Mapping[tuple[str, str, str], Mapping[str, MatrixArm]],
               warnings: list[str]) -> None:
    control = pe6_control(scope.dataset)
    if control is None:
        warnings.append(f"PE6 {scope.name}: protocol.FACTORIAL has no mean-aggregation control for {scope.dataset}")
        return
    named = False
    for sub in COMPARED_SUBSETS:
        cost_tests: list[Test] = []
        cov_tests: list[Test] = []
        done = []
        for (s, regime, policy), arms in assembled.items():
            if s != sub:
                continue
            if control not in arms:
                warnings.append(f"PE6 {scope.name} {sub} {regime} {policy}: the run has no control arm {control}")
                continue
            if not named:
                nums.number(f"PE6.{scope.name}.control_arm", control, "text", source=arms[control].data.sources)
                named = True
            target = arms[control].duplicate_of or control
            q = f"PE6.{scope.name}.{sub}.{segment(regime)}.{policy}"
            compare(nums, q, arms[rev.MAIN_ARM].data, arms[target].data, scope.budgets[:-1],
                    f"PE6 {scope.name} {sub} {regime} {policy}", cost_tests, cov_tests)
            done.append((regime, policy, q, f"E5.{scope.name}.{sub}.{segment(regime)}.{policy}.arm", segment(target)))
        write_tests(nums, cost_tests)
        write_tests(nums, cov_tests)
        if not done:
            continue
        pre = "PE6" if sub == PLANTED else "Not a pre-registered endpoint (PE6 is the planted positives)"
        head = (f"{pre}: {scope.dataset}, the full objective (main) against the mean-aggregation control {control}, "
                f"learned ranker, queries: {subset_note(sub)}; averaged per query over the training seeds, then "
                "paired by query.")
        nums.table(f"PE6.{scope.name}.{sub}.cost",
                   ["Regime", "Policy", "n", "Mean cost main", "Mean cost control", "GM cost ratio main/control", "p",
                    "p (Holm)"],
                   [[regime, policy, ref(nums, f"{q}.n"), ref(nums, f"{arm}.{segment(rev.MAIN_ARM)}.cost.mean", ci=True),
                     ref(nums, f"{arm}.{cseg}.cost.mean", ci=True), ref(nums, f"{q}.cost.gm_ratio", ci=True),
                     ref(nums, f"{q}.cost.p"), ref(nums, f"{q}.cost.p_holm")] for regime, policy, q, arm, cseg in done],
                   notes=f"{head} GM cost ratio: geometric mean over queries of cost main / cost control, 95% "
                         "bootstrap interval (20,000), sign-flip p (100,000), Holm over every p of this table; queries "
                         "with an error on either side are left out.")
        if len(scope.budgets) > 1:
            nums.table(f"PE6.{scope.name}.{sub}.fullcov",
                       ["Regime", "Policy", "K", "n", "FullCov main", "FullCov control", "main - control (pp)", "p",
                        "p (Holm)"],
                       [[regime, policy, str(k), ref(nums, f"{q}.n"),
                         ref(nums, f"{arm}.{segment(rev.MAIN_ARM)}.fullcov.K{k}", ci=True),
                         ref(nums, f"{arm}.{cseg}.fullcov.K{k}", ci=True), ref(nums, f"{q}.fullcov.K{k}.diff", ci=True),
                         ref(nums, f"{q}.fullcov.K{k}.p"), ref(nums, f"{q}.fullcov.K{k}.p_holm")]
                        for regime, policy, q, arm, cseg in done for k in scope.budgets[:-1]],
                       notes=f"{head} FullCov@K in % and the paired difference in pp (an error counts as not covered), "
                             "95% bootstrap interval (20,000), sign-flip p (100,000), Holm over every p of this "
                             "table; every K but P.")


# ----------------------------------------------------------------------------
# Layerwise (E6)
# ----------------------------------------------------------------------------
def layer_x(layer: str, layers: int) -> int:
    """input -1, proj 0, message-passing layer k at k, readout L+1."""
    fixed = {"input": -1, "proj": 0, "readout": int(layers) + 1}
    if layer in fixed:
        return fixed[layer]
    if layer.isdigit() and 1 <= int(layer) <= int(layers):
        return int(layer)
    refuse(f"layer {layer!r} is not one of {oversmoothing.layer_names(layers)}")


def reduce_layerwise(nums: nj.Numbers, runs: Sequence[LayerwiseRun]) -> None:
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], str]]] = {}
    for run in runs:
        for rec in run.records:
            groups.setdefault((str(rec["dataset"]), str(rec["regime"]), str(rec["partition"])), []).append(
                (rec, run.run_id))
    metrics = oversmoothing.metric_names()
    for dataset, regime, partition in sorted(groups):
        items = groups[(dataset, regime, partition)]
        where = f"layerwise {dataset} {regime} {partition}"
        sources = sorted({run_id for _, run_id in items})
        try:
            rows = oversmoothing.summary_rows([rec for rec, _ in items])
        except ValueError as exc:
            refuse(f"{where}: {exc}")
        models: dict[str, list[dict[str, Any]]] = {}
        idents: dict[str, tuple] = {}
        for row in rows:
            label = str(row["arm"]) + (oversmoothing.KNOCKOUT_SUFFIX if row["knockout"] else "")
            ident = (row["arch"], row["layers"], row["residual"], row["knockout"], row["arm"])
            if idents.setdefault(label, ident) != ident:
                refuse(f"{where}: two models are labelled {label} ({idents[label]}, {ident})")
            models.setdefault(label, []).append(row)
        segs = {label: segment(label) for label in models}
        if len(set(segs.values())) != len(segs):
            refuse(f"{where}: model labels {sorted(segs)} clash as key segments")
        styles: dict[str, str] = {}
        for label in models:
            if styles.setdefault(series_key(label), label) != label:
                refuse(f"{where}: models {styles[series_key(label)]} and {label} would share one figure style")
        base = f"{LAYERWISE}.{scope_name(dataset, partition)}.{segment(regime)}"
        order = [arm + suffix for arm in (rev.MAIN_ARM, *architecture_arms(dataset))
                 for suffix in ("", oversmoothing.KNOCKOUT_SUFFIX)]
        figure_rows: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
        for label in ordered(models, order):
            mrows = models[label]
            first = mrows[0]
            prefix = f"{base}.{segs[label]}"
            n_seeds = int(first["n_seeds"])
            nums.number(f"{prefix}.seeds", n_seeds, "count", n=n_seeds, source=sources)
            table = []
            for row in mrows:
                layer = str(row["layer"])
                cells: list[Any] = [layer, str(row["dim"])]
                for metric in metrics:
                    mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
                    if mean is None:
                        cells += [NA, NA]
                        continue
                    key = f"{prefix}.layer-{segment(layer)}.{metric}"
                    digits = LAYERWISE_DIGITS.get(metric, 3)
                    nums.number(f"{key}.mean", float(mean), "number", n=n_seeds, digits=digits, source=sources)
                    cells.append({"key": f"{key}.mean"})
                    if sd is None:
                        cells.append(NA)
                    else:
                        nums.number(f"{key}.sd", float(sd), "number", n=n_seeds, digits=digits, source=sources)
                        cells.append({"key": f"{key}.sd"})
                    figure_rows[metric].append({"series": label, "x": layer_x(layer, int(row["layers"])),
                                                "y": float(mean), "lo": None, "hi": None})
                table.append(cells)
            knockout = (" Knockout: the trained weights evaluated with the residual removed at inference, not a "
                        "model trained without it." if first["knockout"] else "")
            nums.table(prefix, ["Layer", "Width", *[f"{m} {s}" for m in metrics for s in ("mean", "SD")]], table,
                       notes=f"E6 layerwise, {dataset} {regime} ({partition}): {first['arch']}, {first['layers']} "
                             f"layers, residual {'on' if first['residual'] else 'off'}, arm {first['arm']}; mean and "
                             f"sample SD over training seeds {first['seeds']} "
                             f"({'every' if first['seeds_complete'] else 'not every'} protocol seed); "
                             f"{first['sample_blocks']} blocks and {first['sample_nodes']} nodes sampled. Definitions: "
                             "oversmoothing.py. n/a: a null metric (node metrics at readout, pairwise metrics over "
                             f"one row) or the SD of one seed.{knockout}")
        for metric in metrics:
            if figure_rows[metric]:
                nums.figure(f"{base}.{metric}", "curve", figure_rows[metric],
                            x="layer (input -1, proj 0, message passing 1..L, readout L+1)", y=metric,
                            notes=f"E6 layerwise, {dataset} {regime}: {metric}, mean over training seeds per layer, "
                                  "one series per model (+knockout: the residual removed at inference).")


# ----------------------------------------------------------------------------
# Reduction
# ----------------------------------------------------------------------------
def reduce(*, e4: Sequence[str] = (), e5: Sequence[str] = (), e6: Sequence[str] = (), layerwise: Sequence[str] = (),
           out: str, allow_dev: bool = False, by_family: bool = False,
           pattern_audit: Sequence[str] = ()) -> dict[str, Any]:
    """Read and check every input, reduce, and write OUT/numbers.json and the figure CSVs last."""
    check_statistics()
    inputs = [str(d) for d in (*e4, *e5, *e6, *layerwise, *pattern_audit)]
    if not inputs:
        refuse("no inputs: give --e4, --e5, --e6 or --layerwise run directories")
    check_out(str(out), inputs)
    retrieval = [read_retrieval_run(str(d), exp) for exp, dirs in (("E4", e4), ("E5", e5), ("E6", e6)) for d in dirs]
    lw = [read_layerwise_run(str(d)) for d in layerwise]
    audits = [read_audit(str(d)) for d in pattern_audit]
    split = check_split(retrieval, lw, allow_dev)
    audit_of = audits_by_regime(audits, split)
    run_hash = check_hash([*retrieval, *lw, *audits])
    if run_hash != protocol.protocol_hash():  # as reducer.py: numbers come from the code's protocol only
        refuse(f"the runs' protocol hash {run_hash[:12]} is not the code's {protocol.protocol_hash()[:12]}")
    warnings: list[str] = []
    assign_run_ids([*retrieval, *lw, *audits])
    unfrozen = sorted(run.given for run in (*retrieval, *lw, *audits) if not run.protocol_frozen)
    frozen = protocol.FROZEN_PROTOCOL_HASH is not None and run_hash == protocol.FROZEN_PROTOCOL_HASH and not unfrozen
    if unfrozen and protocol.FROZEN_PROTOCOL_HASH is not None and run_hash == protocol.FROZEN_PROTOCOL_HASH:
        warnings.append(f"{len(unfrozen)} inputs were run before the protocol was frozen (manifest protocol_frozen "
                        f"is not true), e.g. {unfrozen[0]}; frozen is false")
    nums = nj.Numbers(split=split, frozen=frozen, protocol_hash=run_hash)
    for run in sorted([*retrieval, *lw, *audits], key=lambda r: r.run_id):
        nums.source(run.run_id, path=Path(run.given).as_posix(), manifest_sha256=run.manifest_sha256,
                    experiment=run.experiment, dataset=run.dataset)
    scopes = merge_scopes(retrieval)
    for audit in audits:
        if not any(s.experiment == "E4" and s.dataset == audit.dataset for s in scopes):
            warnings.append(f"{audit.given} audits {audit.dataset}, which no --e4 run evaluates")
    for scope in scopes:
        if scope.experiment == "E4":
            reduce_e4(nums, scope, by_family, warnings, audit_of)
        else:
            assembled = reduce_matrix(nums, scope)
            if scope.experiment == "E5":
                reduce_pe6(nums, scope, assembled, warnings)
    reduce_layerwise(nums, lw)
    figures_dir = Path(out) / "figures"
    if figures_dir.is_dir():
        stale = sorted(p.stem for p in figures_dir.glob("*.csv") if p.stem not in nums.figures)
        if stale:
            warnings.append(f"{len(stale)} figure CSVs in {figures_dir} are not this reduction's, e.g. {stale[:3]}")
    path = nums.write(out)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return {"out": str(path), "split": split, "frozen": frozen, "protocol_hash": run_hash,
            "sources": sorted(nums.sources), "numbers": len(nums.numbers), "tables": len(nums.tables),
            "figures": len(nums.figures), "warnings": warnings}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, what in (("--e4", "E4"), ("--e5", "E5"), ("--e6", "E6")):
        parser.add_argument(flag, nargs="+", action="extend", default=[], metavar="DIR",
                            help=f"retrieval_eval runs of experiment {what}")
    parser.add_argument("--layerwise", nargs="+", action="extend", default=[], metavar="DIR",
                        help="oversmoothing.py runs (experiment E6_layerwise)")
    parser.add_argument("--out", required=True, help="directory for numbers.json and figures/")
    parser.add_argument("--allow-dev", action="store_true", help="accept the dev and validation splits")
    parser.add_argument("--by-family", action="store_true", help="E4 also per query family (descriptive)")
    parser.add_argument("--pattern-audit", nargs="+", action="extend", default=[], metavar="DIR",
                        help="pattern_audit.py outputs: E4 also on the seen and the novel planted positives")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = reduce(e4=args.e4, e5=args.e5, e6=args.e6, layerwise=args.layerwise, out=args.out,
                     allow_dev=args.allow_dev, by_family=args.by_family, pattern_audit=args.pattern_audit)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
