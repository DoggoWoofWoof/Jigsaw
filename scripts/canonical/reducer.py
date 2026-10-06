"""Reducer: every number of the paper, from validated run directories (plan sections 4, 7 and 8).

    python -m scripts.canonical.reducer --plan PLAN.json --out OUT [--allow-dev] [--allow-missing]
        [--allow-uncertified] [--no-verify] [--allow-pe1-failure]

The plan names every input. scripts/canonical/launch/reduce_plan.py writes it for the canonical layout,
with a list of absolute paths per run entry and offline stage (<id> is a main-arm encoder, the system
whose offline cost E17 reports):

    {"runs": [{"experiment": "E1", "dataset": "arxiv", "dirs": ["RUNS/runs/E1/arxiv/shard0", ...]},
              {"experiment": "E7-classical", "dataset": "arxiv",
               "dirs": ["RUNS/runs/E7/arxiv/classical_cfl", ...]}, ...],
     "certifications": {"arxiv": {"R0": "QUERIES/arxiv/test/certification/R0.json", ...}},
     "e2_root": "RUNS/runs/E2", "e10_root": "RUNS/runs/E10",
     "offline": {"arxiv": {"prepare": ["STORE/arxiv/_prepare_run"],
                           "label_index": ["RUNS/offline/arxiv/label_index"],
                           "rankers": ["RUNS/offline/arxiv/rankers-primary"],
                           "embeddings": ["RUNS/offline/arxiv/embed-<id>-primary", ...],
                           "training": ["RUNS/train/<id>", ...], "router_fit": ["RUNS/router_fit/arxiv"]}},
     "pattern_audit": {"arxiv": ["RUNS/audit/pattern/arxiv-test"]}}

A hand-written plan may give a run dir, an offline stage or an audit as a single path instead of a list,
and any of them as a glob; a relative path is relative to the plan file.

Run experiments are run_experiment's E1, E3x, E7, E8, E9 and E16 and run_external's E7-classical and
E12 (both join E7's analysis). e2_root and e10_root are streaming.py's measurement trees (E2 feeds E19,
PE3 and PE4; E10, the papers100M scale-out, gets the same reduction under the prefix "E10."), the
certifications are certify_negatives.py's files (E14), offline names the run dirs of the offline
stages (E17, costed by offline_build.offline_cost; GNN-PE's offline stage comes from the E12 manifests)
and pattern_audit names pattern_audit.py's output directories (E20).
OUT gets numbers.json and figures/*.csv (numbers_json), tables/*.csv, pe1.json,
exclusions.json, failure_localization.json and a manifest (experiment "reduce"). The reducer never
imports torch or seal (run_experiment, certify_negatives and streaming import both, so the few names
needed from them are repeated here and kept equal by test_reducer).

Refusals. Every check raises ReduceError; the CLI prints it and exits 2 before writing anything.
  * Manifests: status complete, the reducer's protocol hash, outputs that re-hash to the manifest
    (manifest.verify_manifest; --no-verify skips only the re-hashing), the plan entry's experiment
    (E7-classical runs are run_external's "E7"; --allow-dev also takes the ad-hoc "arms" and "external")
    and dataset, and a sensitivity partition only in E16.
  * One split over every run, certification and measured cell, in queries.TEST_SPLITS unless
    --allow-dev; on a test split every run has its experiment's T. One query set (the manifest's
    sha256 of input "queries/<regime>") per dataset and regime over every run, E16's partitions and the
    measured cells included.
  * Records: schema, dataset, regime, arm, method and T as configured; a known status, kind and family;
    one file per configured arm with config.query_sets[regime].selected records; complete shards (i in
    0..n-1 of one n, never beside an unsharded run); no (arm, uid) twice; one uid set for every arm of an
    experiment, dataset and regime (all E16 partitions together; E3x is exempt). --allow-missing reduces
    over the intersection and lists each dropped record in exclusions.json.
  * E1: a seed-averaged system runs the protocol's training seeds (any one set with --allow-dev) on the
    same queries (stats.seed_average's rule); a fixed method has no seed.
  * Certifications (jigsaw-certification-v1, certify_negatives.load_certification's checks): complete;
    the runs' protocol hash, split, dataset, regime and query set (query_set.sha256); "found" only with
    witness_check "ok"; every negative covered with its family and size, and no positive. Negatives
    without one are refused unless --allow-uncertified.
  * E3x reruns only queries E1 ran with the same arm and left undecided, with a longer T.
  * Measured trees: a root holds at least one cell; every rep<k> holds its cell's measure.json of the
    root's experiment (a file without "experiment" is E2's); rep<k>.attemptN are superseded attempts and
    skipped; a completed cell's run/ passes the run checks and ran the cell's system and mode (on a dev
    split its arm may carry the "+backend=X" of a --backend sweep); a failed one (a cap kill, a timeout)
    counts for PE4 only.
  * Offline stages: complete manifests with unchanged outputs (no protocol hash: they precede the
    protocol's final hash) that offline_build.offline_cost can cost; one prepare run per dataset; "disk"
    and "gnnpe" are not stage names. Every E12 GNN-PE run records each regime's build
    (extra.prepare.<regime>.offline_seconds and prepared_utc, gnnpe_adapter.prepare).
  * Pattern audits (E20): a complete manifest of experiment pattern_audit under the reducer's protocol
    hash; a complete jigsaw-pattern-audit-v1 file of the listed dataset and the reduction's split; one
    audit per dataset and regime; the runs' query set; on a test split exactly the main encoders of
    every training seed (no smoke run); every learned arm of E1 and E7 served by an audited encoder
    (config.models, else the arm's model); every E1 and E7 query in the audited set; no seen negative
    certified none (the training query isomorphic to it is an embedding; an uncertified or undecided
    one is noted).

Truth. A positive is planted. A negative takes its certification's label: "none" ones are in the
negative rates; "found" (an embedding exists) and "undecided" ones leave every rate and are listed in
exclusions.json and counted (relabelled, undecided). Without a certification file (--allow-uncertified)
a negative keeps its query set's label: an "uncertified" one enters the negative rates provisionally
(noted on every table of its dataset and regime, counted, and listed under "provisional" in
exclusions.json) unless a record of the reduction holds a verified witness for it, which relabels it
"found". A query has an embedding when it is a positive, a "found" negative, or any query some record
of the reduction found with a verified witness. On an "undecided" negative a verified FOUND is "newly
found" and a NONE proof on one without a witness "newly certified".

Outcomes at t <= T (failure_localize): a positive is solved@t when FOUND with a checked witness by t; a
negative is correct@t when a NONE proof (time_decided set) came by t. An external NONE that proves nothing
(GNN-PE) is undecided; on a query with an embedding it is that method's false NONE, reported apart.
PE1 counts per run, experiment and system the false positives (a rejected witness), false negatives (a
NONE proof on a query with an embedding), contradictions (a verified FOUND on a "none" negative) and
crashes (any other ERROR); it holds when all are 0 on every cascade system (external methods are counted
under "external"). When PE1 fails the CLI writes everything and exits 3, unless --allow-pe1-failure.

Systems. In E1 an arm's system drops "@SEED" (a +model id's "-seed<d>-" becomes "-seed*-"); its replicate
is train_seed, else the model id's seed, else none. Seed-averaged systems are averaged per query before
any test. Elsewhere arms are systems ("jigsaw@7301", "jigsaw@7301+backend=glasgow"); a measured cell's system
is its measure.json system, keyed with its mode (filterall runs in both). A "+backend=X" shared by every
cascade arm of an experiment and dataset (a dev --backend run) is dropped from the names.

Statistics (protocol.STATISTICS through stats.py). Rates (and figure bands): Wilson 95% over queries
(rate_interval wilson_95, unit query). Replicates are training seeds, averaged per query before any
test: a seed-averaged rate is p = the mean over queries of the per-query seed means p_i, with the Wilson
interval of that mean (stats.wilson_interval of sum p_i successes in n = the number of queries). It is
the ordinary Wilson interval when every query's seeds agree and conservative otherwise (Var(p_i) <=
p(1 - p) for p_i in [0, 1]); each seed's own Wilson rate and the seed range are reported beside it.
Counts of a seed-averaged system are summed over its seeds. Paired differences are in percentage points:
stats.paired_compare (bootstrap interval, sign-flip p) and, when a side is seed-averaged,
stats.hierarchical_bootstrap_diff as the pre-registered second interval (diff_hier; the only
hierarchical bootstrap). The p of E7, E8 and E9 is stats.exact_mcnemar (deterministic arms); of PE2 and
E16 the sign-flip p. Holm within each table's family. PE3: stats.paired_log_ratio, with bytes read + 1
(a zero-byte warm read stays positive) and latency = elapsed censored at T (at least 1 microsecond).
E19: stats.quantile_with_ci. Draw counts come from DRAWS; the CLI always uses PROTOCOL_DRAWS. Measured
repeats: repeat 1 (else the lowest completed one) is the measurement for PE3, E18 and E19; PE3 adds the
min and max geometric-mean ratio over every completed repeat as a stability check.

Key grammar (numbers_json.KEY_RE; the paper cites \\jnum{KEY}): dot-separated segments.
  * A name in [A-Za-z0-9_]+ is its own segment. Any other name is escaped character by character: a
    character outside [A-Za-z0-9_] becomes the token "-NAME-" (dash at plus eq dot slash star sp colon
    comma) or "-xHEX-" (its code point): "jigsaw@7301+backend=gql" is "jigsaw-at-7301-plus-backend-eq-gql".
    Every "-" of an escaped name delimits a token, so unescape() inverts escape() and an escaped name
    never equals a plain one. (Bare hex such as "-40" for "@" would let the alias R2-128 read as an escape.)
  * Closed vocabularies keep their names: regimes (R0 R1 R2-512 R2-128 R2-32), families (protocol and
    queries.SM_FAMILIES; others are refused), E9 aliases (full halo0 prune_passes0 components_none
    component_cap50 label_filter0), E16 aliases (jigsaw halo0 halo<N> seeds<N> topk jigsaw_index
    filterall) and partitions (primary kahip_fast bfs_random metis_x0p5 metis_x2).
  * Horizons t<s> (t30; t0p5 for 0.5 s), sizes n<k>, seeds s<seed>, caps cap<GB>|capnone, and pairs
    "<A>_vs_<B>" (a name holding "_vs_" is refused). A horizon above a run's T is never reported; a dev
    run shorter than 30 s reports its "t30" numbers at t<T>.
  With S a system, h a horizon and {a,b} a choice:
  E1.<ds>.<r>.<S>.solved.<h>  (+ .s<seed> .seed_min .seed_max when seed-averaged, on every E1 rate)
  E1.<ds>.<r>.<S>[.<fam>][.n<k>].solved.t30 ; E1.<ds>.<r>.<S>[.<negfam>].correct_none.t30
  E1.<ds>.<r>.<S>.{fp,fn,crash,incomplete,newly_found,newly_certified,timeout_pos.t30,timeout_neg.t30}
  E1.<ds>.<r>.<S>.{cand_nodes,cand_edges,solver_nodes}.{median,p95} ; E1.<ds>.<r>.<S>.time.<stage>.
    {mean,median,solved_mean,solved_median} ; E1.<ds>.<r>.<S>.{route.<route>,selectivity_mean} ;
    E1.<ds>.<r>.{relabelled,undecided,uncertified}
  PE2.<ds>.<r>.jigsaw_vs_filterall.{diff,p,p_holm,noninferior,superior,wins,losses,ties,diff_hier} ;
    PE2.<ds>.<r>.<S>_vs_{filterall,jigsaw_index}.{diff,p,p_holm,wins,losses,ties,diff_hier} ;
    PE2.<ds>.<r>.c2.{<category>,pairs} ; PE2.margin
  E3x.<ds>.<r>.<S>.{solved.t5,solved.t30,solved.t120,solved.t600,rerun,not_rerun,newly_solved}
  E7.<ds>.<r>.<S>.{solved.t30,solved.t120,correct_none.t120,tts_median,fp,fn,crash,none_unverified,
    false_none,peak_rss_bytes.max,peak_rss_bytes.median,route.<route>,selectivity_mean} ;
    E7.<ds>.<r>.<A>_vs_best_classical.{best,b,c,p,p_holm,diff} ; E7.<ds>.<r>.<A>_vs_gnnpe.{b,c,p,diff} ;
    E7.<ds>.selectivity.{q25,q50,q75} ; E7.<ds>.<S>.{quartile.q1..q4,log10.b0..b3}.solved.t120
  E8.<ds>.<r>.<S>[.<fam>.n<k>].solved.{t30,t600} ; E8.<ds>.<r>.<S>.{tts_median,peak_rss_bytes.max} ;
    E8.<ds>.<r>.direct_vs_filterall.<h>.{b,c,p,p_holm,diff}
  E9.<ds>.<r>.<alias>.{solved.t30,correct_none.t30,incomplete,cand_nodes.median,elapsed.median} ;
    E9.<ds>.<r>.<alias>_vs_full.{pos,neg}.{diff,p,p_holm,b,c}
  E16.<ds>.<r>.<partition>.<alias>.{solved.t30,correct_none.t30,cand_nodes.median,elapsed.median} ;
    E16.<ds>.<r>.<partition>.budgets ; E16.<ds>.<r>.primary.<alias>_vs_jigsaw.{diff,p,p_holm} ;
    E16.<ds>.<r>.<partition>.jigsaw_vs_filterall.{diff,p,p_holm}
  E14.<ds>.<r>.{generated,none,found,undecided,rejected_witnesses,errors} ;
    E14.<ds>.<r>.<fam>.n<k>.{generated,none,found,undecided,none.<stage>,found.<stage>} ;
    E14.<ds>.<r>.<stage>.{seconds_median,seconds_max}
  E17.<ds>.<stage>.{wall_seconds,wall_seconds_median,bytes,peak_bytes,runs,machine} (offline_cost; the
    median when a stage has several runs, bytes and peak_bytes when the stage records them) ;
    E17.<ds>.prepare.<prepare stage>.{wall_seconds,peak_bytes} ;
    E17.<ds>.disk.{graph,sigma,meta,partition.<partition>,total} ;
    E17.<ds>.training.{gpu_hours,gpu_hours_per_encoder,gpu} ;
    E17.<ds>.training.{quiet_epoch_share,gpu_hours_quiet,gpu_hours_per_encoder_quiet} (offline_build.quiet_training:
    the share when the logs record the machine's load, the hours when every encoder has enough quiet epochs) ;
    E17.<ds>.gnnpe.<r>.{offline_seconds,builds}
  E18.<exp>.<ds>.<r>.<S>[.<setting>][.<fam>].{pos,neg}.<category>.<h> ;
    E18.<exp>.<ds>.<r>.<S>[.<setting>].overlap_rescued  (setting: E16 partition, E2/E10 <mode>_<cache>)
  [E10.]E19.<ds>.<S>.{<mode>_<cache>,e1_resident}.{<stage>,elapsed,wall_seconds}.{p50,p95,p99}
  [E10.]PE3.<ds>.<cache>.<A>_vs_<B>.{bytes,latency}.{gm_ratio,p,p_holm,median_ratio,n,gm_ratio_min,
    gm_ratio_max,repeats}
  [E10.]PE4.<ds>.<S>.<mode>.<cache>.<cap>.{peak_rss.median,peak_rss.min,peak_rss.max,repeats,success,
    solved.t30} ; [E10.]PE4.<ds>.<cache>.<J>_vs_<R>.{conservative_ratio,median_ratio}
    (J Jigsaw streaming, R a resident system; an E10 key is E2's with "E10." in front)
  PE1.{holds,fp,fn,contradiction,crash,found,witness_checked,witness_passed,none_unverified,false_none,
    witness_pass} ; PE1.<exp>.<metric> ; PE1.external.<metric>
  E20.<ds>.<r>.{queries,positives,negatives,seen_positives,seen_negatives,encoders,seen_share,seen.s<seed>} ;
    E20.<g>.<ds>.<r>.positives.{seen,novel} ; E20.<g>.<ds>.<r>.<S>.{seen,novel}.solved.<h> ;
    E20.E1.<ds>.<r>.jigsaw_vs_{filterall,jigsaw_index}.novel.{diff,p,wins,losses,ties,diff_hier}
    (g is E1 at its primary horizon or E7 at 120 s; seen: some audited encoder of the regime drew an
    isomorphic training query; novel: none did)
  Examples: E1.arxiv.R0.jigsaw.solved.t30, E1.arxiv.R0.jigsaw.k_hop.n50.solved.t30,
  PE2.arxiv.R0.jigsaw_vs_filterall.p_holm, E16.mag.R0.metis_x0p5.jigsaw.solved.t30,
  E10.PE4.papers100m.jigsaw-at-7301.streaming.cold.cap32.success.
Table ids (tables/<id>.csv) and figure ids (figures/<id>.csv) follow the same grammar.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import failure_localize as fl
from . import manifest, numbers_json, offline_build, protocol, queries, stats


class ReduceError(Exception):
    """An input the reducer refuses; the message names it."""


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
RECORD_SCHEMA = "jigsaw-run-record-v1"
CERT_SCHEMA = "jigsaw-certification-v1"
MEASURE_SCHEMA = "jigsaw-e2-measure-v1"
LEARNED_METHODS = ("jigsaw", "jigsaw_learned", "jigsaw_fused", "mean_rrf")
FIXED_METHODS = ("jigsaw_index", "mean_feature", "topo_feature", "random", "filterall")
ABLATIONS = ("", "+halo=0", "+prune_passes=0", "+component_mode=none", "+component_cap=50", "+label_filter=0")
E9_ALIASES = {"": "full", "+halo=0": "halo0", "+prune_passes=0": "prune_passes0",
              "+component_mode=none": "components_none", "+component_cap=50": "component_cap50",
              "+label_filter=0": "label_filter0"}
CLASSICAL = ("classical_gql", "classical_cfl", "classical_dpiso")
EXTERNAL_METHODS = tuple(m for m, spec in protocol.METHODS.items() if "external" in spec)
STATUSES = ("FOUND", "NONE", "TIMEOUT", "ERROR", "INCOMPLETE")
FAMILIES = (*protocol.POSITIVE_FAMILIES, *queries.SM_FAMILIES.values(), *protocol.NEGATIVE_FAMILIES)
GROUP = {"E1": "E1", "E3x": "E3x", "E7": "E7", "E7-classical": "E7", "E12": "E7", "E8": "E8", "E9": "E9",
         "E16": "E16"}
MEASURED = ("E2", "E10")  # streaming.py's experiments: one measurement tree each
MANIFEST_EXPERIMENT = {"E1": "E1", "E3x": "E3x", "E7": "E7", "E7-classical": "E7", "E12": "E12", "E8": "E8",
                       "E9": "E9", "E16": "E16", "E2": "E2", "E10": "E10"}
EXTERNAL_ENTRIES = ("E7-classical", "E12")
DEV_EXPERIMENT = {False: "arms", True: "external"}  # the manifest name of a run without --experiment
PLAN_KEYS = ("runs", "certifications", "e2_root", "e10_root", "offline", "pattern_audit")
RESERVED_STAGES = ("disk", "gnnpe")  # E17.<ds>.disk and E17.<ds>.gnnpe are not offline stages
AUDIT_SCHEMA = "jigsaw-pattern-audit-v1"  # pattern_audit.py's (it imports torch through train_encoder)
AUDIT_EXPERIMENT = "pattern_audit"
AUDIT_FILE = "pattern_audit.json"
E20_GROUPS = ("E1", "E7")
E18_EXPERIMENTS = ("E1", "E7", "E8", "E9", "E16", *MEASURED)
E18_HORIZONS = (1.0, 5.0, 30.0, 120.0)
E3X_HORIZONS = (5.0, 30.0, 120.0, 600.0)
LOG10_BINS = (("b0", 0.0, 1.0), ("b1", 1.0, 2.0), ("b2", 2.0, 3.0), ("b3", 3.0, math.inf))
PE1_VIOLATIONS = ("fp", "fn", "contradiction", "crash")
PE1_COUNTS = (*PE1_VIOLATIONS, "found", "witness_checked", "witness_passed", "none_unverified", "false_none")
EXCLUDED_TRUTHS = {"negative_found": "relabelled", "negative_undecided": "undecided"}
RATE_NEGATIVES = ("negative_none", "negative_uncertified")  # the second only under --allow-uncertified
PROTOCOL_DRAWS = {"boot": 20_000, "perm": 100_000, "quantile": 5_000}
# The draw counts of every resampling. Tests lower them (monkeypatch.setitem); main() restores PROTOCOL_DRAWS.
DRAWS: dict[str, int] = dict(PROTOCOL_DRAWS)
SEED = int(protocol.STATISTICS["bootstrap_seed"])
LATENCY_FLOOR = 1e-6
_TOL = 1e-9


def first_seed() -> int:
    return int(protocol.SEEDS["training"][0])


def first_arm() -> str:
    """The single-seed Jigsaw arm of E2, E3x, E7, E9, E10 and E16."""
    return f"jigsaw@{first_seed()}"


def sensitivity_partitions() -> list[str]:
    """E16's partitions besides the primary one (run_experiment.sensitivity_partitions)."""
    scales = [float(s) for s in protocol.PARTITIONING["partition_scale_for_sensitivity"] if float(s) != 1.0]
    return list(protocol.PARTITIONING["alternatives_for_sensitivity"]) + [f"metis_x{s:g}" for s in scales]


def planned_T(entry: str) -> float:
    """The T of each experiment's runs (run_experiment's and run_external's experiment_plan)."""
    t_max = protocol.TIME["run_T_max_seconds"]
    primary = protocol.TIME["primary_T_seconds"]
    return float({"E1": t_max["E1"], "E3x": t_max["E3_extension"], "E7": t_max["E7"], "E7-classical": t_max["E7"],
                  "E12": t_max.get("E12", t_max["E1"]), "E8": max(t_max["E8"]), "E9": primary, "E16": primary,
                  "E2": primary, "E10": primary}[entry])


def planned_caps(experiment: str) -> list[float | None]:
    """Uncapped, then the registered memory caps of a measured experiment (streaming.plan_caps)."""
    serving = protocol.SERVING if experiment == "E2" else protocol.SERVING["scale_out"]
    return [None, *(float(c) for c in serving["memory_caps_gb"])]


def primary_horizon(T: float) -> float:
    """30 s, or the whole run when a dev run is shorter."""
    return min(float(protocol.TIME["primary_T_seconds"]), float(T))


# ----------------------------------------------------------------------------
# Key grammar
# ----------------------------------------------------------------------------
_PLAIN = re.compile(r"^[A-Za-z0-9_]+$")
_TOKENS = {"-": "dash", "@": "at", "+": "plus", "=": "eq", ".": "dot", "/": "slash", "*": "star", " ": "sp",
           ":": "colon", ",": "comma"}
_CHARS = {name: ch for ch, name in _TOKENS.items()}
_HEX_TOKEN = re.compile(r"^x[0-9a-f]+$")


def escape(name: Any) -> str:
    """A key segment for any non-empty name; a plain name [A-Za-z0-9_]+ is its own segment."""
    text = str(name)
    if not text:
        raise ReduceError("an empty name has no key segment")
    if _PLAIN.match(text):
        return text
    out = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            out.append(ch)
        else:
            out.append("-" + (_TOKENS.get(ch) or f"x{ord(ch):x}") + "-")
    return "".join(out)


def unescape(segment: str) -> str:
    """The name escape() turned into this segment."""
    parts = segment.split("-")
    if len(parts) % 2 == 0:
        raise ValueError(f"{segment!r} is not an escaped name")
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part and not _PLAIN.match(part):
                raise ValueError(f"{segment!r} is not an escaped name")
            out.append(part)
        elif part in _CHARS:
            out.append(_CHARS[part])
        elif _HEX_TOKEN.match(part):
            out.append(chr(int(part[1:], 16)))
        else:
            raise ValueError(f"bad token {part!r} in {segment!r}")
    return "".join(out)


def _number_text(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def seg_horizon(t: float) -> str:
    return "t" + _number_text(t)


def seg_size(size: Any) -> str:
    return f"n{int(size)}"


def seg_seed(seed: Any) -> str:
    return f"s{int(seed)}"


def seg_cap(cap: float | None) -> str:
    return "capnone" if cap is None else "cap" + _number_text(cap)


def seg_regime(regime: str) -> str:
    if regime not in protocol.TRAINED_REGIMES:
        raise ReduceError(f"unknown regime {regime!r}")
    return regime


def partition_aliases() -> dict[str, str]:
    return {"primary": "primary", **{p: p.replace(".", "p") for p in sensitivity_partitions()}}


def seg_partition(partition: str) -> str:
    aliases = partition_aliases()
    if partition not in aliases:
        raise ReduceError(f"unknown partition {partition!r}; E16 runs on {sorted(aliases)}")
    return aliases[partition]


def seg_family(family: str) -> str:
    if family not in FAMILIES:
        raise ReduceError(f"unknown query family {family!r}")
    return family


def seg_system(name: str) -> str:
    if "_vs_" in str(name):
        raise ReduceError(f"a name holding '_vs_' cannot be keyed: {name!r}")
    return escape(name)


def key(*segments: str) -> str:
    return numbers_json.check_key(".".join(str(s) for s in segments))


def measured_key(experiment: str, *segments: str) -> str:
    """E2's families as they are (E19, PE3, PE4); E10's under "E10." (E10.PE4.papers100m...)."""
    return key(*(() if experiment == "E2" else (experiment,)), *segments)


# ----------------------------------------------------------------------------
# Arms and systems
# ----------------------------------------------------------------------------
_MODEL_SEED = re.compile(r"-seed(\d+)-")


def parse_arm(name: str) -> tuple[str, int | None, dict[str, str]]:
    """METHOD[@SEED][+key=value]... as (method, seed, settings); values stay strings."""
    head, *pairs = str(name).split("+")
    method, _, seed = head.partition("@")
    settings: dict[str, str] = {}
    for pair in pairs:
        k, sep, v = pair.partition("=")
        if not sep or not k or k in settings:
            raise ReduceError(f"bad arm name {name!r}")
        settings[k] = v
    if seed and not seed.isdigit():
        raise ReduceError(f"bad arm name {name!r}")
    return method, int(seed) if seed else None, settings


def arm_name(method: str, seed: int | None, settings: Mapping[str, str]) -> str:
    """run_experiment.Arm.name: settings sorted by key."""
    text = method + (f"@{seed}" if seed is not None else "")
    return text + "".join(f"+{k}={v}" for k, v in sorted(settings.items()))


def strip_backend(arm: str, backend: str | None) -> str:
    """The arm without its "+backend=X" when X is backend."""
    method, seed, settings = parse_arm(arm)
    if backend is None or settings.get("backend") != backend:
        return arm
    del settings["backend"]
    return arm_name(method, seed, settings)


def e1_system(arm: str) -> str:
    """The E1 system of an arm: no training seed, a model id's seed as "seed*"."""
    method, _, settings = parse_arm(arm)
    if "model" in settings:
        settings["model"] = _MODEL_SEED.sub("-seed*-", settings["model"])
    return arm_name(method, None, settings)


def replicate_of(rec: Mapping[str, Any]) -> int | None:
    if rec.get("train_seed") is not None:
        return int(rec["train_seed"])
    found = _MODEL_SEED.search(str(rec.get("model_id") or ""))
    return int(found.group(1)) if found else None


def system_order(names: Iterable[str]) -> list[str]:
    rank = {m: i for i, m in enumerate((*LEARNED_METHODS, *FIXED_METHODS, "direct", *EXTERNAL_METHODS))}
    return sorted(names, key=lambda s: (rank.get(parse_arm(s)[0], len(rank)), s))


def e9_alias(system: str) -> tuple[str, tuple[str, int | None]]:
    """(alias, (method, seed)) of an E9 arm; the base must be common to the table."""
    method, seed, settings = parse_arm(system)
    suffix = "".join(f"+{k}={v}" for k, v in sorted(settings.items()))
    if suffix not in E9_ALIASES:
        raise ReduceError(f"E9 arm {system!r} is not one of run_experiment.ABLATIONS")
    return E9_ALIASES[suffix], (method, seed)


def e16_alias(system: str) -> tuple[str, int | None]:
    """(alias, seed) of an E16 arm: jigsaw, halo0, halo<N>, seeds<N>, topk, jigsaw_index or filterall."""
    method, seed, settings = parse_arm(system)
    if method in ("jigsaw_index", "filterall") and seed is None and not settings:
        return method, None
    if method == "jigsaw" and seed is not None:
        if not settings:
            return "jigsaw", seed
        if len(settings) == 1:
            (k, v), = settings.items()
            if k == "halo" and v == "0":
                return "halo0", seed
            if k == "halo_hops" and v.isdigit() and int(v) >= 2:
                return f"halo{int(v)}", seed
            if k == "seed_count" and v.isdigit() and int(v) >= 1:
                return f"seeds{int(v)}", seed
            if k == "policy" and v == "topk":
                return "topk", seed
    raise ReduceError(f"E16 arm {system!r} is not one of experiment_plan('E16')'s")


def main_model_id(dataset: str, regime: str, seed: int) -> str:
    """serving.default_model_id: the main-arm encoder of a training seed that serves regime."""
    trained, arch = protocol.TRAINED_REGIMES[regime], protocol.DATASETS[dataset]["encoder"]
    return f"{dataset}-{trained}-{arch}-seed{int(seed)}-main"


def arm_model_id(run: Run, regime: str, arm: str) -> str | None:
    """The encoder a learned arm served regime with: the run's config.models (run_experiment records the
    loaded model), else the arm's +model, else the main encoder of its seed; None for a fixed method."""
    ident = (run.config.get("models") or {}).get(f"{regime}/{arm}")
    if isinstance(ident, dict) and ident.get("model_id"):
        return str(ident["model_id"])
    method, seed, settings = parse_arm(arm)
    if method not in LEARNED_METHODS:
        return None
    if settings.get("model"):
        return settings["model"]
    return main_model_id(run.dataset, regime, seed) if seed is not None else None


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Flags:
    allow_dev: bool = False
    allow_missing: bool = False
    allow_uncertified: bool = False
    no_verify: bool = False
    allow_pe1_failure: bool = False


@dataclass
class Plan:
    path: Path
    sha256: str
    raw: dict[str, Any]
    entries: list[tuple[str, str, list[Path]]]
    certifications: dict[tuple[str, str], Path]
    measured: dict[str, Path]
    offline: dict[tuple[str, str], list[Path]]
    audits: dict[str, list[Path]] = field(default_factory=dict)


def _expand(base: Path, pattern: Any, what: str) -> list[Path]:
    if not isinstance(pattern, str) or not pattern:
        raise ReduceError(f"{what}: {pattern!r} is not a path")
    full = Path(pattern) if Path(pattern).is_absolute() else base / pattern
    if any(ch in pattern for ch in "*?["):
        # only the plan's pattern is a pattern: a "[" in the plan's own directory is a literal character
        prefix = "" if Path(pattern).is_absolute() else glob.escape(str(base)) + "/"
        hits = sorted(Path(p) for p in glob.glob(prefix + pattern))
        if not hits:
            raise ReduceError(f"{what}: {pattern} matches nothing")
        return hits
    if not full.exists():
        raise ReduceError(f"{what}: {full} does not exist")
    return [full]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ReduceError(f"plan {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReduceError(f"plan {path}: not a JSON object")
    unknown = sorted(set(raw) - set(PLAN_KEYS))
    if unknown:
        raise ReduceError(f"plan {path}: unknown keys {unknown}; known: {', '.join(PLAN_KEYS)}")
    base = path.resolve().parent
    entries: list[tuple[str, str, list[Path]]] = []
    seen: dict[Path, str] = {}
    for i, item in enumerate(raw.get("runs") or []):
        what = f"plan runs[{i}]"
        if not isinstance(item, dict) or set(item) != {"experiment", "dataset", "dirs"}:
            raise ReduceError(f"{what}: needs exactly experiment, dataset and dirs")
        exp, ds = item["experiment"], item["dataset"]
        if exp not in GROUP:
            raise ReduceError(f"{what}: unknown experiment {exp!r}; known: {', '.join(GROUP)} (E2 and E10 are "
                              "read from e2_root and e10_root)")
        if ds not in protocol.DATASETS:
            raise ReduceError(f"{what}: unknown dataset {ds!r}")
        dirs = []
        for pattern in _as_list(item["dirs"]):
            for hit in _expand(base, pattern, what):
                if not hit.is_dir():
                    raise ReduceError(f"{what}: {hit} is not a directory")
                real = hit.resolve()
                if real in seen:
                    raise ReduceError(f"{what}: {real} is listed twice (also in {seen[real]})")
                seen[real] = what
                dirs.append(real)
        entries.append((exp, ds, dirs))
    certs: dict[tuple[str, str], Path] = {}
    for ds, per in (raw.get("certifications") or {}).items():
        if ds not in protocol.DATASETS or not isinstance(per, dict):
            raise ReduceError(f"plan certifications: bad entry {ds!r}")
        for regime, cpath in per.items():
            seg_regime(regime)
            hits = _expand(base, cpath, f"plan certifications {ds}/{regime}")
            if len(hits) != 1 or not hits[0].is_file():
                raise ReduceError(f"plan certifications {ds}/{regime}: needs one file, got {hits}")
            certs[(ds, regime)] = hits[0].resolve()
    measured: dict[str, Path] = {}
    for exp in MEASURED:
        name = f"{exp.lower()}_root"
        if raw.get(name) is not None:
            hits = _expand(base, raw[name], f"plan {name}")
            if len(hits) != 1 or not hits[0].is_dir():
                raise ReduceError(f"plan {name}: needs one directory, got {hits}")
            measured[exp] = hits[0].resolve()
    if len(set(measured.values())) != len(measured):
        raise ReduceError("plan: e2_root and e10_root name one directory (streaming.py keeps one experiment per root)")
    offline: dict[tuple[str, str], list[Path]] = {}
    for ds, stages in (raw.get("offline") or {}).items():
        if ds not in protocol.DATASETS or not isinstance(stages, dict):
            raise ReduceError(f"plan offline: bad entry {ds!r}")
        for stage, dirs in stages.items():
            if not isinstance(stage, str) or not stage:
                raise ReduceError(f"plan offline {ds}: bad stage {stage!r}")
            if stage in RESERVED_STAGES:
                raise ReduceError(f"plan offline {ds}: {stage!r} names E17 keys, not an offline stage")
            offline[(ds, stage)] = [h.resolve() for p in _as_list(dirs)
                                    for h in _expand(base, p, f"plan offline {ds}/{stage}")]
    audits: dict[str, list[Path]] = {}
    for ds, dirs in (raw.get("pattern_audit") or {}).items():
        if ds not in protocol.DATASETS:
            raise ReduceError(f"plan pattern_audit: unknown dataset {ds!r}")
        for pattern in _as_list(dirs):
            for hit in _expand(base, pattern, f"plan pattern_audit {ds}"):
                real = hit.resolve()
                if not real.is_dir():
                    raise ReduceError(f"plan pattern_audit {ds}: {real} is not a directory")
                if real in seen:
                    raise ReduceError(f"plan pattern_audit {ds}: {real} is listed twice (also in {seen[real]})")
                seen[real] = f"plan pattern_audit {ds}"
                audits.setdefault(ds, []).append(real)
    if not (entries or certs or measured or offline or audits):
        raise ReduceError(f"plan {path}: names no runs, certifications, measurement roots, offline stages or "
                          "pattern audits")
    return Plan(path.resolve(), hashlib.sha256(data).hexdigest(), raw, entries, certs, measured, offline, audits)


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
@dataclass
class Run:
    run_id: str
    path: Path
    entry: str
    group: str
    dataset: str
    partition: str
    split: str
    frozen: bool
    T: float
    shard: tuple[int, int] | None
    regimes: list[str]
    arms: list[str]
    external: bool
    manifest_sha256: str
    query_sha: dict[str, str]
    selected: dict[str, int]
    config: dict[str, Any]
    cell: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # the manifest's extra (E12: GNN-PE's builds)


def read_manifest(path: Path, *, H: str | None, flags: Flags) -> tuple[dict[str, Any], str]:
    """A run's manifest: complete, of protocol H (when given), outputs unchanged unless --no-verify."""
    mpath = path / manifest.MANIFEST_NAME
    if not mpath.is_file():
        raise ReduceError(f"{path}: no {manifest.MANIFEST_NAME}")
    data = mpath.read_bytes()
    try:
        doc = json.loads(data)
    except ValueError as exc:
        raise ReduceError(f"{mpath}: {exc}") from exc
    if doc.get("status") != "complete":
        raise ReduceError(f"{path}: manifest status {doc.get('status')!r}, not 'complete'")
    if H is not None and doc.get("protocol_hash") != H:
        raise ReduceError(f"{path}: run under protocol {str(doc.get('protocol_hash'))[:12]}..., "
                          f"the reducer's is {H[:12]}...")
    if not flags.no_verify:
        problems = manifest.verify_manifest(path, expect_protocol_hash=H)
        if problems:
            raise ReduceError(f"{path}: differs from its manifest: {'; '.join(problems[:5])}")
    return doc, hashlib.sha256(data).hexdigest()


def load_run(path: Path, *, entry: str, dataset: str, run_id: str, external: bool, H: str, flags: Flags) -> Run:
    doc, sha = read_manifest(path, H=H, flags=flags)
    name, expected = doc.get("experiment"), MANIFEST_EXPERIMENT[entry]
    if name != expected and not (flags.allow_dev and name == DEV_EXPERIMENT[external]):
        raise ReduceError(f"{path}: manifest experiment {name!r} listed as {entry} (expected {expected!r}"
                          f"{'' if flags.allow_dev else '; ad-hoc runs need --allow-dev'})")
    config = doc.get("config") or {}
    where = f"{path}: config"
    if config.get("schema") != RECORD_SCHEMA:
        raise ReduceError(f"{where} schema {config.get('schema')!r}, not {RECORD_SCHEMA}")
    if config.get("dataset") != dataset:
        raise ReduceError(f"{where} dataset {config.get('dataset')!r} listed as {dataset}")
    group = entry if entry in MEASURED else GROUP[entry]
    partition = str(config.get("partition") or "primary")
    if group == "E16":
        seg_partition(partition)
    elif partition != "primary":
        raise ReduceError(f"{where} partition {partition!r}: only E16 runs on a sensitivity partition")
    split = doc.get("split")
    if not isinstance(split, str) or not split:
        raise ReduceError(f"{path}: manifest has no split")
    try:
        T = float(config["T"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReduceError(f"{where} has no T") from exc
    if not T > 0:
        raise ReduceError(f"{where} T = {T}")
    regimes = config.get("regimes")
    if not isinstance(regimes, list) or not regimes or len(set(regimes)) != len(regimes):
        raise ReduceError(f"{where} regimes {regimes!r}")
    for regime in regimes:
        seg_regime(regime)
        if regime not in protocol.regimes_for(dataset):
            raise ReduceError(f"{where} regime {regime} is not one of {dataset}'s")
    arms = config.get("arms")
    if not isinstance(arms, list) or not arms or len(set(arms)) != len(arms):
        raise ReduceError(f"{where} arms {arms!r}")
    for arm in arms:
        method = parse_arm(arm)[0]
        if method not in protocol.METHODS or external != (method in EXTERNAL_METHODS):
            raise ReduceError(f"{where} arm {arm!r} in a{'n external' if external else ' cascade'} run")
    if external and len(arms) != 1:
        raise ReduceError(f"{where} an external run has one method, got {arms}")
    shard = config.get("shard")
    if shard is not None:
        if (not isinstance(shard, list) or len(shard) != 2 or not all(isinstance(x, int) for x in shard)
                or not 0 <= shard[0] < shard[1]):
            raise ReduceError(f"{where} shard {shard!r}")
        shard = (int(shard[0]), int(shard[1]))
    qsets, inputs = config.get("query_sets") or {}, doc.get("inputs") or {}
    query_sha: dict[str, str] = {}
    selected: dict[str, int] = {}
    for regime in regimes:
        q = qsets.get(regime) or {}
        if q.get("split") is not None and q.get("split") != split:
            raise ReduceError(f"{where} query set of {regime} is split {q.get('split')!r}, the manifest's {split!r}")
        if isinstance(q.get("selected"), bool) or not isinstance(q.get("selected"), int) or q["selected"] < 0:
            raise ReduceError(f"{where} query_sets.{regime}.selected missing")
        hashed = inputs.get(f"queries/{regime}")
        sha = hashed.get("sha256") if isinstance(hashed, dict) else None
        if not sha:
            raise ReduceError(f"{path}: the manifest holds no sha256 of input queries/{regime} "
                              "(a --fast-hashes fingerprint cannot be matched to a certification)")
        query_sha[regime], selected[regime] = str(sha), int(q["selected"])
    if split in queries.TEST_SPLITS and abs(T - planned_T(entry)) > _TOL:
        raise ReduceError(f"{path}: {entry} on the {split} split runs at T = {planned_T(entry):g} s, not {T:g}")
    extra = doc.get("extra") if isinstance(doc.get("extra"), dict) else {}
    return Run(run_id, path, entry, group, dataset, partition, split, bool(doc.get("protocol_frozen")), T, shard,
               list(regimes), list(arms), external, sha, query_sha, selected, config, extra=dict(extra))


def record_file(arm: str, shard: tuple[int, int] | None) -> str:
    """run_experiment.record_name (run_external's method names hold no "/")."""
    safe = arm.replace("/", "_")
    return f"{safe}.shard{shard[0]}of{shard[1]}.jsonl" if shard else f"{safe}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise ReduceError(f"{path}:{n}: {exc}") from exc
    return out


def check_record(rec: Mapping[str, Any], *, run: Run, regime: str, arm: str, path: Path) -> None:
    where = f"{path} ({rec.get('uid')})"
    if rec.get("schema") != RECORD_SCHEMA:
        raise ReduceError(f"{where}: schema {rec.get('schema')!r}, not {RECORD_SCHEMA}")
    if not isinstance(rec.get("uid"), str) or not rec["uid"]:
        raise ReduceError(f"{where}: no uid")
    if rec.get("dataset") != run.dataset or rec.get("regime") != regime:
        raise ReduceError(f"{where}: {rec.get('dataset')}/{rec.get('regime')} in a {run.dataset}/{regime} file")
    if rec.get("arm") != arm:
        raise ReduceError(f"{where}: arm {rec.get('arm')!r} in the file of {arm!r}")
    expected_method = arm if run.external else parse_arm(arm)[0]
    if rec.get("method") != expected_method:
        raise ReduceError(f"{where}: method {rec.get('method')!r}, the arm's is {expected_method!r}")
    try:
        T = float(rec["T"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReduceError(f"{where}: no T") from exc
    if abs(T - run.T) > _TOL:
        raise ReduceError(f"{where}: T = {T:g}, the run's is {run.T:g}")
    if rec.get("status") not in STATUSES:
        raise ReduceError(f"{where}: unknown status {rec.get('status')!r}")
    if rec.get("kind") not in ("positive", "negative"):
        raise ReduceError(f"{where}: unknown kind {rec.get('kind')!r}")
    seg_family(str(rec.get("family")))
    if isinstance(rec.get("size"), bool) or not isinstance(rec.get("size"), int):
        raise ReduceError(f"{where}: size {rec.get('size')!r}")
    for name in ("time_found", "time_decided", "elapsed"):
        value = rec.get(name)
        # the clock may overrun T by the time a solver takes to die; only nonsense is refused
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(float(value)) or float(value) < 0):
            raise ReduceError(f"{where}: {name} = {value!r}")
    if rec["status"] == "FOUND" and rec.get("witness_check") == "ok" and rec.get("time_found") is None:
        raise ReduceError(f"{where}: a verified FOUND without time_found")
    if run.cell is not None:
        mode = rec.get("mode") or "resident"
        if mode != run.cell["mode"] or rec.get("cache") != run.cell["cache"]:
            raise ReduceError(f"{where}: {mode}/{rec.get('cache')} in the {run.cell['mode']}/{run.cell['cache']} cell")


def terminal(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Candidate nodes and edges of the deciding step (else the last started one) and its solver input."""
    steps = rec.get("steps") or []
    started = [i for i, s in enumerate(steps) if s.get("status") != fl.NOT_STARTED]
    out: dict[str, Any] = {"cand_nodes": None, "cand_edges": None, "solver_nodes": None}
    if not started:
        return out
    k = rec.get("decided_step")
    idx = k if isinstance(k, int) and k in started else started[-1]
    cand = steps[idx].get("candidate") or {}
    out["cand_nodes"], out["cand_edges"] = cand.get("candidate_nodes"), cand.get("candidate_edges")
    j = idx
    while j >= 0 and "solver_calls" not in steps[j] and steps[j].get("skipped"):
        j -= 1  # a skipped step repeats the answer of the step it skips to
    if j >= 0 and "solver_calls" in steps[j]:
        out["solver_nodes"] = int(sum(int(c["nodes"]) for c in steps[j]["solver_calls"] if c.get("nodes") is not None))
    return out


# ----------------------------------------------------------------------------
# The reduction: loading and validation
# ----------------------------------------------------------------------------
@dataclass
class Cell:
    """One system's records in one (experiment, dataset, regime, partition): {replicate: {uid: record}}."""
    group: str
    dataset: str
    regime: str
    partition: str
    system: str
    T: float
    external: bool
    reps: dict[int | None, dict[str, dict[str, Any]]] = field(default_factory=dict)
    runs: set[str] = field(default_factory=set)
    options: dict[str, Any] | None = None

    @property
    def seeded(self) -> bool:
        return any(r is not None for r in self.reps)

    def replicates(self) -> list[int | None]:
        return sorted(self.reps, key=lambda r: -1 if r is None else r)

    def uids(self) -> set[str]:
        return set().union(*(set(v) for v in self.reps.values())) if self.reps else set()

    def records(self) -> list[dict[str, Any]]:
        return [rec for rep in self.replicates() for _, rec in sorted(self.reps[rep].items())]


class Reduction:
    """Everything read from the plan's inputs, validated, and the numbers built from it."""

    def __init__(self, plan: Plan, flags: Flags) -> None:
        self.plan, self.flags = plan, flags
        self.H = protocol.protocol_hash()
        self.runs: list[Run] = []
        self.measured: list[dict[str, Any]] = []
        self.m_records: dict[str, dict[str, dict[str, Any]]] = {}
        self.split: str | None = None
        self.frozen = True
        self.query_sha: dict[tuple[str, str], tuple[str, Path]] = {}
        self.certs: dict[tuple[str, str], dict[str, Any]] = {}
        self.truth: dict[tuple[str, str, str], str] = {}
        self.qinfo: dict[tuple[str, str, str], tuple[str, str, int]] = {}
        self.cells: dict[tuple[str, str, str, str, str], Cell] = {}
        self.strip: dict[tuple[str, str], str | None] = {}
        self.uncertified: set[tuple[str, str]] = set()
        self.witnessed: set[tuple[str, str, str]] = set()
        self.promoted: set[tuple[str, str, str]] = set()
        self.exclusions: list[dict[str, Any]] = []
        self.provisional: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.nums: numbers_json.Numbers | None = None
        self.localization: dict[str, Any] = {}
        self.pe1: dict[str, Any] = {}
        self.offline: list[dict[str, Any]] = []
        self.audits: dict[tuple[str, str], dict[str, Any]] = {}
        self.audit_runs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        counters: dict[tuple[str, str], int] = {}
        for entry, ds, dirs in self.plan.entries:
            for path in dirs:
                n = counters[(entry, ds)] = counters.get((entry, ds), -1) + 1
                self.runs.append(load_run(path, entry=entry, dataset=ds, run_id=f"{entry}.{escape(ds)}.{n}",
                                          external=entry in EXTERNAL_ENTRIES, H=self.H, flags=self.flags))
        for exp, root in sorted(self.plan.measured.items()):
            self.load_measured(exp, root)
        self.check_runs()
        self.load_certifications()
        self.decide_backends()
        for run in self.runs:
            self.read_records(run)
        self.apply_witnesses()
        self.list_excluded_truths()
        self.check_shards()
        self.check_uid_sets()
        self.check_replicates()
        self.load_offline()
        self.load_audits()

    def load_measured(self, exp: str, root: Path) -> None:
        cell_re = re.compile(r"^(resident|streaming)-(none|cold|warm)-cap(none|\d+(?:\.\d+)?)$")
        before = len(self.measured)
        for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if ds_dir.name not in protocol.DATASETS:
                raise ReduceError(f"{ds_dir}: {ds_dir.name!r} is not a dataset")
            for sys_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                for cell_dir in sorted(p for p in sys_dir.iterdir() if p.is_dir()):
                    m = cell_re.match(cell_dir.name)
                    if not m:
                        raise ReduceError(f"{cell_dir}: not a <mode>-<cache>-cap<N|none> cell")
                    cap = None if m.group(3) == "none" else float(m.group(3))
                    for rep_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
                        if re.match(r"^rep\d+\.attempt\d+$", rep_dir.name):
                            continue  # streaming.archive_attempt: superseded by the cell's rep<k>
                        rm = re.match(r"^rep(\d+)$", rep_dir.name)
                        if not rm:
                            raise ReduceError(f"{rep_dir}: not a rep<k> directory")
                        self.load_measured_cell(exp, rep_dir, ds_dir.name, sys_dir.name, m.group(1), m.group(2),
                                                cap, int(rm.group(1)))
        if len(self.measured) == before:
            raise ReduceError(f"plan {exp.lower()}_root: {root} holds no measured cell")

    def load_measured_cell(self, exp: str, rep_dir: Path, ds: str, sys_dir: str, mode: str, cache: str,
                           cap: float | None, repeat: int) -> None:
        mpath = rep_dir / "measure.json"
        if not mpath.is_file():
            raise ReduceError(f"{rep_dir}: no measure.json (a cell still running, or killed with its driver)")
        try:
            measure = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ReduceError(f"{mpath}: {exc}") from exc
        if not isinstance(measure, dict):
            raise ReduceError(f"{mpath}: not a JSON object")
        system = str(measure.get("system") or measure.get("arm") or "")
        checks = {"schema": (measure.get("schema"), MEASURE_SCHEMA),
                  "experiment": (measure.get("experiment", "E2"), exp), "dataset": (measure.get("dataset"), ds),
                  "system": (system.replace("/", "_"), sys_dir), "mode": (measure.get("mode"), mode),
                  "cache": (measure.get("cache"), cache), "repeat": (measure.get("repeat"), repeat),
                  "protocol_hash": (measure.get("protocol_hash"), self.H)}
        for name, (got, want) in checks.items():
            if got != want:
                raise ReduceError(f"{mpath}: {name} {got!r}, its place says {want!r}")
        got_cap = measure.get("cap_gb")
        if got_cap is not None and (isinstance(got_cap, bool) or not isinstance(got_cap, (int, float))):
            raise ReduceError(f"{mpath}: cap_gb {got_cap!r}")
        if (got_cap is None) != (cap is None) or (cap is not None and abs(float(got_cap) - cap) > _TOL):
            raise ReduceError(f"{mpath}: cap_gb {got_cap!r} in a {seg_cap(cap)} cell")
        if measure.get("runner") not in ("run_experiment", "run_external"):
            raise ReduceError(f"{mpath}: unknown runner {measure.get('runner')!r}")
        probe = measure.get("memprobe") or {}
        cid = f"{exp}.{escape(ds)}.{escape(system)}.{mode}.{cache}.{seg_cap(cap)}.rep{repeat}"
        completed = measure.get("completed") is True
        cell = {"id": cid, "experiment": exp, "dataset": ds, "system": system, "mode": mode, "cache": cache,
                "cap": cap, "repeat": repeat, "measure": measure, "measure_path": mpath, "completed": completed,
                "success": bool(completed and probe.get("exit_code") == 0 and not probe.get("timed_out")
                                and not measure.get("oom_killed") and not probe.get("oom_killed")),
                "peak": probe.get("peak_rss_bytes"), "run": None,
                "external": measure.get("runner") == "run_external"}
        if completed:
            run = load_run(rep_dir / "run", entry=exp, dataset=ds, run_id=cid, external=cell["external"], H=self.H,
                           flags=self.flags)
            if len(run.arms) != 1:
                raise ReduceError(f"{rep_dir}/run ran {run.arms}; a measured cell runs one arm")
            names = {run.arms[0]}
            if run.split not in queries.TEST_SPLITS:  # a dev sweep's --backend reaches the child's arm names
                names.add(strip_backend(run.arms[0], parse_arm(run.arms[0])[2].get("backend")))
            if system not in names:
                raise ReduceError(f"{rep_dir}/run ran {run.arms[0]!r}, the cell's system is {system!r}")
            if (run.config.get("mode") or "resident") != mode or run.config.get("cache") != cache:
                raise ReduceError(f"{rep_dir}/run ran {run.config.get('mode')}/{run.config.get('cache')}, "
                                  f"the cell is {mode}/{cache}")
            run.cell, cell["run"] = cell, run
            self.runs.append(run)
        self.measured.append(cell)

    def check_runs(self) -> None:
        splits = {run.split for run in self.runs}
        if len(splits) > 1:
            raise ReduceError(f"the runs come from several splits: {sorted(splits)}")
        self.split = splits.pop() if splits else None
        if self.split is not None and self.split not in queries.TEST_SPLITS and not self.flags.allow_dev:
            raise ReduceError(f"split {self.split!r} is not a test split ({', '.join(queries.TEST_SPLITS)}); "
                              "pass --allow-dev for development numbers")
        self.frozen = all(run.frozen for run in self.runs) if self.runs else protocol.is_frozen()
        for run in self.runs:
            for regime, sha in run.query_sha.items():
                known = self.query_sha.setdefault((run.dataset, regime), (sha, run.path))
                if known[0] != sha:
                    raise ReduceError(f"{run.dataset}/{regime}: {run.path} ran query set {sha[:12]}..., "
                                      f"{known[1]} ran {known[0][:12]}... (every run, E16 partitions and "
                                      "measured cells included, must use one query set)")

    def load_certifications(self) -> None:
        for (ds, regime), path in sorted(self.plan.certifications.items()):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ReduceError(f"certification {path}: {exc}") from exc
            if not isinstance(doc, dict) or doc.get("schema") != CERT_SCHEMA:
                raise ReduceError(f"{path}: schema {doc.get('schema') if isinstance(doc, dict) else None!r}, "
                                  f"not {CERT_SCHEMA}")
            if not doc.get("complete"):
                raise ReduceError(f"{path}: certification is partial")
            if doc.get("dataset") != ds or doc.get("regime") != regime:
                raise ReduceError(f"{path}: certifies {doc.get('dataset')}/{doc.get('regime')}, listed as {ds}/{regime}")
            if doc.get("protocol_hash") != self.H:
                raise ReduceError(f"{path}: made under another protocol ({str(doc.get('protocol_hash'))[:12]}...)")
            if self.split is None:
                self.split = doc.get("split")
                if self.split not in queries.TEST_SPLITS and not self.flags.allow_dev:
                    raise ReduceError(f"{path}: split {self.split!r} needs --allow-dev")
            elif doc.get("split") != self.split:
                raise ReduceError(f"{path}: split {doc.get('split')!r}, the runs' is {self.split!r}")
            sha = (doc.get("query_set") or {}).get("sha256")
            known = self.query_sha.get((ds, regime))
            if known is not None and sha != known[0]:
                raise ReduceError(f"{path} certifies query set {str(sha)[:12]}..., the runs of {ds}/{regime} used "
                                  f"{known[0][:12]}...")
            if known is None:
                self.notes.append(f"{path}: no run of {ds}/{regime} to check its query set against")
            entries = doc.get("queries")
            if not isinstance(entries, dict):
                raise ReduceError(f"{path}: no queries")
            for uid, e in entries.items():
                if not isinstance(e, dict) or e.get("certified") not in ("none", "found", "undecided"):
                    raise ReduceError(f"{path}: {uid} certified {e.get('certified') if isinstance(e, dict) else e!r}")
                if e["certified"] == "found" and e.get("witness_check") != "ok":
                    raise ReduceError(f"{path}: {uid} is certified found without a checked witness")
            self.certs[(ds, regime)] = {"doc": doc, "path": path}

    def decide_backends(self) -> None:
        """A backend every cascade arm of an experiment and dataset names is dropped from its systems."""
        arms: dict[tuple[str, str], set[str]] = {}
        for run in self.runs:
            if not run.external:
                arms.setdefault((run.group, run.dataset), set()).update(run.arms)
        for group_ds, names in sorted(arms.items()):
            backends = {parse_arm(a)[2].get("backend") for a in names}
            common = next(iter(backends)) if len(backends) == 1 else None
            self.strip[group_ds] = common
            if common is not None:
                self.notes.append(f"{group_ds[0]} {group_ds[1]}: every arm ran with backend {common}; "
                                  "the systems omit +backend")

    def truth_of(self, run: Run, regime: str, rec: Mapping[str, Any]) -> tuple[str, str]:
        ds, uid = run.dataset, rec["uid"]
        info = (rec["kind"], rec["family"], int(rec["size"]))
        if self.qinfo.setdefault((ds, regime, uid), info) != info:
            raise ReduceError(f"{ds}/{regime} {uid}: kind, family or size differ between runs")
        cert = self.certs.get((ds, regime))
        if rec["kind"] == "positive":
            if rec.get("certified") != "planted":
                raise ReduceError(f"{run.path}: positive {uid} is certified {rec.get('certified')!r}, not planted")
            if cert is not None and uid in cert["doc"]["queries"]:
                raise ReduceError(f"{cert['path']} certifies {uid}, a positive")
            return "positive", "planted"
        if cert is not None:
            entry = cert["doc"]["queries"].get(uid)
            if entry is None:
                raise ReduceError(f"{cert['path']} does not certify negative {uid}")
            if entry.get("family") != rec["family"] or entry.get("size") != rec["size"]:
                raise ReduceError(f"{cert['path']}: {uid} is {entry.get('family')}/{entry.get('size')} there, "
                                  f"{rec['family']}/{rec['size']} in {run.path}")
            label = str(entry["certified"])
        else:
            if not self.flags.allow_uncertified:
                raise ReduceError(f"{ds}/{regime}: negatives without a certification file; pass "
                                  "--allow-uncertified to use the query set's own labels")
            label = str(rec.get("certified"))
            if label not in ("none", "found", "undecided", "uncertified"):
                raise ReduceError(f"{run.path}: negative {uid} is certified {label!r}")
            self.uncertified.add((ds, regime))
        return f"negative_{label}", label

    def slim(self, rec: dict[str, Any], *, run: Run, arm: str, arm_s: str, system: str, replicate: int | None,
             truth: str, certified: str) -> dict[str, Any]:
        """The fields the reduction reads, the E18 categories and the terminal step; the steps are dropped."""
        T = float(rec["T"])
        out = {k: rec.get(k) for k in ("uid", "regime", "family", "size", "kind", "status", "elapsed", "time_found",
                                       "time_decided", "witness_check", "certificate", "method", "train_seed",
                                       "model_id", "backend", "mode", "cache", "wall_seconds", "decided_step",
                                       "peak_rss_bytes")}
        info = rec.get("ranker_info") or {}
        out.update({"dataset": run.dataset, "T": T, "arm": arm, "arm_s": arm_s, "system": system,
                    "replicate": replicate, "run": run.run_id, "truth": truth, "certified": certified,
                    "error": str(rec.get("error"))[:300] if rec.get("error") else None,
                    "stages": {k: float(v) for k, v in (rec.get("stages") or {}).items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)},
                    "io": dict(rec.get("io") or {}), "external": "external" in rec,
                    "none_certified": (rec.get("external") or {}).get("none_certified"),
                    "selectivity": info.get("selectivity"), "route": info.get("route"),
                    "entries_read": entries_read(rec), **terminal(rec)})
        full = dict(rec, certified=certified)
        ph = primary_horizon(T)
        out["e18"], out["e18_detail"] = {}, None
        for h in sorted({h for h in E18_HORIZONS if h <= T + _TOL} | {ph}):
            category, detail = fl.classify(full, h)
            out["e18"][h] = category
            if h == ph:
                out["e18_detail"] = detail
        out["overlap_rescued"] = bool(rec["kind"] == "positive" and fl.overlap_rescued(rec))
        return out

    def read_records(self, run: Run) -> None:
        backend = self.strip.get((run.group, run.dataset))
        for regime in run.regimes:
            rdir = run.path / "records" / regime
            expected = {record_file(a, run.shard): a for a in run.arms}
            present = {p.name for p in rdir.glob("*.jsonl")} if rdir.is_dir() else set()
            if set(expected) - present:
                raise ReduceError(f"{rdir}: missing {sorted(set(expected) - present)}")
            if present - set(expected):
                raise ReduceError(f"{rdir}: files of arms the run did not configure: {sorted(present - set(expected))}")
            for fname, arm in sorted(expected.items()):
                recs = read_jsonl(rdir / fname)
                if len(recs) != run.selected[regime]:
                    if not self.flags.allow_missing:
                        raise ReduceError(f"{rdir / fname}: {len(recs)} records, the run selected "
                                          f"{run.selected[regime]} (pass --allow-missing to reduce what is there)")
                    self.exclusions.append({"reason": "short_run", "experiment": run.group, "dataset": run.dataset,
                                            "regime": regime, "arm": arm, "run": run.run_id,
                                            "detail": f"{len(recs)} of {run.selected[regime]} selected queries"})
                seen: set[str] = set()
                arm_s = arm if run.external else strip_backend(arm, backend)
                if run.cell is not None:
                    system = run.cell["system"]
                else:
                    system = e1_system(arm_s) if run.group == "E1" else arm_s
                seg_system(system)
                if run.group == "E3x":
                    self.cell_for(run, regime, arm, system)  # an extension with nothing to rerun still has a row
                for rec in recs:
                    check_record(rec, run=run, regime=regime, arm=arm, path=rdir / fname)
                    if rec["uid"] in seen:
                        raise ReduceError(f"{rdir / fname}: {rec['uid']} recorded twice")
                    seen.add(rec["uid"])
                    truth, certified = self.truth_of(run, regime, rec)
                    self.truth[(run.dataset, regime, rec["uid"])] = truth
                    replicate = replicate_of(rec) if run.group == "E1" else None
                    slim = self.slim(rec, run=run, arm=arm, arm_s=arm_s, system=system, replicate=replicate,
                                     truth=truth, certified=certified)
                    if run.cell is not None:
                        self.m_records.setdefault(run.run_id, {})[rec["uid"]] = slim
                    else:
                        self.insert(run, regime, arm, system, replicate, slim)

    def cell_for(self, run: Run, regime: str, arm: str, system: str) -> Cell:
        k = (run.group, run.dataset, regime, run.partition, system)
        cell = self.cells.get(k)
        if cell is None:
            cell = self.cells[k] = Cell(run.group, run.dataset, regime, run.partition, system, run.T, run.external)
        if abs(cell.T - run.T) > _TOL:
            raise ReduceError(f"{run.group} {run.dataset}/{regime} {system}: runs at T = {cell.T:g} and {run.T:g}")
        cell.runs.add(run.run_id)
        if cell.options is None:
            cell.options = ((run.config.get("cascade_options") or {}).get(regime) or {}).get(arm)
        return cell

    def insert(self, run: Run, regime: str, arm: str, system: str, replicate: int | None,
               slim: dict[str, Any]) -> None:
        per = self.cell_for(run, regime, arm, system).reps.setdefault(replicate, {})
        if slim["uid"] in per:
            raise ReduceError(f"{run.group} {run.dataset}/{regime}: {arm} ran {slim['uid']} twice (again in {run.path})")
        per[slim["uid"]] = slim

    def all_records(self) -> Iterator[dict[str, Any]]:
        for cell in self.cells.values():
            for per in cell.reps.values():
                yield from per.values()
        for per in self.m_records.values():
            yield from per.values()

    def apply_witnesses(self) -> None:
        """A verified FOUND anywhere in the reduction shows that its query has an embedding: every record of
        the query is marked witnessed (a NONE proof on it is a false negative) and an uncertified negative
        with one becomes "found" (it leaves the negative rates)."""
        self.witnessed = {(r["dataset"], r["regime"], r["uid"]) for r in self.all_records()
                          if r["status"] == "FOUND" and r.get("witness_check") == "ok"}
        for k in sorted(self.witnessed):
            if self.truth.get(k) == "negative_uncertified":
                self.truth[k] = "negative_found"
                self.promoted.add(k)
        for rec in self.all_records():
            k = (rec["dataset"], rec["regime"], rec["uid"])
            if k in self.witnessed:
                rec["witnessed"] = True
                rec["truth"] = self.truth[k]
        if self.promoted:
            self.notes.append(f"{len(self.promoted)} uncertified negatives have a verified witness in this "
                              "reduction: relabelled found and left out of every rate")

    def list_excluded_truths(self) -> None:
        for (ds, regime, uid), truth in sorted(self.truth.items()):
            if truth in EXCLUDED_TRUTHS:
                promoted = (ds, regime, uid) in self.promoted
                self.exclusions.append({"reason": EXCLUDED_TRUTHS[truth], "dataset": ds, "regime": regime, "uid": uid,
                                        "detail": "left out of every positive and negative rate" + (
                                            "; found with a verified witness by a run of this reduction (no "
                                            "certification file)" if promoted else "")})
            elif truth == "negative_uncertified":
                self.provisional.append({"reason": "uncertified", "dataset": ds, "regime": regime, "uid": uid,
                                         "detail": "no certification file: in the negative rates provisionally"})

    def check_shards(self) -> None:
        by: dict[tuple, list[tuple[tuple[int, int] | None, Path]]] = {}
        for run in self.runs:
            if run.cell is not None:
                continue
            for regime in run.regimes:
                for arm in run.arms:
                    k = (run.group, run.dataset, regime, run.partition, arm)
                    by.setdefault(k, []).append((run.shard, run.path))
        for (group, ds, regime, partition, arm), items in sorted(by.items(), key=lambda kv: str(kv[0])):
            where = f"{group} {ds}/{regime}{'' if partition == 'primary' else ' ' + partition} {arm}"
            shards = [s for s, _ in items]
            if None in shards:
                if len(shards) > 1:
                    raise ReduceError(f"{where}: an unsharded run beside {len(shards) - 1} other(s): "
                                      f"{[str(p) for _, p in items]}")
                continue
            counts = {n for _, n in shards}
            if len(counts) != 1:
                raise ReduceError(f"{where}: shards of different counts {sorted(counts)}")
            n = counts.pop()
            got = sorted(i for i, _ in shards)
            if got != list(range(n)):
                missing = sorted(set(range(n)) - set(got))
                raise ReduceError(f"{where}: shards {got} of {n}" + (f", missing {missing}" if missing else ", twice"))

    def check_uid_sets(self) -> None:
        members: dict[tuple[str, str, str], list[tuple[tuple[str, str, Any], set[str]]]] = {}
        for (group, ds, regime, partition, system), cell in self.cells.items():
            if group == "E3x":
                continue  # the extension reruns E1's undecided queries per arm
            for rep, recs in cell.reps.items():
                members.setdefault((group, ds, regime), []).append(((partition, system, rep), set(recs)))
        for (group, ds, regime), items in sorted(members.items()):
            common = set.intersection(*(s for _, s in items))
            union = set.union(*(s for _, s in items))
            if common == union:
                continue
            if not self.flags.allow_missing:
                (partition, system, rep), s = next(m for m in items if m[1] != union)
                lacking = sorted(union - s)
                who = system + (f" (seed {rep})" if rep is not None else "") + (
                    f" on {partition}" if partition != "primary" else "")
                raise ReduceError(f"{group} {ds}/{regime}: {who} lacks {len(lacking)} of the {len(union)} queries "
                                  f"other arms ran (e.g. {lacking[:3]}); pass --allow-missing to reduce over the "
                                  "intersection")
            for (partition, system, rep), s in items:
                cell = self.cells[(group, ds, regime, partition, system)]
                for uid in sorted(s - common):
                    rec = cell.reps[rep].pop(uid)
                    self.exclusions.append({"reason": "missing", "experiment": group, "dataset": ds, "regime": regime,
                                            "partition": partition, "system": system, "arm": rec["arm"], "uid": uid,
                                            "detail": "not run by every arm of the experiment"})
        by_ds: dict[tuple[str, str], list[tuple[str, set[str]]]] = {}
        for cell in self.measured:
            if cell["run"] is not None:
                by_ds.setdefault((cell["experiment"], cell["dataset"]), []).append(
                    (cell["id"], set(self.m_records.get(cell["id"], {}))))
        for (exp, ds), items in sorted(by_ds.items()):
            if len({frozenset(s) for _, s in items}) > 1 and not self.flags.allow_missing:
                raise ReduceError(f"{exp} {ds}: the cells ran different queries (e.g. {items[0][0]}); pass "
                                  "--allow-missing to pair each comparison on the queries both sides ran")

    def check_replicates(self) -> None:
        seeds = {int(s) for s in protocol.SEEDS["training"]}
        for (group, ds, regime, _, system), cell in sorted(self.cells.items(), key=lambda kv: str(kv[0])):
            reps = set(cell.reps)
            where = f"{group} {ds}/{regime} {system}"
            if None in reps and len(reps) > 1:
                raise ReduceError(f"{where}: seeded and unseeded arms in one system")
            if group != "E1" or not cell.seeded:
                continue
            if parse_arm(system)[0] not in LEARNED_METHODS:
                raise ReduceError(f"{where}: a fixed method with training seeds {sorted(reps)}")
            if reps != seeds and not self.flags.allow_dev:
                raise ReduceError(f"{where}: training seeds {sorted(reps)}, the protocol's are {sorted(seeds)}")
            rows = [{"query_uid": u, "arm": system, "train_seed": rep, "solved": 0.0}
                    for rep, recs in cell.reps.items() for u in recs]
            try:
                stats.seed_average(rows)
            except ValueError as exc:
                raise ReduceError(f"{where}: {exc}") from exc

    def load_offline(self) -> None:
        for (ds, stage), dirs in sorted(self.plan.offline.items()):
            if stage == "prepare" and len(dirs) > 1:
                raise ReduceError(f"plan offline {ds}/prepare: {len(dirs)} runs; a dataset's store is prepared by "
                                  "one run (a resume keeps its run dir)")
            for k, path in enumerate(dirs):
                # the offline stages ran before the protocol took its final hash (the router threshold is
                # fit after training), so only completeness and the outputs are checked
                doc, sha = read_manifest(path, H=None, flags=self.flags)
                try:
                    cost = offline_build.offline_cost(stage, path, doc)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    raise ReduceError(f"{path}: offline stage {stage} cannot be costed "
                                      f"({type(exc).__name__}: {exc})") from exc
                self.offline.append({"dataset": ds, "stage": stage, "path": path, "manifest": doc, "sha256": sha,
                                     "cost": cost, "id": f"E17.{escape(ds)}.{escape(stage)}.{k}"})

    def load_audits(self) -> None:
        """pattern_audit.py's audits: the queries some main encoder drew in training (E20)."""
        for ds, dirs in sorted(self.plan.audits.items()):
            for k, path in enumerate(dirs):
                doc_m, sha = read_manifest(path, H=self.H, flags=self.flags)
                if doc_m.get("experiment") != AUDIT_EXPERIMENT:
                    raise ReduceError(f"{path}: manifest experiment {doc_m.get('experiment')!r}, not "
                                      f"{AUDIT_EXPERIMENT!r}")
                self.frozen = self.frozen and bool(doc_m.get("protocol_frozen"))  # E20 numbers read the audit
                fpath = path / AUDIT_FILE
                try:
                    doc = json.loads(fpath.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ReduceError(f"{fpath}: {exc}") from exc
                if not isinstance(doc, dict) or doc.get("schema") != AUDIT_SCHEMA or doc.get("complete") is not True:
                    raise ReduceError(f"{fpath}: not a complete {AUDIT_SCHEMA} audit")
                if doc.get("dataset") != ds or doc.get("protocol_hash") != self.H:
                    raise ReduceError(f"{fpath}: audits {doc.get('dataset')} under protocol "
                                      f"{str(doc.get('protocol_hash'))[:12]}..., listed as {ds} (the reducer's "
                                      f"protocol is {self.H[:12]}...)")
                if self.split is None:
                    self.split = doc.get("split")
                    if self.split not in queries.TEST_SPLITS and not self.flags.allow_dev:
                        raise ReduceError(f"{fpath}: split {self.split!r} needs --allow-dev")
                elif doc.get("split") != self.split:
                    raise ReduceError(f"{fpath}: split {doc.get('split')!r}, the runs' is {self.split!r}")
                run = {"id": f"E20.{escape(ds)}.{k}", "dataset": ds, "path": path, "file": fpath, "sha256": sha}
                self.audit_runs.append(run)
                for regime, entry in sorted((doc.get("regimes") or {}).items()):
                    self.load_audit_regime(run, doc, regime, entry)

    def load_audit_regime(self, run: dict[str, Any], doc: Mapping[str, Any], regime: str,
                          entry: Mapping[str, Any]) -> None:
        ds, where = run["dataset"], f"{run['file']} {regime}"
        seg_regime(regime)
        if (ds, regime) in self.audits:
            raise ReduceError(f"{where}: {ds}/{regime} is also audited by {self.audits[(ds, regime)]['file']}")
        sha = (entry.get("query_set") or {}).get("sha256")
        known = self.query_sha.get((ds, regime))
        if known is not None and sha != known[0]:
            raise ReduceError(f"{where}: audits query set {str(sha)[:12]}..., the runs of {ds}/{regime} used "
                              f"{known[0][:12]}...")
        if known is None:
            self.notes.append(f"{where}: no run of {ds}/{regime} to check the audited query set against")
        encoders = {mid: doc["encoders"][mid] for mid in entry.get("encoders") or [] if mid in doc["encoders"]}
        if len(encoders) != len(entry.get("encoders") or []):
            raise ReduceError(f"{where}: an encoder of the regime has no entry in encoders")
        if self.split in queries.TEST_SPLITS:
            want = {main_model_id(ds, regime, s) for s in protocol.SEEDS["training"]}
            if set(encoders) != want or any(e.get("smoke") for e in encoders.values()):
                raise ReduceError(f"{where}: audits encoders {sorted(encoders)}; a test split needs the main "
                                  f"encoders of every training seed, not smoke runs ({sorted(want)})")
        uids = set(entry.get("uids") or [])
        seen = dict(entry.get("seen") or {})
        by_seed: dict[int, set[str]] = {}
        for uid, per in seen.items():
            if uid not in uids or not per or set(per) - set(encoders):
                raise ReduceError(f"{where}: seen query {uid} is not in the set or names another encoder")
            for mid in per:
                by_seed.setdefault(int(encoders[mid]["seed"]), set()).add(uid)
        for run_ in self.runs:
            if run_.dataset != ds or run_.group not in E20_GROUPS or run_.external or regime not in run_.regimes:
                continue
            for arm in run_.arms:
                mid = arm_model_id(run_, regime, arm)
                if mid is not None and mid not in encoders:
                    raise ReduceError(f"{run_.path}: {arm} served {regime} with {mid}, which {run['file']} did not "
                                      f"audit ({sorted(encoders)})")
        ran = {u for (g, d, r, _, _), cell in self.cells.items() if g in E20_GROUPS and (d, r) == (ds, regime)
               for u in cell.uids()}
        if ran - uids:
            raise ReduceError(f"{where}: {len(ran - uids)} queries of the E1 and E7 runs are not in the audited set, "
                              f"e.g. {sorted(ran - uids)[0]}")
        for uid in seen:
            truth = self.truth.get((ds, regime, uid))
            if truth == "negative_none":
                mid, first = next(iter(sorted(seen[uid].items())))
                raise ReduceError(f"{where}: negative {uid} is certified none, but training query (epoch, step, "
                                  f"draw) {tuple(first['first'])} of {mid} is isomorphic to it: it has an embedding")
            if truth in ("negative_uncertified", "negative_undecided"):
                self.notes.append(f"{where}: negative {uid} ({truth}) is isomorphic to a training query, so it has "
                                  "an embedding")
        self.audits[(ds, regime)] = {"id": run["id"], "file": run["file"], "entry": dict(entry), "seen": set(seen),
                                     "by_seed": by_seed, "encoders": encoders}

    # ------------------------------------------------------------------ views
    def groups(self, group: str) -> dict[tuple[str, str], dict[tuple[str, str], Cell]]:
        out: dict[tuple[str, str], dict[tuple[str, str], Cell]] = {}
        for (g, ds, regime, partition, system), cell in sorted(self.cells.items()):
            if g == group:
                out.setdefault((ds, regime), {})[(partition, system)] = cell
        return out

    def primary(self, group: str) -> dict[tuple[str, str], dict[str, Cell]]:
        return {k: {s: c for (p, s), c in v.items() if p == "primary"} for k, v in self.groups(group).items()}

    def of_truth(self, ds: str, regime: str, uids: Iterable[str], *truths: str) -> list[str]:
        return sorted(u for u in uids if self.truth[(ds, regime, u)] in truths)

    def neg_note(self, ds: str, regime: str) -> str | None:
        if (ds, regime) in self.uncertified:
            return ("uncertified: no certification file (--allow-uncertified); negatives keep the query set's "
                    "label and enter negative rates provisionally")
        return None


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
Outcome = Callable[[Mapping[str, Any]], bool]


def solved_at(t: float) -> Outcome:
    return lambda rec: fl.solved_by(rec, t)


def none_at(t: float) -> Outcome:
    return lambda rec: fl.none_proved_by(rec, t)


def mean_rate(mat: np.ndarray) -> tuple[float, float, float]:
    """(p, lo, hi) of a (seed x query) 0/1 matrix: p = the mean over queries of the per-query seed means
    p_i, and the Wilson 95% interval of that mean, stats.wilson_interval(sum p_i, n) with n the number of
    queries (protocol.STATISTICS: rate_interval wilson_95 per query, seeds averaged per query first).
    Wilson(k, n) when every query's seeds agree; conservative otherwise, as Var(p_i) <= p(1 - p)."""
    n = int(mat.shape[1])
    successes = float(mat.mean(axis=0).sum())
    lo, hi = stats.wilson_interval(successes, n)
    return successes / n, float(lo), float(hi)


def outcome_matrix(cell: Cell, uids: Sequence[str], outcome: Outcome) -> np.ndarray:
    return np.array([[1.0 if outcome(cell.reps[r][u]) else 0.0 for u in uids] for r in cell.replicates()])


def per_query(cell: Cell, uids: Sequence[str], outcome: Outcome) -> dict[str, float]:
    """Per-query outcome in percent, averaged over the cell's seeds."""
    mat = outcome_matrix(cell, uids, outcome)
    return {u: 100.0 * float(v) for u, v in zip(uids, mat.mean(axis=0))}


def replicate_table(cell: Cell, uids: Sequence[str], outcome: Outcome) -> dict[str, dict[str, float]]:
    return {str(r): {u: 100.0 * float(outcome(cell.reps[r][u])) for u in uids} for r in cell.replicates()}


def paired(a: Cell, b: Cell, uids: Sequence[str], outcome: Outcome) -> dict[str, Any]:
    """a minus b on the same queries, in percentage points; McNemar counts when both are deterministic."""
    res = stats.paired_compare(per_query(a, uids, outcome), per_query(b, uids, outcome), n_boot=DRAWS["boot"],
                               n_perm=DRAWS["perm"], seed=SEED)
    out: dict[str, Any] = {"res": res}
    if a.seeded or b.seeded:
        out["hier"] = stats.hierarchical_bootstrap_diff(replicate_table(a, uids, outcome),
                                                         replicate_table(b, uids, outcome),
                                                         n_boot=DRAWS["boot"], seed=SEED)
    else:
        xa = [outcome(a.reps[None][u]) for u in uids]
        xb = [outcome(b.reps[None][u]) for u in uids]
        bb = sum(1 for p, q in zip(xa, xb) if p and not q)
        cc = sum(1 for p, q in zip(xa, xb) if q and not p)
        out["mcnemar"] = (bb, cc, stats.exact_mcnemar(bb, cc))
    return out


def geometric_mean_ratio(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    xa, xb, _ = stats.aligned_pairs(a, b)
    return float(np.exp(np.mean(np.log(xa) - np.log(xb))))


def join_notes(*notes: str | None) -> str | None:
    kept = [n for n in notes if n]
    return "; ".join(kept) if kept else None


def _timed_out(rec: Mapping[str, Any], t: float) -> bool:
    """Undecided at t: still running then, or returned TIMEOUT by then (crashes and wrong answers count apart)."""
    if rec["status"] == "ERROR" or fl.solved_by(rec, t) or fl.none_proved_by(rec, t):
        return False
    return not fl.returned_by(rec, t + _TOL) or rec["status"] == "TIMEOUT"


# ----------------------------------------------------------------------------
# Writing numbers
# ----------------------------------------------------------------------------
class Writer:
    """numbers_json.Numbers with the reducer's conventions (digits, seed-averaged rates, sources)."""

    def __init__(self, nums: numbers_json.Numbers) -> None:
        self.nums = nums

    def count(self, k: str, value: int, sources: Iterable[str], note: str | None = None) -> str:
        return self.nums.number(k, int(value), "count", source=sources, note=note)

    def value(self, k: str, value: float, unit: str, sources: Iterable[str], *, ci: Sequence[float] | None = None,
              n: int | None = None, digits: int | None = None, note: str | None = None) -> str:
        return self.nums.number(k, value, unit, ci=ci, n=n, digits=digits, source=sources, note=note)

    def text(self, k: str, value: str, sources: Iterable[str], note: str | None = None) -> str:
        return self.nums.number(k, str(value), "text", source=sources, note=note)

    def pvalue(self, k: str, p: float, sources: Iterable[str], note: str | None = None) -> str:
        return self.nums.number(k, min(1.0, max(0.0, float(p))), "pvalue", digits=4, source=sources, note=note)

    def rate(self, k: str, cell: Cell, uids: Sequence[str], outcome: Outcome, note: str | None = None, *,
             also: Iterable[str] = ()) -> None:
        """A rate in percent over uids with its Wilson interval; a seed-averaged one as mean_rate(). also: sources
        beside the cell's runs (the audit that chose the uids)."""
        if not uids:
            return
        sources = sorted({*cell.runs, *also})
        if not cell.seeded:
            successes = sum(1 for u in uids if outcome(cell.reps[None][u]))
            self.nums.rate(k, successes, len(uids), source=sources, note=note)
            return
        mat = outcome_matrix(cell, uids, outcome)
        p, lo, hi = mean_rate(mat)
        reps = cell.replicates()
        self.nums.number(k, 100.0 * p, "pct", ci=(100.0 * lo, 100.0 * hi), n=len(uids), digits=1, source=sources,
                         note=join_notes(note, f"mean over queries of the mean over {len(reps)} training seeds; "
                                               "Wilson 95% of that mean over the queries"))
        per_seed = []
        for i, rep in enumerate(reps):
            successes = int(mat[i].sum())
            self.nums.rate(f"{k}.{seg_seed(rep)}", successes, len(uids), source=sources, note=note)
            per_seed.append(100.0 * successes / len(uids))
        self.nums.number(f"{k}.seed_min", min(per_seed), "pct", n=len(uids), digits=1, source=sources, note=note)
        self.nums.number(f"{k}.seed_max", max(per_seed), "pct", n=len(uids), digits=1, source=sources, note=note)

    def paired_keys(self, prefix: str, result: Mapping[str, Any], p_holm: float | None, sources: Iterable[str], *,
                    p_from: str = "signflip", counts: bool = True, note: str | None = None) -> None:
        res = result["res"]
        sources = sorted(sources)
        self.value(f"{prefix}.diff", res.diff, "pp", sources, ci=(res.lo, res.hi), n=res.n_units, digits=1, note=note)
        p = result["mcnemar"][2] if p_from == "mcnemar" else res.p_value
        self.pvalue(f"{prefix}.p", p, sources, note="exact McNemar" if p_from == "mcnemar" else "sign-flip")
        if p_holm is not None:
            self.pvalue(f"{prefix}.p_holm", p_holm, sources)
        if p_from == "mcnemar":
            self.count(f"{prefix}.b", result["mcnemar"][0], sources)
            self.count(f"{prefix}.c", result["mcnemar"][1], sources)
        if counts:
            for name in ("wins", "losses", "ties"):
                self.count(f"{prefix}.{name}", getattr(res, name), sources)
        if "hier" in result:
            point, lo, hi = result["hier"]
            self.value(f"{prefix}.diff_hier", point, "pp", sources, ci=(lo, hi), n=res.n_units, digits=1,
                       note="hierarchical bootstrap over queries and training seeds")

    def cell(self, k: str, ci: bool = False) -> Any:
        """A table cell: the key (with its interval when it has one), or "-" when the number is absent."""
        if k not in self.nums.numbers:
            return "-"
        return {"key": k, "ci": True} if ci and self.nums.numbers[k]["ci"] is not None else {"key": k}


def _sources(cells: Iterable[Cell]) -> list[str]:
    return sorted(set().union(*(c.runs for c in cells)))


def _holm(tests: Mapping[str, float]) -> dict[str, float]:
    return stats.holm(dict(tests)) if tests else {}


def _seed_note(cell: Cell) -> str | None:
    return f"summed over {len(cell.reps)} training seeds" if cell.seeded else None


# ----------------------------------------------------------------------------
# PE1
# ----------------------------------------------------------------------------
def soundness(rec: Mapping[str, Any]) -> str | None:
    """fp, fn, contradiction, crash, false_none (an unproven external NONE on an embedding) or None."""
    status = rec["status"]
    if status == "ERROR":
        return "fp" if str(rec.get("error") or "").startswith(fl.REJECTED) else "crash"
    if status == "FOUND":
        if rec.get("witness_check") != "ok":
            return "fp"
        return "contradiction" if rec["truth"] == "negative_none" else None
    if status == "NONE":
        embedded = rec["truth"] in ("positive", "negative_found") or bool(rec.get("witnessed"))
        if rec.get("time_decided") is not None:
            return "fn" if embedded else None
        return "false_none" if embedded else None
    return None


def _tally(counts: dict[str, int], rec: Mapping[str, Any]) -> str | None:
    verdict = soundness(rec)
    if verdict:
        counts[verdict] += 1
    counts["found"] += int(rec["status"] == "FOUND")
    counts["witness_checked"] += int(rec.get("witness_check") is not None)
    counts["witness_passed"] += int(rec.get("witness_check") == "ok")
    counts["none_unverified"] += int(rec["status"] == "NONE" and rec.get("time_decided") is None)
    return verdict


def reduce_pe1(ctx: Reduction, w: Writer) -> None:
    def zero() -> dict[str, int]:
        return {m: 0 for m in PE1_COUNTS}

    totals, external = zero(), zero()
    per_group: dict[str, dict[str, int]] = {}
    tree: dict[str, Any] = {}
    ext_tree: dict[str, Any] = {}
    per_run: dict[str, dict[str, int]] = {}
    violations: list[dict[str, Any]] = []
    ext_violations: list[dict[str, Any]] = []

    def visit(group: str, ds: str, regime: str, system: str, rec: dict[str, Any], is_external: bool) -> None:
        target = ext_tree if is_external else tree
        node = target.setdefault(group, {}).setdefault(ds, {}).setdefault(regime, {}).setdefault(system, zero())
        verdict = _tally(node, rec)
        _tally(per_run.setdefault(rec["run"], zero()), rec)
        _tally(external if is_external else totals, rec)
        if not is_external:
            _tally(per_group.setdefault(group, zero()), rec)
        if verdict in PE1_VIOLATIONS or (is_external and verdict == "false_none"):
            entry = {"experiment": group, "dataset": ds, "regime": regime, "arm": rec["arm"], "uid": rec["uid"],
                     "kind": verdict, "status": rec["status"], "error": rec.get("error"),
                     "witness_check": rec.get("witness_check"), "truth": rec["truth"],
                     "witnessed": bool(rec.get("witnessed")), "run": rec["run"]}
            (ext_violations if is_external else violations).append(entry)

    for (group, ds, regime, _, system), cell in sorted(ctx.cells.items()):
        for rec in cell.records():
            visit(group, ds, regime, system, rec, cell.external)
    for cell in ctx.measured:
        for _, rec in sorted((ctx.m_records.get(cell["id"]) or {}).items()):
            visit(cell["experiment"], cell["dataset"], str(rec.get("regime")), rec["system"], rec, cell["external"])
    holds = all(totals[m] == 0 for m in PE1_VIOLATIONS)
    ctx.pe1 = {"holds": holds, "totals": totals, "counts": tree, "external_totals": external, "external": ext_tree,
               "runs": per_run, "violations": violations, "external_violations": ext_violations}
    sources = sorted(per_run)
    w.text("PE1.holds", "yes" if holds else "no", sources,
           note="no false positive, false negative, contradiction or crash on any cascade system")
    for metric in PE1_COUNTS:
        w.count(f"PE1.{metric}", totals[metric], sources)
        w.count(f"PE1.external.{metric}", external[metric], sources)
    if totals["witness_checked"]:
        w.nums.rate("PE1.witness_pass", totals["witness_passed"], totals["witness_checked"], source=sources)
    for group, counts in sorted(per_group.items()):
        entries = {group, *(e for e, g in GROUP.items() if g == group)}
        runs = sorted(r for r in per_run if r.split(".")[0] in entries)
        for metric in PE1_COUNTS:
            w.count(f"PE1.{group}.{metric}", counts[metric], runs)


# ----------------------------------------------------------------------------
# E1, E3 and PE2
# ----------------------------------------------------------------------------
def _families(ctx: Reduction, ds: str, regime: str, uids: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for u in uids:
        out.setdefault(ctx.qinfo[(ds, regime, u)][1], []).append(u)
    return {f: sorted(out[f]) for f in FAMILIES if f in out}


def _sizes(ctx: Reduction, ds: str, regime: str, uids: Iterable[str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for u in uids:
        out.setdefault(ctx.qinfo[(ds, regime, u)][2], []).append(u)
    return {s: sorted(out[s]) for s in sorted(out)}


def _group_T(cells: Iterable[Cell], what: str) -> float:
    Ts = {round(c.T, 9) for c in cells}
    if len(Ts) != 1:
        raise ReduceError(f"{what}: systems ran at different T {sorted(Ts)}")
    return float(Ts.pop())


def _uids(cells: Iterable[Cell]) -> list[str]:
    cells = list(cells)
    return sorted(cells[0].uids()) if cells else []


def _cost_keys(w: Writer, base: str, cell: Cell, recs: Sequence[Mapping[str, Any]], *, p95: bool) -> None:
    sources = sorted(cell.runs)
    note = "pooled over training seeds" if cell.seeded else None
    for name in ("cand_nodes", "cand_edges", "solver_nodes"):
        values = [float(r[name]) for r in recs if r.get(name) is not None]
        if not values:
            continue
        w.value(f"{base}.{name}.median", float(np.median(values)), "number", sources, n=len(values), digits=0,
                note=note)
        if p95:
            w.value(f"{base}.{name}.p95", float(np.quantile(values, 0.95)), "number", sources, n=len(values),
                    digits=0, note=note)


def _router_keys(w: Writer, base: str, cell: Cell, recs: Sequence[Mapping[str, Any]]) -> None:
    sources = sorted(cell.runs)
    routes = [str(r["route"]) for r in recs if r.get("route") is not None]
    for route in sorted(set(routes)):
        share = 100.0 * sum(1 for r in routes if r == route) / len(routes)
        w.value(f"{base}.route.{escape(route)}", share, "pct", sources, n=len(routes), digits=1)
    sel = [float(r["selectivity"]) for r in recs if isinstance(r.get("selectivity"), (int, float))]
    if sel:
        w.value(f"{base}.selectivity_mean", float(np.mean(sel)), "number", sources, n=len(sel), digits=2)


def _negatives_text(n: int, note: str | None) -> str:
    return f"{n} certified-none negatives" if note is None else f"{n} negatives in the rates (see the next note)"


def reduce_e1(ctx: Reduction, w: Writer) -> None:
    horizons_all = [float(h) for h in protocol.TIME["reported_T_seconds"]]
    for (ds, regime), cells in ctx.primary("E1").items():
        T = _group_T(cells.values(), f"E1 {ds}/{regime}")
        uids = _uids(cells.values())
        P = ctx.of_truth(ds, regime, uids, "positive")
        N = ctx.of_truth(ds, regime, uids, *RATE_NEGATIVES)
        h30 = primary_horizon(T)
        ht = seg_horizon(h30)
        horizons = [h for h in horizons_all if h <= T + _TOL]
        note = ctx.neg_note(ds, regime)
        base_dr = key("E1", escape(ds), seg_regime(regime))
        everyone = _sources(cells.values())
        for truth, label in (*EXCLUDED_TRUTHS.items(), ("negative_uncertified", "uncertified")):
            n = sum(1 for u in uids if ctx.truth[(ds, regime, u)] == truth)
            if n:
                w.count(f"{base_dr}.{label}", n, everyone,
                        note="negatives left out of every rate" if truth in EXCLUDED_TRUTHS
                        else "negatives without a certification, in the negative rates provisionally")
        pos_fams, neg_fams = _families(ctx, ds, regime, P), _families(ctx, ds, regime, N)
        rows = []
        for system in system_order(cells):
            cell = cells[system]
            base = f"{base_dr}.{seg_system(system)}"
            sources = sorted(cell.runs)
            recs = cell.records()
            for h in horizons:
                w.rate(f"{base}.solved.{seg_horizon(h)}", cell, P, solved_at(h))
            for fam, fam_uids in pos_fams.items():
                w.rate(f"{base}.{fam}.solved.{ht}", cell, fam_uids, solved_at(h30))
                for size, size_uids in _sizes(ctx, ds, regime, fam_uids).items():
                    w.rate(f"{base}.{fam}.{seg_size(size)}.solved.{ht}", cell, size_uids, solved_at(h30))
            for size, size_uids in _sizes(ctx, ds, regime, P).items():
                w.rate(f"{base}.{seg_size(size)}.solved.{ht}", cell, size_uids, solved_at(h30))
            w.rate(f"{base}.correct_none.{ht}", cell, N, none_at(h30), note=note)
            for fam, fam_uids in neg_fams.items():
                w.rate(f"{base}.{fam}.correct_none.{ht}", cell, fam_uids, none_at(h30), note=note)
            seed_note = _seed_note(cell)
            verdicts = [soundness(r) for r in recs]
            for metric in ("fp", "fn", "crash"):
                w.count(f"{base}.{metric}", sum(1 for v in verdicts if v == metric), sources, note=seed_note)
            w.count(f"{base}.incomplete", sum(1 for r in recs if r["status"] == "INCOMPLETE"), sources, note=seed_note)
            w.count(f"{base}.timeout_pos.{ht}", sum(1 for r in recs if r["truth"] == "positive" and _timed_out(r, h30)),
                    sources, note=seed_note)
            w.count(f"{base}.timeout_neg.{ht}",
                    sum(1 for r in recs if r["truth"] in RATE_NEGATIVES and _timed_out(r, h30)), sources,
                    note=join_notes(seed_note, note))
            undecided = [r for r in recs if r["truth"] == "negative_undecided"]
            if undecided:
                w.count(f"{base}.newly_found", sum(1 for r in undecided if fl.solved_by(r, T)), sources, note=seed_note)
                w.count(f"{base}.newly_certified",
                        sum(1 for r in undecided if fl.none_proved_by(r, T) and not r.get("witnessed")), sources,
                        note=seed_note)
            _cost_keys(w, base, cell, recs, p95=True)
            _stage_keys(w, base, cell, recs, T)
            _router_keys(w, base, cell, recs)
            rows.append([system, *[w.cell(f"{base}.solved.{seg_horizon(h)}", ci=True) for h in horizons],
                         w.cell(f"{base}.correct_none.{ht}", ci=True), w.cell(f"{base}.fp"), w.cell(f"{base}.fn"),
                         w.cell(f"{base}.crash")])
        w.nums.table(base_dr, ["system", *[f"solved@{h:g}s" for h in horizons], f"correct NONE@{h30:g}s", "FP", "FN",
                               "crash"], rows,
                     notes=join_notes(f"{len(P)} positives, {_negatives_text(len(N), note)}", note))


def _stage_keys(w: Writer, base: str, cell: Cell, recs: Sequence[Mapping[str, Any]], T: float) -> None:
    sources = sorted(cell.runs)
    for stage in protocol.TIME["stages_counted"]:
        values = [r["stages"][stage] for r in recs if stage in r["stages"]]
        solved = [r["stages"][stage] for r in recs
                  if stage in r["stages"] and r["truth"] == "positive" and fl.solved_by(r, T)]
        for name, vals, fn in (("mean", values, np.mean), ("median", values, np.median),
                               ("solved_mean", solved, np.mean), ("solved_median", solved, np.median)):
            if vals:
                w.value(f"{base}.time.{escape(stage)}.{name}", float(fn(vals)), "seconds", sources, n=len(vals),
                        digits=3)


def _curve_grid(T: float, extra: Iterable[float]) -> list[float]:
    return sorted({round(float(t), 6) for t in np.geomspace(min(0.05, T), T, 60)}
                  | {float(h) for h in extra if h <= T + _TOL})


def reduce_e3(ctx: Reduction, w: Writer) -> None:
    for (ds, regime), cells in ctx.primary("E1").items():
        T = _group_T(cells.values(), f"E1 {ds}/{regime}")
        P = ctx.of_truth(ds, regime, _uids(cells.values()), "positive")
        if not P:
            continue
        grid = _curve_grid(T, protocol.TIME["reported_T_seconds"])
        rows = []
        for system in system_order(cells):
            cell = cells[system]
            for t in grid:
                # one seed row for a fixed method: mean_rate is then the ordinary Wilson rate
                p, lo, hi = mean_rate(outcome_matrix(cell, P, solved_at(t)))
                rows.append({"series": system, "x": t, "y": 100.0 * p, "lo": 100.0 * lo, "hi": 100.0 * hi})
        w.nums.figure(key("E3", escape(ds), seg_regime(regime)), "curve", rows, x="seconds", y="solved (%)", log_x=True,
                      notes=f"{len(P)} positives; Wilson 95% over queries (a seed-averaged system: of the mean over "
                            "queries of the per-query seed means)")


def reduce_pe2(ctx: Reduction, w: Writer) -> None:
    margin = float(protocol.STATISTICS["noninferiority_margin_pp"])
    alpha = float(protocol.STATISTICS["alpha"])
    w.value("PE2.margin", margin, "pp", [], digits=1, note="pre-registered non-inferiority margin")
    e1 = ctx.primary("E1")
    primary: dict[str, dict[str, dict[str, Any]]] = {}
    secondary: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
    c2_rows = []
    for (ds, regime), cells in e1.items():
        if regime not in ("R0", "R1"):
            continue
        h30 = primary_horizon(_group_T(cells.values(), f"E1 {ds}/{regime}"))
        P = ctx.of_truth(ds, regime, _uids(cells.values()), "positive")
        if not P or "filterall" not in cells:
            ctx.notes.append(f"PE2 {ds}/{regime}: no FilterAll or no positives")
            continue
        outcome = solved_at(h30)
        if "jigsaw" in cells:
            primary.setdefault(regime, {})[ds] = paired(cells["jigsaw"], cells["filterall"], P, outcome)
            c2_rows += _claim_c2(ctx, w, ds, regime, cells, P, h30)
        for comparator in ("filterall", "jigsaw_index"):
            if comparator not in cells:
                continue
            for system in system_order(cells):
                if system == comparator or (comparator == "filterall" and system == "jigsaw"):
                    continue
                secondary.setdefault((regime, comparator), {})[(ds, system)] = paired(
                    cells[system], cells[comparator], P, outcome)
    for regime, results in sorted(primary.items()):
        adjusted = _holm({ds: r["res"].p_value for ds, r in results.items()})
        rows = []
        for ds, result in sorted(results.items()):
            cells = e1[(ds, regime)]
            prefix = key("PE2", escape(ds), seg_regime(regime), "jigsaw_vs_filterall")
            sources = _sources([cells["jigsaw"], cells["filterall"]])
            w.paired_keys(prefix, result, adjusted[ds], sources)
            res = result["res"]
            w.text(f"{prefix}.noninferior", "yes" if stats.non_inferior(res, margin) else "no", sources,
                   note=f"lower bound of the paired interval above -{margin:g} pp")
            w.text(f"{prefix}.superior", "yes" if stats.superior(replace(res, p_value=adjusted[ds]), alpha) else "no",
                   sources, note=f"lower bound above 0 and Holm p below {alpha:g}")
            h = seg_horizon(primary_horizon(cells["jigsaw"].T))
            rows.append([ds, w.cell(key("E1", escape(ds), seg_regime(regime), "jigsaw", "solved", h), ci=True),
                         w.cell(key("E1", escape(ds), seg_regime(regime), "filterall", "solved", h), ci=True),
                         w.cell(f"{prefix}.diff", ci=True), w.cell(f"{prefix}.p_holm"), w.cell(f"{prefix}.noninferior"),
                         w.cell(f"{prefix}.superior"), w.cell(f"{prefix}.diff_hier", ci=True)])
        w.nums.table(key("PE2", seg_regime(regime)), ["dataset", "jigsaw", "filterall", "difference (pp)", "p (Holm)",
                                                       "non-inferior", "superior", "hierarchical interval"], rows,
                     notes=f"solved@30s on positives, paired over queries, Holm across datasets; margin {margin:g} pp"
                           + ("" if regime == "R0" else "; secondary regime"))
    for (regime, comparator), results in sorted(secondary.items()):
        adjusted = _holm({f"{ds}|{s}": r["res"].p_value for (ds, s), r in results.items()})
        rows = []
        for (ds, system), result in sorted(results.items()):
            cells = e1[(ds, regime)]
            prefix = key("PE2", escape(ds), seg_regime(regime), f"{seg_system(system)}_vs_{comparator}")
            w.paired_keys(prefix, result, adjusted[f"{ds}|{system}"], _sources([cells[system], cells[comparator]]))
            rows.append([ds, system, w.cell(f"{prefix}.diff", ci=True), w.cell(f"{prefix}.p_holm"),
                         w.cell(f"{prefix}.diff_hier", ci=True)])
        w.nums.table(key("PE2", seg_regime(regime), f"vs_{comparator}"),
                     ["dataset", "system", "difference (pp)", "p (Holm)", "hierarchical interval"], rows,
                     notes=f"every system vs {comparator}, solved@30s on positives; Holm within the table")
    if c2_rows:
        w.nums.table("PE2.c2", ["dataset", "regime", "failure stage", "pairs"], c2_rows,
                     notes="(query, seed) pairs FilterAll solved at 30 s and Jigsaw did not (claim C2)")


def _claim_c2(ctx: Reduction, w: Writer, ds: str, regime: str, cells: Mapping[str, Cell], P: Sequence[str],
              h30: float) -> list[list[Any]]:
    """Where Jigsaw stood on the (query, seed) pairs FilterAll solved by 30 s and it did not."""
    base = key("PE2", escape(ds), seg_regime(regime), "c2")
    counts: dict[str, int] = {}
    pairs = 0
    jig, fa = cells["jigsaw"], cells["filterall"]
    for rep in jig.replicates():
        for u in P:
            rec = jig.reps[rep][u]
            if fl.solved_by(fa.reps[None][u], h30) and not fl.solved_by(rec, h30):
                pairs += 1
                category = rec["e18"][h30]
                counts[category] = counts.get(category, 0) + 1
    sources = _sources([jig, fa])
    w.count(f"{base}.pairs", pairs, sources,
            note="(query, seed) pairs FilterAll solved and Jigsaw did not, by Jigsaw's failure stage (E18)")
    rows = []
    for category, n in sorted(counts.items()):
        w.count(f"{base}.{escape(category)}", n, sources)
        rows.append([ds, regime, category, w.cell(f"{base}.{escape(category)}")])
    return rows


# ----------------------------------------------------------------------------
# E3x
# ----------------------------------------------------------------------------
def reduce_e3x(ctx: Reduction, w: Writer) -> None:
    e1 = ctx.primary("E1")
    table_rows: dict[str, list[list[Any]]] = {}
    for (ds, regime), xcells in ctx.primary("E3x").items():
        if (ds, regime) not in e1:
            raise ReduceError(f"E3x {ds}/{regime}: no E1 run to extend")
        by_arm: dict[str, dict[str, dict[str, Any]]] = {}
        for cell in e1[(ds, regime)].values():
            for rec in cell.records():
                by_arm.setdefault(rec["arm_s"], {})[rec["uid"]] = rec
        T1 = _group_T(e1[(ds, regime)].values(), f"E1 {ds}/{regime}")
        P = ctx.of_truth(ds, regime, _uids(e1[(ds, regime)].values()), "positive")
        e1_runs = set().union(*(c.runs for c in e1[(ds, regime)].values()))
        rows = []
        for system in system_order(xcells):
            xcell = xcells[system]
            first = by_arm.get(system)
            if first is None:
                raise ReduceError(f"E3x {ds}/{regime}: arm {system} has no E1 records to extend")
            T3 = xcell.T
            if T3 <= T1 + _TOL:
                raise ReduceError(f"E3x {ds}/{regime} {system}: T = {T3:g} does not extend E1's {T1:g}")
            xrecs = xcell.reps.get(None, {})  # empty when E1 left nothing to rerun
            for uid in sorted(xrecs):
                rec = first.get(uid)
                if rec is None:
                    raise ReduceError(f"E3x {ds}/{regime} {system}: {uid} was not run (or was excluded) in E1")
                if fl.solved_by(rec, T1) or fl.none_proved_by(rec, T1):
                    raise ReduceError(f"E3x {ds}/{regime} {system}: {uid} was already decided in E1 "
                                      f"({rec['status']} at {rec.get('time_decided')})")

            def solved(u: str, t: float, first: dict = first, xrecs: dict = xrecs, T1: float = T1) -> bool:
                if t <= T1 + _TOL or fl.solved_by(first[u], T1):
                    return fl.solved_by(first[u], min(t, T1))
                return u in xrecs and fl.solved_by(xrecs[u], t)

            base = key("E3x", escape(ds), seg_regime(regime), seg_system(system))
            sources = sorted(xcell.runs | e1_runs)
            for h in E3X_HORIZONS:
                if h <= T3 + _TOL and P:
                    w.nums.rate(f"{base}.solved.{seg_horizon(h)}", sum(1 for u in P if solved(u, h)), len(P),
                                source=sources, note="E1 up to its T, then the E3x rerun of E1's undecided queries")
            open_pos = [u for u in P if not fl.solved_by(first[u], T1) and first[u]["status"] != "ERROR"]
            rerun = [u for u in P if u in xrecs]
            w.count(f"{base}.rerun", len(rerun), sources)
            w.count(f"{base}.not_rerun", len(set(open_pos) - set(xrecs)), sources)
            w.count(f"{base}.newly_solved", sum(1 for u in rerun if fl.solved_by(xrecs[u], T3)), sources)
            for u in sorted(set(open_pos) - set(xrecs)):
                ctx.exclusions.append({"reason": "e3x_not_rerun", "experiment": "E3x", "dataset": ds, "regime": regime,
                                       "system": system, "uid": u, "detail": "keeps its E1 state after E1's T"})
            if P:
                for t in _curve_grid(T3, (*protocol.TIME["reported_T_seconds"], *E3X_HORIZONS)):
                    k = sum(1 for u in P if solved(u, t))
                    lo, hi = stats.wilson_interval(k, len(P))
                    rows.append({"series": system, "x": t, "y": 100.0 * k / len(P), "lo": 100.0 * lo, "hi": 100.0 * hi})
            table_rows.setdefault(ds, []).append([regime, system, *[w.cell(f"{base}.solved.{seg_horizon(h)}", ci=True)
                                                                    for h in E3X_HORIZONS]])
        if rows:
            w.nums.figure(key("E3x", escape(ds), seg_regime(regime)), "curve", rows, x="seconds", y="solved (%)",
                          log_x=True, notes="solved(t) = E1 up to its T, then the E3x rerun; Wilson 95%")
    for ds, rows in sorted(table_rows.items()):
        w.nums.table(key("E3x", escape(ds)), ["regime", "system", *[f"solved@{h:g}s" for h in E3X_HORIZONS]], rows,
                     notes="the 600 s extension on the queries E1 left undecided")


# ----------------------------------------------------------------------------
# E7 (with E7-classical and E12)
# ----------------------------------------------------------------------------
def reduce_e7(ctx: Reduction, w: Writer) -> None:
    first = first_arm()
    groups = ctx.primary("E7")
    best_tests: dict[str, dict[str, tuple[str, dict[str, Any], list[str]]]] = {}
    points: dict[str, list[tuple[str, str, float]]] = {}
    tables: dict[str, list[list[Any]]] = {}
    for (ds, regime), cells in groups.items():
        T = _group_T(cells.values(), f"E7 {ds}/{regime}")
        uids = _uids(cells.values())
        P = ctx.of_truth(ds, regime, uids, "positive")
        N = ctx.of_truth(ds, regime, uids, *RATE_NEGATIVES)
        h30, h120 = primary_horizon(T), min(float(protocol.TIME["run_T_max_seconds"]["E7"]), T)
        note = ctx.neg_note(ds, regime)
        base_dr = key("E7", escape(ds), seg_regime(regime))
        for system in system_order(cells):
            _e7_system(ctx, w, cells[system], f"{base_dr}.{seg_system(system)}", P, N, h30, h120, T, note)
            base = f"{base_dr}.{seg_system(system)}"
            tables.setdefault(ds, []).append(
                [regime, system, *[w.cell(f"{base}.solved.{seg_horizon(h)}", ci=True) for h in (h30, h120)],
                 w.cell(f"{base}.correct_none.{seg_horizon(h120)}", ci=True), w.cell(f"{base}.tts_median"),
                 w.cell(f"{base}.fp"), w.cell(f"{base}.fn"), w.cell(f"{base}.false_none")])
        outcome = solved_at(h120)
        classical = [s for s in cells if parse_arm(s)[0] in CLASSICAL]
        if first in cells and classical and P:
            solved = {s: sum(1 for u in P if fl.solved_by(cells[s].reps[None][u], h120)) for s in classical}
            best = min(classical, key=lambda s: (-solved[s], s))
            best_tests.setdefault(ds, {})[regime] = (best, paired(cells[first], cells[best], P, outcome),
                                                     _sources([cells[first], cells[best]]))
        if first in cells and "gnnpe" in cells and P:
            w.paired_keys(f"{base_dr}.{seg_system(first)}_vs_gnnpe", paired(cells[first], cells["gnnpe"], P, outcome),
                          None, _sources([cells[first], cells["gnnpe"]]), p_from="mcnemar", counts=False)
        if first in cells:
            for u in P:
                sel = cells[first].reps[None][u].get("selectivity")
                if isinstance(sel, (int, float)):
                    points.setdefault(ds, []).append((regime, u, float(sel)))
    for ds, per_regime in sorted(best_tests.items()):
        adjusted = _holm({r: t[1]["mcnemar"][2] for r, t in per_regime.items()})
        rows = []
        for regime, (best, result, sources) in sorted(per_regime.items()):
            prefix = key("E7", escape(ds), seg_regime(regime), f"{seg_system(first)}_vs_best_classical")
            w.text(f"{prefix}.best", best, sources, note="the classical matcher with the highest solved@120s")
            w.paired_keys(prefix, result, adjusted[regime], sources, p_from="mcnemar", counts=False)
            rows.append([regime, best, w.cell(f"{prefix}.b"), w.cell(f"{prefix}.c"), w.cell(f"{prefix}.diff", ci=True),
                         w.cell(f"{prefix}.p_holm")])
        w.nums.table(key("E7", escape(ds), "best_classical"), ["regime", "best classical", "b", "c", "difference (pp)",
                                                                "p (Holm)"], rows,
                     notes=f"{first} vs the best classical matcher, solved@120s; exact McNemar, Holm across regimes")
    for ds, rows in sorted(tables.items()):
        notes = sorted({n for (d, r) in groups if d == ds for n in [ctx.neg_note(d, r)] if n})
        w.nums.table(key("E7", escape(ds)), ["regime", "system", "solved@30s", "solved@120s", "correct NONE@120s",
                                             "median time to solution", "FP", "FN", "false NONE"], rows,
                     notes=join_notes(*notes))
    _selectivity_bins(ctx, w, points)


def _e7_system(ctx: Reduction, w: Writer, cell: Cell, base: str, P: Sequence[str], N: Sequence[str], h30: float,
               h120: float, T: float, note: str | None) -> None:
    recs = cell.records()
    sources = sorted(cell.runs)
    for h in sorted({h30, h120}):
        w.rate(f"{base}.solved.{seg_horizon(h)}", cell, P, solved_at(h))
    w.rate(f"{base}.correct_none.{seg_horizon(h120)}", cell, N, none_at(h120), note=note)
    times = [float(r["time_found"]) for r in recs if r["truth"] == "positive" and fl.solved_by(r, T)]
    if times:
        w.value(f"{base}.tts_median", float(np.median(times)), "seconds", sources, n=len(times), digits=2,
                note="over the positives solved within T")
    verdicts = [soundness(r) for r in recs]
    for metric in ("fp", "fn", "crash"):
        w.count(f"{base}.{metric}", sum(1 for v in verdicts if v == metric), sources)
    w.count(f"{base}.none_unverified", sum(1 for r in recs if r["status"] == "NONE" and r.get("time_decided") is None),
            sources, note="a NONE without proof (GNN-PE's search): undecided")
    w.count(f"{base}.false_none", sum(1 for v in verdicts if v == "false_none"), sources,
            note="a NONE without proof on a query that has an embedding")
    peaks = [float(r["peak_rss_bytes"]) for r in recs if isinstance(r.get("peak_rss_bytes"), (int, float))]
    if peaks:
        w.value(f"{base}.peak_rss_bytes.max", max(peaks), "bytes", sources, n=len(peaks), digits=0)
        w.value(f"{base}.peak_rss_bytes.median", float(np.median(peaks)), "bytes", sources, n=len(peaks), digits=0)
    _router_keys(w, base, cell, recs)


def _selectivity_bins(ctx: Reduction, w: Writer, points: Mapping[str, list[tuple[str, str, float]]]) -> None:
    """Solved@120s per selectivity bin (the router's, of the Jigsaw arm) and system, pooled over regimes."""
    groups = ctx.primary("E7")
    h_max = float(protocol.TIME["run_T_max_seconds"]["E7"])
    for ds, pts in sorted(points.items()):
        values = np.array([p[2] for p in pts])
        if (values < 0).any():
            raise ReduceError(f"E7 {ds}: a negative selectivity (it is a mean of log10(max(1, df)))")
        edges = [float(q) for q in np.quantile(values, [0.25, 0.5, 0.75])]
        runs = _sources([c for (d, _), cells in groups.items() if d == ds for c in cells.values()])
        for name, edge in zip(("q25", "q50", "q75"), edges):
            w.value(key("E7", escape(ds), "selectivity", name), edge, "number", runs, n=len(values), digits=2)
        labelled = {"quartile": [(f"q{int(np.searchsorted(edges, s, side='left')) + 1}", r, u) for r, u, s in pts],
                    "log10": [(next(b for b, lo, hi in LOG10_BINS if lo <= s < hi), r, u) for r, u, s in pts]}
        systems = sorted({s for (d, _), cells in groups.items() if d == ds for s in cells})
        for scheme, members in labelled.items():
            order = [f"q{i}" for i in range(1, 5)] if scheme == "quartile" else [b for b, _, _ in LOG10_BINS]
            rows = []
            for system in system_order(systems):
                for label in order:
                    chosen = [(r, u) for b, r, u in members if b == label and system in groups[(ds, r)]]
                    if not chosen:
                        continue
                    h = min(h_max, min(groups[(ds, r)][system].T for r, _ in chosen))
                    k = sum(1 for r, u in chosen if fl.solved_by(groups[(ds, r)][system].reps[None][u], h))
                    sources = _sources([groups[(ds, r)][system] for r in sorted({r for r, _ in chosen})])
                    w.nums.rate(key("E7", escape(ds), seg_system(system), scheme, label, "solved", seg_horizon(h)), k,
                                len(chosen), source=sources,
                                note=f"positives pooled over regimes, binned by the selectivity of {first_arm()}")
                    lo, hi = stats.wilson_interval(k, len(chosen))
                    rows.append({"group": label, "series": system, "y": 100.0 * k / len(chosen), "lo": 100.0 * lo,
                                 "hi": 100.0 * hi})
            if rows:
                w.nums.figure(key("E7", escape(ds), f"crossover_{scheme}"), "bars", rows, x="selectivity bin",
                              y="solved (%)", notes="quartiles of the pooled selectivities" if scheme == "quartile"
                              else "log10 bins [0,1) [1,2) [2,3) [3,inf) of the selectivity")


# ----------------------------------------------------------------------------
# E8 and E9
# ----------------------------------------------------------------------------
def reduce_e8(ctx: Reduction, w: Writer) -> None:
    for (ds, regime), cells in ctx.primary("E8").items():
        T = _group_T(cells.values(), f"E8 {ds}/{regime}")
        P = ctx.of_truth(ds, regime, _uids(cells.values()), "positive")
        horizons = sorted({primary_horizon(T), T})
        base_dr = key("E8", escape(ds), seg_regime(regime))
        fams = _families(ctx, ds, regime, P)
        for system in system_order(cells):
            cell = cells[system]
            base = f"{base_dr}.{seg_system(system)}"
            sources = sorted(cell.runs)
            for h in horizons:
                w.rate(f"{base}.solved.{seg_horizon(h)}", cell, P, solved_at(h))
            for fam, fam_uids in fams.items():
                for size, size_uids in _sizes(ctx, ds, regime, fam_uids).items():
                    for h in horizons:
                        w.rate(f"{base}.{fam}.{seg_size(size)}.solved.{seg_horizon(h)}", cell, size_uids, solved_at(h))
            recs = cell.records()
            times = [float(r["time_found"]) for r in recs if r["truth"] == "positive" and fl.solved_by(r, T)]
            if times:
                w.value(f"{base}.tts_median", float(np.median(times)), "seconds", sources, n=len(times), digits=2,
                        note="over the positives solved within T")
            peaks = [float(r["peak_rss_bytes"]) for r in recs if isinstance(r.get("peak_rss_bytes"), (int, float))]
            if peaks:
                w.value(f"{base}.peak_rss_bytes.max", max(peaks), "bytes", sources, n=len(peaks), digits=0)
        if "direct" in cells and "filterall" in cells and P:
            results = {h: paired(cells["direct"], cells["filterall"], P, solved_at(h)) for h in horizons}
            adjusted = _holm({seg_horizon(h): r["mcnemar"][2] for h, r in results.items()})
            for h, result in results.items():
                w.paired_keys(f"{base_dr}.direct_vs_filterall.{seg_horizon(h)}", result, adjusted[seg_horizon(h)],
                              _sources([cells["direct"], cells["filterall"]]), p_from="mcnemar", counts=False)
        rows = [[fam, str(size), *[w.cell(f"{base_dr}.{seg_system(s)}.{fam}.{seg_size(size)}.solved.{seg_horizon(h)}",
                                          ci=True) for s in system_order(cells) for h in horizons]]
                for fam, fam_uids in fams.items() for size in _sizes(ctx, ds, regime, fam_uids)]
        columns = [f"{s} solved@{h:g}s" for s in system_order(cells) for h in horizons]
        w.nums.table(base_dr, ["family", "size", *columns],
                     rows, notes="single-step methods on positives; McNemar keys direct_vs_filterall, Holm across "
                                 "horizons")


def reduce_e9(ctx: Reduction, w: Writer) -> None:
    for (ds, regime), cells in ctx.primary("E9").items():
        aliases: dict[str, str] = {}
        bases = set()
        for system in cells:
            alias, base_arm = e9_alias(system)
            aliases[alias] = system
            bases.add(base_arm)
        if len(bases) != 1:
            raise ReduceError(f"E9 {ds}/{regime}: ablations of different arms {sorted(map(str, bases))}")
        if "full" not in aliases:
            raise ReduceError(f"E9 {ds}/{regime}: no unablated arm")
        T = _group_T(cells.values(), f"E9 {ds}/{regime}")
        h = primary_horizon(T)
        ht = seg_horizon(h)
        uids = _uids(cells.values())
        P = ctx.of_truth(ds, regime, uids, "positive")
        N = ctx.of_truth(ds, regime, uids, *RATE_NEGATIVES)
        note = ctx.neg_note(ds, regime)
        base_dr = key("E9", escape(ds), seg_regime(regime))
        order = [a for a in E9_ALIASES.values() if a in aliases]
        for alias in order:
            cell = cells[aliases[alias]]
            base = f"{base_dr}.{alias}"
            recs = cell.records()
            sources = sorted(cell.runs)
            w.rate(f"{base}.solved.{ht}", cell, P, solved_at(h))
            w.rate(f"{base}.correct_none.{ht}", cell, N, none_at(h), note=note)
            w.count(f"{base}.incomplete", sum(1 for r in recs if r["status"] == "INCOMPLETE"), sources)
            _cost_keys(w, base, cell, recs, p95=False)
            elapsed = [float(r["elapsed"]) for r in recs if r.get("elapsed") is not None]
            if elapsed:
                w.value(f"{base}.elapsed.median", float(np.median(elapsed)), "seconds", sources, n=len(elapsed),
                        digits=3)
        full = cells[aliases["full"]]
        for kind, members, outcome in (("pos", P, solved_at(h)), ("neg", N, none_at(h))):
            if not members:
                continue
            results = {a: paired(cells[aliases[a]], full, members, outcome) for a in order if a != "full"}
            adjusted = _holm({a: r["mcnemar"][2] for a, r in results.items()})
            for alias, result in results.items():
                w.paired_keys(f"{base_dr}.{alias}_vs_full.{kind}", result, adjusted[alias],
                              _sources([cells[aliases[alias]], full]), p_from="mcnemar", counts=False,
                              note=note if kind == "neg" else None)
        rows = [[alias, aliases[alias], w.cell(f"{base_dr}.{alias}.solved.{ht}", ci=True),
                 w.cell(f"{base_dr}.{alias}_vs_full.pos.diff", ci=True),
                 w.cell(f"{base_dr}.{alias}_vs_full.pos.p_holm"),
                 w.cell(f"{base_dr}.{alias}.correct_none.{ht}", ci=True),
                 w.cell(f"{base_dr}.{alias}_vs_full.neg.p_holm"), w.cell(f"{base_dr}.{alias}.incomplete"),
                 w.cell(f"{base_dr}.{alias}.cand_nodes.median"), w.cell(f"{base_dr}.{alias}.elapsed.median")]
                for alias in order]
        w.nums.table(base_dr, ["ablation", "arm", f"solved@{h:g}s", "vs full (pp)", "p (Holm)",
                               f"correct NONE@{h:g}s", "p (Holm)", "INCOMPLETE", "median candidate nodes",
                               "median time (s)"], rows,
                     notes=join_notes("exact McNemar vs the unablated arm, Holm across the ablations", note))


# ----------------------------------------------------------------------------
# E16: sensitivity
# ----------------------------------------------------------------------------
def reduce_e16(ctx: Reduction, w: Writer) -> None:
    for (ds, regime), cells in ctx.groups("E16").items():
        aliases: dict[tuple[str, str], Cell] = {}
        seeds = set()
        for (partition, system), cell in cells.items():
            alias, seed = e16_alias(system)
            if seed is not None:
                seeds.add(seed)
            if (partition, alias) in aliases:
                raise ReduceError(f"E16 {ds}/{regime} {partition}: two arms named {alias}")
            aliases[(partition, alias)] = cell
        if len(seeds) > 1:
            raise ReduceError(f"E16 {ds}/{regime}: Jigsaw arms of several training seeds {sorted(seeds)}")
        T = _group_T(cells.values(), f"E16 {ds}/{regime}")
        h = primary_horizon(T)
        ht = seg_horizon(h)
        uids = _uids(cells.values())
        P = ctx.of_truth(ds, regime, uids, "positive")
        N = ctx.of_truth(ds, regime, uids, *RATE_NEGATIVES)
        note = ctx.neg_note(ds, regime)
        base_dr = key("E16", escape(ds), seg_regime(regime))
        for (partition, alias), cell in sorted(aliases.items()):
            _e16_arm(w, cell, f"{base_dr}.{seg_partition(partition)}", alias, P, N, h, note)
        _e16_sensitivity(w, base_dr, aliases, P, h, ht, note)
        _e16_partitions(w, base_dr, aliases, P, h, ht)


def _e16_arm(w: Writer, cell: Cell, base_part: str, alias: str, P: Sequence[str], N: Sequence[str], h: float,
             note: str | None) -> None:
    base = f"{base_part}.{alias}"
    recs = cell.records()
    sources = sorted(cell.runs)
    w.rate(f"{base}.solved.{seg_horizon(h)}", cell, P, solved_at(h))
    w.rate(f"{base}.correct_none.{seg_horizon(h)}", cell, N, none_at(h), note=note)
    _cost_keys(w, base, cell, recs, p95=False)
    elapsed = [float(r["elapsed"]) for r in recs if r.get("elapsed") is not None]
    if elapsed:
        w.value(f"{base}.elapsed.median", float(np.median(elapsed)), "seconds", sources, n=len(elapsed), digits=3)
    budgets = (cell.options or {}).get("budgets")
    if alias == "jigsaw" and budgets:
        w.text(f"{base_part}.budgets", ",".join(str(int(b)) for b in budgets), sources,
               note="the ladder on this partition: ceil(K * B / P) of its B blocks for each primary step K of P")


def _e16_rows(present: set[str]) -> list[tuple[str, str, str]]:
    """Table (a)'s rows by axis: halo hops, hybrid seed count, selection policy, then the references."""
    halo = sorted([(0, "halo0"), (1, "jigsaw")] + [(int(a[4:]), a) for a in present
                                                  if a.startswith("halo") and a[4:].isdigit() and a != "halo0"])
    seed_count = int(protocol.SELECTION["hybrid"]["seed_count"])
    seeds = sorted([(seed_count, "jigsaw")] + [(int(a[5:]), a) for a in present if a.startswith("seeds")])
    rows = [("halo", str(n), a) for n, a in halo] + [("seed_count", str(n), a) for n, a in seeds]
    rows += [("policy", "hybrid", "jigsaw"), ("policy", "topk", "topk"), ("reference", "-", "jigsaw_index"),
             ("reference", "-", "filterall")]
    return [row for row in rows if row[2] in present]


def _e16_sensitivity(w: Writer, base_dr: str, aliases: Mapping[tuple[str, str], Cell], P: Sequence[str], h: float,
                     ht: str, note: str | None) -> None:
    """(a) The primary partition: every setting against the protocol's Jigsaw arm, grouped by axis."""
    present = {alias for (p, alias) in aliases if p == "primary"}
    if "jigsaw" not in present or not P:
        return
    reference = aliases[("primary", "jigsaw")]
    tests = {a: paired(aliases[("primary", a)], reference, P, solved_at(h)) for a in sorted(present - {"jigsaw"})}
    adjusted = _holm({a: r["res"].p_value for a, r in tests.items()})
    for alias, result in tests.items():
        w.paired_keys(f"{base_dr}.primary.{alias}_vs_jigsaw", result, adjusted[alias],
                      _sources([aliases[("primary", alias)], reference]), counts=False)
    rows = []
    for axis, setting, alias in _e16_rows(present):
        base = f"{base_dr}.primary.{alias}"
        rows.append([axis, setting, alias, w.cell(f"{base}.solved.{ht}", ci=True),
                     w.cell(f"{base_dr}.primary.{alias}_vs_jigsaw.diff", ci=True),
                     w.cell(f"{base_dr}.primary.{alias}_vs_jigsaw.p_holm"), w.cell(f"{base}.cand_nodes.median"),
                     w.cell(f"{base}.elapsed.median"), w.cell(f"{base}.correct_none.{ht}", ci=True)])
    w.nums.table(f"{base_dr}.sensitivity", ["axis", "setting", "arm", f"solved@{h:g}s", "vs jigsaw (pp)", "p (Holm)",
                                            "median candidate nodes", "median time (s)", f"negatives decided@{h:g}s"],
                 rows, notes=join_notes("primary partition; paired sign-flip test against the protocol's arm, Holm "
                                        "within the table; candidate nodes of the deciding (else last) step", note))


def _e16_partitions(w: Writer, base_dr: str, aliases: Mapping[tuple[str, str], Cell], P: Sequence[str], h: float,
                    ht: str) -> None:
    """(b) The partitions: the three references on each, and Jigsaw minus FilterAll."""
    order = [p for p in ["primary", *sensitivity_partitions()] if any(q == p for q, _ in aliases)]
    tests = {p: paired(aliases[(p, "jigsaw")], aliases[(p, "filterall")], P, solved_at(h)) for p in order
             if (p, "jigsaw") in aliases and (p, "filterall") in aliases and P}
    adjusted = _holm({p: r["res"].p_value for p, r in tests.items()})
    for partition, result in tests.items():
        w.paired_keys(f"{base_dr}.{seg_partition(partition)}.jigsaw_vs_filterall", result, adjusted[partition],
                      _sources([aliases[(partition, "jigsaw")], aliases[(partition, "filterall")]]), counts=False)
    rows = []
    for partition in order:
        seg = seg_partition(partition)
        rows.append([partition, *[w.cell(f"{base_dr}.{seg}.{a}.solved.{ht}", ci=True)
                                  for a in ("jigsaw", "jigsaw_index", "filterall")],
                     w.cell(f"{base_dr}.{seg}.jigsaw_vs_filterall.diff", ci=True),
                     w.cell(f"{base_dr}.{seg}.jigsaw_vs_filterall.p_holm")])
    if rows:
        w.nums.table(f"{base_dr}.partitions", ["partition", "jigsaw", "jigsaw_index", "filterall",
                                               "jigsaw - filterall (pp)", "p (Holm)"], rows,
                     notes=f"solved@{h:g}s on one query set; sign-flip test, Holm across partitions")


# ----------------------------------------------------------------------------
# E14 and E17
# ----------------------------------------------------------------------------
def reduce_e14(ctx: Reduction, w: Writer) -> None:
    for (ds, regime), cert in sorted(ctx.certs.items()):
        source = f"cert.{escape(ds)}.{seg_regime(regime)}"
        base_dr = key("E14", escape(ds), seg_regime(regime))
        entries = cert["doc"]["queries"]
        totals = {"generated": len(entries), "none": 0, "found": 0, "undecided": 0, "rejected_witnesses": 0,
                  "errors": 0}
        cells: dict[tuple[str, int], dict[str, int]] = {}
        seconds: dict[str, list[float]] = {}
        for e in entries.values():
            label = e["certified"]
            totals[label] += 1
            totals["rejected_witnesses"] += len(e.get("rejected") or [])
            totals["errors"] += len(e.get("errors") or {})
            cell = cells.setdefault((seg_family(str(e.get("family"))), int(e.get("size"))),
                                    {"generated": 0, "none": 0, "found": 0, "undecided": 0})
            cell["generated"] += 1
            cell[label] += 1
            if label in ("none", "found") and e.get("by"):
                by = f"{label}.{escape(e['by'])}"
                cell[by] = cell.get(by, 0) + 1
            for stage, s in (e.get("seconds") or {}).items():
                if isinstance(s, (int, float)) and not isinstance(s, bool):
                    seconds.setdefault(str(stage), []).append(float(s))
        for name, n in totals.items():
            w.count(f"{base_dr}.{name}", n, [source])
        rows = []
        for (fam, size), counts in sorted(cells.items(), key=lambda kv: (FAMILIES.index(kv[0][0]), kv[0][1])):
            base = f"{base_dr}.{fam}.{seg_size(size)}"
            for name, n in sorted(counts.items()):
                w.count(f"{base}.{name}", n, [source])
            rows.append([fam, str(size), *[w.cell(f"{base}.{c}") for c in ("generated", "none", "found", "undecided")]])
        for stage, values in sorted(seconds.items()):
            w.value(f"{base_dr}.{escape(stage)}.seconds_median", float(np.median(values)), "seconds", [source],
                    n=len(values), digits=2)
            w.value(f"{base_dr}.{escape(stage)}.seconds_max", float(np.max(values)), "seconds", [source],
                    n=len(values), digits=2)
        w.nums.table(base_dr, ["family", "size", "generated", "certified none", "relabelled found", "undecided"], rows,
                     notes="negatives decided by complete search (certify_negatives); found ones are relabelled")


E17_NOTES = {
    "prepare": "prepare_data: the summed seconds of its stage markers (a resumed run keeps them); bytes: the store "
               "without the ranker inputs and block embeddings built later",
    "training": "train_encoder: the summed epoch and validation seconds of each kept trajectory (train_log.jsonl, "
                "val_metrics.jsonl), over the encoders",
    **{s: f"offline_build manifest ({s}): extra.wall_seconds, artifact bytes and peak memory"
       for s in offline_build.STAGES},
}


def _e17_quiet(w: Writer, base: str, sources: Sequence[str], quiet: Sequence[Mapping[str, Any]]) -> None:
    """Training time without other work beside it (offline_build.quiet_training), when every encoder has it."""
    measured = sum(int(q.get("measured") or 0) for q in quiet)
    if not measured:
        return  # logs written before train_encoder recorded the machine's load
    below = f"other work below {offline_build.QUIET_OTHER_CPUS:g} busy CPUs"
    w.value(f"{base}.quiet_epoch_share", sum(int(q.get("quiet") or 0) for q in quiet) / measured, "frac", sources,
            n=measured, digits=2, note=f"epochs with {below}, over the epochs with a load record")
    walls = [q.get("wall_seconds") for q in quiet]
    if any(v is None for v in walls):
        return  # an encoder with fewer than QUIET_MIN_EPOCHS quiet epochs: no quiet total
    note = f"the median epoch with {below}, times the epochs, plus the validation seconds"
    w.value(f"{base}.gpu_hours_quiet", sum(walls) / 3600.0, "number", sources, n=len(walls), digits=1,
            note=f"{note}, summed over the encoders")
    w.value(f"{base}.gpu_hours_per_encoder_quiet", float(np.median(walls)) / 3600.0, "number", sources,
            n=len(walls), digits=2, note=f"{note}, the median over the encoders")


def _disk_segment(component: str) -> str:
    """offline_cost's disk component as key segments: graph, sigma, meta or partition.<partition>."""
    head, _, rest = component.partition("/")
    if head != "partition" or not rest:
        return escape(component)
    return f"partition.{partition_aliases().get(rest) or escape(rest)}"


def _e17_prepare(w: Writer, base_ds: str, item: Mapping[str, Any]) -> None:
    sources = [item["id"]]
    cost = item["cost"]
    for name, part in cost["parts"].items():
        w.value(f"{base_ds}.prepare.{escape(name)}.wall_seconds", float(part["wall_seconds"]), "seconds", sources,
                digits=1, note="the stage's marker")
        if part["peak_bytes"] is not None:
            w.value(f"{base_ds}.prepare.{escape(name)}.peak_bytes", int(part["peak_bytes"]), "bytes", sources,
                    digits=0)
    for component, size in sorted(cost["disk"].items()):
        w.value(f"{base_ds}.disk.{_disk_segment(component)}", int(size), "bytes", sources, digits=0)
    if cost["disk"]:
        w.value(f"{base_ds}.disk.total", sum(int(v) for v in cost["disk"].values()), "bytes", sources, digits=0,
                note="the prepared store without the ranker inputs and block embeddings (their builds count them)")


def _gnnpe_builds(ctx: Reduction) -> dict[tuple[str, str], dict[str, Any]]:
    """GNN-PE's offline builds per (dataset, regime) from the E12 manifests: {prepared_utc: seconds}, run ids."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for run in ctx.runs:
        if run.entry != "E12" or run.arms != ["gnnpe"]:
            continue
        prepared = run.extra.get("prepare")
        for regime in run.regimes:
            info = prepared.get(regime) if isinstance(prepared, dict) else None
            info = info if isinstance(info, dict) else {}
            seconds, utc = info.get("offline_seconds"), info.get("prepared_utc")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                    or seconds < 0 or not isinstance(utc, str) or not utc):
                raise ReduceError(f"{run.path}: extra.prepare.{regime} records no GNN-PE build (the offline_seconds "
                                  "and prepared_utc of gnnpe_adapter.prepare)")
            item = out.setdefault((run.dataset, regime), {"builds": {}, "runs": set()})
            known = item["builds"].setdefault(utc, float(seconds))
            if abs(known - float(seconds)) > _TOL:
                raise ReduceError(f"{run.path}: the GNN-PE build of {utc} took {float(seconds):g} s, another E12 run "
                                  f"of {run.dataset}/{regime} records {known:g} s")
            item["runs"].add(run.run_id)
    return out


def reduce_e17(ctx: Reduction, w: Writer) -> None:
    stages: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in ctx.offline:
        stages.setdefault((item["dataset"], item["stage"]), []).append(item)
    rows: dict[str, list[list[Any]]] = {}
    for (ds, stage), items in sorted(stages.items()):
        base = key("E17", escape(ds), escape(stage))
        sources = [i["id"] for i in items]
        costs = [i["cost"] for i in items]
        walls = [float(c["wall_seconds"]) for c in costs]
        w.value(f"{base}.wall_seconds", sum(walls), "seconds", sources, n=len(items), digits=0,
                note=E17_NOTES.get(stage, "extra.wall_seconds of the manifests; bytes: their outputs"))
        if len(items) > 1:
            w.value(f"{base}.wall_seconds_median", float(np.median(walls)), "seconds", sources, n=len(items), digits=0,
                    note="median over the stage's runs")
        sizes = [int(c["bytes"]) for c in costs if c["bytes"] is not None]
        if sizes:
            w.value(f"{base}.bytes", sum(sizes), "bytes", sources, n=len(sizes), digits=0)
        peaks = [int(c["peak_bytes"]) for c in costs if c["peak_bytes"] is not None]
        if peaks:
            w.value(f"{base}.peak_bytes", max(peaks), "bytes", sources, n=len(peaks), digits=0,
                    note="the largest peak memory of the stage's runs")
        w.count(f"{base}.runs", len(items), sources)
        machines = sorted({str((i["manifest"].get("machine") or {}).get("cpu_model") or "unknown") for i in items})
        w.text(f"{base}.machine", "; ".join(machines), sources)
        if stage == "prepare":
            _e17_prepare(w, key("E17", escape(ds)), items[0])
        elif stage == "training":
            gpus = sorted({str(g).split(",")[0].strip() for i in items
                           for g in ((i["manifest"].get("machine") or {}).get("gpus") or [])})
            w.value(f"{base}.gpu_hours", sum(walls) / 3600.0, "number", sources, n=len(items), digits=1,
                    note="training hours summed over the encoders (see E17.<ds>.training.gpu for the device)")
            w.value(f"{base}.gpu_hours_per_encoder", float(np.median(walls)) / 3600.0, "number", sources,
                    n=len(items), digits=2, note="median over the encoders")
            w.text(f"{base}.gpu", "; ".join(gpus) if gpus else "none", sources)
            _e17_quiet(w, base, sources, [c.get("quiet") or {} for c in costs])
        rows.setdefault(ds, []).append([ds, stage, w.cell(f"{base}.wall_seconds"), w.cell(f"{base}.bytes"),
                                        w.cell(f"{base}.peak_bytes"), w.cell(f"{base}.runs")])
    for (ds, regime), item in sorted(_gnnpe_builds(ctx).items()):
        base = key("E17", escape(ds), "gnnpe", seg_regime(regime))
        sources = sorted(item["runs"])
        seconds = sorted(item["builds"].values())
        w.value(f"{base}.offline_seconds", max(seconds), "seconds", sources, n=len(seconds), digits=1,
                note="GNN-PE's offline stage (gnnpe_adapter.prepare): the longest of its builds")
        w.count(f"{base}.builds", len(seconds), sources, note="distinct builds (by prepared_utc) the E12 runs used")
        rows.setdefault(ds, []).append([ds, f"gnnpe {regime}", w.cell(f"{base}.offline_seconds"), "-", "-",
                                        w.cell(f"{base}.builds")])
    if rows:
        w.nums.table("E17", ["dataset", "stage", "wall seconds", "bytes", "peak bytes", "runs"],
                     [row for ds in sorted(rows) for row in rows[ds]],
                     notes="offline cost per stage (offline_build.offline_cost); GNN-PE's rows count its builds")


# ----------------------------------------------------------------------------
# E18: failure localization
# ----------------------------------------------------------------------------
def _e18_members(ctx: Reduction) -> list[tuple[str, str, str, str | None, str, list[dict[str, Any]], bool, list[str]]]:
    """(experiment, dataset, regime, setting, system, records, seeded, sources) of every E18 cell."""
    out = []
    for (group, ds, regime, partition, system), cell in sorted(ctx.cells.items()):
        if group in E18_EXPERIMENTS and cell.records():
            setting = seg_partition(partition) if group == "E16" else None
            out.append((group, ds, regime, setting, system, cell.records(), cell.seeded, sorted(cell.runs)))
    for cell in measured_cells(ctx):
        recs = [rec for _, rec in sorted(ctx.m_records.get(cell["id"], {}).items())]
        for regime in sorted({str(r["regime"]) for r in recs}):
            out.append((cell["experiment"], cell["dataset"], regime, f"{cell['mode']}_{cell['cache']}", cell["system"],
                        [r for r in recs if r["regime"] == regime], False, [cell["id"]]))
    return out


def reduce_e18(ctx: Reduction, w: Writer) -> None:
    tables: dict[tuple[str, str], list[list[Any]]] = {}
    regimes: dict[tuple[str, str], set[str]] = {}
    for group, ds, regime, setting, system, recs, seeded, sources in _e18_members(ctx):
        T = max(r["T"] for r in recs)
        h30 = primary_horizon(T)
        horizons = [h for h in E18_HORIZONS if h <= T + _TOL]
        parts = ["E18", escape(group), escape(ds), seg_regime(regime), seg_system(system)]
        base = key(*parts, *([setting] if setting else []))
        seed_note = ("mean over queries of the per-query share of training seeds; Wilson 95% of that mean"
                     if seeded else None)
        regimes.setdefault((group, ds), set()).add(regime)
        loc = ctx.localization.setdefault(group, {}).setdefault(ds, {}).setdefault(regime, {}).setdefault(
            setting or "-", {})
        for rec in recs:
            loc.setdefault(rec["arm"], {})[rec["uid"]] = {"truth": rec["truth"], "detail": rec["e18_detail"],
                                                          **{seg_horizon(h): c for h, c in rec["e18"].items()}}
        positives = [r for r in recs if r["truth"] == "positive"]
        if positives:
            w.count(f"{base}.overlap_rescued", sum(1 for r in positives if r["overlap_rescued"]), sources,
                    note="positives whose witness first entered a candidate through the halo")
        negatives = [r for r in recs if r["truth"] in RATE_NEGATIVES]
        for kind, members in (("pos", positives), ("neg", negatives)):
            if not members:
                continue
            note = seed_note if kind == "pos" else join_notes(seed_note, ctx.neg_note(ds, regime))
            seen = sorted({r["e18"][h] for r in members for h in horizons})
            for h in horizons:
                for category in seen:
                    _share(w, f"{base}.{kind}.{escape(category)}.{seg_horizon(h)}", members,
                           lambda r, h=h, c=category: r["e18"][h] == c, sources, seeded, note)
            for category in seen:
                tables.setdefault((group, ds), []).append([regime, system, setting or "-", kind, category, *[
                    w.cell(f"{base}.{kind}.{escape(category)}.{seg_horizon(x)}") for x in E18_HORIZONS]])
            by_family: dict[str, list[dict[str, Any]]] = {}
            for r in members:
                by_family.setdefault(r["family"], []).append(r)
            for fam, fam_recs in sorted(by_family.items()):
                for category in sorted({r["e18"][h30] for r in fam_recs}):
                    _share(w, f"{base}.{seg_family(fam)}.{kind}.{escape(category)}.{seg_horizon(h30)}", fam_recs,
                           lambda r, c=category, h=h30: r["e18"][h] == c, sources, seeded, note)
    for (group, ds), rows in sorted(tables.items()):
        neg_notes = sorted({n for r in regimes[(group, ds)] for n in [ctx.neg_note(ds, r)] if n})
        w.nums.table(key("E18", escape(group), escape(ds)), ["regime", "system", "setting", "kind", "category",
                                                             *[f"{h:g}s" for h in E18_HORIZONS]], rows,
                     notes=join_notes("share of queries in each failure-localization category (failure_localize.py)",
                                      *neg_notes))


def _share(w: Writer, k: str, members: Sequence[Mapping[str, Any]], hit: Outcome, sources: Sequence[str],
           seeded: bool, note: str | None) -> None:
    """The share of members that hit, over queries: a seed-averaged cell's per-query means (as mean_rate)."""
    if not seeded:
        w.nums.rate(k, sum(1 for r in members if hit(r)), len(members), source=sources, note=note)
        return
    per: dict[str, list[float]] = {}
    for r in members:
        per.setdefault(str(r["uid"]), []).append(1.0 if hit(r) else 0.0)
    successes = sum(float(np.mean(v)) for v in per.values())
    lo, hi = stats.wilson_interval(successes, len(per))
    w.value(k, 100.0 * successes / len(per), "pct", sources, ci=(100.0 * lo, 100.0 * hi), n=len(per), digits=1,
            note=note)


# ----------------------------------------------------------------------------
# E2 and E10: E19, PE3 and PE4
# ----------------------------------------------------------------------------
def measured_cells(ctx: Reduction, experiment: str | None = None) -> list[dict[str, Any]]:
    """Per (experiment, dataset, system, mode, cache): the uncapped cell of repeat 1, else of the lowest
    completed repeat; the measurement E18 and E19 read."""
    chosen: dict[tuple, dict[str, Any]] = {}
    for cell in sorted(ctx.measured, key=lambda c: c["repeat"]):
        if cell["cap"] is None and cell["run"] is not None and experiment in (None, cell["experiment"]):
            chosen.setdefault((cell["experiment"], cell["dataset"], cell["system"], cell["mode"], cell["cache"]), cell)
    return [chosen[k] for k in sorted(chosen)]


def reduce_measured(ctx: Reduction, w: Writer) -> None:
    for exp in MEASURED:
        if any(c["experiment"] == exp for c in ctx.measured):
            reduce_e19(ctx, w, exp)
            reduce_pe3(ctx, w, exp)
            reduce_pe4(ctx, w, exp)


def reduce_e19(ctx: Reduction, w: Writer, exp: str) -> None:
    percentiles = [int(p) for p in protocol.SERVING["latency_percentiles"]]
    e1 = ctx.primary("E1")
    covered: set[tuple[str, str]] = set()
    for cell in measured_cells(ctx, exp):
        ds, system = cell["dataset"], cell["system"]
        recs = [rec for _, rec in sorted(ctx.m_records.get(cell["id"], {}).items())]
        _quantile_keys(w, measured_key(exp, "E19", escape(ds), seg_system(system), f"{cell['mode']}_{cell['cache']}"),
                       recs, [cell["id"]], percentiles)
        if (ds, system) in covered or (ds, "R0") not in e1:
            continue
        covered.add((ds, system))
        members = [c for c in e1[(ds, "R0")].values() if any(r["arm_s"] == system for r in c.records())]
        reference = [r for c in members for r in c.records() if r["arm_s"] == system]
        if reference:
            _quantile_keys(w, measured_key(exp, "E19", escape(ds), seg_system(system), "e1_resident"), reference,
                           _sources(members), percentiles)


def _quantile_keys(w: Writer, base: str, recs: Sequence[Mapping[str, Any]], sources: Sequence[str],
                   percentiles: Sequence[int]) -> None:
    columns: dict[str, list[float]] = {}
    for r in recs:
        for stage, value in r["stages"].items():
            columns.setdefault(escape(stage), []).append(float(value))
        for name in ("elapsed", "wall_seconds"):
            if isinstance(r.get(name), (int, float)):
                columns.setdefault(name, []).append(float(r[name]))
    for name, values in sorted(columns.items()):
        for p in percentiles:
            point, lo, hi = stats.quantile_with_ci(values, p / 100.0, n_boot=DRAWS["quantile"], seed=SEED)
            w.value(f"{base}.{name}.p{p}", point, "seconds", sources, ci=(lo, hi), n=len(values), digits=4)


def bytes_read(rec: Mapping[str, Any]) -> float | None:
    """Storage bytes the query's process tree read, or None without the counters (PE3 refuses the row)."""
    counters = rec.get("io") or {}
    children = counters.get("children_block_input_bytes")
    if children is None:
        return None
    if counters.get("read_bytes") is not None:
        return float(counters["read_bytes"]) + float(children)
    if counters.get("block_input_bytes") is not None:
        return float(counters["block_input_bytes"]) + float(children)
    return None


def entries_read(rec: Mapping[str, Any]) -> float | None:
    """Adjacency entries the query's candidate steps read: the logical cost, which counts entries and not pages.
    A query decided before any candidate (a certificate) read none; None when a step lacks the counter."""
    total = 0.0
    for step in rec.get("steps") or []:
        value = (step.get("candidate") or {}).get("adjacency_entries_read")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        total += float(value)
    return total


def latency(rec: Mapping[str, Any]) -> float:
    """Elapsed, censored at T (an unfinished query costs T), at least LATENCY_FLOOR."""
    elapsed = rec.get("elapsed")
    value = float(rec["T"]) if elapsed is None else min(float(elapsed), float(rec["T"]))
    return max(value, LATENCY_FLOOR)


def _pe3_values(ctx: Reduction, a_cell: dict[str, Any], b_cell: dict[str, Any], metric: str,
                exclusion: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]] | None:
    ra, rb = ctx.m_records.get(a_cell["id"], {}), ctx.m_records.get(b_cell["id"], {})
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    if metric == "latency":
        return {u: latency(ra[u]) for u in common}, {u: latency(rb[u]) for u in common}
    if metric == "entries":
        ea, eb = {u: ra[u].get("entries_read") for u in common}, {u: rb[u].get("entries_read") for u in common}
        if any(v is None for v in (*ea.values(), *eb.values())):
            ctx.exclusions.append({**exclusion, "reason": "pe3_no_entry_counters",
                                   "detail": "a candidate step without candidate.adjacency_entries_read"})
            return None
        return {u: v + 1.0 for u, v in ea.items()}, {u: v + 1.0 for u, v in eb.items()}
    va, vb = {u: bytes_read(ra[u]) for u in common}, {u: bytes_read(rb[u]) for u in common}
    if any(v is None for v in (*va.values(), *vb.values())):
        ctx.exclusions.append({**exclusion, "reason": "pe3_no_io_counters",
                               "detail": "records without io.read_bytes (or block_input_bytes) and "
                                         "children_block_input_bytes: no bytes-read ratio"})
        return None
    # + 1 byte: a warm query may read nothing from storage, and a ratio needs positive costs
    return {u: v + 1.0 for u, v in va.items()}, {u: v + 1.0 for u, v in vb.items()}


def reduce_pe3(ctx: Reduction, w: Writer, exp: str) -> None:
    pairs = [(first_arm(), "filterall"), ("jigsaw_index", "filterall")]
    cells: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for cell in ctx.measured:
        if (cell["experiment"] == exp and cell["cap"] is None and cell["run"] is not None
                and cell["mode"] == "streaming"):
            cells[(cell["dataset"], cell["system"], cell["cache"], cell["repeat"])] = cell
    results: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = {}
    for ds in sorted({k[0] for k in cells}):
        for cache in ("cold", "warm"):
            for a, b in pairs:
                repeats = sorted(r for (d, s, c, r) in cells if d == ds and s == a and c == cache
                                 and (ds, b, cache, r) in cells)
                for metric in ("bytes", "entries", "latency"):
                    ratios, test = [], None
                    for r in repeats:
                        exclusion = {"experiment": exp, "dataset": ds, "cache": cache, "pair": f"{a} vs {b}",
                                     "repeat": r}
                        values = _pe3_values(ctx, cells[(ds, a, cache, r)], cells[(ds, b, cache, r)], metric, exclusion)
                        if values is None:
                            if metric in ("bytes", "entries"):
                                ratios, test = [], None
                                break
                            continue
                        ratios.append(geometric_mean_ratio(*values))
                        if test is None:
                            test = (r, stats.paired_log_ratio(*values, n_boot=DRAWS["boot"], n_perm=DRAWS["perm"],
                                                              seed=SEED),
                                    [cells[(ds, a, cache, r)]["id"], cells[(ds, b, cache, r)]["id"]])
                    if test is not None:
                        results.setdefault(cache, {})[(ds, a, b, metric)] = {"test": test, "ratios": ratios}
    for cache, family in sorted(results.items()):
        adjusted = _holm({"|".join(k): v["test"][1]["p_value"] for k, v in family.items()})
        rows = []
        for (ds, a, b, metric), item in sorted(family.items()):
            rep, res, sources = item["test"]
            base = measured_key(exp, "PE3", escape(ds), cache, f"{seg_system(a)}_vs_{seg_system(b)}", metric)
            note = f"repeat {rep}; " + {"bytes": "bytes read from storage + 1 (4 KiB pages, readahead included)",
                                       "entries": "adjacency entries read by the candidate steps + 1 (logical)",
                                       "latency": "elapsed, censored at T"}[metric]
            w.value(f"{base}.gm_ratio", res["gm_ratio"], "ratio", sources, ci=(res["lo"], res["hi"]),
                    n=res["n_units"], digits=3, note=note)
            w.pvalue(f"{base}.p", res["p_value"], sources, note="sign-flip on log ratios")
            w.pvalue(f"{base}.p_holm", adjusted[f"{ds}|{a}|{b}|{metric}"], sources)
            w.value(f"{base}.median_ratio", res["median_ratio"], "ratio", sources, n=res["n_units"], digits=3)
            w.count(f"{base}.n", res["n_units"], sources)
            w.value(f"{base}.gm_ratio_min", min(item["ratios"]), "ratio", sources, digits=3,
                    note="over every completed repeat")
            w.value(f"{base}.gm_ratio_max", max(item["ratios"]), "ratio", sources, digits=3,
                    note="over every completed repeat")
            w.count(f"{base}.repeats", len(item["ratios"]), sources)
            rows.append([ds, f"{a} vs {b}", metric, w.cell(f"{base}.gm_ratio", ci=True), w.cell(f"{base}.p_holm"),
                         w.cell(f"{base}.gm_ratio_min"), w.cell(f"{base}.gm_ratio_max")])
        w.nums.table(measured_key(exp, "PE3", cache), ["dataset", "comparison", "metric", "geometric-mean ratio",
                                                       "p (Holm)", "min over repeats", "max over repeats"], rows,
                     notes=f"{exp} streaming, {cache} cache{' (primary)' if cache == 'cold' else ' (secondary)'}; "
                           "paired over queries; Holm within the table")


def reduce_pe4(ctx: Reduction, w: Writer, exp: str) -> None:
    groups: dict[tuple[str, str, str, str, float | None], list[dict[str, Any]]] = {}
    for cell in ctx.measured:
        if cell["experiment"] == exp:
            groups.setdefault((cell["dataset"], cell["system"], cell["mode"], cell["cache"], cell["cap"]),
                              []).append(cell)
    peaks_by: dict[tuple[str, str, str, str], list[float]] = {}
    for (ds, system, mode, cache, cap), members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        base = measured_key(exp, "PE4", escape(ds), seg_system(system), mode, cache, seg_cap(cap))
        sources = [c["id"] for c in members]
        peaks = [float(c["peak"]) for c in members if isinstance(c["peak"], (int, float))]
        if peaks:
            for name, value in (("median", float(np.median(peaks))), ("min", min(peaks)), ("max", max(peaks))):
                w.value(f"{base}.peak_rss.{name}", value, "bytes", sources, n=len(peaks), digits=0)
            if cap is None:
                peaks_by[(ds, system, mode, cache)] = peaks
        w.count(f"{base}.repeats", len(members), sources)
        w.count(f"{base}.success", sum(1 for c in members if c["success"]), sources,
                note="exit code 0, not killed by the cap, not timed out, run complete")
        solved = []
        for c in members:
            positives = [r for r in (ctx.m_records.get(c["id"], {}) if c["success"] else {}).values()
                         if r["truth"] == "positive"]
            if positives:
                h = primary_horizon(positives[0]["T"])
                solved.append((sum(1 for r in positives if fl.solved_by(r, h)), len(positives), h))
        if solved:
            k, n, h = min(solved, key=lambda s: s[0] / s[1])
            w.nums.rate(f"{base}.solved.{seg_horizon(h)}", k, n, source=sources,
                        note="the lowest over the completed repeats")
    _pe4_ratios(ctx, w, exp, groups, peaks_by)
    caps = sorted({*planned_caps(exp), *(k[4] for k in groups)}, key=lambda c: -1.0 if c is None else c)
    rows: dict[str, list[list[Any]]] = {}
    for (ds, system, mode, cache) in sorted({k[:4] for k in groups}):
        rows.setdefault(ds, []).append([system, mode, cache, *[
            w.cell(measured_key(exp, "PE4", escape(ds), seg_system(system), mode, cache, seg_cap(cap), "success"))
            for cap in caps]])
    for ds, table in sorted(rows.items()):
        w.nums.table(measured_key(exp, "PE4", escape(ds), "caps"), ["system", "mode", "cache",
                                                                   *[seg_cap(c) for c in caps]], table,
                     notes=f"{exp}: completed repeats under each memory cap (capnone: no cap)")


def _pe4_ratios(ctx: Reduction, w: Writer, exp: str, groups: Mapping[tuple, list[dict[str, Any]]],
                peaks_by: Mapping[tuple[str, str, str, str], list[float]]) -> None:
    """Jigsaw streaming's peak over each resident system's, uncapped: conservative (its largest over their
    smallest) and median over median."""
    first = first_arm()
    for (ds, system, mode, cache), jpeaks in sorted(peaks_by.items()):
        if system != first or mode != "streaming":
            continue
        for (ds2, other, mode2, cache2), rpeaks in sorted(peaks_by.items()):
            if ds2 != ds or mode2 != "resident" or cache2 != cache or min(rpeaks) <= 0 or min(jpeaks) <= 0:
                continue
            base = measured_key(exp, "PE4", escape(ds), cache, f"{seg_system(first)}_vs_{seg_system(other)}")
            sources = [c["id"] for k in ((ds, first, "streaming", cache, None), (ds, other, "resident", cache, None))
                       for c in groups.get(k, [])]
            w.value(f"{base}.conservative_ratio", max(jpeaks) / min(rpeaks), "ratio", sources, digits=3,
                    note="Jigsaw streaming's largest peak RSS over the resident system's smallest")
            w.value(f"{base}.median_ratio", float(np.median(jpeaks)) / float(np.median(rpeaks)), "ratio", sources,
                    digits=3)


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# E20: the pattern-overlap audit
# ----------------------------------------------------------------------------
def reduce_e20(ctx: Reduction, w: Writer) -> None:
    """A25: the test queries a main encoder drew in training, and the E1 and E7 solve rates on the seen and
    the novel positives. A query is seen when some audited encoder of its regime drew an isomorphic
    training query; a novel query is new to every encoder of the regime."""
    rows = []
    for (ds, regime), audit in sorted(ctx.audits.items()):
        entry, src = audit["entry"], [audit["id"]]
        base = key("E20", escape(ds), seg_regime(regime))
        for name in ("queries", "positives", "negatives", "seen_positives", "seen_negatives"):
            w.count(f"{base}.{name}", int(entry[name]), src)
        w.count(f"{base}.encoders", len(audit["encoders"]), src)
        if int(entry["positives"]):
            w.nums.rate(f"{base}.seen_share", int(entry["seen_positives"]), int(entry["positives"]), source=src,
                        note="positives isomorphic to a training query of some main encoder of the regime")
        for seed, uids in sorted(audit["by_seed"].items()):
            w.count(f"{base}.seen.{seg_seed(seed)}", len(uids), src,
                    note="queries of the set (positives and negatives) the encoder of this training seed drew")
        rows.append([ds, regime, w.cell(f"{base}.positives"), w.cell(f"{base}.seen_positives"),
                     w.cell(f"{base}.seen_share", ci=True), w.cell(f"{base}.negatives"),
                     w.cell(f"{base}.seen_negatives"), w.cell(f"{base}.encoders")])
    if rows:
        w.nums.table("E20.audit", ["dataset", "regime", "positives", "seen positives", "seen share", "negatives",
                                   "seen negatives", "encoders"], rows,
                     notes="seen: isomorphic (every edge, node type and regime label kept) to a training query "
                           "of some main encoder that serves the regime, regenerated from its seed")
    for group in E20_GROUPS:
        for (ds, regime), cells in ctx.primary(group).items():
            audit = ctx.audits.get((ds, regime))
            if audit is None:
                continue
            T = _group_T(cells.values(), f"{group} {ds}/{regime}")
            h = primary_horizon(T) if group == "E1" else min(float(protocol.TIME["run_T_max_seconds"]["E7"]), T)
            ht = seg_horizon(h)
            P = ctx.of_truth(ds, regime, _uids(cells.values()), "positive")
            parts = {"seen": [u for u in P if u in audit["seen"]], "novel": [u for u in P if u not in audit["seen"]]}
            base_dr = key("E20", group, escape(ds), seg_regime(regime))
            everyone = sorted({audit["id"], *_sources(cells.values())})
            for part, members in parts.items():
                w.count(f"{base_dr}.positives.{part}", len(members), everyone)
            table = []
            for system in system_order(cells):
                cell = cells[system]
                base = f"{base_dr}.{seg_system(system)}"
                for part, members in parts.items():
                    w.rate(f"{base}.{part}.solved.{ht}", cell, members, solved_at(h), also=[audit["id"]])
                table.append([system, w.cell(f"{key(group, escape(ds), seg_regime(regime), seg_system(system))}"
                                             f".solved.{ht}", ci=True),
                              w.cell(f"{base}.seen.solved.{ht}", ci=True), w.cell(f"{base}.novel.solved.{ht}", ci=True)])
            if group == "E1" and "jigsaw" in cells and parts["novel"]:
                for ref in ("filterall", "jigsaw_index"):
                    if ref in cells:
                        w.paired_keys(f"{base_dr}.jigsaw_vs_{ref}.novel",
                                      paired(cells["jigsaw"], cells[ref], parts["novel"], solved_at(h)), None,
                                      sorted({audit["id"], *_sources([cells["jigsaw"], cells[ref]])}),
                                      note="on the novel positives alone; descriptive, in no Holm family")
            w.nums.table(base_dr, ["system", f"solved@{h:g}s, all positives", "seen", "novel"], table,
                         notes=f"{len(parts['seen'])} seen and {len(parts['novel'])} novel positives: seen by some "
                               "main encoder of the regime in training (up to isomorphism), novel to all of them")


def reduce_all(ctx: Reduction) -> numbers_json.Numbers:
    if ctx.split is None:
        raise ReduceError("no run, certification or measured cell names a split")
    nums = numbers_json.Numbers(split=ctx.split, frozen=ctx.frozen, protocol_hash=ctx.H)
    for run in ctx.runs:
        nums.source(run.run_id, path=run.path, manifest_sha256=run.manifest_sha256, experiment=run.entry,
                    dataset=run.dataset)
    for cell in ctx.measured:
        if cell["run"] is None:
            nums.source(cell["id"], path=cell["measure_path"], manifest_sha256=None, experiment=cell["experiment"],
                        dataset=cell["dataset"])
    for (ds, regime), cert in ctx.certs.items():
        nums.source(f"cert.{escape(ds)}.{seg_regime(regime)}", path=cert["path"],
                    manifest_sha256=numbers_json.file_sha256(cert["path"]), experiment="E14", dataset=ds)
    for item in ctx.offline:
        nums.source(item["id"], path=item["path"], manifest_sha256=item["sha256"], experiment="E17",
                    dataset=item["dataset"])
    for item in ctx.audit_runs:
        nums.source(item["id"], path=item["path"], manifest_sha256=item["sha256"], experiment="E20",
                    dataset=item["dataset"])
    w = Writer(nums)
    for step in (reduce_pe1, reduce_e1, reduce_pe2, reduce_e3, reduce_e3x, reduce_e7, reduce_e8, reduce_e9, reduce_e16,
                 reduce_e14, reduce_e17, reduce_e18, reduce_measured, reduce_e20):
        step(ctx, w)
    ctx.nums = nums
    return nums


def _render(entry: Mapping[str, Any], ci: bool) -> str:
    value = entry["value"]
    if entry["unit"] == "text":
        return str(value)
    digits = entry.get("digits")
    text = str(int(value)) if entry["unit"] == "count" else (f"{value:.{digits}f}" if digits is not None
                                                             else f"{value:.6g}")
    if ci and entry.get("ci"):
        lo, hi = entry["ci"]
        places = 3 if digits is None else digits
        text += f" [{lo:.{places}f}, {hi:.{places}f}]"
    return text


def table_csv(nums: numbers_json.Numbers, table_id: str) -> str:
    table = nums.tables[table_id]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(table["columns"])
    for row in table["rows"]:
        writer.writerow([c if isinstance(c, str) else _render(nums.numbers[c["key"]], bool(c.get("ci"))) for c in row])
    return buf.getvalue()


def _json(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True, default=str) + "\n"


def check_out(out: Path, ctx: Reduction) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ReduceError(f"{out} exists and is not an empty directory")
    real = out.resolve()
    for path in [run.path for run in ctx.runs] + list(ctx.plan.measured.values()) + [
            p for dirs in ctx.plan.audits.values() for p in dirs]:
        if real == path or path in real.parents:
            raise ReduceError(f"{out} lies inside the input directory {path}")


def write_outputs(ctx: Reduction, out: Path, argv: Sequence[str] | None) -> None:
    nums = ctx.nums
    if nums is None:
        raise ReduceError("nothing reduced")
    inputs: dict[str, Any] = {"plan": ctx.plan.path}
    for run in ctx.runs:
        inputs[f"manifest/{run.run_id}"] = run.path / manifest.MANIFEST_NAME
    for (ds, regime), cert in sorted(ctx.certs.items()):
        inputs[f"certification/{ds}/{regime}"] = cert["path"]
    for cell in ctx.measured:
        inputs[f"measure/{cell['id']}"] = cell["measure_path"]
    for item in ctx.offline:
        inputs[f"offline/{item['id']}"] = item["path"] / manifest.MANIFEST_NAME
    for item in ctx.audit_runs:
        inputs[f"pattern_audit/{item['id']}"] = item["path"] / manifest.MANIFEST_NAME
        inputs[f"pattern_audit/{item['id']}/{AUDIT_FILE}"] = item["file"]
    config = {"plan": ctx.plan.raw, "plan_path": str(ctx.plan.path), "plan_sha256": ctx.plan.sha256,
              "flags": {k: getattr(ctx.flags, k) for k in ("allow_dev", "allow_missing", "allow_uncertified",
                                                             "no_verify", "allow_pe1_failure")},
              "draws": dict(DRAWS)}
    with manifest.RunManifest(out, experiment="reduce", config=config, split=ctx.split, inputs=inputs,
                              argv=argv) as run:
        nums.write(out)
        (out / "tables").mkdir(exist_ok=True)
        for table_id in sorted(nums.tables):
            (out / "tables" / f"{table_id}.csv").write_text(table_csv(nums, table_id), encoding="utf-8")
        (out / "pe1.json").write_text(_json(ctx.pe1), encoding="utf-8")
        (out / "exclusions.json").write_text(_json({"exclusions": ctx.exclusions, "provisional": ctx.provisional,
                                                    "notes": ctx.notes}), encoding="utf-8")
        (out / "failure_localization.json").write_text(_json(ctx.localization), encoding="utf-8")
        run.note(pe1_holds=ctx.pe1["holds"], numbers=len(nums.numbers), tables=len(nums.tables),
                 figures=len(nums.figures), exclusions=len(ctx.exclusions), provisional=len(ctx.provisional),
                 plan_sha256=ctx.plan.sha256)


def execute(plan_path: str | Path, out: str | Path, flags: Flags | None = None,
            argv: Sequence[str] | None = None) -> int:
    """Reduce a plan into OUT; 0, or 3 when PE1 fails (without --allow-pe1-failure). Raises ReduceError."""
    flags = flags or Flags()
    plan = load_plan(plan_path)
    ctx = Reduction(plan, flags)
    check_out(Path(out), ctx)
    ctx.load()
    check_out(Path(out), ctx)
    reduce_all(ctx)
    write_outputs(ctx, Path(out), argv)
    if not ctx.pe1["holds"] and not flags.allow_pe1_failure:
        print(f"PE1 fails: {len(ctx.pe1['violations'])} soundness violations (see {Path(out) / 'pe1.json'})",
              file=sys.stderr)
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, help="PLAN.json naming the run dirs, certifications, roots, offline")
    p.add_argument("--out", required=True, help="a new or empty directory")
    p.add_argument("--allow-dev", action="store_true", help="reduce a dev or validation split")
    p.add_argument("--allow-missing", action="store_true", help="reduce over the queries every arm ran")
    p.add_argument("--allow-uncertified", action="store_true", help="negatives without certification files")
    p.add_argument("--no-verify", action="store_true", help="skip re-hashing run outputs")
    p.add_argument("--allow-pe1-failure", action="store_true", help="exit 0 even when PE1 fails")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    DRAWS.clear()
    DRAWS.update(PROTOCOL_DRAWS)  # the CLI always resamples with the protocol's counts
    flags = Flags(args.allow_dev, args.allow_missing, args.allow_uncertified, args.no_verify, args.allow_pe1_failure)
    try:
        code = execute(args.plan, args.out, flags, argv=sys.argv if argv is None else ["reducer", *argv])
    except (ReduceError, numbers_json.NumbersError) as exc:
        print(f"reducer refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "exit": code}, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
