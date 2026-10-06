"""Pre-registered protocol for the canonical Jigsaw rerun.

Everything that determines a paper number is fixed here: datasets and attribute
schemas, label regimes, query design, seeds, time limits, the budget ladder,
methods, training recipes, the checkpoint rule and the statistics. Every run
manifest records protocol_hash(); the sealed test run refuses to start unless the
hash equals FROZEN_PROTOCOL_HASH (see seal.py).

Parameters listed in DEV_TUNABLE may change during development (dev and
validation seeds only). Their final values are written into the dictionaries
below at the freeze, which changes the hash; nothing changes after the freeze.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

PROTOCOL_VERSION = "canonical-v1"

# Filled by `python -m scripts.canonical.seal commit` before any development run.
SEAL_COMMITMENT: str | None = "2d16ee4edac05f9d2a3827a313e289eacd4beff824618e352f073287c3b16978"
# Filled at the freeze (Phase 3). Excluded from the hash it certifies.
FROZEN_PROTOCOL_HASH: str | None = "71227175e5d54ed1311d17da2dc55b2eaddaa80f0a27d2c4207c0ab2caa2c746"


# ----------------------------------------------------------------------------
# Datasets. `ladder` lists the retrieval budgets K before the final step; the
# final step is always K = partitions (every block, i.e. the FilterAll step).
# ----------------------------------------------------------------------------
DATASETS: dict[str, dict[str, Any]] = {
    "corafull": {
        "loader": "pyg:CoraFull",
        "schema": "corafull_bow_v1",
        "partitions": 20,
        "fine_per_block": 5,
        "ladder": [1, 2, 5, 10],
        "encoder": "graphsage",
        "semantic_labels": "class",
        "tier": "A",
    },
    "arxiv": {
        "loader": "ogb:ogbn-arxiv",
        "schema": "ogb_intrinsic_v1",
        "partitions": 200,
        "fine_per_block": 5,
        "ladder": [5, 10, 20, 50, 100],
        "encoder": "graphsage",
        "semantic_labels": "class",
        "tier": "A",
    },
    "mag": {
        "loader": "ogb:ogbn-mag",
        "schema": "mag_intrinsic_v1",
        "partitions": 2000,
        "fine_per_block": 5,
        "ladder": [20, 50, 100, 200, 500, 1000],
        "encoder": "rgcn",
        "semantic_labels": "venue_for_papers_type_otherwise",
        "tier": "A",
    },
    "products": {
        "loader": "ogb:ogbn-products",
        "schema": "products_intrinsic_v1",
        "partitions": 2500,
        "fine_per_block": 5,
        "ladder": [20, 50, 100, 200, 500, 1000],
        "encoder": "graphsage",
        "semantic_labels": "class",
        "tier": "B",
    },
    "papers100m": {
        "loader": "ogb:ogbn-papers100M",
        "schema": "ogb_intrinsic_v1",
        "partitions": 100000,
        "fine_per_block": 5,
        "ladder": [20, 50, 100, 200, 500, 1000, 2000, 5000],
        "encoder": "graphsage",
        # Most papers100M nodes carry no class label, so only R0 runs there.
        "semantic_labels": None,
        "tier": "B",
    },
}

# Graph normalization applied by prepare_data.py before anything else.
GRAPH_NORMALIZATION = {
    "directed_edges": "symmetrize",
    "self_loops": "drop",
    "multi_edges": "coalesce",
    "edge_labels": "none",  # matching is on node labels only; MAG relation = f(endpoint types)
    "node_order": "primary_partition_block_order",  # canonical ids are block-contiguous
    "index_dtype": "int32",
    "indptr_dtype": "int64",
}

PARTITIONING = {
    "tool": "metis",  # gpmetis 5.1.0, or pymetis with identical options
    "objective": "edge_cut",
    "ufactor": 30,
    "seed": 1,
    "fine_level": "metis_within_block",
    # METIS k-way does not promise non-empty parts (ogbn-mag, P = 2000, seed 1 leaves one empty)
    "empty_parts": "with e empty parts, the e largest parts (lowest id on ties) are each split by a 2-way "
                   "METIS k-way call with the same seed and ufactor; the i-th largest gives its side 1 to the "
                   "i-th empty id; a one-sided split falls back to halving the part in node-id order",
    "alternatives_for_sensitivity": ["kahip_fast", "bfs_random"],
    "partition_scale_for_sensitivity": [0.5, 1.0, 2.0],
}

# ----------------------------------------------------------------------------
# Attribute schemas. Only intrinsic attributes: nothing derived from the graph.
# Structural features for the encoder are computed from the graph it is given.
# ----------------------------------------------------------------------------
ATTRIBUTE_SCHEMAS: dict[str, dict[str, Any]] = {
    "corafull_bow_v1": {
        "columns": "bag_of_words",
        "values": "word presence: x > 0 -> 1 (cora.npz holds L2-normalised weights; PyG CitationFull binarises)",
        "dtype": "uint8",
        "node_types": ["paper"],
    },
    "ogb_intrinsic_v1": {
        "columns": "word2vec_128",
        "dtype": "float32",
        "node_types": ["paper"],
    },
    "mag_intrinsic_v1": {
        "columns": "paper_word2vec_128_zero_for_non_papers",
        "dtype": "float32",
        "node_types": ["paper", "author", "institution", "field_of_study"],
        "dropped_from_legacy": "log1p full-graph relation degrees (8 columns)",
    },
    "products_intrinsic_v1": {
        "columns": "bag_of_words_pca_100",
        "dtype": "float32",
        "node_types": ["product"],
    },
}

# ----------------------------------------------------------------------------
# Label regimes. sigma(v) is a dictionary code: equal codes iff equal keys.
# ----------------------------------------------------------------------------
LABEL_REGIMES: dict[str, dict[str, Any]] = {
    "R0": {
        "kind": "exact",
        "key": "(type, intrinsic attribute bytes)",
        "payload": ["structure", "labels", "attributes"],
        "encoder_input": "attributes",
    },
    "R1": {
        "kind": "semantic",
        "key": "(type, class)",
        "payload": ["structure", "labels"],
        "encoder_input": "type_onehot+label_centroid",
    },
    "R2-512": {
        "kind": "hashed",
        "buckets": 512,
        "key": "(type, blake2b(R0 key) mod 512)",
        "payload": ["structure", "labels"],
        "encoder_input": "type_onehot+label_centroid",
    },
    "R2-128": {
        "kind": "hashed",
        "buckets": 128,
        "key": "(type, blake2b(R0 key) mod 128)",
        "payload": ["structure", "labels"],
        "encoder_input": "type_onehot+label_centroid",
    },
    "R2-32": {
        "kind": "hashed",
        "buckets": 32,
        "key": "(type, blake2b(R0 key) mod 32)",
        "payload": ["structure", "labels"],
        "encoder_input": "type_onehot+label_centroid",
    },
}

# Encoders are trained for these regimes; R2-* reuse the R2-512 encoder.
TRAINED_REGIMES = {"R0": "R0", "R1": "R1", "R2-512": "R2-512", "R2-128": "R2-512", "R2-32": "R2-512"}

# ----------------------------------------------------------------------------
# Queries.
# ----------------------------------------------------------------------------
POSITIVE_FAMILIES = ["single", "multi_fine", "multi_coarse", "k_hop", "random_walk", "degree_k_hop"]
NEGATIVE_FAMILIES = ["neg_absent_label", "neg_swapped_label", "neg_structure"]
QUERY_SIZES = [20, 50, 100]

QUERY_DESIGN: dict[str, Any] = {
    "shape": "induced subgraph on a connected planted node set",
    "min_nodes": "max(5, size - 10)",
    "min_edges": 2,
    "matching": "non-induced, injective, sigma-equality, undirected",
    "per_cell": {
        # (family kind) -> queries per (family, size) cell
        "R0": {"positive": 100, "neg_absent_label": 50, "neg_swapped_label": 100, "neg_structure": 100},
        "R1": {"positive": 50, "neg_absent_label": 25, "neg_swapped_label": 50, "neg_structure": 50},
        "R2": {"positive": 50, "neg_absent_label": 25, "neg_swapped_label": 50, "neg_structure": 50},
    },
    "negatives_must_be_certified": True,
    "certification": {
        "stage1": "FilterAll, sound pruning, no component cap, 3600 s; its single step is closed, so NONE is a proof",
        "stage2": "classical_gql (SubgraphMatching GQL preset) on the whole data graph, 3600 s, on what stage 1 "
                  "leaves open; NONE counts only when the search finished",
        "stage1_T_seconds": 3600.0,
        "stage2_T_seconds": 3600.0,
        "stage2_method": "classical_gql",
        "witness": "every witness is checked against the raw store data (verify.check_embedding)",
        "undecided": "excluded from the negative set and counted",
        "found_match": "relabelled as a positive with the found witness",
    },
    "standard_sm": {
        "family": "sm_standard",
        "generator": "random walk over distinct nodes; dense = induced subgraph with avg degree >= 3, "
                     "sparse = the walk's traversed edges with avg degree < 3; size 4 is not split by density",
        "sizes": [4, 8, 16, 32],
        "densities": ["dense", "sparse"],
        "density_split_min_size": 8,
        "per_cell": 50,
        "dense_min_avg_degree": 3.0,
    },
    "negative_base": "a planted positive of family POSITIVE_FAMILIES[index % 6] from its own seed stream; "
                     "a degree_k_hop base takes tertile (index // 6) % 3",
    "query_node_order": "random permutation of the planted nodes",
    "dedup": "node-set hash unique within and across splits (later splits exclude earlier ones)",
    # Per validation or dev master; test and reserves use per_cell above.
    "split_counts": {
        "validation": {"positive": 34, "neg_absent_label": 12, "neg_swapped_label": 12, "neg_structure": 12,
                       "sm_standard": 0},
        "dev": {"positive": 34, "neg_absent_label": 12, "neg_swapped_label": 12, "neg_structure": 12,
                "sm_standard": 10},
    },
    "max_attempts_per_query": 200,
}

# ----------------------------------------------------------------------------
# Seeds. Query streams derive from a split master via derive_seed().
# ----------------------------------------------------------------------------
SEEDS: dict[str, Any] = {
    "dev": ["20260607", "20260608", "20260609"],  # development rounds (burned by design)
    "validation": ["31415", "27182", "16180"],  # checkpoint selection and router fitting
    "test": "sealed",  # three CSPRNG masters, committed by SEAL_COMMITMENT: test, reserve_a, reserve_b
    "training": [7301, 7302, 7303],
    "partition": 1,
    # Each validation master serves one decision, so no decision reuses another's queries.
    "validation_roles": {
        "31415": "checkpoint_selection",
        "27182": "router_tau_and_rank_fusion",
        "16180": "dev_tunable_selection",
    },
}

# ----------------------------------------------------------------------------
# Per-query time and the anytime ladder.
# ----------------------------------------------------------------------------
TIME: dict[str, Any] = {
    "primary_T_seconds": 30.0,
    "run_T_max_seconds": {"E1": 120.0, "E7": 120.0, "E12": 120.0, "E3_extension": 600.0, "E8": [30.0, 600.0]},
    "reported_T_seconds": [1.0, 5.0, 30.0, 120.0],
    # Non-final ladder steps get an absolute cap independent of T, so one run at
    # T_max yields the exact solved@t curve for every t <= T_max.
    "step_cap_seconds": 2.0,
    "final_step": "remainder of T",
    "stages_counted": ["query_labels", "retrieval", "candidate", "prune", "components", "solve"],
    "stages_not_counted": ["independent_witness_check"],
    "solver_kill": "process-group SIGKILL at the cap (fractional seconds)",
}

LADDER: dict[str, Any] = {
    "final_step": "all_blocks",
    # Sound checks before any step; each proves NONE on its own.
    "global_certificates": ["a query label held fewer times by the data graph than the query needs",
                            "a query edge whose endpoint types never meet on a data edge"],
    "closed_step": "a step whose candidate holds every node carrying a query label gets the remainder of T; "
                   "its NONE proves that no embedding exists",
    "skip_identical_candidates": True,
    "on_step_timeout": "continue_to_next_step",
    "on_step_none": "continue_to_next_step",
    "stop_on": "first_verified_embedding",
    # E16 runs the ladder on sensitivity partitions (PARTITIONING); there a step of K blocks on the
    # primary partition of P blocks takes ceil(K * B / P) of the B blocks, the same share of the graph.
    "sensitivity_partition_budgets": "ceil(K * B / P), at least 1",
    # Planner settings of the learned and index arms, tuned on validation master 16180 and confirmed on
    # dev master 20260609 (DEV_TUNABLE). They are {dataset: {regime or regime family: value}}; a regime
    # without an entry runs the plain ladder, and an exact regime wins over its family. Only the regimes
    # that were tuned carry entries, so the sensitivity regimes R2-128 and R2-32 run the plain ladder.
    # escalate_seconds: seconds after which the next step is the whole-graph step (a ladder step is a rung).
    # Selection rule, fixed before the final probes: on validation take the lowest p95 among the values
    # whose mean is at most 1.05 times the best mean; the confirmation set chose the same values.
    # Arxiv R0 and R2-512 and CoraFull have no escalate entry because skip_below_nodes already sends their
    # queries to the whole-graph step. MAG has no skip entry because every MAG query carries about 1.2M
    # label nodes, so no threshold would fire.
    "escalate_seconds": {
        "mag": {"R0": 1.0, "R1": 0.25, "R2-512": 0.25},
        "arxiv": {"R1": 0.03},
    },
    # skip_below_nodes: a query whose label-carrying nodes number at most this skips retrieval and the ladder.
    # The thresholds are per regime because one absolute threshold does not separate Arxiv R1 (the ladder
    # wins) from Arxiv R2-512 (the whole graph wins); Arxiv R1 has no entry because a threshold of 30000
    # or more lost found queries there and a smaller one gained nothing.
    "skip_below_nodes": {
        "arxiv": {"R0": 1000, "R2-512": 100000},
        "corafull": {"R0": 1000, "R1": 10000, "R2-512": 10000},
    },
}

# ----------------------------------------------------------------------------
# Candidate construction and pruning (all sound).
# ----------------------------------------------------------------------------
CANDIDATE: dict[str, Any] = {
    "nodes": "blocks(S) plus their one-hop neighbours",
    "edges": "induced",
    "filters": ["sigma", "ldf", "nlf"],
    "fixpoint_max_iterations": 20,
    "components": "connected components of the pruned candidate covering the query label multiset",
    "component_order": "smallest_first",
    "component_cap": None,  # no cap in the canonical run; E9 ablates 50
    "component_mode": "separate",
    "label_prune_fallback": False,
}

SELECTION: dict[str, Any] = {
    "policy": "hybrid",
    "hybrid": {
        "seed_count": 20,
        "model_weight": 0.5,
        "boundary_weight_mix": [0.8, 0.2],
        "teleport_every": 10,
    },
    "ablation_policy": "topk",
    "applies_to": "every ranker",
}

RANKERS: dict[str, Any] = {
    "learned": "cosine between the query embedding and block embeddings",
    "index": "IDF-weighted distinct-label coverage of block nodes; ties by block id",
    "index_count": "unweighted distinct-label coverage (the legacy FeatureIndex); ablation of the IDF weight",
    "mean_feature": "cosine between mean encoder inputs of query and block",
    "topo_feature": "minus the Euclidean distance between z-scored statistics of query and block: mean degree, "
                    "degree CV, fraction of degree <= 1, transitivity and node-type fractions; "
                    "z-scores use the mean and std over blocks",
    "fused_learned_index": "RRF over learned and index",
    "fused_learned_mean": "RRF over learned and mean_feature (the former Mean-RRF baseline)",
    "router": "index with ties by learned score if selectivity(Q) <= tau, else fused_learned_index",
    # {dataset: tau}, fit on validation master 27182 with the dev-seed (7301) encoders, because the canonical
    # encoders train under the frozen protocol; `fit_router merge --allow-smoke` printed this line.
    "router_tau": {"arxiv": 3.1179185944934846, "corafull": 0.14718032990042854, "mag": 2.351935451452383},
    "router_tau_fit": "minimise the summed hybrid cost-to-coverage (blocks until FullCov) over validation "
                      "positives of every regime; candidates are the midpoints of sorted distinct selectivities "
                      "plus one value below and one above all of them; ties to the median optimal candidate",
    "random": "uniform permutation seeded by the query uid",
    "rrf_k": 60,
    "idf": "log(1 + B / bdf(label)); B blocks, bdf(label) blocks holding a node with the label",
    "selectivity": "mean over query nodes of log10(max(1, df(sigma(u)))), df the node frequency",
    "ties": "every ranking breaks score ties by the lower block id",
}

VERIFIERS = {
    # gql (SubgraphMatching) is the default because the Glasgow target is n^2 bits per graph, which runs out of
    # memory on the coarse MAG and Arxiv candidates; Glasgow stays available as an alternative arm.
    "default": "gql",
    "alternatives": ["glasgow", "cfl", "dpiso"],  # run on the candidate
    "witness_check": "independent: raw attributes, types, injectivity, every query edge",
}

METHODS: dict[str, dict[str, Any]] = {
    "jigsaw": {"ranker": "router", "policy": "hybrid", "ladder": True},
    "jigsaw_learned": {"ranker": "learned", "policy": "hybrid", "ladder": True},
    "jigsaw_index": {"ranker": "index", "policy": "hybrid", "ladder": True},
    "jigsaw_fused": {"ranker": "fused_learned_index", "policy": "hybrid", "ladder": True},
    "mean_rrf": {"ranker": "fused_learned_mean", "policy": "hybrid", "ladder": True},
    "mean_feature": {"ranker": "mean_feature", "policy": "hybrid", "ladder": True},
    "topo_feature": {"ranker": "topo_feature", "policy": "hybrid", "ladder": True},
    "random": {"ranker": "random", "policy": "hybrid", "ladder": True},
    "filterall": {"ranker": None, "policy": "all", "ladder": False},
    "direct": {"ranker": None, "policy": "none", "ladder": False, "pruning": "solver only"},
    "classical_gql": {"external": "SubgraphMatching", "algorithm": "GQL"},
    "classical_cfl": {"external": "SubgraphMatching", "algorithm": "CFL"},
    "classical_dpiso": {"external": "SubgraphMatching", "algorithm": "DPiso"},
    "gnnpe": {"external": "GNN-PE"},
}

# ----------------------------------------------------------------------------
# Serving modes and measurement.
# ----------------------------------------------------------------------------
SERVING: dict[str, Any] = {
    "resident": "all store arrays loaded into RAM",
    "streaming": "store arrays memory-mapped; numpy query encoder; no torch import",
    "cold": "store files evicted with posix_fadvise(DONTNEED) before each query; drop_caches when root",
    "warm": "one untimed warm-up pass over the query set, then the timed pass",
    "memory_caps_gb": [2, 4, 8],
    "memory_cap_mechanism": "cgroup v2 memory.max with swap disabled",
    "repeats": 3,  # measured processes per E2 cell (system, mode, cache, cap), uncapped cells included
    "rss": "process tree (driver plus solver subprocesses), sampled every 20 ms, plus ru_maxrss",
    "io": "rusage block input of the process tree plus /proc io counters",
    "latency_percentiles": [50, 95, 99],
    "threads": 1,  # every method; the solvers are sequential
    "thread_mechanism": "OMP/OPENBLAS/MKL_NUM_THREADS=1 in the launcher and threadpoolctl.threadpool_limits(1) "
                        "in the driver; the manifest records threadpoolctl.threadpool_info()",
    "encoder": "encoder_numpy for query embeddings (timed, per query) and block embeddings (offline, "
               "serving.py); torch only trains",
    # E10: a store larger than the memory it runs in. One host with at least 256 GB runs every cell; the
    # uncapped cells are the large machine and the 32 GB cap (cgroup v2, swap off) is the small one.
    "scale_out": {"datasets": ["papers100m"], "memory_caps_gb": [32], "repeats": 1,
                  "host_memory_gb_min": 256},
}

# ----------------------------------------------------------------------------
# Training.
# ----------------------------------------------------------------------------
TRAINING: dict[str, Any] = {
    "trainer": "scripts/canonical/train_encoder.py",
    "architectures": {
        "graphsage": {"layers": 6, "hidden": 256, "output": 128, "residual": True, "layer_norm": True,
                      "readout": "mean, max and sum pools of every layer, projected to the output"},
        "rgcn": {"layers": 6, "hidden": 256, "output": 128, "residual": True, "layer_norm": True,
                 "readout": "mean, max and sum pools of every layer, projected to the output",
                 "relations": "derived from endpoint node types"},
    },
    "structural_inputs": "per-relation log1p degree computed on the graph being encoded (query or block)",
    "loss": {
        "fine_nce_weight_alpha": 0.2,
        "node_nce_weight_beta": 0.0,
        "temperature": 0.1,  # block, fine and node InfoNCE
        "coverage_temperature": 0.05,  # coverage cross-entropy, as in the recipe of the deployed encoders
        "coverage_weight_gamma": 1.0,
        "coverage_aggregation": "cvar",
        "cvar_rho": 0.5,
        "hardest_negative_margin_weight": 0.25,
        "topk_barrier_weight": 0.35,
        "topk_barrier_K": {"corafull": 10, "arxiv": 50, "mag": 50, "products": 50, "papers100m": 50},
        "live_positives": {"corafull": 20, "arxiv": 64, "mag": 64, "products": 64, "papers100m": 64},
        "coverage_targets": "overlap (greedy one-hop cover)",
        "negative_encoder_momentum": 0.99,
    },
    "sampler": {
        "family_probabilities": {"k_hop": 0.25, "single": 0.10, "multi_coarse": 0.20,
                                 "random_walk": 0.30, "degree_k_hop": 0.15},
        "sizes": [20, 50, 100],
        "size_jitter": 8,
    },
    "schedule": {"epochs": 90, "steps_per_epoch": 100, "batch_size": 8, "lr": 5e-5, "lr_final": 5e-6,
                 "lr_schedule": "cosine", "warmup_steps": 100,
                 # AdamW without decay is the Adam of the deployed encoders' recipe
                 "optimizer": "adamw", "weight_decay": 0.0, "grad_clip_norm": 1.0},
    # A training draw whose node set equals a query of any validation master is redrawn, and the
    # sealed splits are generated excluding every training node set (train_nodesets.txt).
    "held_out_nodesets": "validation masters, all three",
    "validation_every_epochs": 5,
    "checkpoint_rule": "max mean over the ladder of hybrid FullCov (overlap targets) on validation queries; "
                       "ties to the later epoch; evaluated every 5 epochs and at the end",
    "validated_system": "live encoder on query and block side, hybrid policy, deployed ladder",
}

# E5 objective factorial. Every arm is trained with TRAINING seeds and selected by the rule above.
FACTORIAL: dict[str, Any] = {
    "arxiv": {
        "factors": {"coverage_aggregation": ["mean", "cvar"], "topk_barrier_weight": [0.0, 0.35], "live_positives": [0, 64]},
        "extra_arms": {"infonce_only": {"coverage_weight_gamma": 0.0}},
    },
    "mag": {
        "arms": {
            "full": {},
            "no_cvar": {"coverage_aggregation": "mean"},
            "no_barrier": {"topk_barrier_weight": 0.0},
            "no_live": {"live_positives": 0},
            "no_momentum": {"negative_encoder_momentum": 0.0},
            "infonce_only": {"coverage_weight_gamma": 0.0},
        },
    },
}

# E6 encoder ablation.
ARCHITECTURE_ABLATION: dict[str, Any] = {
    "arxiv": [
        {"arch": "gin", "layers": 6, "residual": True},
        {"arch": "graphsage", "layers": 6, "residual": True},
        {"arch": "gat", "layers": 6, "residual": True},
        {"arch": "graphsage", "layers": 2, "residual": True},
        {"arch": "graphsage", "layers": 4, "residual": True},
        {"arch": "graphsage", "layers": 8, "residual": True},
        {"arch": "graphsage", "layers": 6, "residual": False},
        {"arch": "graphsage", "layers": 8, "residual": False},
    ],
    "mag": [
        {"arch": "rgcn", "layers": 6, "residual": True},
        {"arch": "graphsage", "layers": 6, "residual": True},
        {"arch": "rgcn", "layers": 6, "residual": False},
    ],
    "layerwise_metrics": ["mean_average_distance", "mean_pairwise_cosine", "effective_rank", "block_separation"],
    # the nodes E6 reads per model (oversmoothing.py): up to max_blocks blocks and max_nodes nodes
    "layerwise_sample": {"seed": 0, "max_blocks": 64, "max_nodes": 4096},
}

# ----------------------------------------------------------------------------
# Statistics (see stats.py).
# ----------------------------------------------------------------------------
STATISTICS: dict[str, Any] = {
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

PRIMARY_ENDPOINTS = {
    "PE1": "soundness: 0 false positives and 100% of witnesses pass the independent check, every run",
    "PE2": "Jigsaw vs FilterAll solved@30s on positives, paired non-inferiority, margin 1.0 pp",
    "PE3": "Jigsaw vs FilterAll streaming bytes read and cold/warm latency, geometric-mean ratios",
    "PE4": "peak process-tree RSS of Jigsaw streaming vs each resident system; success under each cap",
    "PE5": "learned/fused/router vs index and mean_rrf: cost to coverage under the hybrid policy, by regime",
    "PE6": "full objective vs mean-aggregation control: cost to coverage and FullCov, hybrid and top-K",
}

# Parameters that may be tuned during development (dev/validation seeds only).
DEV_TUNABLE: dict[str, Any] = {
    "TIME.step_cap_seconds": [0.5, 1.0, 2.0, 5.0],
    "CANDIDATE.component_mode": ["separate", "merged"],
    "RANKERS.router_tau": "fit on validation queries",
    "SELECTION.hybrid.seed_count": [10, 20, 40],
    "VERIFIERS.default": ["glasgow", "gql"],
    "LADDER.escalate_seconds": "fit on validation queries, per dataset and regime family",
    "LADDER.skip_below_nodes": "fit on validation queries, per dataset and regime family",
}

# Enumeration recall of the ladder (E15, enumeration.py). Up to `limit` embeddings of each
# positive query are enumerated on the label-filtered graph, which holds every embedding; the
# recall of a ladder step is the share of them whose nodes all lie in its candidate. The
# candidate is induced and pruning is sound, so exactly those embeddings reach its kept targets.
ENUMERATION: dict[str, Any] = {
    "limit": 1000,
    "positives_per_dataset": 200,
    "T_seconds": 600.0,  # the whole-graph enumeration, and each step's when steps are solved
    "regimes": ["R0"],
    "arms": ["jigsaw@first_training_seed", "jigsaw_index"],
}


def protocol_dict() -> dict[str, Any]:
    """Every protocol setting as a JSON-serialisable dict (FROZEN_PROTOCOL_HASH excluded)."""
    spec = {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "SEAL_COMMITMENT": SEAL_COMMITMENT,
        "DATASETS": DATASETS,
        "GRAPH_NORMALIZATION": GRAPH_NORMALIZATION,
        "PARTITIONING": PARTITIONING,
        "ATTRIBUTE_SCHEMAS": ATTRIBUTE_SCHEMAS,
        "LABEL_REGIMES": LABEL_REGIMES,
        "TRAINED_REGIMES": TRAINED_REGIMES,
        "POSITIVE_FAMILIES": POSITIVE_FAMILIES,
        "NEGATIVE_FAMILIES": NEGATIVE_FAMILIES,
        "QUERY_SIZES": QUERY_SIZES,
        "QUERY_DESIGN": QUERY_DESIGN,
        "SEEDS": SEEDS,
        "TIME": TIME,
        "LADDER": LADDER,
        "CANDIDATE": CANDIDATE,
        "SELECTION": SELECTION,
        "RANKERS": RANKERS,
        "VERIFIERS": VERIFIERS,
        "METHODS": METHODS,
        "SERVING": SERVING,
        "TRAINING": TRAINING,
        "FACTORIAL": FACTORIAL,
        "ARCHITECTURE_ABLATION": ARCHITECTURE_ABLATION,
        "STATISTICS": STATISTICS,
        "PRIMARY_ENDPOINTS": PRIMARY_ENDPOINTS,
        "DEV_TUNABLE": DEV_TUNABLE,
        "ENUMERATION": ENUMERATION,
    }
    return copy.deepcopy(spec)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_hash() -> str:
    return hashlib.sha256(canonical_json(protocol_dict()).encode("ascii")).hexdigest()


def is_frozen() -> bool:
    return FROZEN_PROTOCOL_HASH is not None and FROZEN_PROTOCOL_HASH == protocol_hash()


def derive_seed(master: str | int, *parts: Any) -> int:
    """A 63-bit seed derived from a split master and a tuple of labels."""
    material = canonical_json([str(master), [str(part) for part in parts]]).encode("ascii")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def ladder_for(dataset: str) -> list[int]:
    """Budgets K for a dataset including the final all-blocks step."""
    spec = DATASETS[dataset]
    budgets = [int(k) for k in spec["ladder"] if int(k) < int(spec["partitions"])]
    return budgets + [int(spec["partitions"])]


def router_tau_for(dataset: str) -> float | None:
    """The frozen router threshold of a dataset, or None before the freeze."""
    taus = RANKERS.get("router_tau")
    if isinstance(taus, dict) and taus.get(dataset) is not None:
        return float(taus[dataset])
    return None


def ladder_setting(name: str, dataset: str, regime: str) -> float | None:
    """A frozen planner setting (LADDER[name]) of a dataset and regime, or None before the freeze.

    The table maps a dataset to {regime: value}, where the key is the exact regime ("R2-512") or its
    family ("R2"); the exact regime wins.
    """
    table = LADDER.get(name)
    row = table.get(dataset) if isinstance(table, dict) else None
    if not isinstance(row, dict):
        return None
    for key in (regime, regime_family(regime)):
        if row.get(key) is not None:
            return float(row[key])
    return None


def regime_family(regime: str) -> str:
    return "R2" if regime.startswith("R2") else regime


def regimes_for(dataset: str) -> list[str]:
    if DATASETS[dataset]["semantic_labels"] is None:
        return ["R0"]
    return list(LABEL_REGIMES)


if __name__ == "__main__":
    print(f"protocol {PROTOCOL_VERSION} sha256={protocol_hash()}")
    print(f"seal commitment: {SEAL_COMMITMENT or 'NOT SET'}")
    print(f"frozen: {is_frozen()}")
    for name in DATASETS:
        print(f"  {name}: ladder={ladder_for(name)} regimes={regimes_for(name)}")
