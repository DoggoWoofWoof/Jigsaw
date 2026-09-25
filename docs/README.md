# docs/ index

Design docs, runbooks, and method notes for the Jigsaw paper.

## Benchmark metrics
- [benchmark_metrics_reference.md](benchmark_metrics_reference.md) — definitions of the benchmark metrics/columns.

## Method & protocol references
- [jigsaw_loss_and_retrieval_method.md](jigsaw_loss_and_retrieval_method.md) — FullCov objective + retrieval method.
- [production_benchmark_protocol.md](production_benchmark_protocol.md) — production benchmark protocol (seeds, families, sizes).
- [query_generator_alignment.md](query_generator_alignment.md) — query-generator alignment notes.
- [arxiv_khop_control_vs_scheduler_seed42.md](arxiv_khop_control_vs_scheduler_seed42.md), [multiview_retrieval_ablation.md](multiview_retrieval_ablation.md) — specific ablation notes.

## Runbooks
- [targeted_experiment_runbook.md](targeted_experiment_runbook.md) — how to run targeted experiments cheaply (don't re-run the full grid); offline probe + selective/bridge overlap flags.
- [walk_aware_retrain_runbook.md](walk_aware_retrain_runbook.md) — staged walk-aware MAG retraining (the random_walk fix) + cheap eval.

## Repo organization
- [scripts_cleanup_plan.md](scripts_cleanup_plan.md) — scripts/ consolidation plan (now executed: see scripts/README.md).

Related: [scripts/README.md](../scripts/README.md) (scripts layout), [README.md](../README.md) (repo overview), and the diagnostic findings in `runs/diagnostics/` (e.g. `candidate_shrinkage_findings.md`, `random_walk_analysis.md`).
