# Reproducing Jigsaw

This repository has two reproduction tiers. The first runs on a laptop from a clean
clone in about two minutes and needs no GPU, no cloud, and none of the (gitignored)
`runs/` directory. The second retrains the encoders and reruns the full Glasgow
benchmark, and needs real compute. We are explicit about which claims each tier covers.

---

## Tier 1 - Laptop reproduction (clean clone, ~2 min, no GPU)

Everything here reads only files committed to the repo (per-layer JSON probes, verified
in-script constants, and the canonical CSV bundle under `benchmarks/paper_results/`).

```bash
python -m pip install -r requirements-repro.txt

# 1. Regenerate every figure in the paper from committed data.
python scripts/reproduce_figures.py

# 2. Re-check the headline numbers.
#    On a clean clone (no runs/) this prints where the committed snapshot lives and exits 0.
#    On a machine that has the run bundle it re-derives every number from the raw per-query CSVs.
python scripts/analysis/reproduce_paper_numbers.py
```

`scripts/reproduce_figures.py` rebuilds the five figures the submission includes:

| Figure | Source | Committed input |
|---|---|---|
| `fig_jigsaw_pipeline.png` | `generate_submission_figures.pipeline_figure()` | hand-drawn schematic (no data) |
| `fig_encoder_architecture.png` | `generate_submission_figures.architecture_figure()` | hand-drawn schematic (no data) |
| `fig_scaling_feasibility.png` | `generate_scaling_figure.main()` | Table 3 constants (falls back when `runs/` absent) |
| `fig_memory_latency.png` | `generate_ablation_pareto_figures.memory_latency()` | verified in-script constants |
| `fig_oversmoothing.png` | `generate_oversmoothing_figure.main()` | `ablations/oversmoothing_{corafull,arxiv,mag}.json` |

### Headline numbers on a clean clone

The raw per-query solver traces (many GB) live under `runs/` and are gitignored, so
`reproduce_paper_numbers.py` cannot re-derive the numbers from scratch on a clean clone.
The validated headline is instead committed in two places:

- `benchmarks/paper_results/final_results/HEADLINE_NUMBERS.csv` - the curated canonical
  headline for all three datasets (per method, per family, cross-dataset selector,
  foreclosure), traced row by row in `benchmarks/paper_results/CANONICAL_SOURCES.md`.
- `benchmarks/paper_results/mag_walkaware_headline_summary.csv` - an auto-emitted
  snapshot written by `reproduce_paper_numbers.py --freeze` when the run bundle is
  present. It adds the MAG per-size slices (93.2 / 90.7 / 82.0) and the paired McNemar
  (b=92, c=35, p=4.36e-7) that the curated file omits.

The per-claim map from every headline number to its committed evidence file and its
one-line reproduction command is the table in [`README.md`](README.md)
("Paper claims -> evidence -> reproduction"). Additional fail-fast validators live in
`scripts/analysis/validate_*.py` (budget fairness, benchmark denominators, query-derived
pruning, raw-attribute acceptance audit, matched costs).

> Do not aggregate the legacy `benchmarks/paper_results/final_results/final_*_summary.csv`
> grids for headline claims. Their MAG neural rows are the earlier seed7202 checkpoint
> (aggregate 86.0%), not the deployed seed7203 model (88.6%). See
> `benchmarks/paper_results/final_results/README.md`.

---

## Tier 2 - Compute reproduction (GPU + cloud, NOT laptop-reproducible)

These steps regenerate the artifacts that Tier 1 consumes. They are gated on hardware
and large data and are intentionally out of scope for a clean-clone check.

- **Encoder training** (GraphSAGE for CoraFull/Arxiv, RGCN for MAG, FullCov objective):
  needs a GPU and the stack in `requirements_lightning_rgcn.txt` (PyTorch 2.2.1 + PyG +
  pymetis). Launchers under `scripts/launchers/`. Trained checkpoints are gitignored.
- **Full production benchmark** (six methods x eight query families x three sizes x two
  seeds, per dataset): needs the trained checkpoints, the OGBN datasets, the METIS
  partitions, and the compiled Glasgow Subgraph Solver binary. The raw per-query CSVs
  land under `runs/` and are summarized into `benchmarks/paper_results/final_results/`.
- **Depth / oversmoothing probes**: CPU on the deployed weights; procedure in
  `benchmarks/paper_results/ablations/oversmoothing_summary.md`. The committed JSON is
  the output, so Tier 1 redraws Figure 5 without rerunning this.

### Known compute-gated evaluation gaps (deferred)

The submission audit flagged three acceptance-relevant experiments that need compute we
do not currently have. They are deferred, not claimed:

1. The canonical SM benchmark suite (Yeast / Human / HPRD / ...).
2. An external exact matcher and a GNN-PE head-to-head in the main results table.
3. A non-citation dense graph, to test generality beyond citation networks.

---

## Environment used for the committed figures

Python 3.13.5, matplotlib 3.10.5, numpy 2.3.2, pandas 2.3.1 (see `requirements-repro.txt`).
