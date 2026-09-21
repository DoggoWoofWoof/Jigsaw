#!/usr/bin/env python3
"""Regenerate every figure in paper/jigsaw_vldb2027.tex from committed data.

This is the clean-clone figure path: it needs only what is checked into the repo
(committed JSON probes + verified in-script constants), never the gitignored runs/
directory, a GPU, or the Glasgow binary. On a machine that *does* have runs/, the
scaling figure additionally reads the raw per-query CSVs; when runs/ is absent it
falls back to the canonical Table 3 constants, so the output is identical either way.

Figures produced (the five \\includegraphics in the submission):
  fig_jigsaw_pipeline.png       generate_submission_figures.pipeline_figure()
  fig_encoder_architecture.png  generate_submission_figures.architecture_figure()
  fig_scaling_feasibility.png   generate_scaling_figure.main()
  fig_memory_latency.png        generate_ablation_pareto_figures.memory_latency()
  fig_oversmoothing.png         generate_oversmoothing_figure.main()

Run from the repo root:  python scripts/reproduce_figures.py
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: set before any generator imports pyplot

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# (paper figure filename, module, callable name). Order matches the paper.
FIGURES = [
    ("fig_jigsaw_pipeline.png", "generate_submission_figures", "pipeline_figure"),
    ("fig_encoder_architecture.png", "generate_submission_figures", "architecture_figure"),
    ("fig_scaling_feasibility.png", "generate_scaling_figure", "main"),
    ("fig_memory_latency.png", "generate_ablation_pareto_figures", "memory_latency"),
    ("fig_oversmoothing.png", "generate_oversmoothing_figure", "main"),
]


def main() -> int:
    paper_dir = SCRIPTS.parent / "paper"
    ok = True
    print("=" * 72)
    print("Regenerating paper figures from committed data (no runs/ / GPU needed)")
    print("=" * 72)
    for fname, modname, funcname in FIGURES:
        try:
            module = importlib.import_module(modname)
            getattr(module, funcname)()
            out = paper_dir / fname
            status = "ok" if out.exists() else "MISSING OUTPUT"
            ok &= out.exists()
            print(f"  [{status:>13}] {fname:<30} <- {modname}.{funcname}()")
        except Exception:  # noqa: BLE001 - report and continue to the next figure
            ok = False
            print(f"  [       FAILED] {fname:<30} <- {modname}.{funcname}()")
            traceback.print_exc()
    print("=" * 72)
    print("ALL FIGURES REGENERATED" if ok else "SOME FIGURES FAILED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
