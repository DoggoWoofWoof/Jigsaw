"""Generate paper/fig_oversmoothing.png from the committed per-layer depth probes in
benchmarks/paper_results/ablations/oversmoothing_{corafull,arxiv,mag}.json.

Three legible panels (one row) tell the whole Section 5.4 story:
  (a) partition-summary distance vs depth (residual ON): the quantity retrieval uses;
      preserved on CoraFull, rising on Arxiv/MAG -> no oversmoothing.
  (b) partition-summary effective rank vs depth (residual ON): reinforces (a).
  (c) layer-6 node cosine, residual ON vs OFF: collapse to the textbook regime appears
      only on the GraphSAGE encoders (CoraFull/Arxiv) when the residual is removed;
      the MAG RGCN is robust without it -> the residual is the anti-collapse mechanism.

Reads only committed JSON, so it reproduces on a clean clone (no GPU/run dirs needed).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "paper_results" / "ablations"
OUT = ROOT / "paper" / "fig_oversmoothing.png"

# (json stem, display label, encoder tag, colour)
DATASETS = [
    ("corafull", "CoraFull", "SAGE", "#1f77b4"),
    ("arxiv", "Arxiv", "SAGE", "#ff7f0e"),
    ("mag", "MAG", "RGCN", "#2ca02c"),
]
COLLAPSE = 0.90  # textbook near-collapse reference line for the L6 node cosine


def load(stem):
    return json.loads((DATA / f"oversmoothing_{stem}.json").read_text())


def main():
    data = {stem: load(stem) for stem, *_ in DATASETS}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # ---- Panel (a): partition-summary distance vs depth (residual ON) ----
    for stem, label, enc, color in DATASETS:
        on = data[stem]["residual_ON"]
        layers = [r["layer"] for r in on]
        dist = [r["part_summary_dist"] for r in on]
        ax[0].plot(layers, dist, "o-", color=color, lw=2, ms=6,
                   label=f"{label} ({enc})")
    ax[0].set_title("(a) Partition-summary distance rises or holds\nwith depth (no oversmoothing)",
                    fontsize=11)
    ax[0].set_xlabel("Message-passing layer")
    ax[0].set_ylabel("Mean pairwise partition distance")
    ax[0].legend(frameon=False, fontsize=9.5, loc="upper left")

    # ---- Panel (b): partition-summary effective rank vs depth (residual ON) ----
    for stem, label, enc, color in DATASETS:
        on = data[stem]["residual_ON"]
        layers = [r["layer"] for r in on]
        rank = [r["part_summary_effrank"] for r in on]
        ax[1].plot(layers, rank, "s-", color=color, lw=2, ms=6,
                   label=f"{label} ({enc})")
    ax[1].set_title("(b) Partition-summary effective rank\n(what FAISS discriminates)",
                    fontsize=11)
    ax[1].set_xlabel("Message-passing layer")
    ax[1].set_ylabel("Effective rank of partition summaries")
    ax[1].legend(frameon=False, fontsize=9.5, loc="lower right")

    # ---- Panel (c): L6 node cosine, residual ON vs OFF ----
    labels = [d[1] for d in DATASETS]
    encs = [d[2] for d in DATASETS]
    on_cos = [data[stem]["residual_ON"][-1]["node_meancos"] for stem, *_ in DATASETS]
    off_cos = [data[stem]["residual_OFF"][-1]["node_meancos"] for stem, *_ in DATASETS]
    x = np.arange(len(labels))
    w = 0.36
    ax[2].bar(x - w / 2, on_cos, w, label="residual ON", color="#1f77b4")
    ax[2].bar(x + w / 2, off_cos, w, label="residual OFF", color="#d62728")
    ax[2].axhline(COLLAPSE, ls=":", color="#555", lw=1.3)
    ax[2].text(2.35, COLLAPSE + 0.005, "textbook collapse", ha="right", va="bottom",
               fontsize=8.5, color="#555")
    for i, (o, f) in enumerate(zip(on_cos, off_cos)):
        ax[2].text(i - w / 2, o + 0.015, f"{o:.2f}", ha="center", va="bottom", fontsize=9)
        note = "  collapse" if f >= COLLAPSE else "  robust"
        ax[2].text(i + w / 2, f + 0.015, f"{f:.2f}", ha="center", va="bottom", fontsize=9)
        if f < COLLAPSE:
            ax[2].text(i + w / 2, f - 0.06, "robust", ha="center", va="top",
                       fontsize=8.5, color="#2ca02c", fontweight="bold")
    ax[2].set_ylim(0, 1.08)
    ax[2].set_title("(c) Residual is the anti-collapse mechanism\n(layer-6 node cosine; →1 is collapse)",
                    fontsize=11)
    ax[2].set_ylabel("Layer-6 mean node cosine")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{l}\n({e})" for l, e in zip(labels, encs)])
    ax[2].legend(frameon=False, fontsize=9.5, loc="upper left")

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.3)
    ax[0].set_xticks(range(7))
    ax[1].set_xticks(range(7))

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
