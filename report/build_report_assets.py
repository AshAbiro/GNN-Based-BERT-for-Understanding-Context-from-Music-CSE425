"""Render report figures from implementation schematics and saved experiment outputs.

Run from any directory: python report/build_report_assets.py
No model training, prediction, dimensionality reduction, or new experiment is performed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#263746", "text.color": "#263746",
    "savefig.facecolor": "white",
})
BLUE, TEAL, GRAY = "#245c8a", "#21786c", "#6b7782"


def canvas(w=8.0, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.10",
                              linewidth=1.2, edgecolor=color, facecolor=color + "12"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            linespacing=1.35)


def arrow(ax, start, end, label=None, curve=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.1, color=GRAY,
                                connectionstyle=f"arc3,rad={curve}"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18,
                label, ha="center", fontsize=8, backgroundcolor="white")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", dpi=190, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def schematics():
    fig, ax = canvas(8.0, 3.0)
    box(ax, .1, 2.65, 1.45, .85, "MusicCaps\ncaption")
    box(ax, 2.2, 2.65, 2.05, .85, "DistilBERT\ntoken states")
    box(ax, .1, .55, 1.45, .85, "Paired\naudio clip", TEAL)
    box(ax, 2.2, .55, 2.05, .85, "Segment graph\nGraphSAGE", TEAL)
    box(ax, 5.0, 1.60, 2.15, .95, "Text / graph\nrepresentations")
    box(ax, 7.9, 2.70, 1.9, .85, "Supervised tags\n124 probabilities", fs=9)
    box(ax, 7.9, .5, 1.9, .95, "Dual-encoder\nretrieval", TEAL)
    arrow(ax, (1.55, 3.075), (2.2, 3.075))
    arrow(ax, (1.55, .975), (2.2, .975))
    arrow(ax, (4.25, 3.075), (5.0, 2.22))
    arrow(ax, (4.25, .975), (5.0, 1.85))
    arrow(ax, (7.15, 2.22), (7.9, 3.075))
    arrow(ax, (7.15, 1.85), (7.9, .975))
    ax.text(5.0, .18, "Classification and retrieval were trained as separate experiments.",
            ha="center", fontsize=8.5)
    save(fig, "fig1_system")

    fig, ax = canvas(8, 2.6)
    box(ax, .12, 2.15, 2.1, 1.0, "Audio\nmono / resample\ncrop or pad", TEAL)
    box(ax, 2.85, 2.15, 2.25, 1.0, "MFCC + chroma\n20 temporal groups\nmean and std.", TEAL)
    box(ax, 5.75, 2.15, 1.8, 1.0, "Node matrix\n20 × 64", TEAL)
    box(ax, 8.2, 2.15, 1.55, 1.0, "Music\nstructure\ngraph", TEAL)
    box(ax, 5.65, .45, 3.3, .8, "Adjacent segments + similarity\nbidirectional edges", TEAL)
    arrow(ax, (2.22, 2.65), (2.85, 2.65))
    arrow(ax, (5.1, 2.65), (5.75, 2.65))
    arrow(ax, (7.55, 2.65), (8.2, 2.65))
    arrow(ax, (6.65, 2.15), (6.65, 1.25))
    arrow(ax, (8.95, .85), (9.25, 2.15), curve=.25)
    ax.text(2.65, .85, "Similarity links are acoustic proxies;\nno musical-form annotation is assumed.",
            ha="center", va="center", fontsize=9)
    save(fig, "fig2_graph_pipeline")

    fig, ax = canvas(8, 2.8)
    box(ax, .1, 2.7, 1.65, .85, "Graph readout\n$g$", TEAL)
    box(ax, 2.35, 2.7, 1.65, .85, "Query\nprojection")
    box(ax, .1, .75, 2.2, .85, "DistilBERT sequence\n$H_{\\mathrm{text}}$")
    box(ax, 4.55, 1.6, 2.15, 1.1, "Cross-attention\nquery: graph\nkeys/values: text")
    box(ax, 7.2, 1.6, 1.1, 1.1, "Concat\n$[g;c]$")
    box(ax, 8.85, 1.6, 1.05, 1.1, "FFN\nTags")
    arrow(ax, (1.75, 3.125), (2.35, 3.125))
    arrow(ax, (4, 3.125), (4.55, 2.4))
    arrow(ax, (2.3, 1.175), (4.55, 1.9), "padding mask")
    arrow(ax, (6.7, 2.15), (7.2, 2.15))
    arrow(ax, (8.3, 2.15), (8.85, 2.15))
    arrow(ax, (1.05, 3.55), (7.75, 2.7), curve=-.13)
    ax.text(5.0, .24, "Early concatenation substitutes the first-token text state for attended text.",
            ha="center", fontsize=8.5)
    save(fig, "fig4_fusion")

    fig, ax = canvas(8, 2.7)
    box(ax, .15, 2.65, 1.65, .85, "Audio graph", TEAL)
    box(ax, 2.45, 2.65, 2.2, .85, "GraphSAGE\nmean pooling", TEAL)
    box(ax, .15, .65, 1.65, .85, "Caption")
    box(ax, 2.45, .65, 2.2, .85, "DistilBERT\nfirst token")
    box(ax, 5.3, 2.65, 1.9, .85, "256-D projection\nL2 normalization", TEAL, 9)
    box(ax, 5.3, .65, 1.9, .85, "256-D projection\nL2 normalization", BLUE, 9)
    box(ax, 7.9, 1.45, 1.95, 1.25, "Cosine scores\nlearnable $\\tau$\nsymmetric InfoNCE", fs=9)
    for yy in (3.075, 1.075):
        arrow(ax, (1.8, yy), (2.45, yy))
        arrow(ax, (4.65, yy), (5.3, yy))
    arrow(ax, (7.2, 3.075), (7.9, 2.35))
    arrow(ax, (7.2, 1.075), (7.9, 1.8))
    ax.text(5.0, .13, "Matching pairs are positives; other pairs in the batch supply negatives.",
            ha="center", fontsize=8.5)
    save(fig, "fig6_contrastive")


def measured_plots():
    path = ROOT / "artifacts/task1_bert/learning_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.65), layout="constrained")
    epochs = [int(r["epoch"]) for r in rows]
    for ax, metric, title in zip(axes, ("macro_f1", "micro_f1"), ("Macro-F1", "Micro-F1")):
        for split, color, marker in (("train", BLUE, "o"), ("val", TEAL, "s")):
            ax.plot(epochs, [float(r[f"{split}_{metric}"]) for r in rows],
                    label="Training (online)" if split == "train" else "Validation",
                    color=color, marker=marker, markersize=3.5, linewidth=1.5)
        ax.set(title=title, xlabel="Epoch", ylabel="F1", ylim=(0, .55), xticks=epochs)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8, frameon=False)
    save(fig, "fig3_bert_curves")

    with (ROOT / "artifacts/task3_fusion/ablation_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = {"fusion": "Cross-attention", "bert": "BERT only", "gnn": "GNN only", "early": "Early concat."}
    fig, ax = plt.subplots(figsize=(7, 2.9), layout="constrained")
    x = np.arange(len(rows))
    for dx, metric, color, label in ((-.18, "macro_f1", BLUE, "Macro-F1"), (.18, "micro_f1", TEAL, "Micro-F1")):
        heights = [float(r[metric]) for r in rows]
        bars = ax.bar(x + dx, heights, .34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in heights], fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=[names[r["variant"]] for r in rows],
           ylabel="Test F1", ylim=(0, .40))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "fig5_ablation")

    samples = json.loads((ROOT / "artifacts/task1_bert/prediction_samples.json").read_text(encoding="utf-8"))
    sample = samples[0]
    tokens, weights = sample["tokens"], np.array(sample["cls_attention"])
    width = 12
    nrows = int(np.ceil(len(tokens) / width))
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 0.75 * nrows + .55), squeeze=False,
                             layout="constrained")
    for i, ax in enumerate(axes[:, 0]):
        a, b = i * width, min((i + 1) * width, len(tokens))
        patch = np.full(width, np.nan)
        patch[:b-a] = weights[a:b]
        ax.imshow(patch[None], cmap="Blues", vmin=0, vmax=max(weights), aspect="auto")
        labels = list(tokens[a:b]) + [""] * (width - (b-a))
        ax.set_xticks(range(width), labels, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for j, value in enumerate(weights[a:b]):
            ax.text(j, 0, f"{value:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if value > max(weights) * .58 else "#263746")
    fig.suptitle(f"Saved attention sample: {sample['id']}", fontsize=10)
    save(fig, "fig7_attention")
    print(f"Replotted {len(tokens)} token weights for saved sample {sample['id']}.")


if __name__ == "__main__":
    schematics()
    measured_plots()
    print(f"Saved seven figures in PDF and PNG format to {OUT}.")
