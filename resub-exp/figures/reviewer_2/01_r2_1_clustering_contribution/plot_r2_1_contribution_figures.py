#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXP_ROOT
FIGURES_DIR = EXP_ROOT / "reproduced"

CLUSTER_CSV = RESULTS_DIR / "r2_1_strengthened_clustering_test.csv"

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#8A8F98"
DARK = "#2F3A45"
GRID_STYLE = {"color": "#BDBDBD", "linestyle": "--", "alpha": 0.45, "linewidth": 0.50}
SPINE_WIDTH = 0.85
PANEL_CAPTION_SIZE = 13
PANEL_CAPTION_Y = -0.20
DELAY_Y_MIN = 8000
DELAY_Y_MAX = 12000
REDUCTION_Y_MIN = 70
REDUCTION_Y_MAX = 85


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def short_cluster_label(method):
    mapping = {
        "No clustering/global domain": "No\nclustering",
        "Static geographic partition": "Static\npartition",
        "K-means": "K-means",
        "Spectral clustering": "Spectral\nclustering",
        "FPDWGC-tuned": "FPDWGC",
    }
    return mapping.get(method, method)


def style_axis(ax):
    ax.grid(axis="y", **GRID_STYLE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#000000")
    ax.tick_params(axis="both", width=0.75, length=3.0, pad=2)


def add_panel_caption(ax, caption, y):
    ax.text(
        0.5,
        y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_CAPTION_SIZE,
    )


def draw_clustering_figure(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [short_cluster_label(row["method"]) for row in rows]
    x = np.arange(len(rows))
    colors = [GRAY, TEAL, BLUE, PURPLE, ORANGE]

    delays = [as_float(row, "mean_fullflow_delay") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    bars = axes[0].bar(x, delays, color=colors, edgecolor="white", linewidth=0.8)
    bars[-1].set_edgecolor(DARK)
    bars[-1].set_linewidth(1.4)
    axes[0].set_ylabel("End-to-end delay (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0, ha="center")
    axes[0].set_ylim(DELAY_Y_MIN, DELAY_Y_MAX)
    style_axis(axes[0])
    add_panel_caption(axes[0], "(a) End-to-end delay", PANEL_CAPTION_Y)

    reduction_rows = [row for row in rows if row["method"] != "No clustering/global domain"]
    reduction_x = np.arange(len(reduction_rows))
    reduction_labels = [short_cluster_label(row["method"]) for row in reduction_rows]
    reduction_values = [as_float(row, "candidate_reduction_vs_global_pct") for row in reduction_rows]
    reduction_colors = colors[1:]

    bars = axes[1].bar(
        reduction_x,
        reduction_values,
        color=reduction_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    bars[-1].set_edgecolor(DARK)
    bars[-1].set_linewidth(1.4)
    axes[1].set_ylabel("Candidate-space reduction (%)", labelpad=8)
    axes[1].set_xticks(reduction_x)
    axes[1].set_xticklabels(reduction_labels, rotation=0, ha="center")
    axes[1].set_ylim(REDUCTION_Y_MIN, REDUCTION_Y_MAX)
    style_axis(axes[1])
    add_panel_caption(axes[1], "(b) Candidate-space reduction", PANEL_CAPTION_Y)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.97, bottom=0.31, wspace=0.30)
    png = FIGURES_DIR / "r2_1_clustering_contribution.png"
    pdf = FIGURES_DIR / "r2_1_clustering_contribution.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    for path in draw_clustering_figure(read_csv(CLUSTER_CSV)):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
