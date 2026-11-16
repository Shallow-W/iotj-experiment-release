#!/usr/bin/env python3
"""Publication-style R3-2 clustering baseline figure.

The plot reads the R3-2 baseline CSV and renders only the three measures that
directly answer the reviewer: end-to-end delay, cross-domain ratio, and the
FPDWGC-objective cut loss. The single global domain is intentionally excluded
from the two partition-quality panels, where it is not a meaningful baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXP_ROOT
FIGURES_DIR = EXP_ROOT / "reproduced"

INPUT_CSV = RESULTS_DIR / "r3_2_clustering_baseline_metrics.csv"
# Keep the main response documents on a single canonical figure path.
OUT_BASE = FIGURES_DIR / "r3_2_clustering_baseline_metrics"


PAPER_RED = "#D7301F"
PAPER_YELLOW = "#E5A83A"
PAPER_NAVY = "#1B2A38"
PAPER_TEAL = "#5AA6A0"
PAPER_GREY = "#9AA0A6"
EDGE = "#333333"

METHOD_COLORS = {
    "No clustering/global domain": PAPER_GREY,
    "Static geographic partition": PAPER_YELLOW,
    "K-means": PAPER_NAVY,
    "Spectral clustering": PAPER_TEAL,
    "FPDWGC-tuned": PAPER_RED,
}

SHORT_LABELS = {
    "No clustering/global domain": "No\nclustering",
    "Static geographic partition": "Static",
    "K-means": "K-means",
    "Spectral clustering": "Spectral",
    "FPDWGC-tuned": "FPDWGC",
}

# ===== 画图超参数：手动润色时优先调整这一块 =====
# 整图大小和输出分辨率；与 R2-2 的单行三子图统一。
FIGURE_SIZE = (10.8, 3.25)
FIGURE_DPI = 600
# 四周边距和子图间距。底部预留给横轴标签和下方子图标题。
SUBPLOT_LEFT = 0.070
SUBPLOT_RIGHT = 0.990
SUBPLOT_TOP = 0.965
SUBPLOT_BOTTOM = 0.240
# 一行三子图的横向间距，与 R2-2 三子图保持一致。
SUBPLOT_WSPACE = 0.30
# 子图标题统一放在下方；-0.20 对应标题与 x 轴的标准间距。
PANEL_TITLE_Y = -0.20
# 字号严格遵循 FIGURE_STYLE_STANDARD.md。
PANEL_TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 12.5
TICK_LABEL_FONT_SIZE = 11
# 分类标签与坐标刻度同字号；标签较长时以换行而非缩小字体处理。
X_TICK_LABEL_FONT_SIZE = 11
# 柱状图、边框和网格样式。
BAR_WIDTH = 0.64
BAR_EDGE_WIDTH = 0.55
SPINE_WIDTH = 0.85
GRID_COLOR = "#BDBDBD"
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.45
GRID_LINEWIDTH = 0.50
# 三个面板的纵轴下界。截断空白区以便辨认方法间差异；精确数据仍由结果表报告。
DELAY_Y_MIN = 8000.0
PSI_Y_MIN = 0.15
CUT_LOSS_Y_MIN = 0.60

PANEL_TITLES = [
    "(a) End-to-end response delay",
    r"(b) Cross-domain ratio $\psi$",
    "(c) FPDWGC-objective cut loss",
]


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "axes.linewidth": SPINE_WIDTH,
            "xtick.major.width": SPINE_WIDTH,
            "ytick.major.width": SPINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def draw_panel(ax, labels, values, panel_title, ylabel, ylim_pad=0.10, ymin=0.0):
    x = np.arange(len(labels))
    colors = [METHOD_COLORS[label] for label in labels]
    bars = ax.bar(x, values, width=BAR_WIDTH, color=colors, edgecolor=EDGE, linewidth=BAR_EDGE_WIDTH)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[label] for label in labels], rotation=0, ha="center")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=SPINE_WIDTH, length=3.0, pad=2)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(X_TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y", color=GRID_COLOR, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#000000")
    ax.text(
        0.5,
        PANEL_TITLE_Y,
        panel_title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_TITLE_FONT_SIZE,
    )

    ymax = max(values) if len(values) else 1.0
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(ymin, ymax * (1 + ylim_pad))

    # Highlight the proposed clustering method using the same red as MPD3QN in
    # the original manuscript figures, without adding a heavy visual outline.
    for label, bar in zip(labels, bars):
        if label == "FPDWGC-tuned":
            bar.set_linewidth(0.9)
            bar.set_edgecolor("#7A160D")
    return bars


def main() -> None:
    set_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    labels = df["method"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [1.24, 1.0, 1.0]})
    non_global = df[df["method"] != "No clustering/global domain"].copy()

    draw_panel(
        axes[0],
        labels,
        df["mean_fullflow_delay"].to_numpy(),
        PANEL_TITLES[0],
        "Average response delay (ms)",
        ymin=DELAY_Y_MIN,
    )
    draw_panel(
        axes[1],
        non_global["method"].tolist(),
        non_global["psi"].to_numpy(),
        PANEL_TITLES[1],
        r"Cross-domain ratio $\psi$",
        ymin=PSI_Y_MIN,
    )
    draw_panel(
        axes[2],
        non_global["method"].tolist(),
        non_global["tuned_fpdwgc_objective_cut_loss"].to_numpy(),
        PANEL_TITLES[2],
        "FPDWGC-objective cut loss",
        ymin=CUT_LOSS_Y_MIN,
    )

    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
        wspace=SUBPLOT_WSPACE,
    )

    fig.savefig(OUT_BASE.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    print(OUT_BASE.with_suffix(".png"))
    print(OUT_BASE.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
