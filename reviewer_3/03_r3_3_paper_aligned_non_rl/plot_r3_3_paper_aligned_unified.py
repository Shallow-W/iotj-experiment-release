#!/usr/bin/env python3
"""Draw the paper-axis R3-3 delay figure.

The CSV also records load variance, but the default R3-3 figure intentionally
uses delay only.  R3-3 asks for an additional non-RL baseline; the paper's
load-variance panels use a different historical reporting definition, so a
new load-variance panel should be regenerated only after that definition is
fully re-aligned.
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
INPUT_CSV = RESULTS_DIR / "r3_3_paper_aligned_unified_metrics.csv"

DELAY_PNG = FIGURES_DIR / "r3_3_paper_aligned_unified_delay.png"
DELAY_PDF = FIGURES_DIR / "r3_3_paper_aligned_unified_delay.pdf"

# ===== 可调绘图参数：与 resub-exp/experiments/FIGURE_STYLE_STANDARD.md 对齐 =====
FIGURE_SIZE = (10.6, 6.7)
FIGURE_DPI = 600
AXIS_LABEL_SIZE = 12.5
TICK_LABEL_SIZE = 11
PANEL_TITLE_SIZE = 13
LEGEND_SIZE = 10
SPINE_WIDTH = 0.85
BAR_WIDTH = 0.105
PANEL_TITLE_Y = -0.18
SUBPLOT_WSPACE = 0.24
SUBPLOT_HSPACE = 0.36
SUBPLOT_TOP = 0.965
SUBPLOT_BOTTOM = 0.165
GRID_STYLE = {"color": "#BDBDBD", "linestyle": "--", "alpha": 0.45, "linewidth": 0.50}

# 方法顺序和论文原图保持一致：先放四个 RL 方法，再补充非 RL baseline。
# 非 RL baseline 使用低饱和实心色，不使用斜纹/点纹，避免和论文原柱状图风格冲突。
METHODS = [
    ("mpd3qn", "MPD3QN", "#D23918"),
    ("dqn", "DQN", "#E5A84B"),
    ("sac", "SAC", "#151D29"),
    ("gddpg", "GDDPG", "#5DA39D"),
    ("first_fit", "First-fit", "#B8C0C8"),
    ("load_greedy", "Load-aware", "#8EA6BF"),
]
PANELS = [
    ("server_count", "Number of MEC servers", "(a) Impact of server quantity"),
    ("request_level", "Number of requests", "(b) Impact of request quantity"),
    ("microservice_types", "Number of microservice types", "(c) Impact of microservice variety"),
    ("base_station_count", "Number of base stations", "(d) Impact of base station quantity"),
]


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": SPINE_WIDTH,
            "xtick.major.width": SPINE_WIDTH,
            "ytick.major.width": SPINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def style_axis(axis) -> None:
    axis.grid(axis="y", **GRID_STYLE)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", width=SPINE_WIDTH, length=3.0, pad=2)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#000000")


def panel_frame(df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    panel = df[df["sweep"] == sweep].copy()
    return panel.sort_values(["x_value", "method_id"])


def draw_metric(df: pd.DataFrame, metric: str, ylabel: str, outputs: tuple[Path, Path]) -> None:
    set_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    handles = []
    labels = []

    for axis, (sweep, xlabel, caption) in zip(axes.ravel(), PANELS):
        panel = panel_frame(df, sweep)
        x_values = sorted(panel["x_value"].unique())
        positions = np.arange(len(x_values), dtype=float)
        offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * BAR_WIDTH

        for index, (method_id, label, color) in enumerate(METHODS):
            values = []
            for x_value in x_values:
                selected = panel[(panel["x_value"] == x_value) & (panel["method_id"] == method_id)]
                if selected.empty:
                    values.append(np.nan)
                else:
                    values.append(float(selected.iloc[0][metric]))
            bars = axis.bar(
                positions + offsets[index],
                values,
                width=BAR_WIDTH,
                label=label,
                color=color,
                edgecolor="#000000",
                linewidth=0.75,
            )
            if len(handles) < len(METHODS):
                handles.append(bars[0])
                labels.append(label)

        axis.set_ylabel(ylabel)
        axis.set_xlabel(xlabel, labelpad=0)
        axis.set_xticks(positions)
        axis.set_xticklabels([str(value) for value in x_values], rotation=0)
        axis.margins(x=0.05, y=0.10)
        axis.text(
            0.5,
            PANEL_TITLE_Y,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_TITLE_SIZE,
        )
        style_axis(axis)

    legend = axes.ravel()[1].legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D9D9D9",
        fontsize=LEGEND_SIZE,
        borderpad=0.38,
        labelspacing=0.32,
        handlelength=1.7,
        handletextpad=0.65,
        columnspacing=1.00,
    )
    legend.get_frame().set_linewidth(0.60)
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )
    for output in outputs:
        fig.savefig(output, dpi=FIGURE_DPI)
        print(output)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    draw_metric(df, "mean_end_to_end_delay_ms", "Average response delay (ms)", (DELAY_PNG, DELAY_PDF))


if __name__ == "__main__":
    main()
