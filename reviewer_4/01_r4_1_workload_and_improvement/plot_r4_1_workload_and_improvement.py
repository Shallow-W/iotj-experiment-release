#!/usr/bin/env python3
"""绘制 R4-1 工作负载与性能改善四子图。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR
FIGURES_DIR = EXP_DIR / "reproduced"
SLOT_CSV = RESULTS_DIR / "r4_1_controlled_hotspot_slot_metrics.csv"
FIG_PNG = FIGURES_DIR / "r4_1_workload_and_improvement.png"
FIG_PDF = FIGURES_DIR / "r4_1_workload_and_improvement.pdf"

# ---------------------------------------------------------------------------
# 用户可调绘图参数：2 x 2 工作负载与动态改善主图
# ---------------------------------------------------------------------------
CONTROL_WINDOW = 10
FIGURE_SIZE = (7.60, 5.45)
FIGURE_DPI = 600
X_TICKS = [1, 50, 100, 150, 200, 250, 300]
FONT_STYLE = {
    "font.size": 10.8,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
}
LINE_WIDTH = 1.45
MARKER_SIZE = 3.8
SPINE_WIDTH = 0.85
GRID_STYLE = {
    "color": "#BDBDBD",
    "linestyle": "--",
    "alpha": 0.45,
    "linewidth": 0.50,
}
PANEL_CAPTION_Y = -0.22
PANEL_CAPTION_SIZE = 12.8
PAPER_RED = "#D7301F"
PAPER_TEAL = "#009E73"
PAPER_BLUE = "#0072B2"
PAPER_ORANGE = "#D55E00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            **FONT_STYLE,
            "axes.linewidth": SPINE_WIDTH,
            "xtick.major.width": SPINE_WIDTH,
            "ytick.major.width": SPINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def style_axis(ax: Any) -> None:
    ax.grid(axis="both", **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=SPINE_WIDTH, length=3.0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#000000")


def add_panel_caption(ax: Any, caption: str) -> None:
    ax.text(
        0.5,
        PANEL_CAPTION_Y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_CAPTION_SIZE,
    )


def window_means(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window_ends: list[int] = []
    means: list[float] = []
    for start_idx in range(0, len(values), CONTROL_WINDOW):
        end_idx = min(start_idx + CONTROL_WINDOW, len(values))
        window_ends.append(end_idx)
        means.append(float(np.mean(values[start_idx:end_idx])))
    return np.array(window_ends, dtype=int), np.array(means, dtype=float)


def window_improvement(
    static_values: np.ndarray,
    dynamic_values: np.ndarray,
    percentage_points: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    window_ends: list[int] = []
    values: list[float] = []
    for start_idx in range(0, len(static_values), CONTROL_WINDOW):
        end_idx = min(start_idx + CONTROL_WINDOW, len(static_values))
        static_mean = float(np.mean(static_values[start_idx:end_idx]))
        dynamic_mean = float(np.mean(dynamic_values[start_idx:end_idx]))
        if percentage_points:
            improvement = (static_mean - dynamic_mean) * 100.0
        else:
            improvement = (
                (static_mean - dynamic_mean) / static_mean * 100.0
                if abs(static_mean) > 1e-12
                else 0.0
            )
        window_ends.append(end_idx)
        values.append(improvement)
    return np.array(window_ends, dtype=int), np.array(values, dtype=float)


def draw_line_panel(
    ax: Any,
    slots: np.ndarray,
    values: np.ndarray,
    color: str,
    ylabel: str,
    caption: str,
    ylim: tuple[float, float],
    zero_line: bool,
) -> None:
    if zero_line:
        ax.axhline(0.0, color="#333333", linewidth=0.80)
    ax.plot(
        slots,
        values,
        color=color,
        linewidth=LINE_WIDTH,
        marker="s",
        markersize=MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=0.35,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("PeMS 5-min time slot")
    ax.set_xlim(1, 300)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(*ylim)
    style_axis(ax)
    add_panel_caption(ax, caption)


def make_figure() -> None:
    rows = read_csv(SLOT_CSV)
    static = sorted(
        [row for row in rows if row["method_id"] == "initial_static_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    dynamic = sorted(
        [row for row in rows if row["method_id"] == "causal_dynamic_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    if len(static) != len(dynamic):
        raise ValueError("Static and dynamic slot counts differ")

    workload = np.array(
        [float(row["long_chain_request_share"]) * 100.0 for row in dynamic],
        dtype=float,
    )
    workload_slots, workload_mean = window_means(workload)

    panel_specs = [
        (
            "mean_fullflow_delay",
            PAPER_RED,
            "Delay improvement (%)",
            "(b) End-to-end delay",
            (-3.0, 10.0),
            False,
        ),
        (
            "load_variance",
            PAPER_TEAL,
            "Load-variance improvement (%)",
            "(c) Load variance",
            (-5.0, 50.0),
            False,
        ),
        (
            "psi",
            PAPER_BLUE,
            r"$\psi$ reduction (percentage points)",
            r"(d) Cross-domain ratio $\psi$",
            (-7.0, 13.0),
            True,
        ),
    ]

    set_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    draw_line_panel(
        axes[0, 0],
        workload_slots,
        workload_mean,
        PAPER_ORANGE,
        "Long-chain request share (%)",
        "(a) Long-chain workload share",
        (0.0, 100.0),
        False,
    )

    for ax, (metric, color, ylabel, caption, ylim, percentage_points) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]], panel_specs
    ):
        static_values = np.array([float(row[metric]) for row in static], dtype=float)
        dynamic_values = np.array([float(row[metric]) for row in dynamic], dtype=float)
        slots, values = window_improvement(
            static_values,
            dynamic_values,
            percentage_points=percentage_points,
        )
        draw_line_panel(
            ax,
            slots,
            values,
            color,
            ylabel,
            caption,
            ylim,
            True,
        )

    fig.subplots_adjust(
        left=0.105,
        right=0.990,
        top=0.975,
        bottom=0.125,
        hspace=0.52,
        wspace=0.28,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(FIG_PNG)
