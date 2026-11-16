#!/usr/bin/env python3
"""绘制 R4-1 简洁版动态改善趋势图。"""

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
EVENT_CSV = RESULTS_DIR / "r4_1_controlled_hotspot_update_events.csv"
FIG_PNG = FIGURES_DIR / "r4_1_simple_dynamic_improvement.png"
FIG_PDF = FIGURES_DIR / "r4_1_simple_dynamic_improvement.pdf"

# ---------------------------------------------------------------------------
# 用户可调绘图参数：1 x 3 简洁动态趋势图
# ---------------------------------------------------------------------------
HOTSPOT_START_SLOT = 61
CONTROL_WINDOW = 10
FIGURE_SIZE = (9.80, 3.25)
FIGURE_DPI = 600
X_TICKS = list(range(60, 301, 30))
FONT_STYLE = {
    "font.size": 10.8,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
}
LINE_WIDTH = 1.45
MARKER_SIZE = 4.0
SPINE_WIDTH = 0.85
GRID_STYLE = {
    "color": "#BDBDBD",
    "linestyle": "--",
    "alpha": 0.45,
    "linewidth": 0.50,
}
PANEL_CAPTION_Y = -0.24
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


def hotspot_window_values(
    static_values: np.ndarray,
    dynamic_values: np.ndarray,
    percentage_points: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    start_idx = HOTSPOT_START_SLOT - 1
    window_ends: list[int] = []
    values: list[float] = []
    for window_start in range(start_idx, len(static_values), CONTROL_WINDOW):
        window_end = min(window_start + CONTROL_WINDOW, len(static_values))
        static_mean = float(np.mean(static_values[window_start:window_end]))
        dynamic_mean = float(np.mean(dynamic_values[window_start:window_end]))
        if percentage_points:
            value = (static_mean - dynamic_mean) * 100.0
        else:
            value = (
                (static_mean - dynamic_mean) / static_mean * 100.0
                if abs(static_mean) > 1e-12 else 0.0
            )
        window_ends.append(window_end)
        values.append(value)
    return np.array(window_ends, dtype=int), np.array(values, dtype=float)


def draw_panel(
    ax: Any,
    window_ends: np.ndarray,
    values: np.ndarray,
    color: str,
    ylabel: str,
    caption: str,
    ylim: tuple[float, float],
    update_slots: list[int],
) -> None:
    ax.axvline(HOTSPOT_START_SLOT, color="#555555", linestyle=":", linewidth=0.85)
    for slot in update_slots:
        ax.axvline(slot, color=PAPER_ORANGE, linestyle="-.", linewidth=0.85, alpha=0.78)
    ax.axhline(0.0, color="#333333", linewidth=0.80)
    ax.plot(
        window_ends,
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
    ax.set_xlim(60, 300)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(*ylim)
    style_axis(ax)
    add_panel_caption(ax, caption)


def make_figure() -> None:
    rows = read_csv(SLOT_CSV)
    events = read_csv(EVENT_CSV)
    static = sorted(
        [row for row in rows if row["method_id"] == "initial_static_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    dynamic = sorted(
        [row for row in rows if row["method_id"] == "causal_dynamic_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    update_slots = [int(event["applied_slot"]) for event in events]

    panel_specs = [
        ("mean_fullflow_delay", PAPER_RED, "Delay improvement (%)", "(a) End-to-end delay", (-3.0, 10.0), False),
        ("load_variance", PAPER_TEAL, "Load-variance improvement (%)", "(b) Load variance", (-5.0, 50.0), False),
        ("psi", PAPER_BLUE, r"$\psi$ reduction (percentage points)", r"(c) Cross-domain ratio $\psi$", (-7.0, 13.0), True),
    ]

    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    for ax, (metric, color, ylabel, caption, ylim, percentage_points) in zip(axes, panel_specs):
        static_values = np.array([float(row[metric]) for row in static], dtype=float)
        dynamic_values = np.array([float(row[metric]) for row in dynamic], dtype=float)
        window_ends, values = hotspot_window_values(
            static_values,
            dynamic_values,
            percentage_points=percentage_points,
        )
        draw_panel(
            ax,
            window_ends,
            values,
            color,
            ylabel,
            caption,
            ylim,
            update_slots,
        )

    fig.subplots_adjust(left=0.068, right=0.992, top=0.955, bottom=0.230, wspace=0.27)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(FIG_PNG)
