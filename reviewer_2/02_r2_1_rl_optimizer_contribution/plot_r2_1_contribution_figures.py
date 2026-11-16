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

RL_CSV = RESULTS_DIR / "r2_1_strengthened_rl_test.csv"

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
RL_DELAY_Y_MIN = 9000
RL_DELAY_Y_MAX = 12500


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


def short_rl_label(method):
    mapping = {
        "DQN": "DQN",
        "D3QN": "D3QN",
        "D3QN+PER": "D3QN\n+PER",
        "D3QN+mask": "D3QN\n+mask",
        "MPD3QN": "MPD3QN",
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


def draw_rl_figure(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [short_rl_label(row["method"]) for row in rows]
    x = np.arange(len(rows))
    colors = [GRAY, BLUE, PURPLE, TEAL, ORANGE]

    delays = [as_float(row, "mean_fullflow_delay") for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.6))

    bars = ax.bar(x, delays, color=colors, edgecolor="white", linewidth=0.8)
    bars[-1].set_edgecolor(DARK)
    bars[-1].set_linewidth(1.4)
    ax.set_ylabel("End-to-end delay (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(RL_DELAY_Y_MIN, RL_DELAY_Y_MAX)
    style_axis(ax)
    add_panel_caption(ax, "RL optimizer contribution", PANEL_CAPTION_Y)

    fig.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.28)
    png = FIGURES_DIR / "r2_1_rl_optimizer_contribution.png"
    pdf = FIGURES_DIR / "r2_1_rl_optimizer_contribution.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    for path in draw_rl_figure(read_csv(RL_CSV)):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
