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

FACTORIAL_CSV = RESULTS_DIR / "r2_3_module_factorial_ablation.csv"

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#8A8F98"
RED = "#D55E00"
DARK = "#2F3A45"
GRID_STYLE = {"color": "#BDBDBD", "linestyle": "--", "alpha": 0.45, "linewidth": 0.50}
SPINE_WIDTH = 0.85
PANEL_CAPTION_Y = -0.20
PANEL_CAPTION_SIZE = 13
LEGEND_STYLE = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.95,
    "facecolor": "white",
    "edgecolor": "#D9D9D9",
    "borderpad": 0.38,
    "borderaxespad": 0.45,
    "labelspacing": 0.32,
    "handlelength": 1.7,
    "handletextpad": 0.65,
}


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def short_label(row):
    mapping = {
        "all_off": "All\noff",
        "fpdwgc_only": "F",
        "per_only": "P",
        "mask_only": "M",
        "fpdwgc_per": "F+P",
        "fpdwgc_mask": "F+M",
        "per_mask_no_fpdwgc": "P+M",
        "full_mpd3qn": "Full",
    }
    return mapping.get(row["ablation_id"], row["label"])


def set_bar_chart_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def style_bar_axis(ax):
    ax.grid(axis="y", **GRID_STYLE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#000000")
    ax.tick_params(axis="both", width=0.75, length=3.0, pad=2)


def add_panel_caption(ax, caption):
    ax.text(
        0.5,
        PANEL_CAPTION_Y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_CAPTION_SIZE,
    )


def style_legend(legend):
    legend.get_frame().set_linewidth(0.60)


def draw_factorial_metrics(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_bar_chart_style()
    factorial = [row for row in rows if int(row["is_factorial"]) == 1]
    labels = [short_label(row) for row in factorial]
    x = np.arange(len(factorial))
    colors = [GRAY] * len(factorial)
    colors[0] = "#A3A9B3"
    colors[-1] = ORANGE

    delays = [row["mean_fullflow_delay"] for row in factorial]
    load_vars = [row["load_variance"] for row in factorial]
    resource_shortfall = [(1.0 - row["resource_satisfaction_rate"]) * 100.0 for row in factorial]
    invalid_rates = [row["invalid_action_rate"] * 100.0 for row in factorial]
    reward_costs = [-row["reward_proxy"] for row in factorial]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.5))
    ax = axes[0, 0]
    bars = ax.bar(x, delays, color=colors, edgecolor="white", linewidth=0.8)
    bars[-1].set_edgecolor(DARK)
    bars[-1].set_linewidth(1.4)
    ax.set_ylabel("Mean end-to-end delay (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(7500, max(delays) * 1.18)
    style_bar_axis(ax)
    add_panel_caption(ax, "(a) End-to-end delay")

    ax = axes[0, 1]
    bars = ax.bar(x, load_vars, color=colors, edgecolor="white", linewidth=0.8)
    bars[-1].set_edgecolor(DARK)
    bars[-1].set_linewidth(1.4)
    ax.set_ylabel("Load variance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, max(load_vars) * 1.20)
    style_bar_axis(ax)
    add_panel_caption(ax, "(b) Load balance")

    ax = axes[1, 0]
    width = 0.36
    ax.bar(
        x - width / 2,
        resource_shortfall,
        width,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.8,
        label="Resource shortfall",
    )
    ax.bar(
        x + width / 2,
        invalid_rates,
        width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="Invalid-action rate",
    )
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, max(max(resource_shortfall), max(invalid_rates), 1.0) * 1.30)
    style_bar_axis(ax)
    style_legend(ax.legend(loc="upper right", **LEGEND_STYLE))
    add_panel_caption(ax, "(c) Feasibility violations")

    ax = axes[1, 1]
    bars = ax.bar(x, reward_costs, color=colors, edgecolor="white", linewidth=0.8)
    bars[-1].set_edgecolor(DARK)
    bars[-1].set_linewidth(1.4)
    ax.set_ylabel("Reward-cost proxy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, max(reward_costs) * 1.18)
    style_bar_axis(ax)
    add_panel_caption(ax, "(d) Reward-cost proxy")

    fig.subplots_adjust(left=0.105, right=0.99, top=0.97, bottom=0.12, wspace=0.35, hspace=0.35)
    png = FIGURES_DIR / "r2_3_factorial_module_metrics.png"
    pdf = FIGURES_DIR / "r2_3_factorial_module_metrics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    for path in draw_factorial_metrics(read_csv(FACTORIAL_CSV)):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
