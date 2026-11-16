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

EFFECTS_CSV = RESULTS_DIR / "r2_3_single_factor_effects.csv"

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


def effect_label(row):
    mapping = {
        "FPDWGC dynamic clustering": "FPDWGC",
        "PER priority replay": "PER",
        "Feasibility action mask": "Mask",
        "Full combination vs all modules off": "Full\nvs All off",
    }
    return mapping.get(row["module"], row["module"])


def draw_single_factor_contribution(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_bar_chart_style()
    module_rows = [row for row in rows if row["factor_key"] != "combination"]
    labels = [effect_label(row) for row in module_rows]
    x = np.arange(len(module_rows))

    main_effect = [row["main_effect_delay_improvement_pct"] for row in module_rows]
    removal_effect = [row["single_remove_delay_increase_pct_vs_full"] for row in module_rows]
    sat_gain = [row["main_effect_resource_satisfaction_delta"] * 100.0 for row in module_rows]
    invalid_reduction = [row["main_effect_invalid_rate_reduction"] * 100.0 for row in module_rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    width = 0.36

    ax = axes[0]
    ax.bar(x - width / 2, main_effect, width, color=BLUE, edgecolor="white", linewidth=0.8, label="Mean effect")
    ax.bar(
        x + width / 2,
        removal_effect,
        width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="Increase if removed",
    )
    ax.set_ylabel("Delay change (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, 18.5)
    style_bar_axis(ax)
    style_legend(ax.legend(loc="upper left", ncol=2, columnspacing=1.10, **LEGEND_STYLE))
    add_panel_caption(ax, "(a) Delay contribution")

    ax = axes[1]
    ax.bar(
        x - width / 2,
        sat_gain,
        width,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.8,
        label="Resource satisfaction",
    )
    ax.bar(
        x + width / 2,
        invalid_reduction,
        width,
        color=TEAL,
        edgecolor="white",
        linewidth=0.8,
        label="Invalid action reduction",
    )
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.set_ylabel("Feasibility change\n(percentage points)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(-2.5, 7.0)
    style_bar_axis(ax)
    style_legend(ax.legend(loc="upper left", **LEGEND_STYLE))
    add_panel_caption(ax, "(b) Feasibility contribution")

    fig.subplots_adjust(left=0.085, right=0.99, top=0.97, bottom=0.31, wspace=0.35)
    png = FIGURES_DIR / "r2_3_single_factor_contribution.png"
    pdf = FIGURES_DIR / "r2_3_single_factor_contribution.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    for path in draw_single_factor_contribution(read_csv(EFFECTS_CSV)):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
