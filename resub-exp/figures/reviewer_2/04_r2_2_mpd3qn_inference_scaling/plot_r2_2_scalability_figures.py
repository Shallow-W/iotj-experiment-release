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
SUMMARY_CSV = RESULTS_DIR / "r2_2_runtime_summary.csv"

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
DARK = "#2F3A45"

# ===== 字体与标题参数：与 FIGURE_STYLE_STANDARD.md 保持一致 =====
# 所有图内文本使用 Times 风格 serif；子图标题位于 x 轴标题下方。
BASE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 12.5
TICK_LABEL_FONT_SIZE = 11
LEGEND_FONT_SIZE = 10
PANEL_TITLE_Y = -0.24
PANEL_TITLE_FONT_SIZE = 13
# 背景网格采用统一的浅灰虚线，不抢曲线和数据点的视觉重点。
GRID_COLOR = "#BDBDBD"
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.45
GRID_LINEWIDTH = 0.50
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
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def scenario_rows(rows, scenario, sort_key):
    selected = [row for row in rows if row["scenario"] == scenario]
    return sorted(selected, key=lambda row: row[sort_key])


def style_axis(ax):
    ax.grid(
        axis="both",
        color=GRID_COLOR,
        linestyle=GRID_LINESTYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_caption(ax, caption):
    ax.text(
        0.5,
        PANEL_TITLE_Y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_TITLE_FONT_SIZE,
    )


def style_legend(legend):
    legend.get_frame().set_linewidth(0.60)


def style_request_scale_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(ticks) * 0.82, max(ticks) * 1.18)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])


def plot_mpd3qn_inference(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("mec_sweep", "mec_servers", "MEC servers", "(a) MEC-server scale"),
        ("bs_sweep", "bs_count", "BS count", "(b) BS scale"),
        ("request_sweep", "request_scale", "Request/instance scale", "(c) Request scale"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    for ax, (scenario, x_key, x_label, caption) in zip(axes, panels):
        data = scenario_rows(rows, scenario, x_key)
        x = [row[x_key] for row in data]
        mpd3qn = [row["mpd3qn_online_inference_ms_median"] for row in data]
        global_baseline = [row["global_exhaustive_inference_ms_median"] for row in data]
        ax.plot(x, mpd3qn, marker="o", color=TEAL, linewidth=2.0, markersize=5, label="MPD3QN")
        ax.plot(
            x,
            global_baseline,
            marker="s",
            color=ORANGE,
            linewidth=2.0,
            markersize=4.8,
            label="Global candidate evaluation",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Online inference time (ms)")
        ax.set_ylim(0, max(global_baseline) * 1.22)
        style_axis(ax)
        add_panel_caption(ax, caption)
        if x_key == "request_scale":
            style_request_scale_axis(ax, x)
    style_legend(axes[0].legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, **LEGEND_STYLE))

    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.25, wspace=0.30)
    png = FIGURES_DIR / "r2_2_mpd3qn_inference_scaling.png"
    pdf = FIGURES_DIR / "r2_2_mpd3qn_inference_scaling.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    for path in plot_mpd3qn_inference(read_rows()):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
