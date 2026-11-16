#!/usr/bin/env python3
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXP_ROOT
FIGURES_DIR = EXP_ROOT / "reproduced"

BLUE = "#0072B2"
ORANGE = "#E69F00"
DEFAULT_LINE = "#4D4D4D"


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


# ---------------------------------------------------------------------------
# 默认切片趋势图：用户可调绘图参数
# 修改这部分即可调整字体、标题位置、间距和线条，不会改动 full-flow 实验数据。
# ---------------------------------------------------------------------------

SLICE_FIGURE_SIZE = (8.1, 6.8)
SLICE_FIGURE_DPI = 600
SLICE_FONT_STYLE = {
    "font.size": 11.0,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 10.0,
}
SLICE_TEXT_POSITION = {
    "x_axis_label_y": -0.13,
    "panel_caption_y": -0.25,
    "panel_caption_size": 12.0,
}
SLICE_SUBPLOT_LAYOUT = {
    "left": 0.08,
    "right": 0.99,
    "top": 0.98,
    "bottom": 0.10,
    "wspace": 0.31,
    "hspace": 0.40,
}
SLICE_LINE_STYLE = {
    "line_width": 1.45,
    "marker_size": 5.0,
    "marker_edge_width": 0.45,
    "default_line_width": 0.8,
    "spine_width": 0.85,
}
SLICE_GRID_STYLE = {
    "color": "#BDBDBD",
    "linestyle": "--",
    "alpha": 0.45,
    "linewidth": 0.50,
}
SLICE_X_LABEL = r"Swept weight scale $\sigma$"


def set_slice_paper_style():
    """仅为默认切片趋势图应用 experiments 目录的统一绘图规范。"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            **SLICE_FONT_STYLE,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": SLICE_FIGURE_DPI,
        }
    )


def make_slice_figure(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_slice_paper_style()
    delay_sweep = sorted(
        (row for row in rows if row["slice"] == "sweep_sigma_delay_at_sigma_res_1.0"),
        key=lambda row: row["swept_sigma"],
    )
    res_sweep = sorted(
        (row for row in rows if row["slice"] == "sweep_sigma_res_at_sigma_delay_1.0"),
        key=lambda row: row["swept_sigma"],
    )

    metrics = [
        ("mean_fullflow_delay", "(a) End-to-end delay trend", "End-to-end delay (ms)"),
        ("cut_loss", "(b) Cut loss trend", "Cut loss"),
        ("psi", r"(c) Cross-domain ratio $\psi$ trend", r"Cross-domain ratio $\psi$"),
        ("load_variance", "(d) Load variance trend", "Load variance"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=SLICE_FIGURE_SIZE)
    for ax, (metric, panel_caption, ylabel) in zip(axes.ravel(), metrics):
        ax.plot(
            [row["sigma_delay"] for row in delay_sweep],
            [row[metric] for row in delay_sweep],
            marker="o",
            linewidth=SLICE_LINE_STYLE["line_width"],
            markersize=SLICE_LINE_STYLE["marker_size"],
            markeredgewidth=SLICE_LINE_STYLE["marker_edge_width"],
            color=BLUE,
            label=r"Vary $\sigma_{\mathrm{delay}}$ ($\sigma_{\mathrm{res}}=1.0$)",
        )
        ax.plot(
            [row["sigma_res"] for row in res_sweep],
            [row[metric] for row in res_sweep],
            marker="s",
            linewidth=SLICE_LINE_STYLE["line_width"],
            markersize=SLICE_LINE_STYLE["marker_size"],
            markeredgewidth=SLICE_LINE_STYLE["marker_edge_width"],
            color=ORANGE,
            label=r"Vary $\sigma_{\mathrm{res}}$ ($\sigma_{\mathrm{delay}}=1.0$)",
        )
        ax.axvline(1.0, color=DEFAULT_LINE, linestyle="--", linewidth=SLICE_LINE_STYLE["default_line_width"])
        ax.set_xlabel(SLICE_X_LABEL, labelpad=3)
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0])
        ax.grid(axis="both", **SLICE_GRID_STYLE)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", width=0.75, length=3.0, pad=2, direction="out")
        ax.xaxis.set_label_coords(0.5, SLICE_TEXT_POSITION["x_axis_label_y"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(SLICE_LINE_STYLE["spine_width"])
            spine.set_color("#000000")
        ax.text(
            0.5,
            SLICE_TEXT_POSITION["panel_caption_y"],
            panel_caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=SLICE_TEXT_POSITION["panel_caption_size"],
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            linewidth=SLICE_LINE_STYLE["line_width"],
            markersize=SLICE_LINE_STYLE["marker_size"],
            label=r"Vary $\sigma_{\mathrm{delay}}$ ($\sigma_{\mathrm{res}}=1.0$)",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="s",
            linewidth=SLICE_LINE_STYLE["line_width"],
            markersize=SLICE_LINE_STYLE["marker_size"],
            label=r"Vary $\sigma_{\mathrm{res}}$ ($\sigma_{\mathrm{delay}}=1.0$)",
        ),
        Line2D(
            [0],
            [0],
            color=DEFAULT_LINE,
            linestyle="--",
            linewidth=SLICE_LINE_STYLE["default_line_width"],
            label="Default value (1.0)",
        ),
    ]
    axes[0, 1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        handlelength=1.5,
        labelspacing=0.35,
    )
    fig.subplots_adjust(**SLICE_SUBPLOT_LAYOUT)
    png = FIGURES_DIR / "r1_2_sigma_default_slice_trends.png"
    pdf = FIGURES_DIR / "r1_2_sigma_default_slice_trends.pdf"
    fig.savefig(png, dpi=SLICE_FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    rows = pd.read_csv(RESULTS_DIR / "r1_2_sigma_default_slices.csv").to_dict("records")
    for path in make_slice_figure(rows):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
