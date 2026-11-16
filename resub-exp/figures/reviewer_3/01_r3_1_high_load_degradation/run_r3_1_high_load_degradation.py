#!/usr/bin/env python3
"""Redraw R3-1 high-load degradation from the bundled sweep results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR
FIGURES_DIR = EXP_DIR / "reproduced"

LOAD_SWEEP_CSV = RESULTS_DIR / "r3_1_high_load_degradation_sweep.csv"
FIG_PNG = FIGURES_DIR / "r3_1_high_load_degradation.png"
FIG_PDF = FIGURES_DIR / "r3_1_high_load_degradation.pdf"

# ===== 画图超参数：你可以优先手动调整这一块 =====
# 图像整体大小，单位是 inch；增大宽/高可以让子图更松散。
FIGURE_SIZE = (11.6, 7.2)
# 输出 PNG 的分辨率；普通预览 300 dpi，论文终稿可按规范调到 600 dpi。
FIGURE_DPI = 300
# 子图四周边距和子图间距；hspace 控制上下间距，wspace 控制左右间距。
SUBPLOT_LEFT = 0.075
SUBPLOT_RIGHT = 0.985
SUBPLOT_TOP = 0.965
SUBPLOT_BOTTOM = 0.12
SUBPLOT_HSPACE = 0.36
SUBPLOT_WSPACE = 0.30
# 子图标题放在图下方；数值越接近 0，标题越靠近 x 轴。
PANEL_TITLE_Y = -0.22
# 四子图的标题略大于默认值，保证插入 Word/论文双栏后仍可清晰辨认。
PANEL_TITLE_FONT_SIZE = 12.5
# x/y 轴标题和刻度数字大小；调大可以增强论文插图可读性。
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 12
X_AXIS_LABEL_PAD = 2
# 图例字体大小，只在第一个子图显示图例。
LEGEND_FONT_SIZE = 8
# 图例遵循统一规范：白底圆角、浅灰细边框，避免文字悬浮在曲线上。
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
# 统一 x 轴标题、rho=1 参考线和网格样式。
X_AXIS_LABEL = r"MEC Server Utilization ($\rho$)"
RHO_REFERENCE = 1.0
REFERENCE_LINE_COLOR = "#111827"
REFERENCE_LINE_WIDTH = 0.8
# 网格按 resub-exp/experiments/FIGURE_STYLE_STANDARD.md 使用浅灰虚线。
GRID_COLOR = "#BDBDBD"
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.45
GRID_LINEWIDTH = 0.50
# 曲线标记样式和每组退化参数的颜色。
LINE_MARKER = "o"
RUN_ORDER = ["B01", "B02", "B05", "B07"]
RUN_COLORS = {
    "B01": "#2563eb",
    "B02": "#7c3aed",
    "B05": "#dc2626",
    "B07": "#059669",
}
# 子图标题统一放在每个子图下方。
PANEL_TITLES = [
    "(a) End-to-end response delay",
    "(b) Load-balancing variance",
    "(c) Queueing/execution delay",
    "(d) Effective capacity ratio",
]

def make_figure(rows: list[dict[str, Any]]) -> None:
    fig, axis_grid = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    axes = axis_grid.ravel()

    for run_id in RUN_ORDER:
        series = [row for row in rows if row["run_id"] == run_id]
        x = [float(row["actual_p95_raw_rho"]) for row in series]
        label = f"{run_id}: {series[0]['label']}" if series else run_id
        axes[0].plot(x, [float(row["mean_fullflow_delay"]) for row in series], marker=LINE_MARKER, color=RUN_COLORS[run_id], label=label)
        axes[1].plot(x, [float(row["load_variance"]) for row in series], marker=LINE_MARKER, color=RUN_COLORS[run_id])
        axes[2].plot(x, [float(row["mean_queue_delay"]) for row in series], marker=LINE_MARKER, color=RUN_COLORS[run_id])
        axes[3].plot(x, [float(row["mean_effective_capacity_ratio"]) for row in series], marker=LINE_MARKER, color=RUN_COLORS[run_id])

    for ax, panel_title in zip(axes, PANEL_TITLES):
        ax.axvline(RHO_REFERENCE, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=REFERENCE_LINE_WIDTH, alpha=0.7)
        ax.grid(color=GRID_COLOR, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
        ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=X_AXIS_LABEL_PAD)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        ax.text(0.5, PANEL_TITLE_Y, panel_title, transform=ax.transAxes, ha="center", va="top", fontsize=PANEL_TITLE_FONT_SIZE)

    axes[0].set_ylabel("Average response delay", fontsize=AXIS_LABEL_FONT_SIZE)
    legend = axes[0].legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, **LEGEND_STYLE)
    legend.get_frame().set_linewidth(0.60)

    axes[1].set_ylabel("Load variance", fontsize=AXIS_LABEL_FONT_SIZE)

    axes[2].set_ylabel("Mean queue delay", fontsize=AXIS_LABEL_FONT_SIZE)

    axes[3].set_ylabel("Mean f(rho)", fontsize=AXIS_LABEL_FONT_SIZE)

    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
        hspace=SUBPLOT_HSPACE,
        wspace=SUBPLOT_WSPACE,
    )
    fig.savefig(FIG_PNG, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with LOAD_SWEEP_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(rows)
    print(FIG_PNG)


if __name__ == "__main__":
    main()
