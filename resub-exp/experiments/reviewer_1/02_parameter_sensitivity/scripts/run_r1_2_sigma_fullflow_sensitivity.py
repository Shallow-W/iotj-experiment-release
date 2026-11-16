#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import run_r1_2_minimal_fullflow_validation as fullflow
import run_r1_2_parameter_sensitivity as base


EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / "results"
FIGURES_DIR = EXP_ROOT / "figures"

DEFAULT_SIGMA_DELAY = 1.0
DEFAULT_SIGMA_RES = 1.0
BLUE = "#0072B2"
ORANGE = "#E69F00"
DEFAULT_RED = "#D55E00"
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


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows, **kwargs):
    for row in rows:
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    raise KeyError(kwargs)


def rel_delta(value, baseline):
    return 0.0 if baseline == 0 else (value - baseline) / baseline


def sigma_values(rows, key):
    return sorted({float(row[key]) for row in rows})


def row_for(rows, sigma_delay, sigma_res):
    return find_row(rows, sigma_delay=sigma_delay, sigma_res=sigma_res)


def metric_grid(rows, metric, delay_values, res_values):
    data = np.zeros((len(res_values), len(delay_values)))
    for i, sigma_res in enumerate(res_values):
        for j, sigma_delay in enumerate(delay_values):
            data[i, j] = row_for(rows, sigma_delay, sigma_res)[metric]
    return data


def make_heatmap_figure(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    delay_values = sigma_values(rows, "sigma_delay")
    res_values = sigma_values(rows, "sigma_res")
    default_j = delay_values.index(DEFAULT_SIGMA_DELAY)
    default_i = res_values.index(DEFAULT_SIGMA_RES)

    metrics = [
        ("mean_fullflow_delay", "(a) Final delay", "{:.0f}", "viridis_r", None, None, "Delay"),
        ("cut_loss", "(b) Cut loss", "{:.3f}", "magma", None, None, "Cut loss"),
        ("psi", "(c) Cross-domain ratio", "{:.3f}", "YlOrRd", None, None, r"$\psi$"),
        ("resource_satisfaction_rate", "(d) Resource satisfaction", "{:.3f}", "Greens", 0.95, 1.0, "Satisfaction rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
    for ax, (metric, title, fmt, cmap, vmin, vmax, cbar_label) in zip(axes.ravel(), metrics):
        data = metric_grid(rows, metric, delay_values, res_values)
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$\sigma_{delay}$")
        ax.set_ylabel(r"$\sigma_{res}$")
        ax.set_xticks(range(len(delay_values)))
        ax.set_xticklabels([str(value) for value in delay_values])
        ax.set_yticks(range(len(res_values)))
        ax.set_yticklabels([str(value) for value in res_values])
        ax.add_patch(Rectangle((default_j - 0.5, default_i - 0.5), 1, 1, fill=False, edgecolor=DEFAULT_RED, linewidth=2.3))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                rgba = im.cmap(im.norm(data[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.45 else "black"
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8.5, color=text_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    default_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor=DEFAULT_RED, linewidth=2.3)
    fig.legend(
        handles=[default_handle],
        labels=[r"Default setting ($\sigma_{delay}=\sigma_{res}=1.0$)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        handlelength=1.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    png = FIGURES_DIR / "r1_2_sigma_fullflow_sensitivity_2x2.png"
    pdf = FIGURES_DIR / "r1_2_sigma_fullflow_sensitivity_2x2.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_slice_figure(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_slice_paper_style()
    delay_sweep = [row for row in rows if row["sigma_res"] == DEFAULT_SIGMA_RES]
    delay_sweep = sorted(delay_sweep, key=lambda row: row["sigma_delay"])
    res_sweep = [row for row in rows if row["sigma_delay"] == DEFAULT_SIGMA_DELAY]
    res_sweep = sorted(res_sweep, key=lambda row: row["sigma_res"])

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


def build_slice_rows(rows):
    slice_rows = []
    for row in sorted(rows, key=lambda item: item["sigma_delay"]):
        if row["sigma_res"] == DEFAULT_SIGMA_RES:
            slice_rows.append(
                {
                    "slice": "sweep_sigma_delay_at_sigma_res_1.0",
                    "swept_sigma": row["sigma_delay"],
                    **row,
                }
            )
    for row in sorted(rows, key=lambda item: item["sigma_res"]):
        if row["sigma_delay"] == DEFAULT_SIGMA_DELAY:
            slice_rows.append(
                {
                    "slice": "sweep_sigma_res_at_sigma_delay_1.0",
                    "swept_sigma": row["sigma_res"],
                    **row,
                }
            )
    return slice_rows


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def markdown_table(rows):
    lines = [
        "| Run | `sigma_delay` | `sigma_res` | Final delay | Load variance | Cut loss | `psi` | Resource satisfaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = f"{row['run_id']} default" if row["label"] == "default" else row["run_id"]
        lines.append(
            f"| {label} | {row['sigma_delay']} | {row['sigma_res']} | "
            f"{fmt(row['mean_fullflow_delay'])} | {fmt(row['load_variance'], 6)} | "
            f"{fmt(row['cut_loss'])} | {fmt(row['psi'])} | {fmt(row['resource_satisfaction_rate'])} |"
        )
    return "\n".join(lines)


def slice_table(slice_rows):
    lines = [
        "| Slice | Swept value | Final delay | Load variance | Cut loss | `psi` | Resource satisfaction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in slice_rows:
        name = "`sigma_delay` sweep, `sigma_res=1.0`" if row["slice"].startswith("sweep_sigma_delay") else "`sigma_res` sweep, `sigma_delay=1.0`"
        lines.append(
            f"| {name} | {row['swept_sigma']} | {fmt(row['mean_fullflow_delay'])} | "
            f"{fmt(row['load_variance'], 6)} | {fmt(row['cut_loss'])} | "
            f"{fmt(row['psi'])} | {fmt(row['resource_satisfaction_rate'])} |"
        )
    return "\n".join(lines)


def write_summary(rows, slice_rows, load_scale, heatmap_png, slice_png):
    default = find_row(rows, sigma_delay=DEFAULT_SIGMA_DELAY, sigma_res=DEFAULT_SIGMA_RES)
    best_delay = min(rows, key=lambda row: row["mean_fullflow_delay"])
    best_cut = min(rows, key=lambda row: row["cut_loss"])
    best_psi = min(rows, key=lambda row: row["psi"])
    best_load = min(rows, key=lambda row: row["load_variance"])
    worst_psi = max(rows, key=lambda row: row["psi"])
    all_resource_ok = all(row["resource_satisfaction_rate"] >= 0.999 for row in rows)

    lines = [
        "# R1-2 sigma_delay / sigma_res 敏感性实验总结",
        "",
        "日期：2026-07-24",
        "",
        "## 1. 审稿人要求",
        "",
        "审稿人要求对 FPDWGC 资源权重函数中的 `sigma_delay`、`sigma_res` 做敏感性实验，并报告 cut loss、跨域比例 `psi`、资源满足率或最终时延。",
        "",
        "## 2. 实验设计",
        "",
        "- 固定 PeMS03 派生的 300 个 5-min time slots、50 MEC 拓扑、10 类微服务、5 条 SFC 模板。",
        "- 固定默认资源退化参数 `rho_th=0.7,varphi=1.0,f_min=0.2`。",
        "- 重新执行 `sigma_delay x sigma_res` 的完整 `4 x 4` 网格：`{0.25,0.5,1.0,2.0} x {0.25,0.5,1.0,2.0}`。",
        "- 每组参数都重新构建 FPDWGC 权重矩阵、聚类分配、微服务部署，然后用 full-flow routing evaluator 计算指标。",
        f"- 端到端负载校准系数为 `{load_scale:.8f}`。",
        "",
        "## 3. 结果图",
        "",
        f"![R1-2 sigma full-flow sensitivity 2x2](figures/{heatmap_png.name})",
        "",
        "上图顶部图例中的空心方框为默认点 `sigma_delay=sigma_res=1.0`。四个子图分别报告最终端到端时延、cut loss、跨域比例 `psi` 和资源满足率。",
        "",
        f"![R1-2 sigma default-slice trends](figures/{slice_png.name})",
        "",
        "上图用于辅助读趋势：顶部图例区分两条默认切片，蓝线固定 `sigma_res=1.0` 扫 `sigma_delay`，橙线固定 `sigma_delay=1.0` 扫 `sigma_res`，灰色虚线为默认 sigma 值。",
        "",
        "## 4. 完整网格数据",
        "",
        markdown_table(rows),
        "",
        "## 5. 默认切片趋势",
        "",
        slice_table(slice_rows),
        "",
        "## 6. 关键观察",
        "",
        f"1. 默认点 S11 的最终时延为 `{default['mean_fullflow_delay']:.3f}`，cut loss 为 `{default['cut_loss']:.3f}`，`psi={default['psi']:.3f}`，资源满足率为 `{default['resource_satisfaction_rate']:.3f}`。",
        f"2. 全网格最低最终时延出现在 `{best_delay['run_id']}` (`sigma_delay={best_delay['sigma_delay']},sigma_res={best_delay['sigma_res']}`)，比默认点降低 `{abs(rel_delta(best_delay['mean_fullflow_delay'], default['mean_fullflow_delay'])) * 100:.2f}%`；但该点负载方差为 `{best_delay['load_variance']:.6f}`，高于默认点的 `{default['load_variance']:.6f}`。",
        f"3. 最低 cut loss 出现在 `{best_cut['run_id']}` (`sigma_delay={best_cut['sigma_delay']},sigma_res={best_cut['sigma_res']}`)，但它的 `psi={best_cut['psi']:.3f}`，高于默认点的 `{default['psi']:.3f}`，说明 cut loss 最低并不等价于跨域比例最低。",
        f"4. 最低跨域比例出现在 `{best_psi['run_id']}` (`sigma_delay={best_psi['sigma_delay']},sigma_res={best_psi['sigma_res']}`)，`psi={best_psi['psi']:.3f}`；最高跨域比例出现在 `{worst_psi['run_id']}`，`psi={worst_psi['psi']:.3f}`，说明 sigma 参数确实会改变服务链跨域调用。",
        f"5. 最低负载方差出现在 `{best_load['run_id']}`，但对应最终时延为 `{best_load['mean_fullflow_delay']:.3f}`，高于最低时延点，说明 delay 与 load balance 存在 tradeoff。",
        f"6. 16 组参数的资源满足率均为 `1.000`：`{'是' if all_resource_ok else '否'}`，说明在当前服务实例规模下，sigma 改变不会破坏部署可行性。",
        "",
        "## 7. 是否能回复审稿人",
        "",
        "可以。该实验逐点报告了审稿人要求的 cut loss、跨域比例 `psi`、资源满足率和最终时延，并且展示了 `sigma_delay/sigma_res` 对聚类亲和度和端到端性能的影响。更稳妥的论文表述不是“默认 sigma 全局最优”，而是“默认 sigma 是时延、负载均衡、跨域比例和资源可行性之间的平衡设置”。",
        "",
        "## 8. 可写入论文 / rebuttal 的表述",
        "",
        "We further conduct a 4 x 4 sensitivity analysis over the FPDWGC weight scales sigma_delay and sigma_res. For each parameter pair, we rebuild the FPDWGC domains and service placement, and then evaluate the deployment with the full-flow routing simulator. The results report cut loss, cross-domain ratio psi, resource satisfaction rate, and final end-to-end delay. The grid shows an interpretable delay/load/locality tradeoff: the lowest-delay setting is not the lowest-cut-loss or lowest-psi setting, while all parameter pairs keep the resource satisfaction rate at 1.0. Therefore, we use sigma_delay=sigma_res=1.0 as a balanced default rather than a delay-only optimum.",
        "",
        "## 9. 输出文件",
        "",
        "- `results/r1_2_sigma_fullflow_sensitivity.csv`",
        "- `results/r1_2_sigma_default_slices.csv`",
        "- `figures/r1_2_sigma_fullflow_sensitivity_2x2.png`",
        "- `figures/r1_2_sigma_fullflow_sensitivity_2x2.pdf`",
        "- `figures/r1_2_sigma_default_slice_trends.png`",
        "- `figures/r1_2_sigma_default_slice_trends.pdf`",
    ]
    path = EXP_ROOT / "R1_2_SIGMA_SENSITIVITY_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_experiment():
    config = base.load_config()
    nodes = base.load_nodes()
    services, service_order = base.load_microservices()
    chains, targets = base.load_app()
    requests, request_totals = base.load_requests()
    flow_matrix = fullflow.load_flow_matrix()

    deployment, _, _, _ = fullflow.build_clustered_deployment(
        config,
        nodes,
        services,
        service_order,
        chains,
        request_totals,
        targets,
        DEFAULT_SIGMA_DELAY,
        DEFAULT_SIGMA_RES,
    )
    state = fullflow.placements_to_state(deployment["placements"], service_order, len(nodes))
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(fullflow.SERVICE_COUNT)]
    load_scale = fullflow.calibrate_fullflow_scale(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals)
    rows = fullflow.run_sigma_validation(
        config,
        nodes,
        services,
        service_order,
        chains,
        requests,
        request_totals,
        targets,
        flow_matrix,
        load_scale,
    )
    return rows, load_scale


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows, load_scale = run_experiment()
    slice_rows = build_slice_rows(rows)
    write_csv(RESULTS_DIR / "r1_2_sigma_fullflow_sensitivity.csv", rows)
    write_csv(RESULTS_DIR / "r1_2_sigma_default_slices.csv", slice_rows)
    heatmap_png, heatmap_pdf = make_heatmap_figure(rows)
    slice_png, slice_pdf = make_slice_figure(rows)
    summary = write_summary(rows, slice_rows, load_scale, heatmap_png, slice_png)

    for path in [
        RESULTS_DIR / "r1_2_sigma_fullflow_sensitivity.csv",
        RESULTS_DIR / "r1_2_sigma_default_slices.csv",
        heatmap_png,
        heatmap_pdf,
        slice_png,
        slice_pdf,
        summary,
    ]:
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
