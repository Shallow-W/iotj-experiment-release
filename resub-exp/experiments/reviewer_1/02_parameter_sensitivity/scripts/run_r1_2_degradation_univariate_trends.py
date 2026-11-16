#!/usr/bin/env python3
"""R1-2 univariate degradation-parameter trend experiment.

This script keeps the topology, deployment, full-flow routing evaluator, and two
of the three degradation parameters fixed while sweeping the remaining one.
The resulting line plots are meant to answer the reviewer's request directly:
how average delay and load variance change as rho_th, varphi, or f_min changes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import run_r1_2_minimal_fullflow_validation as fullflow


EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / "results"
FIGURES_DIR = EXP_ROOT / "figures"

# ---------------------------------------------------------------------------
# 用户可调绘图参数
# 手动润色论文图片时，优先只改这一段；下面的实验数据和绘图逻辑通常不用动。
# ---------------------------------------------------------------------------

# 是否显示顶部图例。单变量单 y 轴图通常不需要图例；虚线含义可在图注中说明。
SHOW_LEGEND = False

# 整张六子图的画布尺寸，单位是 inch；第二个值越大，整体高度越高。
FIGURE_SIZE = (7.25, 6.8)

# 字体大小设置。axes.labelsize 控制横纵轴标题；xtick/ytick 控制坐标刻度数字；
# legend.fontsize 控制顶部图例字体。
FONT_STYLE = {
    "font.size": 10.5,
    "axes.titlesize": 10.8,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.2,
}

# 曲线、散点、默认参数虚线和坐标轴边框的粗细/大小。
LINE_STYLE = {
    "delay_marker_size": 3.8,
    "load_marker_size": 3.6,
    "line_width": 1.35,
    "marker_edge_width": 0.35,
    "default_line_width": 0.8,
    "spine_width": 0.85,
}

# 横纵网格线样式。alpha 越大网格越明显，linewidth 越大线越粗。
GRID_STYLE = {
    "color": "#BDBDBD",
    "linestyle": "--",
    "alpha": 0.65,
    "linewidth": 0.55,
}

# 顶部图例的位置和内部间距。仅当 SHOW_LEGEND=True 时生效。
# bbox_to_anchor 的第二个值越大，图例越靠上。
LEGEND_STYLE = {
    "bbox_to_anchor": (0.5, 1.01),
    "borderpad": 0.35,
    "handlelength": 1.5,
    "columnspacing": 1.2,
}

# 横轴标题和子图标题的位置。数值越负，文本越往下。
# x_axis_label_y 控制横轴标题位置；panel_caption_y 控制 (a)-(f) 子图标题位置。
TEXT_POSITION = {
    "x_axis_label_y": -0.155,
    "panel_caption_y": -0.335,
    "panel_caption_size": 11.2,
}

# 六子图整体布局边距。hspace 控制上下子图间距，wspace 控制左右子图间距。
SUBPLOT_LAYOUT = {
    "left": 0.085,
    "right": 0.99,
    "top": 0.97,
    "bottom": 0.11,
    "wspace": 0.29,
    "hspace": 0.5,
}

DEFAULT_PARAMS = {"rho_th": 0.7, "varphi": 1.0, "f_min": 0.2}
SWEEPS = {
    "rho_th": [0.5, 0.6, 0.7, 0.8, 0.9],
    "varphi": [0.5, 0.75, 1.0, 1.5, 2.0],
    "f_min": [0.1, 0.2, 0.3, 0.4, 0.5],
}

PAPER_RED = "#D7301F"
PAPER_TEAL = "#5AA6A0"
EDGE = "#333333"
DEFAULT_LINE = "#555555"
PARAM_LABELS = {
    "rho_th": r"$\rho_{\mathrm{th}}$",
    "varphi": r"$\varphi$",
    "f_min": r"$f_{\min}$",
}
X_AXIS_LABELS = {
    "rho_th": r"Trigger threshold $\rho_{\mathrm{th}}$",
    "varphi": r"Degradation slope $\varphi$",
    "f_min": r"Capacity floor $f_{\min}$",
}
def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            **FONT_STYLE,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def build_default_state(config, nodes, services, service_order, chains, request_totals, targets):
    cluster_cfg = config["fpdwgc_clustering"]
    sigma_delay = float(cluster_cfg["sigma_delay"]["value"])
    sigma_res = float(cluster_cfg["sigma_res"]["value"])
    deployment, assignment, cut_loss, selected_weight_ratio = fullflow.build_clustered_deployment(
        config, nodes, services, service_order, chains, request_totals, targets, sigma_delay, sigma_res
    )
    state = fullflow.placements_to_state(deployment["placements"], service_order, len(nodes))
    return deployment, assignment, state, cut_loss, selected_weight_ratio


def run_sweeps():
    config = fullflow.base.load_config()
    nodes = fullflow.base.load_nodes()
    services, service_order = fullflow.base.load_microservices()
    chains, targets = fullflow.base.load_app()
    requests, request_totals = fullflow.base.load_requests()
    flow_matrix = fullflow.load_flow_matrix()
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(fullflow.SERVICE_COUNT)]

    deployment, assignment, state, cut_loss, selected_weight_ratio = build_default_state(
        config, nodes, services, service_order, chains, request_totals, targets
    )
    load_scale = fullflow.calibrate_fullflow_scale(
        nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals
    )
    psi = fullflow.base.compute_psi(chains, request_totals, deployment["placements"], assignment)

    baseline_metrics = fullflow.evaluate_fullflow(
        nodes,
        services_by_idx,
        state,
        flow_matrix,
        targets,
        requests,
        request_totals,
        DEFAULT_PARAMS,
        load_scale,
    )
    baseline_delay = baseline_metrics["mean_fullflow_delay"]
    baseline_var = baseline_metrics["load_variance"]

    rows = []
    for parameter, values in SWEEPS.items():
        for value in values:
            params = dict(DEFAULT_PARAMS)
            params[parameter] = value
            metrics = fullflow.evaluate_fullflow(
                nodes,
                services_by_idx,
                state,
                flow_matrix,
                targets,
                requests,
                request_totals,
                params,
                load_scale,
            )
            rows.append(
                {
                    "sweep_parameter": parameter,
                    "sweep_value": value,
                    "rho_th": params["rho_th"],
                    "varphi": params["varphi"],
                    "f_min": params["f_min"],
                    "mean_fullflow_delay": round(metrics["mean_fullflow_delay"], 6),
                    "delay_delta_pct_vs_default": round(
                        (metrics["mean_fullflow_delay"] / baseline_delay - 1.0) * 100.0, 6
                    ),
                    "mean_queue_delay": round(metrics["mean_queue_delay"], 6),
                    "load_variance": round(metrics["load_variance"], 8),
                    "load_variance_delta_pct_vs_default": round(
                        (metrics["load_variance"] / baseline_var - 1.0) * 100.0, 6
                    ),
                    "min_effective_capacity_ratio": round(metrics["min_effective_capacity_ratio"], 6),
                    "p95_slot_max_effective_utilization": round(
                        metrics["p95_slot_max_effective_utilization"], 6
                    ),
                    "overload_node_slot_ratio": round(metrics["overload_node_slot_ratio"], 8),
                    "psi": round(psi, 6),
                    "cut_loss": round(cut_loss, 6),
                    "selected_weight_ratio": round(selected_weight_ratio, 6),
                    "deployment_success_ratio": round(deployment["deployment_success_ratio"], 6),
                    "failed_instances": deployment["failed_instances"],
                    "load_scale": round(load_scale, 8),
                    "is_default": parameter in params and value == DEFAULT_PARAMS[parameter],
                }
            )
    return pd.DataFrame(rows)


def plot_trends(df):
    set_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, sharex=False)
    panel_captions = [
        r"(a) Delay trend of $\rho_{\mathrm{th}}$",
        r"(b) Load variance trend of $\rho_{\mathrm{th}}$",
        r"(c) Delay trend of $\varphi$",
        r"(d) Load variance trend of $\varphi$",
        r"(e) Delay trend of $f_{\min}$",
        r"(f) Load variance trend of $f_{\min}$",
    ]
    for row_idx, parameter in enumerate(["rho_th", "varphi", "f_min"]):
        subset = df[df["sweep_parameter"] == parameter].sort_values("sweep_value")
        x = subset["sweep_value"].to_numpy()
        delay = subset["mean_fullflow_delay"].to_numpy()
        load_var = subset["load_variance"].to_numpy()
        x_margin = (x.max() - x.min()) * 0.06
        tick_labels = [f"{value:.2f}".rstrip("0").rstrip(".") for value in x]

        ax = axes[row_idx, 0]
        ax.plot(
            x,
            delay,
            marker="o",
            markersize=LINE_STYLE["delay_marker_size"],
            linewidth=LINE_STYLE["line_width"],
            color=PAPER_RED,
            markerfacecolor=PAPER_RED,
            markeredgecolor=EDGE,
            markeredgewidth=LINE_STYLE["marker_edge_width"],
        )
        ax.axvline(
            DEFAULT_PARAMS[parameter],
            color=DEFAULT_LINE,
            linestyle="--",
            linewidth=LINE_STYLE["default_line_width"],
        )
        ax.set_ylabel("Delay (ms)")
        ax.set_xlabel(X_AXIS_LABELS[parameter], labelpad=3)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.grid(axis="both", **GRID_STYLE)
        ax.set_axisbelow(True)

        ax = axes[row_idx, 1]
        ax.plot(
            x,
            load_var,
            marker="s",
            markersize=LINE_STYLE["load_marker_size"],
            linewidth=LINE_STYLE["line_width"],
            color=PAPER_TEAL,
            markerfacecolor=PAPER_TEAL,
            markeredgecolor=EDGE,
            markeredgewidth=LINE_STYLE["marker_edge_width"],
        )
        ax.axvline(
            DEFAULT_PARAMS[parameter],
            color=DEFAULT_LINE,
            linestyle="--",
            linewidth=LINE_STYLE["default_line_width"],
        )
        ax.set_ylabel("Load variance")
        ax.set_xlabel(X_AXIS_LABELS[parameter], labelpad=3)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.grid(axis="both", **GRID_STYLE)
        ax.set_axisbelow(True)

        for panel_ax in (axes[row_idx, 0], axes[row_idx, 1]):
            panel_ax.tick_params(axis="both", labelrotation=0, pad=2, direction="out")
            panel_ax.xaxis.set_label_coords(0.5, TEXT_POSITION["x_axis_label_y"])
            for spine in panel_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(LINE_STYLE["spine_width"])
                spine.set_color("#000000")

        for col_idx, panel_ax in enumerate((axes[row_idx, 0], axes[row_idx, 1])):
            panel_ax.text(
                0.5,
                TEXT_POSITION["panel_caption_y"],
                panel_captions[row_idx * 2 + col_idx],
                transform=panel_ax.transAxes,
                ha="center",
                va="top",
                fontsize=TEXT_POSITION["panel_caption_size"],
            )

    if SHOW_LEGEND:
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=PAPER_RED,
                marker="o",
                linewidth=LINE_STYLE["line_width"],
                markersize=LINE_STYLE["delay_marker_size"],
                label="Delay",
            ),
            plt.Line2D(
                [0],
                [0],
                color=PAPER_TEAL,
                marker="s",
                linewidth=LINE_STYLE["line_width"],
                markersize=LINE_STYLE["load_marker_size"],
                label="Load variance",
            ),
            plt.Line2D(
                [0],
                [0],
                color=DEFAULT_LINE,
                linestyle="--",
                linewidth=LINE_STYLE["default_line_width"],
                label="Default",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=True,
            framealpha=1.0,
            edgecolor="#CCCCCC",
            fancybox=False,
            **LEGEND_STYLE,
        )
    fig.subplots_adjust(**SUBPLOT_LAYOUT)
    png = FIGURES_DIR / "r1_2_degradation_univariate_trends.png"
    pdf = FIGURES_DIR / "r1_2_degradation_univariate_trends.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def format_table(subset):
    lines = [
        "| value | Mean full-flow delay | Delta vs default | Load variance | Delta vs default | Min f |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in subset.sort_values("sweep_value").iterrows():
        lines.append(
            f"| {row['sweep_value']:.2f} | {row['mean_fullflow_delay']:.3f} | "
            f"{row['delay_delta_pct_vs_default']:+.2f}% | {row['load_variance']:.6f} | "
            f"{row['load_variance_delta_pct_vs_default']:+.1f}% | "
            f"{row['min_effective_capacity_ratio']:.3f} |"
        )
    return lines


def write_summary(df, png, pdf):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = RESULTS_DIR / "r1_2_degradation_univariate_trends_summary.md"
    lines = [
        "# R1-2 退化函数参数单变量趋势实验",
        "",
        "日期：2026-07-24",
        "",
        "## 实验目的",
        "",
        "该实验直接回应审稿人关于“报告平均时延和负载方差随退化函数参数变化趋势”的要求。与 B01/B02/B05/B07 代表组合表不同，本实验每次只改变一个参数，固定另外两个参数和部署/路由评估口径。",
        "",
        "默认参数固定为：`rho_th=0.7,varphi=1.0,f_min=0.2`。",
        "",
        "固定内容：PeMS03 300 个 time slots、50 MEC 拓扑、默认 FPDWGC 部署、full-flow distance-weighted routing evaluator、同一 load scale。",
        "",
        "## 1. rho_th 单变量趋势",
        "",
    ]
    lines.extend(format_table(df[df["sweep_parameter"] == "rho_th"]))
    lines.extend(
        [
            "",
            "读法：`rho_th` 越小，容量退化越早触发，因此平均时延和负载方差整体更高；`rho_th` 提高后，退化触发更晚，时延和负载方差下降。默认 `rho_th=0.7` 位于中间区域。",
            "",
            "## 2. varphi 单变量趋势",
            "",
        ]
    )
    lines.extend(format_table(df[df["sweep_parameter"] == "varphi"]))
    lines.extend(
        [
            "",
            "读法：`varphi` 越大，超过阈值后的容量退化越陡，平均时延和负载方差随之上升。该参数对负载方差尤其敏感。",
            "",
            "## 3. f_min 单变量趋势",
            "",
        ]
    )
    lines.extend(format_table(df[df["sweep_parameter"] == "f_min"]))
    lines.extend(
        [
            "",
            "读法：`f_min` 越大，有效容量下界越高，高负载时队列/执行时间被限制得更强，因此平均时延下降；负载方差在当前负载分布下变化较小，说明 `f_min` 主要提供最坏情况保护。",
            "",
            "## 4. 是否更适合放论文",
            "",
            "是。相比原来的代表组合图，这组单变量趋势更直接，也更符合审稿人“随参数变化的趋势”的措辞。建议正文主图优先使用 `r1_2_degradation_univariate_trends`，原 B01/B02/B05/B07 表作为补充验证或附录材料。",
            "",
            "## 5. 建议正文/回复表述",
            "",
            "> We further conduct a univariate sensitivity study for the bounded degradation function. With the topology, deployment, traffic trace, and routing evaluator fixed, we sweep one parameter at a time while keeping the other two at their default values. The results show that earlier degradation triggering and a steeper degradation slope increase both the mean full-flow delay and load variance, whereas a larger capacity floor mainly reduces high-load delay while keeping the load-variance trend stable.",
            "",
            "## 输出文件",
            "",
            "- `results/r1_2_degradation_univariate_trends.csv`",
            f"- `{png.relative_to(EXP_ROOT)}`",
            f"- `{pdf.relative_to(EXP_ROOT)}`",
        ]
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = run_sweeps()
    csv_path = RESULTS_DIR / "r1_2_degradation_univariate_trends.csv"
    df.to_csv(csv_path, index=False)
    png, pdf = plot_trends(df)
    summary = write_summary(df, png, pdf)
    for path in [csv_path, summary, png, pdf]:
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
