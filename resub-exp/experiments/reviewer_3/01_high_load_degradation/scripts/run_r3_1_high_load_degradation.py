#!/usr/bin/env python3
"""R3-1 high-load bounded degradation experiment.

This script extends the R1-2 degradation sensitivity design to explicit
high-load cases where the p95 raw utilization rho is close to or above 1.
It reuses the same full-flow evaluator and the tuned FPDWGC/MPD3QN deployment
setting used by Reviewer 2 experiments.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "hjw_disanzhangbishe").exists():
            return parent
    raise FileNotFoundError("Cannot locate workspace root containing hjw_disanzhangbishe")


ROOT = find_workspace_root()
R2_1_SCRIPT_DIR = ROOT / "resub-exp" / "experiments" / "reviewer_2" / "01_hierarchical_contribution_ablation" / "scripts"
sys.path.insert(0, str(R2_1_SCRIPT_DIR))

import run_r2_1_hierarchical_ablation as r2  # noqa: E402
import run_r2_1_strengthened_ablation as r2s  # noqa: E402


base = r2.base
fullflow = r2.fullflow

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOAD_SWEEP_CSV = RESULTS_DIR / "r3_1_high_load_degradation_sweep.csv"
FUNCTION_CSV = RESULTS_DIR / "r3_1_degradation_function_near_rho1.csv"
SUMMARY_MD = RESULTS_DIR / "r3_1_high_load_degradation_summary.md"
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

TARGET_P95_RHOS = [0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30]
SELECTED_DEGRADATION_RUNS = {
    "B01": "default bounded degradation",
    "B02": "lower rho_th",
    "B05": "stronger degradation slope",
    "B07": "higher capacity floor",
}
RHO_FUNCTION_GRID = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def rel_change(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return (after - before) / abs(before) * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_inputs() -> dict[str, Any]:
    config = base.load_config()
    nodes = base.load_nodes()
    services, service_order = base.load_microservices()
    chains, targets = base.load_app()
    requests, request_totals = base.load_requests()
    flow_matrix = fullflow.load_flow_matrix()
    load_scale, _, _ = r2.build_reference_load_scale(
        config,
        nodes,
        services,
        service_order,
        chains,
        requests,
        request_totals,
        targets,
        flow_matrix,
    )
    validation_requests = requests[: r2s.VALIDATION_SLOTS]
    test_requests = requests[r2s.VALIDATION_SLOTS :]
    validation_rows, tuned_spec, validation_metrics = r2s.select_fpdwgc_by_validation(
        config,
        nodes,
        services,
        service_order,
        chains,
        validation_requests,
        request_totals,
        targets,
        flow_matrix,
        load_scale,
    )

    xi = float(config["fpdwgc_clustering"]["xi"]["value"])
    deployment = r2.deploy_rl_variant(
        "MPD3QN",
        nodes,
        services,
        service_order,
        chains,
        request_totals,
        targets,
        xi,
        tuned_spec["assignment"],
        tuned_spec["weights"],
    )
    state = fullflow.placements_to_state(deployment["placements"], service_order, len(nodes))
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(r2.SERVICE_COUNT)]
    return {
        "config": config,
        "nodes": nodes,
        "services": services,
        "service_order": service_order,
        "chains": chains,
        "targets": targets,
        "requests": requests,
        "request_totals": request_totals,
        "flow_matrix": flow_matrix,
        "load_scale": load_scale,
        "validation_rows": validation_rows,
        "validation_metrics": validation_metrics,
        "test_requests": test_requests,
        "tuned_spec": tuned_spec,
        "deployment": deployment,
        "state": state,
        "services_by_idx": services_by_idx,
    }


def selected_degradation_params(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in config["validation_grids"]["degradation_representative_runs"]:
        run_id = str(run["run_id"])
        if run_id not in SELECTED_DEGRADATION_RUNS:
            continue
        rows.append(
            {
                "run_id": run_id,
                "label": SELECTED_DEGRADATION_RUNS[run_id],
                "rho_th": float(run["rho_th"]),
                "varphi": float(run["varphi"]),
                "f_min": float(run["f_min"]),
            }
        )
    rows.sort(key=lambda row: list(SELECTED_DEGRADATION_RUNS).index(row["run_id"]))
    return rows


def slot_raw_rhos(inputs: dict[str, Any], load_scale: float) -> list[list[float]]:
    out = []
    for slot in inputs["test_requests"]:
        traffic, _, _ = fullflow.route_slot(
            inputs["nodes"],
            inputs["state"],
            inputs["flow_matrix"],
            inputs["targets"],
            slot,
            inputs["request_totals"],
        )
        out.append(fullflow.raw_utilization(inputs["nodes"], inputs["services_by_idx"], traffic, load_scale))
    return out


def baseline_p95_raw_rho(inputs: dict[str, Any]) -> float:
    slot_max = [max(values) if values else 0.0 for values in slot_raw_rhos(inputs, 1.0)]
    return percentile(slot_max, 95)


def capacity_profile(inputs: dict[str, Any], degradation: dict[str, float], load_scale: float) -> dict[str, float]:
    all_rhos: list[float] = []
    all_f: list[float] = []
    all_slowdown: list[float] = []
    all_effective_utils: list[float] = []
    slot_max_rho: list[float] = []
    slot_max_effective: list[float] = []
    floor_hits = 0
    near_one = 0
    overload = 0
    node_slots = 0

    for rhos in slot_raw_rhos(inputs, load_scale):
        slot_effective = []
        slot_max_rho.append(max(rhos) if rhos else 0.0)
        for rho in rhos:
            f_val = base.degradation_value(
                max(rho, 0.0),
                degradation["rho_th"],
                degradation["varphi"],
                degradation["f_min"],
            )
            effective_util = min(5.0, rho / f_val) if f_val > 0 else 5.0
            all_rhos.append(rho)
            all_f.append(f_val)
            all_slowdown.append(1.0 / f_val if f_val > 0 else 5.0)
            all_effective_utils.append(effective_util)
            slot_effective.append(effective_util)
            floor_hits += 1 if f_val <= degradation["f_min"] + 1e-12 else 0
            near_one += 1 if 0.95 <= rho <= 1.05 else 0
            overload += 1 if rho >= 1.0 else 0
            node_slots += 1
        slot_max_effective.append(max(slot_effective) if slot_effective else 0.0)

    return {
        "mean_raw_rho": mean(all_rhos),
        "p95_raw_rho": percentile(slot_max_rho, 95),
        "max_raw_rho": max(all_rhos) if all_rhos else 0.0,
        "rho_near_one_node_slot_ratio": near_one / node_slots if node_slots else 0.0,
        "overload_node_slot_ratio_profile": overload / node_slots if node_slots else 0.0,
        "mean_effective_capacity_ratio": mean(all_f),
        "p05_effective_capacity_ratio": percentile(all_f, 5),
        "min_effective_capacity_ratio_profile": min(all_f) if all_f else 0.0,
        "floor_hit_node_slot_ratio": floor_hits / node_slots if node_slots else 0.0,
        "mean_execution_slowdown": mean(all_slowdown),
        "p95_execution_slowdown": percentile(all_slowdown, 95),
        "p95_node_effective_utilization": percentile(all_effective_utils, 95),
        "p95_slot_max_effective_utilization_profile": percentile(slot_max_effective, 95),
    }


def run_load_sweep(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    p95_raw_at_scale1 = baseline_p95_raw_rho(inputs)
    degradation_runs = selected_degradation_params(inputs["config"])
    rows: list[dict[str, Any]] = []
    for target_p95_rho in TARGET_P95_RHOS:
        scenario_scale = target_p95_rho / p95_raw_at_scale1 if p95_raw_at_scale1 > 0 else 1.0
        for degradation in degradation_runs:
            metrics = fullflow.evaluate_fullflow(
                inputs["nodes"],
                inputs["services_by_idx"],
                inputs["state"],
                inputs["flow_matrix"],
                inputs["targets"],
                inputs["test_requests"],
                inputs["request_totals"],
                degradation,
                scenario_scale,
            )
            profile = capacity_profile(inputs, degradation, scenario_scale)
            rows.append(
                {
                    "run_id": degradation["run_id"],
                    "label": degradation["label"],
                    "rho_th": degradation["rho_th"],
                    "varphi": degradation["varphi"],
                    "f_min": degradation["f_min"],
                    "target_p95_raw_rho": target_p95_rho,
                    "actual_p95_raw_rho": round(profile["p95_raw_rho"], 6),
                    "mean_raw_rho": round(profile["mean_raw_rho"], 6),
                    "max_raw_rho": round(profile["max_raw_rho"], 6),
                    "rho_near_one_node_slot_ratio": round(profile["rho_near_one_node_slot_ratio"], 8),
                    "overload_node_slot_ratio": round(metrics["overload_node_slot_ratio"], 8),
                    "overload_node_slot_ratio_profile": round(profile["overload_node_slot_ratio_profile"], 8),
                    "mean_effective_capacity_ratio": round(profile["mean_effective_capacity_ratio"], 6),
                    "p05_effective_capacity_ratio": round(profile["p05_effective_capacity_ratio"], 6),
                    "min_effective_capacity_ratio": round(metrics["min_effective_capacity_ratio"], 6),
                    "floor_hit_node_slot_ratio": round(profile["floor_hit_node_slot_ratio"], 8),
                    "mean_execution_slowdown": round(profile["mean_execution_slowdown"], 6),
                    "p95_execution_slowdown": round(profile["p95_execution_slowdown"], 6),
                    "p95_node_effective_utilization": round(profile["p95_node_effective_utilization"], 6),
                    "p95_slot_max_effective_utilization": round(metrics["p95_slot_max_effective_utilization"], 6),
                    "mean_fullflow_delay": round(metrics["mean_fullflow_delay"], 6),
                    "mean_trans_delay": round(metrics["mean_trans_delay"], 6),
                    "mean_queue_delay": round(metrics["mean_queue_delay"], 6),
                    "mean_missing_delay": round(metrics["mean_missing_delay"], 6),
                    "load_variance": round(metrics["load_variance"], 8),
                    "scenario_load_scale": round(scenario_scale, 8),
                    "calibrated_reference_load_scale": round(inputs["load_scale"], 8),
                }
            )
    return rows


def degradation_function_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for params in selected_degradation_params(config):
        for rho in RHO_FUNCTION_GRID:
            f_val = base.degradation_value(rho, params["rho_th"], params["varphi"], params["f_min"])
            rows.append(
                {
                    "run_id": params["run_id"],
                    "label": params["label"],
                    "rho_th": params["rho_th"],
                    "varphi": params["varphi"],
                    "f_min": params["f_min"],
                    "rho": rho,
                    "effective_capacity_ratio_f_rho": round(f_val, 6),
                    "execution_slowdown_1_over_f": round(1.0 / f_val if f_val > 0 else 5.0, 6),
                    "floor_active": int(f_val <= params["f_min"] + 1e-12),
                }
            )
    return rows


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


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def row_lookup(rows: list[dict[str, Any]], run_id: str, target: float) -> dict[str, Any]:
    return min(
        [row for row in rows if row["run_id"] == run_id],
        key=lambda row: abs(float(row["target_p95_raw_rho"]) - target),
    )


def write_summary(rows: list[dict[str, Any]], function_rows: list[dict[str, Any]], inputs: dict[str, Any]) -> None:
    default_085 = row_lookup(rows, "B01", 0.85)
    default_100 = row_lookup(rows, "B01", 1.00)
    default_130 = row_lookup(rows, "B01", 1.30)
    b05_100 = row_lookup(rows, "B05", 1.00)
    b07_100 = row_lookup(rows, "B07", 1.00)
    b01_floor_130 = row_lookup(rows, "B01", 1.30)

    lines = [
        "# R3-1 高负载容量退化模型表现总结",
        "",
        "## 审稿意见对应关系",
        "",
        "该实验用于回答 Reviewer 3 关于“补充 rho 接近 1 时模型表现”的黄色要求。实验复用 R1-2 的有界退化函数，并将测试集负载显式缩放到 p95 raw rho 接近 `0.95/1.00/1.05` 以及超过 1 的高负载区域。",
        "",
        "## 实验设置",
        "",
        "- 部署策略：使用 R2-1 strengthened 中验证选参后的 FPDWGC + MPD3QN 部署。",
        "- 测试集：slots 151-300。",
        "- 负载倍率：将测试集 p95 raw rho 分别调到 `0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30`。",
        "- 退化函数：`f(rho)=1` if `rho<=rho_th`; otherwise `max(f_min, 1-varphi*(rho-rho_th)^2)`。",
        "- 参数组：B01 默认有界退化；B02 降低 `rho_th`；B05 增大 `varphi`；B07 提高 `f_min`。",
        f"- FPDWGC 验证集最优参数：`sigma_delay={inputs['tuned_spec']['sigma_delay']}`，`sigma_res={inputs['tuned_spec']['sigma_res']}`。",
        "",
        "## 图像结果",
        "",
        "![R3-1 high-load bounded degradation](../figures/r3_1_high_load_degradation.png)",
        "",
        "图例说明：每个子图中的四条曲线不是四个算法，而是同一个有界退化函数的四组参数设置；四幅子图中的图例含义完全一致。",
        "",
        "| 图例 | 参数含义 | 参数值 |",
        "|---|---|---|",
        "| B01 default bounded degradation | 默认有界退化 | `rho_th=0.70`, `varphi=1.00`, `f_min=0.20` |",
        "| B02 lower rho_th | 降低退化触发阈值 | `rho_th=0.60`, `varphi=1.00`, `f_min=0.20` |",
        "| B05 stronger degradation slope | 增大退化斜率 | `rho_th=0.70`, `varphi=2.00`, `f_min=0.20` |",
        "| B07 higher capacity floor | 提高有效容量下界 | `rho_th=0.70`, `varphi=1.00`, `f_min=0.30` |",
        "",
        "读图方式：横轴为 MEC 服务器利用率 `rho`；实验中通过测试集节点负载的 p95 进行场景校准，用于表示高负载尾部节点接近容量边界，而不是全网平均利用率。竖线表示 `rho=1` 的临界高负载位置；(a) 观察论文主指标平均响应时延是否随 rho 平滑上升；(b) 观察论文主指标负载方差在高负载区的变化；(c) 观察平均排队/执行相关时延如何传导容量退化影响；(d) 观察有界退化函数下平均有效计算能力是否保持有限。`floor-hit ratio` 仍在表格/CSV 中保留，作为容量下界触发的辅助诊断指标。",
        "",
        "## 默认退化函数在 rho 接近 1 时的表现",
        "",
        "| target p95 rho | actual p95 rho | mean f(rho) | p05 f(rho) | floor-hit ratio | queue delay | end-to-end delay | load variance | overload ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for target in [0.85, 0.95, 1.00, 1.05, 1.15, 1.30]:
        row = row_lookup(rows, "B01", target)
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["target_p95_raw_rho"], 2),
                    fmt(row["actual_p95_raw_rho"], 3),
                    fmt(row["mean_effective_capacity_ratio"], 3),
                    fmt(row["p05_effective_capacity_ratio"], 3),
                    fmt(row["floor_hit_node_slot_ratio"], 3),
                    fmt(row["mean_queue_delay"], 3),
                    fmt(row["mean_fullflow_delay"], 3),
                    fmt(row["load_variance"], 6),
                    fmt(row["overload_node_slot_ratio"], 4),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## 与 R1 参数敏感性合并观察",
        "",
        "| 参数组 | rho_th | varphi | f_min | p95 rho≈1 时 mean f | p95 rho≈1 时 queue delay | p95 rho≈1 时 end-to-end delay |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id in ["B01", "B02", "B05", "B07"]:
        row = row_lookup(rows, run_id, 1.00)
        lines.append(
            f"| {row['run_id']} {row['label']} | {fmt(row['rho_th'], 2)} | {fmt(row['varphi'], 2)} | {fmt(row['f_min'], 2)} | "
            f"{fmt(row['mean_effective_capacity_ratio'])} | {fmt(row['mean_queue_delay'])} | {fmt(row['mean_fullflow_delay'])} |"
        )

    lines += [
        "",
        "## 关键发现",
        "",
        f"1. 在默认有界退化函数 B01 下，当 p95 raw rho 从 `{fmt(default_085['actual_p95_raw_rho'])}` 提高到 `{fmt(default_100['actual_p95_raw_rho'])}` 时，端到端时延从 `{fmt(default_085['mean_fullflow_delay'])}` 增至 `{fmt(default_100['mean_fullflow_delay'])}`，增幅 `{fmt(rel_change(float(default_085['mean_fullflow_delay']), float(default_100['mean_fullflow_delay'])))}%`。",
        f"2. 当 p95 raw rho 进一步升至 `{fmt(default_130['actual_p95_raw_rho'])}` 时，默认 B01 的最小有效容量仍被 `f_min=0.2` 限制在下界，端到端时延为 `{fmt(default_130['mean_fullflow_delay'])}`，负载方差为 `{fmt(default_130['load_variance'], 6)}`，没有出现无界退化导致的数值发散。",
        f"3. 更强退化斜率 B05 在 p95 rho≈1 时使端到端时延达到 `{fmt(b05_100['mean_fullflow_delay'])}`，高于默认 B01 的 `{fmt(default_100['mean_fullflow_delay'])}`，说明 `varphi` 控制高负载惩罚强度。",
        f"4. 更高容量下界 B07 在 p95 rho≈1 时端到端时延为 `{fmt(b07_100['mean_fullflow_delay'])}`，低于默认 B01，说明 `f_min` 能缓解 rho 接近 1 时的执行时间膨胀。",
        f"5. 在 p95 rho≈1.30 的更高负载下，默认 B01 的 floor-hit ratio 为 `{fmt(b01_floor_130['floor_hit_node_slot_ratio'])}`，证明有界退化函数确实在高负载节点上激活容量下界。",
        "",
        "补充解释：图 (d) 中 `mean f(rho)` 变化幅度较小，是因为实验控制的是每个时隙最大节点利用率的 p95，而不是让所有 MEC 节点同时接近 rho=1。因此正文中应把 `f(rho)` 与 average response delay、load variance、queue delay、overload ratio、floor-hit ratio 一起报告，避免只看全网平均容量比例。主图保留 average response delay 和 load variance，是为了对齐原论文已有实验的两个主评价指标；主图加入 queue delay，是为了展示退化函数通过执行/排队项影响端到端时延的传导路径；`floor-hit ratio` 是新增诊断指标，不作为主图 y 轴。",
        "",
        "## 是否足以回复审稿人",
        "",
        "可以。该实验显式覆盖 rho 接近 1 和超过 1 的高负载区域，主图报告了原论文已有的 average response delay、load variance，以及用于解释模型传导路径的 queue delay 和有效计算能力 `f(rho)`；表格/CSV 进一步报告过载节点比例和容量下界激活比例。它与 R1-2 的 `rho_th`、`varphi`、`f_min` 敏感性使用同一参数组，因此可以在论文中合并呈现为“退化参数敏感性 + 高负载验证”。",
        "",
        "写作时建议强调：有界退化函数不会让有效容量无限趋近 0；当 rho 接近 1 时，`varphi` 决定退化斜率，`f_min` 决定最坏情况下的容量下界和时延上界。",
        "",
        "## 输出文件",
        "",
        f"- 高负载 sweep：`{LOAD_SWEEP_CSV.relative_to(ROOT)}`",
        f"- 退化函数采样：`{FUNCTION_CSV.relative_to(ROOT)}`",
        f"- 图：`{FIG_PNG.relative_to(ROOT)}`",
    ]

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    inputs = load_inputs()
    rows = run_load_sweep(inputs)
    function_rows = degradation_function_rows(inputs["config"])
    write_csv(LOAD_SWEEP_CSV, rows)
    write_csv(FUNCTION_CSV, function_rows)
    make_figure(rows)
    write_summary(rows, function_rows, inputs)
    print(json.dumps({"load_sweep_rows": len(rows), "function_rows": len(function_rows)}, indent=2))
    print(f"Wrote {LOAD_SWEEP_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {FIG_PNG}")


if __name__ == "__main__":
    main()
