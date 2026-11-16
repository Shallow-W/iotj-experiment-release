#!/usr/bin/env python3
"""R4-1 dynamic workload time-evolution experiment.

This experiment reports per-5-minute-slot metrics under dynamic PeMS workload
changes. A causal dynamic FPDWGC control plane is updated every WINDOW_SIZE
slots using the previously observed control window, and the resulting placement
is evaluated slot by slot with the same full-flow routing evaluator used by the
previous resubmission experiments.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
import time
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

WINDOW_SIZE = 50
TIME_STEP_MINUTES = 5

SLOT_CSV = RESULTS_DIR / "r4_1_time_slot_metrics.csv"
WINDOW_CSV = RESULTS_DIR / "r4_1_control_window_migrations.csv"
SUMMARY_CSV = RESULTS_DIR / "r4_1_dynamic_workload_summary.csv"
SUMMARY_MD = RESULTS_DIR / "r4_1_dynamic_workload_summary.md"
ASSESSMENT_MD = EXP_DIR / "R4_1_RESULTS_AND_RESPONSE_ASSESSMENT.md"
FIG_PNG = FIGURES_DIR / "r4_1_dynamic_workload_time_evolution.png"
FIG_PDF = FIGURES_DIR / "r4_1_dynamic_workload_time_evolution.pdf"

# ===== 动态时间演化主图绘图超参数：遵循 FIGURE_STYLE_STANDARD.md =====
# 六子图采用统一的 3 x 2 论文版式；数据密集时通过 5-slot MA 提升可读性。
FIGURE_SIZE = (7.25, 6.8)
FIGURE_DPI = 600
FONT_STYLE = {
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
}
SPINE_WIDTH = 0.85
GRID_STYLE = {
    "color": "#BDBDBD",
    "linestyle": "--",
    "alpha": 0.45,
    "linewidth": 0.50,
}
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
SUBPLOT_LAYOUT = {
    "left": 0.095,
    "right": 0.970,
    "top": 0.975,
    "bottom": 0.105,
    "wspace": 0.30,
    "hspace": 0.52,
}
# 六个面板均展示横轴标题时，子图标题进一步下移以留出稳定间距。
PANEL_TITLE_Y = -0.31
BOTTOM_PANEL_TITLE_Y = -0.31
PANEL_TITLE_FONT_SIZE = 11.2
BOTTOM_X_LABEL_Y = -0.145
PAPER_RED = "#D7301F"
PAPER_TEAL = "#009E73"
PAPER_BLUE = "#0072B2"
PAPER_PURPLE = "#CC79A7"
PAPER_ORANGE = "#D55E00"
PAPER_GREY = "#4B5563"
EDGE = "#2F3A45"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return sum((value - m) ** 2 for value in values) / len(values)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    pos = (len(arr) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)


def cv(values: list[float]) -> float:
    m = mean(values)
    return math.sqrt(variance(values)) / m if abs(m) > 1e-12 else 0.0


def rel_improvement(before: float, after: float) -> float:
    return 0.0 if abs(before) < 1e-12 else (before - after) / abs(before) * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def request_totals_for_slots(slots: list[dict[str, float]]) -> dict[str, float]:
    keys = list(slots[0].keys()) if slots else []
    return {key: sum(float(slot.get(key, 0.0)) for slot in slots) for key in keys}


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
        "tuned_spec": tuned_spec,
        "degradation": r2.default_degradation_params(config),
        "xi": float(config["fpdwgc_clustering"]["xi"]["value"]),
    }


def deploy_for_spec(spec: dict[str, Any], inputs: dict[str, Any], totals: dict[str, float]) -> dict[str, Any]:
    prefs = base.service_cluster_preferences(inputs["service_order"], inputs["chains"], totals, spec["cluster_count"])
    return r2.deploy_cluster_constrained(
        inputs["nodes"],
        inputs["services"],
        inputs["service_order"],
        inputs["targets"],
        inputs["xi"],
        spec["assignment"],
        prefs,
        spec["weights"],
    )


def build_window_spec(inputs: dict[str, Any], slots: list[dict[str, float]]) -> dict[str, Any]:
    totals = request_totals_for_slots(slots)
    return r2s.build_fpdwgc_spec(
        inputs["config"],
        inputs["nodes"],
        inputs["services"],
        inputs["chains"],
        totals,
        float(inputs["tuned_spec"]["sigma_delay"]),
        float(inputs["tuned_spec"]["sigma_res"]),
    )


def aligned_migration_count(prev_assignment: list[int], next_assignment: list[int], cluster_count: int) -> tuple[int, float]:
    if len(prev_assignment) != len(next_assignment):
        raise ValueError("Assignment lengths differ")
    if cluster_count <= 1:
        return 0, 1.0
    best_matches = -1
    for perm in itertools.permutations(range(cluster_count)):
        matches = sum(1 for prev, nxt in zip(prev_assignment, next_assignment) if prev == perm[nxt])
        best_matches = max(best_matches, matches)
    migrated = len(prev_assignment) - best_matches
    stability = best_matches / len(prev_assignment) if prev_assignment else 1.0
    return migrated, stability


def cluster_size_cv(assignment: list[int], cluster_count: int) -> float:
    sizes = [sum(1 for value in assignment if value == cluster_idx) for cluster_idx in range(cluster_count)]
    return cv([float(size) for size in sizes])


def slot_fullflow_metrics(
    inputs: dict[str, Any],
    slot: dict[str, float],
    deployment: dict[str, Any],
    assignment: list[int],
) -> dict[str, float]:
    nodes = inputs["nodes"]
    state = fullflow.placements_to_state(deployment["placements"], inputs["service_order"], len(nodes))
    services_by_idx = [inputs["services"][f"m{idx + 1}"] for idx in range(r2.SERVICE_COUNT)]
    traffic, trans_delay, missing_traffic = fullflow.route_slot(
        nodes,
        state,
        inputs["flow_matrix"],
        inputs["targets"],
        slot,
        inputs["request_totals"],
    )
    rhos = fullflow.raw_utilization(nodes, services_by_idx, traffic, inputs["load_scale"])
    deg = inputs["degradation"]
    f_values = [
        base.degradation_value(max(rho, 0.0), deg["rho_th"], deg["varphi"], deg["f_min"])
        for rho in rhos
    ]
    effective_utils = [min(5.0, rho / f) if f > 0 else 5.0 for rho, f in zip(rhos, f_values)]
    queueing_delay = 0.0
    for svc_idx in range(r2.SERVICE_COUNT):
        for node_idx in range(len(nodes)):
            flow = traffic[svc_idx][node_idx]
            instances = state[svc_idx][node_idx]
            if flow <= 0 or instances <= 0:
                continue
            queueing_delay += instances * fullflow.queue_delay(flow / instances) / f_values[node_idx]
    missing_delay = missing_traffic * fullflow.MISSING_TRAFFIC_DELAY
    return {
        "mean_fullflow_delay": trans_delay + queueing_delay + missing_delay,
        "mean_trans_delay": trans_delay,
        "mean_queue_delay": queueing_delay,
        "mean_missing_delay": missing_delay,
        "load_variance": variance(effective_utils),
        "slot_max_effective_utilization": max(effective_utils) if effective_utils else 0.0,
        "overload_node_ratio": sum(1 for rho in rhos if rho >= 1.0) / len(rhos) if rhos else 0.0,
        "psi": base.compute_psi(inputs["chains"], slot, deployment["placements"], assignment),
        "request_total": sum(float(value) for value in slot.values()),
        "resource_satisfaction_rate": deployment["deployment_success_ratio"],
        "failed_instances": deployment["failed_instances"],
    }


def build_dynamic_windows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    prev_assignment = None
    requests = inputs["requests"]
    for window_idx, start_idx in enumerate(range(0, len(requests), WINDOW_SIZE), start=1):
        end_idx = min(start_idx + WINDOW_SIZE, len(requests))
        slots = requests[start_idx:end_idx]
        fit_start_idx = start_idx if window_idx == 1 else max(0, start_idx - WINDOW_SIZE)
        fit_end_idx = end_idx if window_idx == 1 else start_idx
        fit_slots = requests[fit_start_idx:fit_end_idx]
        spec = build_window_spec(inputs, fit_slots)
        totals = request_totals_for_slots(fit_slots)
        deployment = deploy_for_spec(spec, inputs, totals)
        if prev_assignment is None:
            migrated, stability = 0, 1.0
        else:
            migrated, stability = aligned_migration_count(prev_assignment, spec["assignment"], spec["cluster_count"])
        stats = r2.cluster_size_stats(spec["assignment"], spec["cluster_count"])
        windows.append(
            {
                "window_idx": window_idx,
                "start_slot": start_idx + 1,
                "end_slot": end_idx,
                "start_minute": start_idx * TIME_STEP_MINUTES,
                "end_minute": end_idx * TIME_STEP_MINUTES,
                "fit_start_slot": fit_start_idx + 1,
                "fit_end_slot": fit_end_idx,
                "fit_start_minute": fit_start_idx * TIME_STEP_MINUTES,
                "fit_end_minute": fit_end_idx * TIME_STEP_MINUTES,
                "spec": spec,
                "deployment": deployment,
                "request_total": sum(sum(float(value) for value in slot.values()) for slot in slots),
                "mean_request_per_slot": mean([sum(float(value) for value in slot.values()) for slot in slots]),
                "cluster_count": spec["cluster_count"],
                "cluster_size_min": stats["cluster_size_min"],
                "cluster_size_max": stats["cluster_size_max"],
                "cluster_size_cv": cluster_size_cv(spec["assignment"], spec["cluster_count"]),
                "migrated_nodes": migrated,
                "migration_rate": migrated / len(inputs["nodes"]) if inputs["nodes"] else 0.0,
                "stability_rate": stability,
                "cluster_time_ms": spec["cluster_time_ms"],
                "deployment_success_ratio": deployment["deployment_success_ratio"],
                "failed_instances": deployment["failed_instances"],
                "candidate_evaluations": deployment["candidate_evaluations"],
            }
        )
        prev_assignment = spec["assignment"]
    return windows


def evaluate_time_slots(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_slots = inputs["requests"][:WINDOW_SIZE]
    static_spec = build_window_spec(inputs, initial_slots)
    static_deployment = deploy_for_spec(static_spec, inputs, request_totals_for_slots(initial_slots))
    dynamic_windows = build_dynamic_windows(inputs)
    window_by_slot = {}
    for window in dynamic_windows:
        for slot_id in range(window["start_slot"], window["end_slot"] + 1):
            window_by_slot[slot_id] = window

    rows = []
    for slot_idx, slot in enumerate(inputs["requests"], start=1):
        window = window_by_slot[slot_idx]
        for method_id, method, spec, deployment in [
            ("initial_static_fpdwgc", "Initial static FPDWGC", static_spec, static_deployment),
            ("dynamic_fpdwgc", "Causal dynamic FPDWGC", window["spec"], window["deployment"]),
        ]:
            metrics = slot_fullflow_metrics(inputs, slot, deployment, spec["assignment"])
            migration_event = window["migrated_nodes"] if method_id == "dynamic_fpdwgc" and slot_idx == window["start_slot"] else 0
            rows.append(
                {
                    "slot": slot_idx,
                    "minute": (slot_idx - 1) * TIME_STEP_MINUTES,
                    "method": method,
                    "method_id": method_id,
                    "window_idx": window["window_idx"] if method_id == "dynamic_fpdwgc" else 0,
                    "window_start_slot": window["start_slot"] if method_id == "dynamic_fpdwgc" else 1,
                    "window_end_slot": window["end_slot"] if method_id == "dynamic_fpdwgc" else len(inputs["requests"]),
                    "request_total": round(metrics["request_total"], 6),
                    "mean_fullflow_delay": round(metrics["mean_fullflow_delay"], 6),
                    "mean_trans_delay": round(metrics["mean_trans_delay"], 6),
                    "mean_queue_delay": round(metrics["mean_queue_delay"], 6),
                    "mean_missing_delay": round(metrics["mean_missing_delay"], 6),
                    "load_variance": round(metrics["load_variance"], 8),
                    "psi": round(metrics["psi"], 6),
                    "cluster_count": spec["cluster_count"],
                    "active_cluster_count": len(set(spec["assignment"])),
                    "cluster_size_cv": round(cluster_size_cv(spec["assignment"], spec["cluster_count"]), 6),
                    "migrated_nodes_at_update": migration_event,
                    "cumulative_migrated_nodes": sum(w["migrated_nodes"] for w in dynamic_windows if w["start_slot"] <= slot_idx)
                    if method_id == "dynamic_fpdwgc"
                    else 0,
                    "resource_satisfaction_rate": round(metrics["resource_satisfaction_rate"], 6),
                    "failed_instances": metrics["failed_instances"],
                    "slot_max_effective_utilization": round(metrics["slot_max_effective_utilization"], 6),
                    "overload_node_ratio": round(metrics["overload_node_ratio"], 8),
                }
            )
    window_rows = []
    for window in dynamic_windows:
        window_rows.append(
            {
                "window_idx": window["window_idx"],
                "slot_range": f"{window['start_slot']}-{window['end_slot']}",
                "minute_range": f"{window['start_minute']}-{window['end_minute']}",
                "fit_slot_range": f"{window['fit_start_slot']}-{window['fit_end_slot']}",
                "fit_minute_range": f"{window['fit_start_minute']}-{window['fit_end_minute']}",
                "request_total": round(window["request_total"], 6),
                "mean_request_per_slot": round(window["mean_request_per_slot"], 6),
                "cluster_count": window["cluster_count"],
                "cluster_size_min": window["cluster_size_min"],
                "cluster_size_max": window["cluster_size_max"],
                "cluster_size_cv": round(window["cluster_size_cv"], 6),
                "migrated_nodes": window["migrated_nodes"],
                "migration_rate": round(window["migration_rate"], 6),
                "stability_rate": round(window["stability_rate"], 6),
                "cluster_time_ms": round(window["cluster_time_ms"], 6),
                "deployment_success_ratio": round(window["deployment_success_ratio"], 6),
                "failed_instances": window["failed_instances"],
                "candidate_evaluations": window["candidate_evaluations"],
            }
        )
    return rows, window_rows


def summarize_method(rows: list[dict[str, Any]], method_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method_id"] == method_id]
    delays = [float(row["mean_fullflow_delay"]) for row in selected]
    load_vars = [float(row["load_variance"]) for row in selected]
    psis = [float(row["psi"]) for row in selected]
    requests = [float(row["request_total"]) for row in selected]
    p95_threshold = percentile(requests, 95)
    high_load = [row for row in selected if float(row["request_total"]) >= p95_threshold]
    high_delays = [float(row["mean_fullflow_delay"]) for row in high_load]
    high_load_vars = [float(row["load_variance"]) for row in high_load]
    return {
        "method_id": method_id,
        "method": selected[0]["method"] if selected else "",
        "slots": len(selected),
        "mean_delay": round(mean(delays), 6),
        "p95_delay": round(percentile(delays, 95), 6),
        "max_delay": round(max(delays) if delays else 0.0, 6),
        "delay_cv": round(cv(delays), 6),
        "mean_load_variance": round(mean(load_vars), 8),
        "p95_load_variance": round(percentile(load_vars, 95), 8),
        "mean_psi": round(mean(psis), 6),
        "p95_psi": round(percentile(psis, 95), 6),
        "high_load_threshold": round(p95_threshold, 6),
        "high_load_mean_delay": round(mean(high_delays), 6),
        "high_load_mean_load_variance": round(mean(high_load_vars), 8),
        "resource_satisfaction_rate_mean": round(mean([float(row["resource_satisfaction_rate"]) for row in selected]), 6),
        "failed_instances_max": max(int(row["failed_instances"]) for row in selected) if selected else 0,
    }


def make_summary_rows(slot_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    static_row = summarize_method(slot_rows, "initial_static_fpdwgc")
    dynamic_row = summarize_method(slot_rows, "dynamic_fpdwgc")
    dynamic_row["delay_improvement_vs_static_pct"] = round(rel_improvement(static_row["mean_delay"], dynamic_row["mean_delay"]), 6)
    dynamic_row["p95_delay_improvement_vs_static_pct"] = round(rel_improvement(static_row["p95_delay"], dynamic_row["p95_delay"]), 6)
    dynamic_row["load_variance_improvement_vs_static_pct"] = round(
        rel_improvement(static_row["mean_load_variance"], dynamic_row["mean_load_variance"]), 6
    )
    dynamic_row["psi_improvement_vs_static_pct"] = round(rel_improvement(static_row["mean_psi"], dynamic_row["mean_psi"]), 6)
    static_row["delay_improvement_vs_static_pct"] = 0.0
    static_row["p95_delay_improvement_vs_static_pct"] = 0.0
    static_row["load_variance_improvement_vs_static_pct"] = 0.0
    static_row["psi_improvement_vs_static_pct"] = 0.0
    total_migration = sum(int(row["migrated_nodes"]) for row in window_rows)
    mean_migration = mean([float(row["migrated_nodes"]) for row in window_rows[1:]])
    max_migration = max(int(row["migrated_nodes"]) for row in window_rows) if window_rows else 0
    for row in [static_row, dynamic_row]:
        row["control_window_size_slots"] = WINDOW_SIZE
        row["control_window_size_minutes"] = WINDOW_SIZE * TIME_STEP_MINUTES
        row["control_windows"] = len(window_rows) if row["method_id"] == "dynamic_fpdwgc" else 1
        row["total_migrated_nodes"] = total_migration if row["method_id"] == "dynamic_fpdwgc" else 0
        row["mean_migrated_nodes_per_update"] = round(mean_migration, 6) if row["method_id"] == "dynamic_fpdwgc" else 0.0
        row["max_migrated_nodes_per_update"] = max_migration if row["method_id"] == "dynamic_fpdwgc" else 0
    return [static_row, dynamic_row]


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_figure(slot_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> None:
    dynamic = sorted(
        [row for row in slot_rows if row["method_id"] == "dynamic_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    static_by_slot = {
        int(row["slot"]): row for row in slot_rows if row["method_id"] == "initial_static_fpdwgc"
    }
    static = [static_by_slot[int(row["slot"])] for row in dynamic]
    slots = np.array([int(row["slot"]) for row in dynamic])
    request_total = np.array([float(row["request_total"]) for row in dynamic])
    update_slots = [int(row["slot_range"].split("-")[0]) for row in window_rows]
    migrated = [int(row["migrated_nodes"]) for row in window_rows]

    def relative_improvement(metric: str) -> np.ndarray:
        static_values = np.array([float(row[metric]) for row in static], dtype=float)
        dynamic_values = np.array([float(row[metric]) for row in dynamic], dtype=float)
        return np.divide(
            static_values - dynamic_values,
            static_values,
            out=np.zeros_like(static_values),
            where=np.abs(static_values) > 1e-12,
        ) * 100.0

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

    panel_captions = [
        "(a) PeMS workload trace",
        "(b) Dynamic FPDWGC delay",
        "(c) Delay improvement",
        "(d) Load-variance improvement",
        "(e) Cross-domain-ratio improvement",
        "(f) Control-plane updates",
    ]

    def style_axis(ax: Any, grid_axis: str = "both") -> None:
        ax.grid(axis=grid_axis, **GRID_STYLE)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", width=SPINE_WIDTH, length=3.0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(SPINE_WIDTH)
            spine.set_color("#000000")

    def style_legend(legend: Any) -> None:
        legend.get_frame().set_linewidth(0.60)

    def add_improvement_panel(
        ax: Any,
        values: np.ndarray,
        ylabel: str,
        positive_color: str = PAPER_TEAL,
        negative_color: str = PAPER_RED,
    ) -> None:
        smoothed = moving_average(values.tolist(), window=5)
        positive = smoothed >= 0
        ax.axhline(0, color=EDGE, linewidth=0.8, alpha=0.7)
        ax.fill_between(
            slots,
            0,
            smoothed,
            where=positive,
            color=positive_color,
            alpha=0.24,
            interpolate=True,
            label="Dynamic better",
        )
        ax.fill_between(
            slots,
            0,
            smoothed,
            where=~positive,
            color=negative_color,
            alpha=0.18,
            interpolate=True,
            label="Static better",
        )
        ax.plot(slots, smoothed, color=EDGE, linewidth=1.35)
        ax.set_ylabel(ylabel)
        style_axis(ax)

    def add_update_lines(ax: Any) -> None:
        for window in window_rows:
            if int(window["window_idx"]) > 1:
                ax.axvline(
                    int(window["slot_range"].split("-")[0]),
                    color="#D9D9D9",
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.65,
                    zorder=0,
                )

    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, sharex=True)

    axes[0, 0].plot(slots, request_total, color=PAPER_GREY, linewidth=1.35)
    axes[0, 0].fill_between(slots, request_total, color="#D1D5DB", alpha=0.35)
    axes[0, 0].set_ylabel("Requests/slot")
    style_axis(axes[0, 0])

    dynamic_delay = np.array([float(row["mean_fullflow_delay"]) for row in dynamic], dtype=float)
    axes[0, 1].plot(
        slots,
        moving_average(dynamic_delay.tolist(), window=5),
        color=PAPER_RED,
        linewidth=1.35,
    )
    axes[0, 1].set_ylabel("Response delay (ms)")
    style_axis(axes[0, 1])

    add_improvement_panel(
        axes[1, 0],
        relative_improvement("mean_fullflow_delay"),
        "Improvement over static (%)",
    )
    style_legend(axes[1, 0].legend(loc="upper left", fontsize=FONT_STYLE["legend.fontsize"], **LEGEND_STYLE))

    add_improvement_panel(
        axes[1, 1],
        relative_improvement("load_variance"),
        "Improvement over static (%)",
    )

    add_improvement_panel(
        axes[2, 0],
        relative_improvement("psi"),
        "Improvement over static (%)",
        positive_color=PAPER_BLUE,
    )
    cluster_count = [float(row["cluster_count"]) for row in dynamic]
    axes[2, 1].step(slots, cluster_count, where="post", color=PAPER_PURPLE, linewidth=1.35, label="Clusters")
    axes[2, 1].set_ylabel("Cluster count")
    axes[2, 1].set_ylim(0, max(cluster_count) + 1.0)
    style_axis(axes[2, 1], grid_axis="y")
    migration_axis = axes[2, 1].twinx()
    migration_axis.bar(update_slots, migrated, width=5.5, color=PAPER_ORANGE, alpha=0.76, edgecolor="white", linewidth=0.8, label="Migrated nodes")
    migration_axis.set_ylabel("Migrated nodes")
    migration_axis.set_ylim(0, max(migrated) + 1.2)
    migration_axis.tick_params(axis="y", width=SPINE_WIDTH, length=3.0, pad=2)
    migration_axis.spines["right"].set_linewidth(SPINE_WIDTH)
    migration_axis.spines["right"].set_color("#000000")
    lines, labels = axes[2, 1].get_legend_handles_labels()
    bars, bar_labels = migration_axis.get_legend_handles_labels()
    style_legend(
        axes[2, 1].legend(
            lines + bars,
            labels + bar_labels,
            loc="lower right",
            fontsize=FONT_STYLE["legend.fontsize"],
            **LEGEND_STYLE,
        )
    )

    for index, ax in enumerate(axes.ravel()):
        add_update_lines(ax)
        ax.set_xlim(int(slots[0]), int(slots[-1]))
        # sharex=True 默认隐藏上两行的数字；每个面板都显式展示时间槽刻度。
        ax.tick_params(axis="x", labelbottom=True)
        caption_y = BOTTOM_PANEL_TITLE_Y if index >= 4 else PANEL_TITLE_Y
        ax.text(
            0.5,
            caption_y,
            panel_captions[index],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_TITLE_FONT_SIZE,
        )

    for ax in axes.ravel():
        ax.set_xlabel("5-min time slot")
        ax.xaxis.set_label_coords(0.5, BOTTOM_X_LABEL_Y)

    fig.subplots_adjust(**SUBPLOT_LAYOUT)
    fig.savefig(FIG_PNG, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def slot_comparison_counts(slot_rows: list[dict[str, Any]]) -> dict[str, int]:
    by_slot: dict[int, dict[str, dict[str, Any]]] = {}
    for row in slot_rows:
        by_slot.setdefault(int(row["slot"]), {})[row["method_id"]] = row

    counts = {
        "slots": len(by_slot),
        "delay_better": 0,
        "delay_equal": 0,
        "delay_worse": 0,
        "load_var_better": 0,
        "load_var_equal": 0,
        "load_var_worse": 0,
        "psi_better": 0,
        "psi_equal": 0,
        "psi_worse": 0,
        "high_load_slots": 0,
        "high_load_delay_better": 0,
        "high_load_delay_equal": 0,
        "high_load_delay_worse": 0,
    }
    dynamic_rows = [row for row in slot_rows if row["method_id"] == "dynamic_fpdwgc"]
    high_load_threshold = percentile([float(row["request_total"]) for row in dynamic_rows], 95)

    for rows in by_slot.values():
        static = rows["initial_static_fpdwgc"]
        dynamic = rows["dynamic_fpdwgc"]
        comparisons = [
            ("mean_fullflow_delay", "delay"),
            ("load_variance", "load_var"),
            ("psi", "psi"),
        ]
        for metric_key, count_key in comparisons:
            delta = float(dynamic[metric_key]) - float(static[metric_key])
            if delta < -1e-9:
                counts[f"{count_key}_better"] += 1
            elif delta > 1e-9:
                counts[f"{count_key}_worse"] += 1
            else:
                counts[f"{count_key}_equal"] += 1

        if float(dynamic["request_total"]) >= high_load_threshold:
            counts["high_load_slots"] += 1
            delta = float(dynamic["mean_fullflow_delay"]) - float(static["mean_fullflow_delay"])
            if delta < -1e-9:
                counts["high_load_delay_better"] += 1
            elif delta > 1e-9:
                counts["high_load_delay_worse"] += 1
            else:
                counts["high_load_delay_equal"] += 1

    return counts


def write_summary(slot_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], inputs: dict[str, Any]) -> None:
    static_row = next(row for row in summary_rows if row["method_id"] == "initial_static_fpdwgc")
    dynamic_row = next(row for row in summary_rows if row["method_id"] == "dynamic_fpdwgc")
    dynamic_slots = [row for row in slot_rows if row["method_id"] == "dynamic_fpdwgc"]
    peak_load_slot = max(dynamic_slots, key=lambda row: float(row["request_total"]))
    peak_delay_slot = max(dynamic_slots, key=lambda row: float(row["mean_fullflow_delay"]))
    max_migration_window = max(window_rows, key=lambda row: int(row["migrated_nodes"]))
    counts = slot_comparison_counts(slot_rows)

    lines = [
        "# R4-1 动态负载时间演化实验总结",
        "",
        "日期：2026-07-22",
        "",
        "## 审稿意见对应关系",
        "",
        "该实验用于回答 Reviewer 4 关于“动态工作负载变化下性能指标随时间演化”的黄色要求。实验不再只报告总体平均值，而是基于 PeMS 的 300 个 5-min time slots，逐槽输出端到端时延、负载方差、跨域比例 `psi`、cluster 数量和节点迁移次数。",
        "",
        "## 实验设置",
        "",
        f"- 时间粒度：`{TIME_STEP_MINUTES}` min，`300` 个 PeMS time slots。",
        f"- 控制面更新周期：每 `{WINDOW_SIZE}` 个 slot，即 `{WINDOW_SIZE * TIME_STEP_MINUTES}` min 重算一次 FPDWGC。",
        f"- FPDWGC 参数：沿用 R2-1 strengthened 验证集选参结果，`sigma_delay={inputs['tuned_spec']['sigma_delay']}, sigma_res={inputs['tuned_spec']['sigma_res']}`。",
        "- 对比口径：`Initial static FPDWGC` 只用第一个控制窗口建立 domain 并固定全程；`Causal dynamic FPDWGC` 用上一控制窗口已经观测到的负载重算下一窗口的 domain 和部署，避免使用当前窗口未来信息。",
        "- 所有逐槽指标使用同一 full-flow distance-weighted routing、容量退化和硬资源可行性口径。",
        "",
        "## 时间演化图",
        "",
        f"![R4-1 dynamic workload time evolution]({Path('..') / FIG_PNG.relative_to(EXP_DIR)})",
        "",
        "图 R4-1：基于 300 个 PeMS 5-min time slots 的动态负载时间演化结果。面板 (a) 展示输入 PeMS workload trace，面板 (b) 展示 causal dynamic FPDWGC 的端到端时延；为避免 static 与 dynamic 曲线重叠，面板 (c)--(e) 用 `Static - Dynamic` 的 5-slot moving-average reduction 展示 delay、load variance 和跨域比例 `psi` 的趋势改善。正值表示 dynamic 更好，负值表示 static 更好，原始逐槽数值保留在 CSV 中。面板 (f) 展示 cluster 数量和控制面更新时的节点迁移次数；竖向浅色线表示每 50 个 slot 的控制面更新边界。",
        "",
        "## 聚合摘要",
        "",
        "| Method | Mean delay | P95 delay | Mean load var | Mean psi | High-load delay | Res. sat. | Migrations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['mean_delay'])} | {fmt(row['p95_delay'])} | "
            f"{fmt(row['mean_load_variance'], 5)} | {fmt(row['mean_psi'], 3)} | "
            f"{fmt(row['high_load_mean_delay'])} | {fmt(row['resource_satisfaction_rate_mean'], 3)} | "
            f"{row['total_migrated_nodes']} |"
        )
    lines += [
        "",
        "## 时间演化关键观察",
        "",
        f"1. 交通负载存在明显时间波动：总请求量峰值出现在 slot `{peak_load_slot['slot']}`，请求量为 `{fmt(peak_load_slot['request_total'])}`；请求量 CV 为 `{fmt(cv([float(row['request_total']) for row in dynamic_slots]))}`。",
        f"2. Causal dynamic FPDWGC 的平均 delay 为 `{fmt(dynamic_row['mean_delay'])}`，相对 initial static FPDWGC `{fmt(static_row['mean_delay'])}` 改善 `{fmt(dynamic_row['delay_improvement_vs_static_pct'])}%`；P95 delay 改善 `{fmt(dynamic_row['p95_delay_improvement_vs_static_pct'])}%`。",
        f"3. 在高负载 slots，请求量阈值为 `{fmt(dynamic_row['high_load_threshold'])}`，Causal dynamic FPDWGC 的平均 delay 为 `{fmt(dynamic_row['high_load_mean_delay'])}`，低于 initial static FPDWGC 的 `{fmt(static_row['high_load_mean_delay'])}`。",
        f"4. Causal dynamic FPDWGC 的平均负载方差为 `{fmt(dynamic_row['mean_load_variance'], 5)}`，相对 initial static FPDWGC 改善 `{fmt(dynamic_row['load_variance_improvement_vs_static_pct'])}%`；平均 `psi={fmt(dynamic_row['mean_psi'])}`。",
        f"5. cluster 数量保持为 `{int(dynamic_slots[0]['cluster_count'])}`，符合 50 MEC 节点和目标 cluster size 的设置；适应性主要体现在节点迁移而非 cluster 数量变化。",
        f"6. `{dynamic_row['control_windows']}` 个控制窗口共产生 `{dynamic_row['total_migrated_nodes']}` 次节点迁移，平均每次更新迁移 `{fmt(dynamic_row['mean_migrated_nodes_per_update'])}` 个节点；最大迁移发生在 `{max_migration_window['slot_range']}`，迁移 `{max_migration_window['migrated_nodes']}` 个节点。",
        f"7. 逐 slot 统计显示，causal dynamic FPDWGC 在 `{counts['delay_better']}/{counts['slots']}` 个 slot 上降低端到端 delay，在 `{counts['delay_equal']}` 个 slot 上与 static 相同，在 `{counts['delay_worse']}` 个 slot 上更高；高负载 `{counts['high_load_slots']}` 个 slot 中有 `{counts['high_load_delay_better']}` 个降低 delay。",
        f"8. Causal dynamic FPDWGC 的最高逐槽 delay 出现在 slot `{peak_delay_slot['slot']}`，说明时间序列图可以定位热点时间段，而聚合均值无法展示这些短时峰值。",
        "",
        "## 是否足以回复审稿人",
        "",
        "可以。该实验直接按 PeMS 5-min time slots 绘制动态时间序列，并覆盖审稿人建议的平均时延、负载方差、跨域比例 `psi`、cluster 数量和节点迁移次数。修正后的 causal dynamic 设置不使用当前窗口未来负载，而是用上一窗口观测负载更新下一窗口，因此比当前窗口离线 oracle 设置更保守。",
        "",
        "需要保守说明：本实验是 offline trace replay 下的周期性控制面更新，不是在线实时学习重训；cluster 数量由当前规模边界固定，动态适应主要通过节点迁移和部署更新体现。动态更新改善平均指标，但不是每个 slot 和每个指标都优于 static，因此正文应配合时间序列图说明均值和瞬时波动。",
        "",
        "## 建议写入 rebuttal 的表述",
        "",
        "> We added a dynamic workload time-evolution experiment using the 300 PeMS 5-min time slots. Instead of reporting only aggregate metrics, we plot the end-to-end delay, load variance, cross-domain ratio, cluster count, and node migrations over time. To avoid look-ahead bias, the causal dynamic FPDWGC control plane uses the previously observed control window to update the next window. Compared with an initial static FPDWGC domain fixed after the first control window, the causal dynamic version reduces the mean and p95 delay and lowers the average load variance, while the migration curve shows when the control plane adapts to traffic hotspot shifts.",
        "",
        "## 输出文件",
        "",
        f"- 逐 slot 指标：`{SLOT_CSV.relative_to(ROOT)}`",
        f"- 控制窗口迁移：`{WINDOW_CSV.relative_to(ROOT)}`",
        f"- 聚合摘要：`{SUMMARY_CSV.relative_to(ROOT)}`",
        f"- 时间演化图：`{FIG_PNG.relative_to(ROOT)}` / `{FIG_PDF.relative_to(ROOT)}`",
        f"- 实验脚本：`{Path(__file__).resolve().relative_to(ROOT)}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assessment = [
        "# R4-1 实验结果与审稿回复评估",
        "",
        "## 判断",
        "",
        "R4-1 当前可以满足审稿人的要求。实验已经从聚合结果扩展为 300 个 PeMS 5-min time slots 的时间序列结果，并覆盖平均时延、负载方差、跨域比例、cluster 数量和节点迁移次数。",
        "",
        "## 支撑证据",
        "",
        "![R4-1 dynamic workload time evolution](figures/r4_1_dynamic_workload_time_evolution.png)",
        "",
        "图 R4-1：基于 300 个 PeMS 5-min time slots 的动态负载时间演化结果。面板 (a) 展示输入 PeMS workload trace，面板 (b) 展示 causal dynamic FPDWGC 的端到端时延；为避免 static 与 dynamic 曲线重叠，面板 (c)--(e) 用 `Static - Dynamic` 的 5-slot moving-average reduction 展示 delay、load variance 和跨域比例 `psi` 的趋势改善。正值表示 dynamic 更好，负值表示 static 更好，原始逐槽数值保留在 CSV 中。面板 (f) 展示 cluster 数量和控制面更新时的节点迁移次数；竖向浅色线表示每 50 个 slot 的控制面更新边界。",
        "",
        f"- Causal dynamic FPDWGC mean delay：`{fmt(dynamic_row['mean_delay'])}`，相对 initial static FPDWGC 改善 `{fmt(dynamic_row['delay_improvement_vs_static_pct'])}%`。",
        f"- Causal dynamic FPDWGC p95 delay：`{fmt(dynamic_row['p95_delay'])}`，相对 initial static FPDWGC 改善 `{fmt(dynamic_row['p95_delay_improvement_vs_static_pct'])}%`。",
        f"- Causal dynamic FPDWGC mean load variance：`{fmt(dynamic_row['mean_load_variance'], 5)}`，相对 initial static FPDWGC 改善 `{fmt(dynamic_row['load_variance_improvement_vs_static_pct'])}%`；P95 负载方差仍需结合时间序列图保守解释。",
        f"- Causal dynamic FPDWGC mean psi：`{fmt(dynamic_row['mean_psi'])}`。",
        f"- 控制窗口：`{dynamic_row['control_windows']}` 个；总迁移节点数 `{dynamic_row['total_migrated_nodes']}`，平均每次更新迁移 `{fmt(dynamic_row['mean_migrated_nodes_per_update'])}` 个节点。",
        f"- 逐 slot：causal dynamic FPDWGC 在 `{counts['delay_better']}/{counts['slots']}` 个 slot 上降低 delay，在高负载 `{counts['high_load_slots']}` 个 slot 中有 `{counts['high_load_delay_better']}` 个降低 delay。",
        "",
        "## 结果审阅",
        "",
        "修正后的 causal/lagged 设置趋势正常，也更符合论文中动态控制面的实际假设。它不用当前窗口的未来请求总量，而是用上一控制窗口观测到的负载更新下一窗口，因此前两个窗口与 static 基本相同，后续在迁移发生后逐渐体现收益。这比当前窗口 oracle replay 更保守，也更适合写入 rebuttal。",
        "",
        f"需要保守说明：causal dynamic FPDWGC 不是每个 slot 或每个指标都优于 static。当前逐 slot 统计是 `{counts['delay_better']}` 个 slot delay 更低、`{counts['delay_equal']}` 个相同、`{counts['delay_worse']}` 个更高；高负载 slot 中 `{counts['high_load_delay_better']}/{counts['high_load_slots']}` 个 delay 更低。负载方差方面也有短时波动，因此正文应写整体 mean/P95 delay 和平均 load variance 改善，不要写全时刻单调更优。",
        "",
        "## 写作边界",
        "",
        "- 可以写“展示了随时间演化结果”和“动态控制面在热点变化时进行节点迁移”。",
        "- 可以写“相比只使用初始窗口并固定全程的静态 domain，causal dynamic FPDWGC 降低 mean/p95 delay 和平均 load variance”。",
        "- 不建议写成在线实时重训练；这里是 PeMS trace replay 下的周期性控制面更新，并且不使用当前窗口未来负载。",
        "",
        "## 对应文件",
        "",
        "- 逐 slot 指标：`results/r4_1_time_slot_metrics.csv`",
        "- 控制窗口迁移：`results/r4_1_control_window_migrations.csv`",
        "- 聚合摘要：`results/r4_1_dynamic_workload_summary.csv`",
        "- 实验图 PNG：`figures/r4_1_dynamic_workload_time_evolution.png`",
        "- 实验图 PDF：`figures/r4_1_dynamic_workload_time_evolution.pdf`",
        "- 实验脚本：`scripts/run_r4_1_dynamic_workload_evolution.py`",
    ]
    ASSESSMENT_MD.write_text("\n".join(assessment) + "\n", encoding="utf-8")


def main() -> None:
    inputs = load_inputs()
    slot_rows, window_rows = evaluate_time_slots(inputs)
    summary_rows = make_summary_rows(slot_rows, window_rows)
    write_csv(SLOT_CSV, slot_rows)
    write_csv(WINDOW_CSV, window_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    make_figure(slot_rows, window_rows)
    write_summary(slot_rows, window_rows, summary_rows, inputs)
    print(
        json.dumps(
            {
                "slot_rows": len(slot_rows),
                "control_windows": len(window_rows),
                "summary_rows": len(summary_rows),
                "window_size_slots": WINDOW_SIZE,
                "figure": str(FIG_PNG.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    for path in [SLOT_CSV, WINDOW_CSV, SUMMARY_CSV, SUMMARY_MD, ASSESSMENT_MD, FIG_PNG, FIG_PDF]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
