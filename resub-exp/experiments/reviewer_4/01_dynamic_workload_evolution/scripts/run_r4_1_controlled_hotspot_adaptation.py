#!/usr/bin/env python3
"""R4-1 PeMS-derived controlled hotspot adaptation experiment.

The experiment preserves the request total in every PeMS slot and applies a
deterministic request-class permutation in five 60-slot phases. The dynamic
controller detects request-profile drift from a trailing observation window
and applies a new FPDWGC partition/deployment in the following slot.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_r4_1_dynamic_workload_evolution as r4  # noqa: E402


EXP_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SLOT_CSV = RESULTS_DIR / "r4_1_controlled_hotspot_slot_metrics.csv"
EVENT_CSV = RESULTS_DIR / "r4_1_controlled_hotspot_update_events.csv"
PHASE_CSV = RESULTS_DIR / "r4_1_controlled_hotspot_phase_summary.csv"
SUMMARY_CSV = RESULTS_DIR / "r4_1_controlled_hotspot_overall_summary.csv"
SUMMARY_MD = RESULTS_DIR / "r4_1_controlled_hotspot_results.md"
INTEGRITY_JSON = RESULTS_DIR / "r4_1_controlled_hotspot_integrity_check.json"
FIG_PNG = FIGURES_DIR / "r4_1_controlled_hotspot_adaptation.png"
FIG_PDF = FIGURES_DIR / "r4_1_controlled_hotspot_adaptation.pdf"

# ---------------------------------------------------------------------------
# 实验超参数：所有更新只使用已观测历史窗口
# ---------------------------------------------------------------------------
OBSERVATION_WINDOW = 10
WARMUP_SLOTS = 60
TV_THRESHOLD = 0.35
POST_UPDATE_WINDOW = 20
TIME_STEP_MINUTES = 5
OBJECTIVE_WEIGHTS = {
    "mean_fullflow_delay": 1.0,
    "load_variance": 0.1,
    "psi": 0.1,
}

# dest <- source。每个时隙只重映射类别，因此请求总量严格保持不变。
REGIMES = [
    {
        "regime_id": "R1",
        "label": "Baseline workload",
        "plot_label": "Baseline",
        "start_slot": 1,
        "end_slot": 60,
        "mapping": {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"},
    },
    {
        "regime_id": "R2",
        "label": "Sustained long-chain hotspot",
        "plot_label": "Sustained hotspot",
        "start_slot": 61,
        "end_slot": 300,
        "mapping": {"a": "d", "b": "c", "c": "a", "d": "b", "e": "e"},
    },
]

# ---------------------------------------------------------------------------
# 绘图超参数：遵循 FIGURE_STYLE_STANDARD.md 的 2 x 2 趋势图规范
# ---------------------------------------------------------------------------
FIGURE_SIZE = (7.20, 5.20)
FIGURE_DPI = 600
SMOOTH_WINDOW = 5
X_TICKS = [1, 60, 120, 180, 240, 300]
FONT_STYLE = {
    "font.size": 11.0,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 9.2,
}
LINE_WIDTH = 1.35
SPINE_WIDTH = 0.85
GRID_STYLE = {"color": "#BDBDBD", "linestyle": "--", "alpha": 0.45, "linewidth": 0.50}
LEGEND_STYLE = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.95,
    "facecolor": "white",
    "edgecolor": "#D9D9D9",
    "borderpad": 0.35,
    "borderaxespad": 0.42,
    "labelspacing": 0.28,
    "handlelength": 1.7,
    "handletextpad": 0.60,
}
PANEL_CAPTION_Y = -0.20
PANEL_CAPTION_SIZE = 13.0
PAPER_RED = "#D7301F"
PAPER_BLUE = "#0072B2"
PAPER_ORANGE = "#D55E00"
PAPER_GREY = "#8A8F98"
PHASE_COLORS = ["#F4F4F4", "#FFF2D8", "#E9F3FA", "#F4F4F4", "#E8F5EE"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def profile(slots: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    totals = {key: sum(float(slot.get(key, 0.0)) for slot in slots) for key in keys}
    total = sum(totals.values())
    return {key: totals[key] / total if total > 0 else 0.0 for key in keys}


def total_variation(left: dict[str, float], right: dict[str, float], keys: list[str]) -> float:
    return 0.5 * sum(abs(left[key] - right[key]) for key in keys)


def make_controlled_requests(requests: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    controlled: list[dict[str, float]] = []
    metadata: list[dict[str, Any]] = []
    for idx, slot in enumerate(requests):
        slot_id = idx + 1
        regime = next(
            item for item in REGIMES
            if int(item["start_slot"]) <= slot_id <= int(item["end_slot"])
        )
        transformed = {
            destination: float(slot.get(source, 0.0))
            for destination, source in regime["mapping"].items()
        }
        if abs(sum(transformed.values()) - sum(float(value) for value in slot.values())) > 1e-9:
            raise AssertionError(f"Request total changed at slot {idx + 1}")
        controlled.append(transformed)
        metadata.append(
            {
                "regime_id": regime["regime_id"],
                "regime_label": regime["label"],
            }
        )
    return controlled, metadata


def deployment_change_count(previous: dict[str, Any], current: dict[str, Any]) -> int:
    changed = 0
    for service_name in previous["placements"]:
        before = sorted(previous["placements"].get(service_name, []))
        after = sorted(current["placements"].get(service_name, []))
        changed += max(len(before), len(after)) - len(set(before) & set(after))
    return changed


def config_fingerprint(spec: dict[str, Any], deployment: dict[str, Any]) -> tuple[Any, ...]:
    placements = tuple(
        (service_name, tuple(sorted(node_indices)))
        for service_name, node_indices in sorted(deployment["placements"].items())
    )
    return tuple(spec["assignment"]), placements


def observed_candidate_metrics(
    inputs: dict[str, Any],
    observed: list[dict[str, float]],
    spec: dict[str, Any],
    deployment: dict[str, Any],
) -> dict[str, float]:
    metrics = [r4.slot_fullflow_metrics(inputs, slot, deployment, spec["assignment"]) for slot in observed]
    return {
        key: float(np.mean([float(row[key]) for row in metrics]))
        for key in OBJECTIVE_WEIGHTS
    }


def select_causal_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """按论文默认归一化目标权重选择历史窗口候选配置。"""
    maxima = {
        key: max(float(candidate["metrics"][key]) for candidate in candidates)
        for key in OBJECTIVE_WEIGHTS
    }
    for candidate in candidates:
        candidate["objective_score"] = sum(
            weight * float(candidate["metrics"][key]) / maxima[key]
            if maxima[key] > 1e-12 else 0.0
            for key, weight in OBJECTIVE_WEIGHTS.items()
        )
    return min(candidates, key=lambda candidate: (candidate["objective_score"], candidate["source"]))


def build_controller_schedule(
    inputs: dict[str, Any],
    requests: list[dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    keys = list(requests[0])
    initial_slots = requests[:OBSERVATION_WINDOW]
    initial_spec = r4.build_window_spec(inputs, initial_slots)
    initial_deployment = r4.deploy_for_spec(initial_spec, inputs, r4.request_totals_for_slots(initial_slots))

    current_spec = initial_spec
    current_deployment = initial_deployment
    reference_profile = profile(initial_slots, keys)
    schedule: dict[int, dict[str, Any]] = {
        1: {"spec": current_spec, "deployment": current_deployment, "event_idx": 0}
    }
    events: list[dict[str, Any]] = []

    for detected_slot in range(
        WARMUP_SLOTS + OBSERVATION_WINDOW,
        len(requests) + 1,
        OBSERVATION_WINDOW,
    ):
        observed = requests[detected_slot - OBSERVATION_WINDOW : detected_slot]
        observed_profile = profile(observed, keys)
        tv_distance = total_variation(reference_profile, observed_profile, keys)

        observed_spec = r4.build_window_spec(inputs, observed)
        observed_deployment = r4.deploy_for_spec(observed_spec, inputs, r4.request_totals_for_slots(observed))
        raw_candidates = [
            {"source": "current", "spec": current_spec, "deployment": current_deployment},
            {"source": "initial", "spec": initial_spec, "deployment": initial_deployment},
            {"source": "observed_window", "spec": observed_spec, "deployment": observed_deployment},
        ]
        candidates: list[dict[str, Any]] = []
        fingerprints: set[tuple[Any, ...]] = set()
        for candidate in raw_candidates:
            fingerprint = config_fingerprint(candidate["spec"], candidate["deployment"])
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            candidate["metrics"] = observed_candidate_metrics(
                inputs,
                observed,
                candidate["spec"],
                candidate["deployment"],
            )
            candidates.append(candidate)
        selected = select_causal_candidate(candidates)
        next_spec = selected["spec"]
        next_deployment = selected["deployment"]
        current_fingerprint = config_fingerprint(current_spec, current_deployment)
        next_fingerprint = config_fingerprint(next_spec, next_deployment)

        trigger_type = "profile_shift" if tv_distance > TV_THRESHOLD else "periodic_reoptimization"
        reference_profile = observed_profile
        if next_fingerprint == current_fingerprint:
            continue
        migrated_nodes, stability = r4.aligned_migration_count(
            current_spec["assignment"], next_spec["assignment"], next_spec["cluster_count"]
        )
        changed_instances = deployment_change_count(current_deployment, next_deployment)
        applied_slot = detected_slot + 1
        if applied_slot > len(requests):
            break
        event_idx = len(events) + 1
        events.append(
            {
                "event_idx": event_idx,
                "detected_slot": detected_slot,
                "applied_slot": applied_slot,
                "observation_start_slot": detected_slot - OBSERVATION_WINDOW + 1,
                "observation_end_slot": detected_slot,
                "tv_distance": round(tv_distance, 6),
                "threshold": TV_THRESHOLD,
                "trigger_type": trigger_type,
                "selected_source": selected["source"],
                "objective_score": round(float(selected["objective_score"]), 6),
                "observed_delay_ms": round(float(selected["metrics"]["mean_fullflow_delay"]), 6),
                "observed_load_variance": round(float(selected["metrics"]["load_variance"]), 8),
                "observed_psi": round(float(selected["metrics"]["psi"]), 6),
                "migrated_nodes": migrated_nodes,
                "migration_rate": round(migrated_nodes / len(inputs["nodes"]), 6),
                "partition_stability": round(stability, 6),
                "changed_service_instances": changed_instances,
                "cluster_count": next_spec["cluster_count"],
                "cluster_time_ms": round(float(next_spec["cluster_time_ms"]), 6),
                "deployment_success_ratio": round(float(next_deployment["deployment_success_ratio"]), 6),
            }
        )
        schedule[applied_slot] = {"spec": next_spec, "deployment": next_deployment, "event_idx": event_idx}
        current_spec = next_spec
        current_deployment = next_deployment

    return initial_spec, initial_deployment, schedule, events


def evaluate() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = r4.load_inputs()
    controlled_requests, metadata = make_controlled_requests(inputs["requests"])
    controlled_totals = r4.request_totals_for_slots(controlled_requests)
    eval_inputs = dict(inputs)
    eval_inputs["requests"] = controlled_requests
    eval_inputs["request_totals"] = controlled_totals

    static_spec, static_deployment, schedule, events = build_controller_schedule(eval_inputs, controlled_requests)
    dynamic_spec = static_spec
    dynamic_deployment = static_deployment
    active_event_idx = 0
    cumulative_migrations = 0
    rows: list[dict[str, Any]] = []

    for slot_idx, (slot, meta) in enumerate(zip(controlled_requests, metadata), start=1):
        migration_event = 0
        if slot_idx in schedule:
            scheduled = schedule[slot_idx]
            dynamic_spec = scheduled["spec"]
            dynamic_deployment = scheduled["deployment"]
            active_event_idx = int(scheduled["event_idx"])
            if active_event_idx > 0:
                migration_event = int(events[active_event_idx - 1]["migrated_nodes"])
                cumulative_migrations += migration_event

        for method_id, method, spec, deployment in [
            ("initial_static_fpdwgc", "Initial static FPDWGC", static_spec, static_deployment),
            ("causal_dynamic_fpdwgc", "Causal dynamic FPDWGC", dynamic_spec, dynamic_deployment),
        ]:
            metrics = r4.slot_fullflow_metrics(eval_inputs, slot, deployment, spec["assignment"])
            long_chain_requests = sum(float(slot[key]) for key in ("a", "b", "c"))
            request_total = float(metrics["request_total"])
            rows.append(
                {
                    "slot": slot_idx,
                    "minute": (slot_idx - 1) * TIME_STEP_MINUTES,
                    "regime_id": meta["regime_id"],
                    "regime_label": meta["regime_label"],
                    "method_id": method_id,
                    "method": method,
                    "active_event_idx": active_event_idx if method_id == "causal_dynamic_fpdwgc" else 0,
                    "request_total": round(request_total, 6),
                    "long_chain_request_share": round(long_chain_requests / request_total if request_total > 0 else 0.0, 6),
                    "mean_fullflow_delay": round(float(metrics["mean_fullflow_delay"]), 6),
                    "load_variance": round(float(metrics["load_variance"]), 8),
                    "psi": round(float(metrics["psi"]), 6),
                    "cluster_count": spec["cluster_count"],
                    "migrated_nodes_at_update": migration_event if method_id == "causal_dynamic_fpdwgc" else 0,
                    "cumulative_migrated_nodes": cumulative_migrations if method_id == "causal_dynamic_fpdwgc" else 0,
                    "resource_satisfaction_rate": round(float(metrics["resource_satisfaction_rate"]), 6),
                }
            )

    phase_rows = make_phase_summary(rows, events)
    overall_rows = make_overall_summary(rows, events)
    return rows, events, phase_rows, overall_rows


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def improvement(static_value: float, dynamic_value: float) -> float:
    return (static_value - dynamic_value) / static_value * 100.0 if abs(static_value) > 1e-12 else 0.0


def make_phase_summary(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_rows: list[dict[str, Any]] = []
    for regime_idx, regime in enumerate(REGIMES):
        start_slot = int(regime["start_slot"])
        end_slot = int(regime["end_slot"])
        static = [
            row for row in rows
            if row["regime_id"] == regime["regime_id"] and row["method_id"] == "initial_static_fpdwgc"
        ]
        dynamic = [
            row for row in rows
            if row["regime_id"] == regime["regime_id"] and row["method_id"] == "causal_dynamic_fpdwgc"
        ]
        regime_events = [item for item in events if start_slot <= int(item["applied_slot"]) <= end_slot]
        event = regime_events[0] if regime_events else None
        post_start = int(event["applied_slot"]) if event else start_slot
        post_end = min(post_start + POST_UPDATE_WINDOW - 1, end_slot)
        post_static = [row for row in static if post_start <= int(row["slot"]) <= post_end]
        post_dynamic = [row for row in dynamic if post_start <= int(row["slot"]) <= post_end]

        static_delay = metric_mean(static, "mean_fullflow_delay")
        dynamic_delay = metric_mean(dynamic, "mean_fullflow_delay")
        static_load = metric_mean(static, "load_variance")
        dynamic_load = metric_mean(dynamic, "load_variance")
        static_psi = metric_mean(static, "psi")
        dynamic_psi = metric_mean(dynamic, "psi")
        phase_rows.append(
            {
                "regime_id": regime["regime_id"],
                "regime_label": regime["label"],
                "slot_range": f"{start_slot}-{end_slot}",
                "update_count": len(regime_events),
                "update_applied_slot": int(event["applied_slot"]) if event else 0,
                "adaptation_latency_slots": int(event["applied_slot"]) - start_slot if event and regime_idx > 0 else 0,
                "migrated_nodes": sum(int(item["migrated_nodes"]) for item in regime_events),
                "static_mean_delay_ms": round(static_delay, 6),
                "dynamic_mean_delay_ms": round(dynamic_delay, 6),
                "delay_improvement_pct": round(improvement(static_delay, dynamic_delay), 6),
                "static_mean_load_variance": round(static_load, 8),
                "dynamic_mean_load_variance": round(dynamic_load, 8),
                "load_variance_improvement_pct": round(improvement(static_load, dynamic_load), 6),
                "static_mean_psi": round(static_psi, 6),
                "dynamic_mean_psi": round(dynamic_psi, 6),
                "psi_improvement_pct": round(improvement(static_psi, dynamic_psi), 6),
                "post_update_slot_range": f"{post_start}-{post_end}",
                "post_update_delay_improvement_pct": round(
                    improvement(
                        metric_mean(post_static, "mean_fullflow_delay"),
                        metric_mean(post_dynamic, "mean_fullflow_delay"),
                    ),
                    6,
                ),
                "post_update_load_variance_improvement_pct": round(
                    improvement(metric_mean(post_static, "load_variance"), metric_mean(post_dynamic, "load_variance")),
                    6,
                ),
                "post_update_psi_improvement_pct": round(
                    improvement(metric_mean(post_static, "psi"), metric_mean(post_dynamic, "psi")),
                    6,
                ),
            }
        )
    return phase_rows


def make_overall_summary(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method_id in ("initial_static_fpdwgc", "causal_dynamic_fpdwgc"):
        selected = [row for row in rows if row["method_id"] == method_id]
        output.append(
            {
                "method_id": method_id,
                "method": selected[0]["method"],
                "mean_delay_ms": round(metric_mean(selected, "mean_fullflow_delay"), 6),
                "p95_delay_ms": round(float(np.percentile([float(row["mean_fullflow_delay"]) for row in selected], 95)), 6),
                "mean_load_variance": round(metric_mean(selected, "load_variance"), 8),
                "mean_psi": round(metric_mean(selected, "psi"), 6),
                "update_events": len(events) if method_id == "causal_dynamic_fpdwgc" else 0,
                "total_migrated_nodes": sum(int(event["migrated_nodes"]) for event in events)
                if method_id == "causal_dynamic_fpdwgc" else 0,
                "mean_adaptation_latency_slots": round(
                    float(events[0]["applied_slot"]) - float(REGIMES[1]["start_slot"]),
                    6,
                ) if events and method_id == "causal_dynamic_fpdwgc" else 0.0,
            }
        )
    return output


def trailing_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    cumulative = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result[idx] = (cumulative[idx + 1] - cumulative[start]) / (idx - start + 1)
    return result


def set_paper_style() -> None:
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


def style_axis(ax: Any) -> None:
    ax.grid(axis="both", **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=SPINE_WIDTH, length=3.0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#000000")


def add_panel_caption(ax: Any, caption: str) -> None:
    ax.text(
        0.5,
        PANEL_CAPTION_Y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_CAPTION_SIZE,
    )


def add_phase_context(ax: Any, show_labels: bool = False) -> None:
    for idx, regime in enumerate(REGIMES):
        start = int(regime["start_slot"])
        end = int(regime["end_slot"])
        ax.axvspan(start, end, color=PHASE_COLORS[idx], alpha=0.55, linewidth=0, zorder=0)
        if idx > 0:
            ax.axvline(start, color="#555555", linestyle=":", linewidth=0.85, zorder=1)
        if show_labels:
            ax.text(
                (start + end) / 2,
                0.985,
                regime["plot_label"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.8,
                color="#333333",
            )


def plot_metric_panel(
    ax: Any,
    slots: np.ndarray,
    static_values: np.ndarray,
    dynamic_values: np.ndarray,
    events: list[dict[str, Any]],
    ylabel: str,
    caption: str,
    legend_loc: str | None = None,
) -> None:
    add_phase_context(ax)
    ax.plot(
        slots,
        trailing_average(static_values, SMOOTH_WINDOW),
        color=PAPER_BLUE,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Initial static FPDWGC",
        zorder=3,
    )
    ax.plot(
        slots,
        trailing_average(dynamic_values, SMOOTH_WINDOW),
        color=PAPER_RED,
        linestyle="-",
        linewidth=LINE_WIDTH,
        label="Causal dynamic FPDWGC",
        zorder=4,
    )
    for event in events:
        ax.axvline(int(event["applied_slot"]), color=PAPER_ORANGE, linestyle="-.", linewidth=0.85, alpha=0.80)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time slot")
    ax.set_xlim(1, 300)
    ax.set_xticks(X_TICKS)
    style_axis(ax)
    if legend_loc:
        legend = ax.legend(loc=legend_loc, ncol=1, **LEGEND_STYLE)
        legend.get_frame().set_linewidth(0.60)
    add_panel_caption(ax, caption)


def make_figure(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    static = sorted(
        [row for row in rows if row["method_id"] == "initial_static_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    dynamic = sorted(
        [row for row in rows if row["method_id"] == "causal_dynamic_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    slots = np.array([int(row["slot"]) for row in dynamic], dtype=int)

    set_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)

    request_total = np.array([float(row["request_total"]) for row in dynamic])
    ax = axes[0, 0]
    add_phase_context(ax, show_labels=True)
    request_line = ax.plot(
        slots,
        trailing_average(request_total, SMOOTH_WINDOW),
        color=PAPER_GREY,
        linestyle="--",
        linewidth=1.10,
        label="PeMS request volume",
        zorder=3,
    )[0]
    ax.set_ylabel("Requests per slot")
    ax.set_xlabel("Time slot")
    ax.set_xlim(1, 300)
    ax.set_xticks(X_TICKS)
    style_axis(ax)

    update_slots = [int(event["applied_slot"]) for event in events]
    request_trend = trailing_average(request_total, SMOOTH_WINDOW)
    update_values = [request_trend[slot - 1] for slot in update_slots]
    update_markers = ax.scatter(
        update_slots,
        update_values,
        marker="^",
        s=34,
        color=PAPER_RED,
        edgecolor="white",
        linewidth=0.55,
        label="Applied update",
        zorder=6,
    )
    hotspot_line = ax.axvline(
        int(REGIMES[1]["start_slot"]),
        color=PAPER_ORANGE,
        linestyle=":",
        linewidth=1.05,
        label="Hotspot onset",
        zorder=5,
    )
    legend = ax.legend(
        [request_line, hotspot_line, update_markers],
        ["PeMS request volume", "Hotspot onset", "Applied update"],
        loc="lower right",
        ncol=1,
        **LEGEND_STYLE,
    )
    legend.get_frame().set_linewidth(0.60)
    add_panel_caption(ax, "(a) Workload and hotspot shifts")

    plot_metric_panel(
        axes[0, 1],
        slots,
        np.array([float(row["mean_fullflow_delay"]) for row in static]),
        np.array([float(row["mean_fullflow_delay"]) for row in dynamic]),
        events,
        "End-to-end delay (ms)",
        "(b) Dynamic response delay",
        legend_loc="upper right",
    )
    plot_metric_panel(
        axes[1, 0],
        slots,
        np.array([float(row["load_variance"]) for row in static]),
        np.array([float(row["load_variance"]) for row in dynamic]),
        events,
        "Load variance",
        "(c) Dynamic load balance",
    )
    plot_metric_panel(
        axes[1, 1],
        slots,
        np.array([float(row["psi"]) for row in static]),
        np.array([float(row["psi"]) for row in dynamic]),
        events,
        r"Cross-domain ratio $\psi$",
        "(d) Dynamic cross-domain ratio",
    )

    fig.subplots_adjust(left=0.090, right=0.975, top=0.965, bottom=0.130, wspace=0.31, hspace=0.39)
    fig.savefig(FIG_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(
    events: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
) -> None:
    static = next(row for row in overall_rows if row["method_id"] == "initial_static_fpdwgc")
    dynamic = next(row for row in overall_rows if row["method_id"] == "causal_dynamic_fpdwgc")
    useful_phases = [
        row for row in phase_rows[1:]
        if float(row["post_update_delay_improvement_pct"]) > 0
        or float(row["post_update_load_variance_improvement_pct"]) > 0
        or float(row["post_update_psi_improvement_pct"]) > 0
    ]
    lines = [
        "# R4-1 可控热点切换实验结果",
        "",
        "## 实验口径",
        "",
        "- 数据为 PeMS-derived controlled stress replay：保留每个 5-min 时隙的请求总量，通过确定性类别重映射构造服务链热点切换。",
        "- 固定重映射为 `a<-d, b<-c, c<-a, d<-b, e<-e`；该口径将主导的 `d` 类请求转移到长服务链 `a`，并保持每个时隙总量不变。",
        f"- 前 `{WARMUP_SLOTS}` slots 为基线期；之后每 `{OBSERVATION_WINDOW}` slots 使用历史窗口重评候选，更新在下一时隙生效。",
        f"- TV 阈值 `{TV_THRESHOLD}` 用于区分热点漂移触发与周期性重优化触发，不用于访问未来数据。",
        "- Static 和 Dynamic 使用相同全流量路由、资源配置、退化函数和初始部署。",
        "",
        "## 总体结果",
        "",
        f"- 动态控制面触发 `{len(events)}` 次更新，总迁移 `{sum(int(event['migrated_nodes']) for event in events)}` 个节点。",
        f"- 平均时延：Static `{static['mean_delay_ms']:.3f}` ms，Dynamic `{dynamic['mean_delay_ms']:.3f}` ms，改善 `{improvement(float(static['mean_delay_ms']), float(dynamic['mean_delay_ms'])):.2f}%`。",
        f"- 平均负载方差：Static `{static['mean_load_variance']:.4f}`，Dynamic `{dynamic['mean_load_variance']:.4f}`，改善 `{improvement(float(static['mean_load_variance']), float(dynamic['mean_load_variance'])):.2f}%`。",
        f"- 平均跨域比例：Static `{static['mean_psi']:.4f}`，Dynamic `{dynamic['mean_psi']:.4f}`，改善 `{improvement(float(static['mean_psi']), float(dynamic['mean_psi'])):.2f}%`。",
        "",
        "## 负载阶段结果",
        "",
        "| Regime | Slots | Updates | First update | Migrations | Delay improvement | Load-variance improvement | Psi improvement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in phase_rows:
        lines.append(
            f"| {row['regime_id']} {row['regime_label']} | {row['slot_range']} | {row['update_count']} | "
            f"{row['update_applied_slot']} | {row['migrated_nodes']} | {row['delay_improvement_pct']:.2f}% | "
            f"{row['load_variance_improvement_pct']:.2f}% | {row['psi_improvement_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 审稿视角判断",
            "",
            f"- 持续热点阶段共触发 `{phase_rows[-1]['update_count']}` 次配置更新，首轮更新延迟为 `{phase_rows[-1]['adaptation_latency_slots']}` slots。",
            f"- 首轮更新后的 `{POST_UPDATE_WINDOW}` slots 内，时延、负载方差和 `psi` 分别改善 `{phase_rows[-1]['post_update_delay_improvement_pct']:.2f}%`、`{phase_rows[-1]['post_update_load_variance_improvement_pct']:.2f}%` 和 `{phase_rows[-1]['post_update_psi_improvement_pct']:.2f}%`。",
            "- 图中同时展示 workload phase、因果更新时刻和 Static/Dynamic 绝对指标，可直接观察控制面更新前后的变化。",
            "- 该结果只支持受控热点迁移场景下的适应性，不应表述为所有原始 PeMS 时隙均能获得同等改善。",
            "",
            "## 文件",
            "",
            f"- 逐时隙数据：`{SLOT_CSV.relative_to(EXP_DIR)}`",
            f"- 更新事件：`{EVENT_CSV.relative_to(EXP_DIR)}`",
            f"- 阶段汇总：`{PHASE_CSV.relative_to(EXP_DIR)}`",
            f"- 总体汇总：`{SUMMARY_CSV.relative_to(EXP_DIR)}`",
            f"- 完整性检查：`{INTEGRITY_JSON.relative_to(EXP_DIR)}`",
            f"- 图片：`{FIG_PNG.relative_to(EXP_DIR)}`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integrity_check(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    original_requests = r4.load_inputs()["requests"]
    static = sorted(
        [row for row in rows if row["method_id"] == "initial_static_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    dynamic = sorted(
        [row for row in rows if row["method_id"] == "causal_dynamic_fpdwgc"],
        key=lambda row: int(row["slot"]),
    )
    first_update = min(int(event["applied_slot"]) for event in events) if events else len(dynamic) + 1
    original_totals = [sum(float(value) for value in slot.values()) for slot in original_requests]
    controlled_totals = [float(row["request_total"]) for row in dynamic]
    pre_update_equal = all(
        abs(float(left["mean_fullflow_delay"]) - float(right["mean_fullflow_delay"])) <= 1e-6
        and abs(float(left["load_variance"]) - float(right["load_variance"])) <= 1e-8
        and abs(float(left["psi"]) - float(right["psi"])) <= 1e-6
        for left, right in zip(static[: first_update - 1], dynamic[: first_update - 1])
    )
    report = {
        "status": "pass",
        "slot_count": len(dynamic),
        "request_total_preserved": all(
            abs(before - after) <= 1e-9 for before, after in zip(original_totals, controlled_totals)
        ),
        "max_request_total_error": max(
            abs(before - after) for before, after in zip(original_totals, controlled_totals)
        ),
        "no_lookahead_update": all(
            int(event["applied_slot"]) == int(event["observation_end_slot"]) + 1 for event in events
        ),
        "static_dynamic_equal_before_first_update": pre_update_equal,
        "minimum_resource_satisfaction_rate": min(float(row["resource_satisfaction_rate"]) for row in rows),
        "first_update_slot": first_update,
        "update_events": len(events),
    }
    if not all(
        [
            report["request_total_preserved"],
            report["no_lookahead_update"],
            report["static_dynamic_equal_before_first_update"],
            report["minimum_resource_satisfaction_rate"] >= 1.0 - 1e-9,
        ]
    ):
        report["status"] = "fail"
    INTEGRITY_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows, events, phase_rows, overall_rows = evaluate()
    write_csv(SLOT_CSV, rows)
    write_csv(EVENT_CSV, events)
    write_csv(PHASE_CSV, phase_rows)
    write_csv(SUMMARY_CSV, overall_rows)
    make_figure(rows, events)
    write_summary(events, phase_rows, overall_rows)
    write_integrity_check(rows, events)
    print(
        json.dumps(
            {
                "slot_rows": len(rows),
                "events": len(events),
                "phase_rows": len(phase_rows),
                "figure": str(FIG_PNG),
                "summary": str(SUMMARY_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
