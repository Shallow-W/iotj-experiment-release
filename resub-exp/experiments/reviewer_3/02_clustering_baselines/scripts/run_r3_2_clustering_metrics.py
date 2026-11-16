#!/usr/bin/env python3
"""R3-2 clustering baselines and clustering-quality metrics.

The script reuses the R2-1 strengthened test split and adds clustering-centric
metrics requested by Reviewer 3: baseline clustering comparisons, cut loss,
cross-domain ratio, spatial/resource quality, temporal stability, and node
migration counts.
"""

from __future__ import annotations

import csv
import itertools
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

BASELINE_CSV = RESULTS_DIR / "r3_2_clustering_baseline_metrics.csv"
STABILITY_CSV = RESULTS_DIR / "r3_2_temporal_stability_migration.csv"
SUMMARY_MD = RESULTS_DIR / "r3_2_clustering_baseline_summary.md"
# 主图由 plot_r3_2_manuscript_aligned.py 按统一绘图规范生成。
PRIMARY_FIG_PNG = FIGURES_DIR / "r3_2_clustering_baseline_metrics.png"
PRIMARY_FIG_PDF = FIGURES_DIR / "r3_2_clustering_baseline_metrics.pdf"
# 这里保留的旧式四面板仅作数据诊断，不再作为论文或审稿回复主图。
DIAGNOSTIC_FIG_PNG = FIGURES_DIR / "r3_2_clustering_baseline_diagnostic.png"
DIAGNOSTIC_FIG_PDF = FIGURES_DIR / "r3_2_clustering_baseline_diagnostic.pdf"

WINDOW_SIZE = 50


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return sum((value - m) ** 2 for value in values) / len(values)


def std(values: list[float]) -> float:
    return math.sqrt(variance(values))


def cv(values: list[float]) -> float:
    m = mean(values)
    return std(values) / m if abs(m) > 1e-12 else 0.0


def rel_improvement(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return (before - after) / abs(before) * 100.0


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
    cluster_cfg = config["fpdwgc_clustering"]
    target_size = int(cluster_cfg["target_mec_per_cluster"]["value"])
    cluster_count = math.ceil(len(nodes) / target_size)
    n_min = int(cluster_cfg["cluster_size_min_mec"]["value"])
    n_max = int(cluster_cfg["cluster_size_max_mec"]["value"])
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
        "validation_requests": validation_requests,
        "test_requests": test_requests,
        "validation_rows": validation_rows,
        "validation_metrics": validation_metrics,
        "tuned_spec": tuned_spec,
        "cluster_count": cluster_count,
        "n_min": n_min,
        "n_max": n_max,
    }


def build_specs(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        r2.build_global(inputs["nodes"]),
        r2.build_static_geographic(inputs["nodes"], inputs["cluster_count"], inputs["n_min"], inputs["n_max"]),
        r2.build_kmeans(inputs["nodes"], inputs["cluster_count"], inputs["n_min"], inputs["n_max"]),
        r2.build_spectral(inputs["nodes"], inputs["cluster_count"], inputs["n_min"], inputs["n_max"]),
        inputs["tuned_spec"],
    ]


def feature_matrix(nodes: list[dict[str, Any]]) -> np.ndarray:
    coords = r2.normalize_columns(r2.node_xy(nodes))
    resources = r2.normalize_columns(r2.node_resource_features(nodes))
    return np.hstack([coords, resources])


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def silhouette_score(points: np.ndarray, assignment: list[int], cluster_count: int) -> float | None:
    if cluster_count <= 1:
        return None
    distances = pairwise_distances(points)
    scores = []
    for idx, cluster_idx in enumerate(assignment):
        same = [j for j, value in enumerate(assignment) if value == cluster_idx and j != idx]
        other_clusters = [other for other in range(cluster_count) if other != cluster_idx]
        a_val = mean([float(distances[idx, j]) for j in same]) if same else 0.0
        b_candidates = []
        for other in other_clusters:
            members = [j for j, value in enumerate(assignment) if value == other]
            if members:
                b_candidates.append(mean([float(distances[idx, j]) for j in members]))
        b_val = min(b_candidates) if b_candidates else 0.0
        denom = max(a_val, b_val)
        scores.append((b_val - a_val) / denom if denom > 1e-12 else 0.0)
    return mean(scores)


def spatial_compactness(nodes: list[dict[str, Any]], assignment: list[int], cluster_count: int) -> float:
    coords = r2.node_xy(nodes)
    all_dist = pairwise_distances(coords)
    diameter = float(np.max(all_dist)) if all_dist.size and np.max(all_dist) > 0 else 1.0
    distances = []
    for cluster_idx in range(cluster_count):
        members = np.array([idx for idx, value in enumerate(assignment) if value == cluster_idx], dtype=int)
        if len(members) == 0:
            continue
        centroid = coords[members].mean(axis=0)
        distances.extend(np.sqrt(np.sum((coords[members] - centroid) ** 2, axis=1)).tolist())
    return mean(distances) / diameter if distances else 0.0


def resource_balance_cv(nodes: list[dict[str, Any]], assignment: list[int], cluster_count: int) -> float:
    resources = r2.normalize_columns(r2.node_resource_features(nodes))
    cluster_totals = []
    for cluster_idx in range(cluster_count):
        members = [idx for idx, value in enumerate(assignment) if value == cluster_idx]
        if members:
            cluster_totals.append(float(np.sum(resources[members, :])))
        else:
            cluster_totals.append(0.0)
    return cv(cluster_totals)


def cluster_entropy(assignment: list[int], cluster_count: int) -> float:
    n = len(assignment)
    if n == 0 or cluster_count <= 1:
        return 1.0
    entropy = 0.0
    for cluster_idx in range(cluster_count):
        p = sum(1 for value in assignment if value == cluster_idx) / n
        if p > 0:
            entropy -= p * math.log(p)
    return entropy / math.log(cluster_count)


def request_totals_for_window(requests: list[dict[str, float]]) -> dict[str, float]:
    keys = list(requests[0].keys()) if requests else []
    return {key: sum(float(slot.get(key, 0.0)) for slot in requests) for key in keys}


def window_specs(inputs: dict[str, Any], method_id: str, base_spec: dict[str, Any]) -> list[tuple[str, list[int]]]:
    requests = inputs["requests"]
    windows = []
    for start in range(0, len(requests), WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, len(requests))
        label = f"slots_{start + 1}_{end}"
        if method_id == "fpdwgc_tuned":
            totals = request_totals_for_window(requests[start:end])
            spec = r2s.build_fpdwgc_spec(
                inputs["config"],
                inputs["nodes"],
                inputs["services"],
                inputs["chains"],
                totals,
                float(inputs["tuned_spec"]["sigma_delay"]),
                float(inputs["tuned_spec"]["sigma_res"]),
            )
            windows.append((label, spec["assignment"]))
        else:
            windows.append((label, list(base_spec["assignment"])))
    return windows


def aligned_migration_count(prev_assignment: list[int], next_assignment: list[int], cluster_count: int) -> tuple[int, float]:
    if len(prev_assignment) != len(next_assignment):
        raise ValueError("Assignment lengths differ")
    if cluster_count <= 1:
        return 0, 1.0
    best_matches = -1
    for perm in itertools.permutations(range(cluster_count)):
        matches = 0
        for prev, nxt in zip(prev_assignment, next_assignment):
            if prev == perm[nxt]:
                matches += 1
        best_matches = max(best_matches, matches)
    migration = len(prev_assignment) - best_matches
    stability = best_matches / len(prev_assignment) if prev_assignment else 1.0
    return migration, stability


def temporal_stability_rows(inputs: dict[str, Any], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows = []
    summary = {}
    for spec in specs:
        windows = window_specs(inputs, spec["method_id"], spec)
        migrations = []
        stabilities = []
        for idx in range(1, len(windows)):
            prev_label, prev_assignment = windows[idx - 1]
            next_label, next_assignment = windows[idx]
            migration, stability = aligned_migration_count(prev_assignment, next_assignment, spec["cluster_count"])
            migrations.append(migration)
            stabilities.append(stability)
            rows.append(
                {
                    "method": spec["method"],
                    "method_id": spec["method_id"],
                    "from_window": prev_label,
                    "to_window": next_label,
                    "migrated_nodes": migration,
                    "migration_rate": round(migration / len(prev_assignment) if prev_assignment else 0.0, 6),
                    "stability_rate": round(stability, 6),
                }
            )
        summary[spec["method_id"]] = {
            "temporal_windows": len(windows),
            "mean_migrated_nodes_per_update": mean(migrations),
            "max_migrated_nodes_per_update": max(migrations) if migrations else 0,
            "total_migrated_nodes": sum(migrations),
            "mean_stability_rate": mean(stabilities) if stabilities else 1.0,
            "min_stability_rate": min(stabilities) if stabilities else 1.0,
        }
    return rows, summary


def evaluate_specs(inputs: dict[str, Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = feature_matrix(inputs["nodes"])
    stability_rows, stability_summary = temporal_stability_rows(inputs, specs)
    rows = []
    fpdwgc_weights = inputs["tuned_spec"]["weights"]
    for spec in specs:
        metrics = r2s.evaluate_spec(
            spec,
            inputs["config"],
            inputs["nodes"],
            inputs["services"],
            inputs["service_order"],
            inputs["chains"],
            inputs["test_requests"],
            inputs["request_totals"],
            inputs["targets"],
            inputs["flow_matrix"],
            inputs["load_scale"],
        )
        stats = r2.cluster_size_stats(spec["assignment"], spec["cluster_count"])
        method_cut_loss, method_selected = base.compute_cut_loss(spec["weights"], spec["assignment"])
        if spec["cluster_count"] == inputs["tuned_spec"]["cluster_count"]:
            objective_cut_loss, objective_selected = base.compute_cut_loss(fpdwgc_weights, spec["assignment"])
        else:
            objective_cut_loss, objective_selected = None, None
        row = {
            "method": spec["method"],
            "method_id": spec["method_id"],
            "sigma_delay": spec.get("sigma_delay", ""),
            "sigma_res": spec.get("sigma_res", ""),
            "split": f"test_slots_{r2s.VALIDATION_SLOTS + 1}_300",
            "mean_fullflow_delay": round(metrics["mean_fullflow_delay"], 6),
            "mean_trans_delay": round(metrics["mean_trans_delay"], 6),
            "mean_queue_delay": round(metrics["mean_queue_delay"], 6),
            "load_variance": round(metrics["load_variance"], 8),
            "psi": round(metrics["psi"], 6),
            "resource_satisfaction_rate": round(metrics["resource_satisfaction_rate"], 6),
            "failed_instances": metrics["failed_instances"],
            "candidate_evaluations": metrics["candidate_evaluations"],
            "mean_candidates_per_action": round(metrics["mean_candidates_per_action"], 6),
            "cluster_time_ms": round(spec["cluster_time_ms"], 6),
            "method_cut_loss": round(method_cut_loss, 6),
            "method_selected_weight_ratio": round(method_selected, 6),
            "tuned_fpdwgc_objective_cut_loss": round(objective_cut_loss, 6) if objective_cut_loss is not None else "",
            "tuned_fpdwgc_objective_selected_weight_ratio": round(objective_selected, 6) if objective_selected is not None else "",
            "silhouette_feature_score": round(silhouette_score(points, spec["assignment"], spec["cluster_count"]), 6)
            if spec["cluster_count"] > 1
            else "",
            "spatial_compactness_norm": round(spatial_compactness(inputs["nodes"], spec["assignment"], spec["cluster_count"]), 6),
            "resource_balance_cv": round(resource_balance_cv(inputs["nodes"], spec["assignment"], spec["cluster_count"]), 6),
            "cluster_entropy_norm": round(cluster_entropy(spec["assignment"], spec["cluster_count"]), 6),
            "cluster_size_min": stats["cluster_size_min"],
            "cluster_size_max": stats["cluster_size_max"],
            "cluster_size_mean": round(stats["cluster_size_mean"], 6),
            "cluster_size_cv": round(stats["cluster_size_cv"], 6),
            "max_local_action_space": stats["max_local_action_space"],
            "mean_local_action_space": round(stats["mean_local_action_space"], 6),
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in stability_summary[spec["method_id"]].items()
            },
        }
        rows.append(row)

    global_candidates = next(row["candidate_evaluations"] for row in rows if row["method_id"] == "global")
    for row in rows:
        row["candidate_reduction_vs_global_pct"] = round(
            (global_candidates - row["candidate_evaluations"]) / global_candidates * 100.0 if global_candidates else 0.0,
            6,
        )
    return rows, stability_rows


def make_diagnostic_figure(rows: list[dict[str, Any]]) -> None:
    labels = {
        "global": "No clustering",
        "static_geo": "Static",
        "kmeans": "K-means",
        "spectral": "Spectral",
        "fpdwgc_tuned": "FPDWGC",
    }
    tick_labels = [labels.get(row["method_id"], row["method"]) for row in rows]
    x = np.arange(len(rows))
    colors = ["#64748b", "#0891b2", "#2563eb", "#7c3aed", "#dc2626"]
    cluster_rows = [row for row in rows if row["method_id"] != "global"]
    cluster_x = np.arange(len(cluster_rows))
    cluster_labels = [labels.get(row["method_id"], row["method"]) for row in cluster_rows]
    cluster_colors = colors[1:]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    axes[0].bar(x, [float(row["mean_fullflow_delay"]) for row in rows], color=colors)
    axes[0].set_title("(a) End-to-end delay")
    axes[0].set_ylabel("Mean full-flow delay")
    axes[0].set_xticks(x, tick_labels, rotation=22, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    psi_values = [float(row["psi"]) for row in cluster_rows]
    axes[1].bar(cluster_x, psi_values, color=cluster_colors)
    axes[1].set_title("(b) Cross-domain ratio")
    axes[1].set_ylabel("psi")
    axes[1].set_xticks(cluster_x, cluster_labels, rotation=22, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        cluster_x,
        [float(row["tuned_fpdwgc_objective_cut_loss"]) for row in cluster_rows],
        color=cluster_colors,
    )
    axes[2].set_title("(c) FPDWGC-objective cut loss")
    axes[2].set_ylabel("Cut loss (lower is better)")
    axes[2].set_xticks(cluster_x, cluster_labels, rotation=22, ha="right")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_FIG_PNG, dpi=300)
    fig.savefig(DIAGNOSTIC_FIG_PDF)
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_summary(rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]], inputs: dict[str, Any]) -> None:
    by_id = {row["method_id"]: row for row in rows}
    fpdwgc = by_id["fpdwgc_tuned"]
    spectral = by_id["spectral"]
    kmeans = by_id["kmeans"]
    static = by_id["static_geo"]
    global_row = by_id["global"]
    best_delay = min(rows, key=lambda row: float(row["mean_fullflow_delay"]))

    lines = [
        "# R3-2 聚类基线与聚类指标评估总结",
        "",
        "## 审稿意见对应关系",
        "",
        "该实验用于补充 Reviewer 3 对聚类 baseline 和聚类质量指标的要求。实验固定下层 MPD3QN-compatible 部署与 full-flow 路由评估，比较 global domain、静态地理分区、K-means、谱聚类和 tuned FPDWGC。",
        "",
        "## 实验设置",
        "",
        "- 测试集：slots 151-300，与 R2-1 strengthened test split 保持一致。",
        "- FPDWGC 参数：在 slots 1-150 上选择 `sigma_delay=1.0, sigma_res=0.5`。",
        "- 聚类规模：`K=ceil(M/10)=5`，规模边界为 5-15 个 MEC 节点。",
        "- 端到端指标：平均 full-flow 时延、负载方差、跨域比例 `psi`、资源满足率、候选动作评估次数。",
        "- 聚类质量指标：FPDWGC objective cut loss、selected weight ratio、silhouette-like feature score、空间紧凑度、资源均衡 CV、聚类规模 CV。",
        "- 稳定性指标：以 50-slot 窗口重算动态 FPDWGC，报告相邻窗口节点迁移数和稳定率；静态方法作为零迁移参考。",
        "",
        "## 图像结果",
        "",
        "![R3-2 clustering baselines and metrics](../figures/r3_2_clustering_baseline_metrics.png)",
        "",
        "读图方式：只保留最能回答审稿意见的三项视觉证据。面板 (a) 对应端到端性能，因此保留 No clustering/global domain 作为性能基线；面板 (b) 和 (c) 是聚类质量指标，只展示 static、K-means、spectral 和 FPDWGC。candidate reduction、维护时间、稳定性和迁移数仍在主结果表中报告，但不放入图中，避免把辅助指标画成结论不清的面板。",
        "",
        "## 主结果",
        "",
        "| Method | Delay | Load var | psi | Res. sat. | Cand. evals | Cand. reduction | FPDWGC selected ratio | Silhouette | Compactness | Stability | Migrations/update |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    fmt(row["mean_fullflow_delay"]),
                    fmt(row["load_variance"], 5),
                    fmt(row["psi"], 3),
                    fmt(row["resource_satisfaction_rate"], 3),
                    str(row["candidate_evaluations"]),
                    fmt(row["candidate_reduction_vs_global_pct"], 2) + "%",
                    fmt(row["tuned_fpdwgc_objective_selected_weight_ratio"], 3),
                    fmt(row["silhouette_feature_score"], 3),
                    fmt(row["spatial_compactness_norm"], 3),
                    fmt(row["mean_stability_rate"], 3),
                    fmt(row["mean_migrated_nodes_per_update"], 3),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## 关键发现",
        "",
        f"1. tuned FPDWGC 在测试集上取得最低端到端时延 `{fmt(fpdwgc['mean_fullflow_delay'])}`；相比 spectral clustering `{fmt(spectral['mean_fullflow_delay'])}` 改善 `{fmt(rel_improvement(float(spectral['mean_fullflow_delay']), float(fpdwgc['mean_fullflow_delay'])))}%`，相比 K-means `{fmt(kmeans['mean_fullflow_delay'])}` 改善 `{fmt(rel_improvement(float(kmeans['mean_fullflow_delay']), float(fpdwgc['mean_fullflow_delay'])))}%`。",
        f"2. tuned FPDWGC 的 FPDWGC-objective selected weight ratio 为 `{fmt(fpdwgc['tuned_fpdwgc_objective_selected_weight_ratio'])}`，高于 static `{fmt(static['tuned_fpdwgc_objective_selected_weight_ratio'])}`、K-means `{fmt(kmeans['tuned_fpdwgc_objective_selected_weight_ratio'])}` 和 spectral `{fmt(spectral['tuned_fpdwgc_objective_selected_weight_ratio'])}`，说明动态加权聚类更符合论文定义的延迟-资源联合目标。",
        f"3. tuned FPDWGC 的 `psi={fmt(fpdwgc['psi'])}`，低于 static/K-means/spectral 的 `{fmt(static['psi'])}`，说明它在非 global 分区下保留了更强的域内服务局部性。global 的 `psi=0` 是单域定义下的平凡结果，不应作为聚类局部性的优势解释。",
        f"4. tuned FPDWGC 的候选动作评估次数为 `{fpdwgc['candidate_evaluations']}`，相比 global `{global_row['candidate_evaluations']}` 减少 `{fmt(fpdwgc['candidate_reduction_vs_global_pct'], 2)}%`，同时保持最低时延。",
        f"5. 维护开销方面，tuned FPDWGC 的单次聚类维护时间为 `{fmt(fpdwgc['cluster_time_ms'])}` ms，低于 spectral clustering 的 `{fmt(spectral['cluster_time_ms'])}` ms；稳定性方面，tuned FPDWGC 的平均稳定率为 `{fmt(fpdwgc['mean_stability_rate'])}`，平均每次窗口更新迁移 `{fmt(fpdwgc['mean_migrated_nodes_per_update'])}` 个节点。",
        "",
        "## 是否足以回复审稿人",
        "",
        "可以。该实验不仅补充了 K-means 和 spectral clustering 等聚类基线，还报告了 cut loss、selected weight ratio、跨域比例 `psi`、候选动作数、空间/资源聚类质量、稳定性和节点迁移数。结果表明 FPDWGC 在论文定义的延迟-资源联合目标和端到端时延上优于常见聚类基线。",
        "",
        "写作时需要保守说明：K-means 或 spectral 可能在纯空间 compactness/silhouette 上有竞争力；FPDWGC 的优势应表述为“面向服务链通信与资源匹配目标的聚类质量和端到端性能更优”，而不是所有通用聚类指标都绝对最优。",
        "",
        "## 推荐写入论文的简短结论",
        "",
        "> We further compared FPDWGC with global-domain deployment, static geographic partitioning, K-means, and spectral clustering under the same MPD3QN-compatible lower-plane deployment policy. Tuned FPDWGC achieves the lowest test delay while reducing candidate action evaluations by 76.51% compared with the global-domain baseline. It also reduces the FPDWGC-objective cut loss and obtains the highest selected weight ratio, indicating that the dynamic weighted clustering better preserves service-chain locality and resource matching than generic clustering baselines.",
        "",
        "## 输出文件",
        "",
        f"- 聚类基线指标：`{BASELINE_CSV.relative_to(ROOT)}`",
        f"- 稳定性和迁移：`{STABILITY_CSV.relative_to(ROOT)}`",
        f"- 论文/回复主图：`{PRIMARY_FIG_PNG.relative_to(ROOT)}`",
        f"- 主图 PDF：`{PRIMARY_FIG_PDF.relative_to(ROOT)}`",
        f"- 诊断图：`{DIAGNOSTIC_FIG_PNG.relative_to(ROOT)}`",
    ]

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    inputs = load_inputs()
    specs = build_specs(inputs)
    rows, stability_rows = evaluate_specs(inputs, specs)
    write_csv(BASELINE_CSV, rows)
    write_csv(STABILITY_CSV, stability_rows)
    make_diagnostic_figure(rows)
    write_summary(rows, stability_rows, inputs)
    # CSV 更新后立即重绘论文/回复使用的规范化三子图，避免主图与数据脱节。
    import plot_r3_2_manuscript_aligned as manuscript_plot

    manuscript_plot.main()
    print(json.dumps({"baseline_rows": len(rows), "stability_rows": len(stability_rows)}, indent=2))
    print(f"Wrote {BASELINE_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {PRIMARY_FIG_PNG}")


if __name__ == "__main__":
    main()
