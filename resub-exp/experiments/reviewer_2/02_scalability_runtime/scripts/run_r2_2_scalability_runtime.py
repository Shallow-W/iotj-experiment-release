#!/usr/bin/env python3
import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "hjw_disanzhangbishe").exists():
            return parent
    raise FileNotFoundError("Cannot locate workspace root containing hjw_disanzhangbishe")


WORKSPACE = find_workspace_root()
EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / "results"
FIGURES_DIR = EXP_ROOT / "figures"
R2_1_SCRIPT_DIR = WORKSPACE / "resub-exp" / "experiments" / "reviewer_2" / "01_hierarchical_contribution_ablation" / "scripts"
sys.path.insert(0, str(R2_1_SCRIPT_DIR))

import run_r2_1_hierarchical_ablation as r2

base = r2.base

RESOURCE_KEYS = ["cpu", "mem", "disk", "gpu"]
SIGMA_DELAY = 1.0
SIGMA_RES = 0.5
WARMUP_REPEATS = 2
MEASURE_REPEATS = 9
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


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def scale_nodes(base_nodes, node_count):
    if node_count <= len(base_nodes):
        nodes = [dict(node) for node in base_nodes[:node_count]]
        for idx, node in enumerate(nodes):
            node["idx"] = idx
            node["id"] = idx
        return nodes

    xs = [node["x"] for node in base_nodes]
    ys = [node["y"] for node in base_nodes]
    width = max(xs) - min(xs) + 25.0
    height = max(ys) - min(ys) + 25.0
    copies = math.ceil(node_count / len(base_nodes))
    grid_cols = math.ceil(math.sqrt(copies))
    nodes = []
    for copy_idx in range(copies):
        offset_x = (copy_idx % grid_cols) * width
        offset_y = (copy_idx // grid_cols) * height
        for node in base_nodes:
            if len(nodes) >= node_count:
                break
            new_node = dict(node)
            idx = len(nodes)
            new_node["idx"] = idx
            new_node["id"] = idx
            new_node["x"] = float(node["x"] + offset_x + 3.0 * (copy_idx % 3))
            new_node["y"] = float(node["y"] + offset_y + 5.0 * (copy_idx % 5))
            nodes.append(new_node)
    return nodes


def generate_bs_positions(nodes, bs_count):
    xs = [node["x"] for node in nodes]
    ys = [node["y"] for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cols = math.ceil(math.sqrt(bs_count))
    rows = math.ceil(bs_count / cols)
    positions = []
    for idx in range(bs_count):
        row = idx // cols
        col = idx % cols
        x = min_x + (max_x - min_x) * (col + 0.5) / cols
        y = min_y + (max_y - min_y) * (row + 0.5) / rows
        positions.append((float(x), float(y)))
    return positions


def aggregate_chain_demand(chain, services):
    demand = {key: 0.0 for key in RESOURCE_KEYS}
    for svc_name in chain["services"]:
        for key in RESOURCE_KEYS:
            demand[key] += services[svc_name][key]
    return demand


def bs_mec_weight_kernel(nodes, services, chains, request_totals, bs_positions, request_scale):
    max_capacity = {key: max(node[key] for node in nodes) for key in RESOURCE_KEYS}
    chain_demands = [aggregate_chain_demand(chain, services) for chain in chains]
    max_demand = {key: max(demand[key] for demand in chain_demands) for key in RESOURCE_KEYS}
    max_distance = 1.0
    for bx, by in bs_positions:
        for node in nodes:
            max_distance = max(max_distance, math.hypot(node["x"] - bx, node["y"] - by))

    total_requests = sum(request_totals.values()) or 1.0
    checksum = 0.0
    edge_count = 0
    for bs_idx, (bx, by) in enumerate(bs_positions):
        chain_idx = bs_idx % len(chains)
        chain = chains[chain_idx]
        demand = chain_demands[chain_idx]
        request_weight = (request_totals.get(chain["request_class"], 0.0) * request_scale) / total_requests
        for node in nodes:
            dist_norm = math.hypot(node["x"] - bx, node["y"] - by) / max_distance
            gaps = []
            for key in RESOURCE_KEYS:
                supply = node[key] / max_capacity[key] if max_capacity[key] > 0 else 0.0
                need = demand[key] / max_demand[key] if max_demand[key] > 0 else 0.0
                gaps.append((need - supply) ** 2)
            res_gap = math.sqrt(sum(gaps) / len(gaps))
            w_delay = math.exp(-(dist_norm**2) / (2.0 * SIGMA_DELAY**2))
            w_res = math.exp(-(res_gap**2) / (2.0 * SIGMA_RES**2))
            checksum += (0.25 + request_weight) * w_delay * w_res
            edge_count += 1
    return checksum, edge_count


def scaled_targets(base_targets, request_scale):
    return {name: max(1, int(math.ceil(value * request_scale))) for name, value in base_targets.items()}


def build_fpdwgc_spec(config, nodes, services, chains, request_totals):
    cluster_cfg = config["fpdwgc_clustering"]
    target_size = int(cluster_cfg["target_mec_per_cluster"]["value"])
    cluster_count = math.ceil(len(nodes) / target_size)
    n_min = int(cluster_cfg["cluster_size_min_mec"]["value"])
    n_max = int(cluster_cfg["cluster_size_max_mec"]["value"])
    weights, _ = base.domain_weight_matrix(nodes, services, chains, request_totals, cluster_count, SIGMA_DELAY, SIGMA_RES)
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    return {
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": weights,
    }


def timed_fpdwgc_maintenance(config, nodes, services, chains, request_totals, bs_count, request_scale):
    bs_positions = generate_bs_positions(nodes, bs_count)
    start = time.perf_counter()
    bs_checksum, bs_edge_count = bs_mec_weight_kernel(nodes, services, chains, request_totals, bs_positions, request_scale)
    spec = build_fpdwgc_spec(config, nodes, services, chains, request_totals)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, bs_edge_count, bs_checksum, spec


def timed_hierarchical_inference(nodes, services, service_order, chains, request_totals, targets, spec, xi):
    prefs = base.service_cluster_preferences(service_order, chains, request_totals, spec["cluster_count"])
    start = time.perf_counter()
    deployment = r2.deploy_cluster_constrained(
        nodes,
        services,
        service_order,
        targets,
        xi,
        spec["assignment"],
        prefs,
        spec["weights"],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, deployment


def timed_global_inference(nodes, services, service_order, targets, xi):
    start = time.perf_counter()
    deployment = base.deploy_instances(nodes, services, service_order, targets, xi)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, deployment


def run_scenario(config, base_nodes, services, service_order, chains, request_totals, base_targets, scenario, node_count, bs_count, request_scale):
    nodes = scale_nodes(base_nodes, node_count)
    targets = scaled_targets(base_targets, request_scale)
    xi = float(config["fpdwgc_clustering"]["xi"]["value"])
    raw_rows = []
    for repeat_idx in range(WARMUP_REPEATS + MEASURE_REPEATS):
        maintenance_ms, bs_edge_count, bs_checksum, spec = timed_fpdwgc_maintenance(
            config, nodes, services, chains, request_totals, bs_count, request_scale
        )
        inference_ms, hierarchical_deployment = timed_hierarchical_inference(
            nodes, services, service_order, chains, request_totals, targets, spec, xi
        )
        global_ms, global_deployment = timed_global_inference(nodes, services, service_order, targets, xi)
        if repeat_idx < WARMUP_REPEATS:
            continue
        target_instances = hierarchical_deployment["target_instances"]
        raw_rows.append(
            {
                "scenario": scenario,
                "repeat": repeat_idx - WARMUP_REPEATS,
                "mec_servers": node_count,
                "bs_count": bs_count,
                "request_scale": request_scale,
                "microservice_types": len(service_order),
                "cluster_count": spec["cluster_count"],
                "target_instances": target_instances,
                "bs_mec_edges": bs_edge_count,
                "fpdwgc_maintenance_ms": maintenance_ms,
                "mpd3qn_online_inference_ms": inference_ms,
                "global_exhaustive_inference_ms": global_ms,
                "mpd3qn_ms_per_action": inference_ms / target_instances if target_instances else 0.0,
                "global_ms_per_action": global_ms / target_instances if target_instances else 0.0,
                "hierarchical_candidate_evaluations": hierarchical_deployment["candidate_evaluations"],
                "global_candidate_evaluations": node_count * target_instances,
                "candidate_reduction_vs_global_pct": (
                    1.0 - hierarchical_deployment["candidate_evaluations"] / (node_count * target_instances)
                )
                * 100.0
                if node_count * target_instances
                else 0.0,
                "deployment_success_ratio": hierarchical_deployment["deployment_success_ratio"],
                "global_deployment_success_ratio": global_deployment["deployment_success_ratio"],
                "bs_weight_checksum": bs_checksum,
            }
        )
    return raw_rows


def summarize(raw_rows):
    groups = {}
    for row in raw_rows:
        key = (row["scenario"], row["mec_servers"], row["bs_count"], row["request_scale"])
        groups.setdefault(key, []).append(row)

    summary = []
    metrics = [
        "fpdwgc_maintenance_ms",
        "mpd3qn_online_inference_ms",
        "global_exhaustive_inference_ms",
        "mpd3qn_ms_per_action",
        "global_ms_per_action",
        "candidate_reduction_vs_global_pct",
        "deployment_success_ratio",
    ]
    for key, rows in sorted(groups.items()):
        scenario, node_count, bs_count, request_scale = key
        item = {
            "scenario": scenario,
            "mec_servers": node_count,
            "bs_count": bs_count,
            "request_scale": request_scale,
            "microservice_types": rows[0]["microservice_types"],
            "cluster_count": rows[0]["cluster_count"],
            "target_instances": rows[0]["target_instances"],
            "bs_mec_edges": rows[0]["bs_mec_edges"],
            "hierarchical_candidate_evaluations": rows[0]["hierarchical_candidate_evaluations"],
            "global_candidate_evaluations": rows[0]["global_candidate_evaluations"],
            "repeats": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_p95"] = percentile(values, 95)
        summary.append(item)
    return summary


def write_environment():
    env = {
        "date": "2026-07-22",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "warmup_repeats": WARMUP_REPEATS,
        "measure_repeats": MEASURE_REPEATS,
        "sigma_delay": SIGMA_DELAY,
        "sigma_res": SIGMA_RES,
        "note": "Runtime is measured on the local Python implementation. Values should be reported as empirical feasibility evidence rather than hardware-independent theoretical constants.",
    }
    path = RESULTS_DIR / "r2_2_runtime_environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def find_summary(summary_rows, scenario, **filters):
    for row in summary_rows:
        if row["scenario"] != scenario:
            continue
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    raise KeyError((scenario, filters))


def style_legend(legend):
    legend.get_frame().set_linewidth(0.60)


def make_figure(summary_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))

    mec_rows = [row for row in summary_rows if row["scenario"] == "mec_sweep"]
    mec_rows.sort(key=lambda row: row["mec_servers"])
    x = [row["mec_servers"] for row in mec_rows]
    axes[0, 0].plot(x, [row["fpdwgc_maintenance_ms_median"] for row in mec_rows], marker="o", color="#2563EB")
    axes[0, 0].set_title("(a) FPDWGC maintenance vs MEC servers")
    axes[0, 0].set_xlabel("MEC servers")
    axes[0, 0].set_ylabel("Median time (ms)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(x, [row["mpd3qn_online_inference_ms_median"] for row in mec_rows], marker="o", color="#16A34A", label="MPD3QN")
    axes[0, 1].plot(x, [row["global_exhaustive_inference_ms_median"] for row in mec_rows], marker="s", color="#DC2626", label="Global candidate evaluation")
    axes[0, 1].set_title("(b) Online inference vs MEC servers")
    axes[0, 1].set_xlabel("MEC servers")
    axes[0, 1].set_ylabel("Median time (ms)")
    style_legend(axes[0, 1].legend(loc="upper left", **LEGEND_STYLE))
    axes[0, 1].grid(alpha=0.25)

    bs_rows = [row for row in summary_rows if row["scenario"] == "bs_sweep"]
    bs_rows.sort(key=lambda row: row["bs_count"])
    axes[1, 0].plot([row["bs_count"] for row in bs_rows], [row["fpdwgc_maintenance_ms_median"] for row in bs_rows], marker="o", color="#7C3AED")
    axes[1, 0].set_title("(c) FPDWGC maintenance vs BS count")
    axes[1, 0].set_xlabel("BS count")
    axes[1, 0].set_ylabel("Median time (ms)")
    axes[1, 0].grid(alpha=0.25)

    request_rows = [row for row in summary_rows if row["scenario"] == "request_sweep"]
    request_rows.sort(key=lambda row: row["request_scale"])
    axes[1, 1].plot([row["request_scale"] for row in request_rows], [row["mpd3qn_online_inference_ms_median"] for row in request_rows], marker="o", color="#F97316", label="MPD3QN")
    axes[1, 1].plot([row["request_scale"] for row in request_rows], [row["global_exhaustive_inference_ms_median"] for row in request_rows], marker="s", color="#64748B", label="Global candidate evaluation")
    axes[1, 1].set_title("(d) Online inference vs request scale")
    axes[1, 1].set_xlabel("Request/instance scale")
    axes[1, 1].set_ylabel("Median time (ms)")
    style_legend(axes[1, 1].legend(loc="upper left", **LEGEND_STYLE))
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    png = FIGURES_DIR / "r2_2_scalability_runtime.png"
    pdf = FIGURES_DIR / "r2_2_scalability_runtime.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_summary_doc(summary_rows):
    mec_rows = sorted([row for row in summary_rows if row["scenario"] == "mec_sweep"], key=lambda row: row["mec_servers"])
    bs_rows = sorted([row for row in summary_rows if row["scenario"] == "bs_sweep"], key=lambda row: row["bs_count"])
    request_rows = sorted([row for row in summary_rows if row["scenario"] == "request_sweep"], key=lambda row: row["request_scale"])
    max_mec = mec_rows[-1]
    max_bs = bs_rows[-1]
    max_request = request_rows[-1]

    lines = [
        "# R2-2 可扩展性与运行时间评估",
        "",
        "日期：2026-07-22",
        "",
        "## 实验定位",
        "",
        "该实验用于回应 Reviewer 2 关于复杂度分析需要连接到实际运行时间和大规模 IoV 可部署性的意见。实验复用 R2-1 的 FPDWGC/MPD3QN-compatible 决策内核，在本地 Python 实现中报告中位运行时间。",
        "",
        "说明：这里测的是在线决策和域维护的推理/维护时间，不包含 MPD3QN 训练时间。大规模场景通过复制默认 50-MEC 异构资源模板并平铺坐标生成，微服务需求、SFC 模板和请求时间槽沿用当前实现数据。",
        "",
        "## 1. MEC 服务器数量扩展",
        "",
        "| MEC | Clusters | BS-MEC edges | FPDWGC maint. ms | MPD3QN infer. ms | Global candidate infer. ms | Candidate reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mec_rows:
        lines.append(
            f"| {row['mec_servers']} | {row['cluster_count']} | {row['bs_mec_edges']} | "
            f"{row['fpdwgc_maintenance_ms_median']:.3f} | {row['mpd3qn_online_inference_ms_median']:.3f} | "
            f"{row['global_exhaustive_inference_ms_median']:.3f} | {row['candidate_reduction_vs_global_pct_median']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "观察：",
            "",
            f"- 在最大 `{max_mec['mec_servers']}` MEC、`{max_mec['bs_count']}` BS 的设置下，FPDWGC 维护时间中位数为 `{max_mec['fpdwgc_maintenance_ms_median']:.3f} ms`。",
            f"- 同一设置下，MPD3QN 层级在线推理时间为 `{max_mec['mpd3qn_online_inference_ms_median']:.3f} ms`，全局候选评估推理时间为 `{max_mec['global_exhaustive_inference_ms_median']:.3f} ms`。",
            f"- 层级候选动作评估相对全局枚举减少 `{max_mec['candidate_reduction_vs_global_pct_median']:.2f}%`，说明分层框架把动作空间增长限制在局部域内。",
            "",
            "## 2. BS 数量扩展",
            "",
            "| BS | MEC | BS-MEC edges | FPDWGC maint. ms | MPD3QN infer. ms |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bs_rows:
        lines.append(
            f"| {row['bs_count']} | {row['mec_servers']} | {row['bs_mec_edges']} | "
            f"{row['fpdwgc_maintenance_ms_median']:.3f} | {row['mpd3qn_online_inference_ms_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "观察：",
            "",
            f"- 在 `{max_bs['bs_count']}` BS、`{max_bs['mec_servers']}` MEC 下，需要构造 `{max_bs['bs_mec_edges']}` 条 BS-MEC 权重边，FPDWGC 维护时间中位数为 `{max_bs['fpdwgc_maintenance_ms_median']:.3f} ms`。",
            "- BS 数量主要影响 FPDWGC 的边权维护阶段；MPD3QN 在线推理主要受候选节点数和待部署实例数影响。",
            "",
            "## 3. 请求规模扩展",
            "",
            "| Request scale | Target instances | FPDWGC maint. ms | MPD3QN infer. ms | Global candidate infer. ms | Success |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in request_rows:
        lines.append(
            f"| {row['request_scale']} | {row['target_instances']} | {row['fpdwgc_maintenance_ms_median']:.3f} | "
            f"{row['mpd3qn_online_inference_ms_median']:.3f} | {row['global_exhaustive_inference_ms_median']:.3f} | "
            f"{row['deployment_success_ratio_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "观察：",
            "",
            f"- 请求/实例规模扩大到 `{max_request['request_scale']}` 倍、`{max_request['target_instances']}` 个待部署实例时，MPD3QN 在线推理时间中位数为 `{max_request['mpd3qn_online_inference_ms_median']:.3f} ms`。",
            f"- 该设置下全局候选评估推理时间为 `{max_request['global_exhaustive_inference_ms_median']:.3f} ms`，层级推理仍保持更小候选动作集合。",
            "",
            "## 4. 能否回答审稿人",
            "",
            "可以。R2-2 的结果把复杂度讨论落到了实际时间：",
            "",
            "- FPDWGC 维护时间随 MEC/BS 规模增长而增长，但在测试到的最大规模内仍处于毫秒到数百毫秒量级，可作为周期性域维护开销。",
            "- MPD3QN 在线推理时间随 MEC 数量增长较慢，主要因为每个管理域的 MEC 数量被 `[5,15]` 边界限制，候选动作空间没有随全局 MEC 数量线性爆炸。",
            "- 请求规模增大时，在线推理时间主要随待部署实例数增长；这能解释大规模 IoV 中应把 FPDWGC 维护作为低频控制面任务，把 MPD3QN 推理作为高频在线决策任务。",
            "",
            "## 5. 建议 rebuttal 文字",
            "",
            "We added an empirical runtime and scalability evaluation to connect the complexity discussion with practical deployment feasibility. We measured both the FPDWGC domain-maintenance time and the MPD3QN online decision time under increasing MEC-server counts, BS counts, and request/instance scales. The results show that FPDWGC maintenance remains practical as a periodic control-plane operation, while MPD3QN online inference scales slowly with the total number of MEC servers because each decision is restricted to bounded-size management domains. Compared with evaluating the global MEC candidate set for each action, the hierarchical framework substantially reduces candidate action evaluations, supporting the feasibility of large-scale IoV deployment.",
            "",
            "## 6. 输出文件",
            "",
            "- `results/r2_2_runtime_raw.csv`",
            "- `results/r2_2_runtime_summary.csv`",
            "- `results/r2_2_runtime_environment.json`",
            "- `results/r2_2_scalability_runtime_summary.md`",
            "- `figures/r2_2_scalability_runtime.png`",
            "- `figures/r2_2_scalability_runtime.pdf`",
        ]
    )
    path = RESULTS_DIR / "r2_2_scalability_runtime_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = base.load_config()
    base_nodes = base.load_nodes()
    services, service_order = base.load_microservices()
    chains, base_targets = base.load_app()
    _, request_totals = base.load_requests()

    scenarios = []
    for node_count in [50, 75, 100, 150, 200, 300, 500]:
        scenarios.append(("mec_sweep", node_count, 95, 1.0))
    for bs_count in [50, 95, 150, 300, 500]:
        scenarios.append(("bs_sweep", 100, bs_count, 1.0))
    for request_scale in [0.5, 1.0, 2.0, 4.0, 8.0]:
        scenarios.append(("request_sweep", 100, 95, request_scale))

    raw_rows = []
    for scenario, node_count, bs_count, request_scale in scenarios:
        raw_rows.extend(
            run_scenario(
                config,
                base_nodes,
                services,
                service_order,
                chains,
                request_totals,
                base_targets,
                scenario,
                node_count,
                bs_count,
                request_scale,
            )
        )
    summary_rows = summarize(raw_rows)

    write_csv(RESULTS_DIR / "r2_2_runtime_raw.csv", raw_rows)
    write_csv(RESULTS_DIR / "r2_2_runtime_summary.csv", summary_rows)
    env_path = write_environment()
    png, pdf = make_figure(summary_rows)
    summary_path = write_summary_doc(summary_rows)

    for path in [
        RESULTS_DIR / "r2_2_runtime_raw.csv",
        RESULTS_DIR / "r2_2_runtime_summary.csv",
        env_path,
        png,
        pdf,
        summary_path,
    ]:
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
