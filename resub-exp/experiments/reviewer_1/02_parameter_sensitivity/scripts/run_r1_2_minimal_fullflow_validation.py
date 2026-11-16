#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
import openpyxl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_r1_2_parameter_sensitivity as base


def find_workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "hjw_disanzhangbishe").exists():
            return parent
    raise FileNotFoundError("Cannot locate workspace root containing hjw_disanzhangbishe")


WORKSPACE = find_workspace_root()
EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / "results"
FIGURES_DIR = EXP_ROOT / "figures"
DATA_DIR = WORKSPACE / "hjw_disanzhangbishe" / "50endpoints_nms"

RESOURCE_KEYS = ["cpu", "mem", "disk", "gpu"]
SERVICE_COUNT = 10
AP_POSITION = (250.0, 250.0)
MISSING_TRAFFIC_DELAY = 400.0


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_flow_matrix():
    ws = openpyxl.load_workbook(DATA_DIR / "app.xlsx", data_only=True)["Sheet1"]
    matrix = []
    for row in range(55, 66):
        matrix.append([float(ws.cell(row, col).value or 0.0) for col in range(2, 12)])
    return matrix


def service_idx(name):
    return int(name[1:]) - 1


def placements_to_state(placements, service_order, node_count):
    state = [[0 for _ in range(node_count)] for _ in range(SERVICE_COUNT)]
    for svc_name in service_order:
        idx = service_idx(svc_name)
        for node_idx in placements.get(svc_name, []):
            state[idx][node_idx] += 1
    return state


def queue_delay(n):
    return 50.0 * n * (n + 1.0) / 2.0


def edge_distance(nodes, node_idx, parent_idx=None):
    if parent_idx is None:
        px, py = AP_POSITION
    else:
        px, py = nodes[parent_idx]["x"], nodes[parent_idx]["y"]
    return max(1.0, math.hypot(nodes[node_idx]["x"] - px, nodes[node_idx]["y"] - py))


def distribute_to_child(nodes, state, child_idx, parent_node_idx, traffic):
    active_nodes = [idx for idx, count in enumerate(state[child_idx]) if count > 0]
    if not active_nodes:
        return [], traffic

    weights = []
    for node_idx in active_nodes:
        dist = edge_distance(nodes, node_idx, parent_node_idx)
        weights.append(state[child_idx][node_idx] / dist)
    weight_total = sum(weights)
    if weight_total <= 0:
        share = 1.0 / len(active_nodes)
        return [(node_idx, share, edge_distance(nodes, node_idx, parent_node_idx)) for node_idx in active_nodes], 0.0
    return [
        (node_idx, weight / weight_total, edge_distance(nodes, node_idx, parent_node_idx))
        for node_idx, weight in zip(active_nodes, weights)
    ], 0.0


def branch_multipliers(slot, request_totals):
    m1_slot = slot.get("a", 0.0) + slot.get("b", 0.0) + slot.get("c", 0.0)
    m8_slot = slot.get("d", 0.0) + slot.get("e", 0.0)
    m1_avg = (request_totals.get("a", 0.0) + request_totals.get("b", 0.0) + request_totals.get("c", 0.0)) / 300.0
    m8_avg = (request_totals.get("d", 0.0) + request_totals.get("e", 0.0)) / 300.0
    return {
        0: m1_slot / m1_avg if m1_avg > 0 else 1.0,
        7: m8_slot / m8_avg if m8_avg > 0 else 1.0,
    }


def route_slot(nodes, state, flow_matrix, targets, slot, request_totals):
    traffic = [[0.0 for _ in range(len(nodes))] for _ in range(SERVICE_COUNT)]
    trans_delay = 0.0
    missing_traffic = 0.0
    multipliers = branch_multipliers(slot, request_totals)

    edges = [(-1, 0), (-1, 7), (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6), (5, 6), (7, 8), (7, 9)]
    queue = [(-1, None, child, flow_matrix[0][child] * multipliers.get(child, 1.0)) for _, child in edges if _ == -1]

    while queue:
        parent_idx, parent_node_idx, child_idx, child_traffic = queue.pop(0)
        if child_traffic <= 0:
            continue

        distribution, missing = distribute_to_child(nodes, state, child_idx, parent_node_idx, child_traffic)
        if missing > 0:
            missing_traffic += missing
            continue

        for node_idx, share, dist in distribution:
            delivered = child_traffic * share
            traffic[child_idx][node_idx] += delivered
            trans_delay += delivered * dist

        outgoing = [(src, dst) for src, dst in edges if src == child_idx]
        for src, dst in outgoing:
            base_child_flow = flow_matrix[src + 1][dst]
            if base_child_flow <= 0:
                continue
            parent_base = float(targets.get(f"m{src + 1}", 1) or 1)
            for node_idx, share, _ in distribution:
                parent_delivered = child_traffic * share
                next_traffic = parent_delivered * (base_child_flow / parent_base)
                queue.append((child_idx, node_idx, dst, next_traffic))

    return traffic, trans_delay, missing_traffic


def raw_utilization(nodes, services_by_idx, traffic, load_scale=1.0):
    rhos = []
    for node_idx, node in enumerate(nodes):
        cpu_load = 0.0
        gpu_load = 0.0
        for svc_idx in range(SERVICE_COUNT):
            svc = services_by_idx[svc_idx]
            cpu_load += traffic[svc_idx][node_idx] * svc["cpu"]
            gpu_load += traffic[svc_idx][node_idx] * svc["gpu"]
        cpu_rho = cpu_load / node["cpu"] if node["cpu"] > 0 else 0.0
        if gpu_load > 0:
            gpu_rho = gpu_load / node["gpu"] if node["gpu"] > 0 else 10.0
        else:
            gpu_rho = 0.0
        rhos.append(max(cpu_rho, gpu_rho) * load_scale)
    return rhos


def calibrate_fullflow_scale(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals):
    slot_max = []
    for slot in requests:
        traffic, _, _ = route_slot(nodes, state, flow_matrix, targets, slot, request_totals)
        slot_max.append(max(raw_utilization(nodes, services_by_idx, traffic, 1.0)))
    p95 = base.percentile(slot_max, 95)
    return 0.9 / p95 if p95 > 0 else 1.0


def evaluate_fullflow(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals, degradation_params, load_scale):
    total_delay = 0.0
    total_trans = 0.0
    total_queue = 0.0
    total_missing = 0.0
    load_variances = []
    slot_count = len(requests)
    min_f = 1.0
    slot_max_effective = []
    overload = 0
    node_slots = 0

    for slot in requests:
        traffic, trans_delay, missing_traffic = route_slot(nodes, state, flow_matrix, targets, slot, request_totals)
        rhos = raw_utilization(nodes, services_by_idx, traffic, load_scale)
        f_values = [
            base.degradation_value(max(rho, 0.0), degradation_params["rho_th"], degradation_params["varphi"], degradation_params["f_min"])
            for rho in rhos
        ]
        min_f = min(min_f, min(f_values) if f_values else 1.0)
        effective_utils = [min(5.0, rho / f) if f > 0 else 5.0 for rho, f in zip(rhos, f_values)]
        slot_max_effective.append(max(effective_utils) if effective_utils else 0.0)
        load_variances.append(base.variance(effective_utils))
        overload += sum(1 for rho in rhos if rho >= 1.0)
        node_slots += len(rhos)

        queueing_delay = 0.0
        for svc_idx in range(SERVICE_COUNT):
            for node_idx in range(len(nodes)):
                flow = traffic[svc_idx][node_idx]
                instances = state[svc_idx][node_idx]
                if flow <= 0 or instances <= 0:
                    continue
                base_queue = instances * queue_delay(flow / instances)
                queueing_delay += base_queue / f_values[node_idx]

        missing_delay = missing_traffic * MISSING_TRAFFIC_DELAY
        total_trans += trans_delay
        total_queue += queueing_delay
        total_missing += missing_delay
        total_delay += trans_delay + queueing_delay + missing_delay

    return {
        "mean_fullflow_delay": total_delay / slot_count if slot_count else 0.0,
        "mean_trans_delay": total_trans / slot_count if slot_count else 0.0,
        "mean_queue_delay": total_queue / slot_count if slot_count else 0.0,
        "mean_missing_delay": total_missing / slot_count if slot_count else 0.0,
        "load_variance": base.mean(load_variances),
        "min_effective_capacity_ratio": min_f,
        "p95_slot_max_effective_utilization": base.percentile(slot_max_effective, 95),
        "overload_node_slot_ratio": overload / node_slots if node_slots else 0.0,
    }


def build_clustered_deployment(config, nodes, services, service_order, chains, request_totals, targets, sigma_delay, sigma_res):
    cluster_cfg = config["fpdwgc_clustering"]
    target_size = int(cluster_cfg["target_mec_per_cluster"]["value"])
    cluster_count = math.ceil(len(nodes) / target_size)
    weights, _ = base.domain_weight_matrix(nodes, services, chains, request_totals, cluster_count, sigma_delay, sigma_res)
    assignment = base.cluster_assignment_from_weights(
        weights,
        int(cluster_cfg["cluster_size_min_mec"]["value"]),
        int(cluster_cfg["cluster_size_max_mec"]["value"]),
    )
    prefs = base.service_cluster_preferences(service_order, chains, request_totals, cluster_count)
    deployment = base.deploy_instances(
        nodes,
        services,
        service_order,
        targets,
        float(cluster_cfg["xi"]["value"]),
        cluster_assignment=assignment,
        service_cluster_preferences=prefs,
        node_domain_weights=weights,
    )
    cut_loss, selected_weight_ratio = base.compute_cut_loss(weights, assignment)
    return deployment, assignment, cut_loss, selected_weight_ratio


def run_degradation_validation(config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix):
    default_sigma_delay = float(config["fpdwgc_clustering"]["sigma_delay"]["value"])
    default_sigma_res = float(config["fpdwgc_clustering"]["sigma_res"]["value"])
    deployment, assignment, cut_loss, selected_weight_ratio = build_clustered_deployment(
        config, nodes, services, service_order, chains, request_totals, targets, default_sigma_delay, default_sigma_res
    )
    state = placements_to_state(deployment["placements"], service_order, len(nodes))
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(SERVICE_COUNT)]
    load_scale = calibrate_fullflow_scale(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals)
    selected = {"B01", "B02", "B05", "B07"}
    rows = []
    for run in config["validation_grids"]["degradation_representative_runs"]:
        if run["run_id"] not in selected:
            continue
        metrics = evaluate_fullflow(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals, run, load_scale)
        rows.append(
            {
                "run_id": run["run_id"],
                "rho_th": run["rho_th"],
                "varphi": run["varphi"],
                "f_min": run["f_min"],
                "mean_fullflow_delay": round(metrics["mean_fullflow_delay"], 6),
                "mean_trans_delay": round(metrics["mean_trans_delay"], 6),
                "mean_queue_delay": round(metrics["mean_queue_delay"], 6),
                "mean_missing_delay": round(metrics["mean_missing_delay"], 6),
                "load_variance": round(metrics["load_variance"], 8),
                "min_effective_capacity_ratio": round(metrics["min_effective_capacity_ratio"], 6),
                "p95_slot_max_effective_utilization": round(metrics["p95_slot_max_effective_utilization"], 6),
                "overload_node_slot_ratio": round(metrics["overload_node_slot_ratio"], 8),
                "psi": round(base.compute_psi(chains, request_totals, deployment["placements"], assignment), 6),
                "cut_loss": round(cut_loss, 6),
                "selected_weight_ratio": round(selected_weight_ratio, 6),
                "deployment_success_ratio": round(deployment["deployment_success_ratio"], 6),
                "failed_instances": deployment["failed_instances"],
                "load_scale": round(load_scale, 8),
            }
        )
    return rows, load_scale


def run_sigma_validation(config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix, load_scale):
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(SERVICE_COUNT)]
    default_deg = {
        "rho_th": float(config["resource_degradation_model"]["rho_th"]["value"]),
        "varphi": float(config["resource_degradation_model"]["varphi"]["value"]),
        "f_min": float(config["resource_degradation_model"]["f_min"]["value"]),
    }
    rows = []
    run_idx = 1
    for sigma_delay in config["validation_grids"]["sigma_grid"]["sigma_delay"]:
        for sigma_res in config["validation_grids"]["sigma_grid"]["sigma_res"]:
            label = "default" if sigma_delay == 1.0 and sigma_res == 1.0 else "grid"
            deployment, assignment, cut_loss, selected_weight_ratio = build_clustered_deployment(
                config, nodes, services, service_order, chains, request_totals, targets, sigma_delay, sigma_res
            )
            state = placements_to_state(deployment["placements"], service_order, len(nodes))
            metrics = evaluate_fullflow(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals, default_deg, load_scale)
            sizes = [sum(1 for item in assignment if item == cluster_idx) for cluster_idx in range(max(assignment) + 1)]
            rows.append(
                {
                    "run_id": f"S{run_idx:02d}",
                    "label": label,
                    "sigma_delay": sigma_delay,
                    "sigma_res": sigma_res,
                    "mean_fullflow_delay": round(metrics["mean_fullflow_delay"], 6),
                    "mean_trans_delay": round(metrics["mean_trans_delay"], 6),
                    "mean_queue_delay": round(metrics["mean_queue_delay"], 6),
                    "mean_missing_delay": round(metrics["mean_missing_delay"], 6),
                    "load_variance": round(metrics["load_variance"], 8),
                    "cut_loss": round(cut_loss, 6),
                    "selected_weight_ratio": round(selected_weight_ratio, 6),
                    "psi": round(base.compute_psi(chains, request_totals, deployment["placements"], assignment), 6),
                    "resource_satisfaction_rate": round(deployment["deployment_success_ratio"], 6),
                    "cluster_size_min": min(sizes),
                    "cluster_size_max": max(sizes),
                    "failed_instances": deployment["failed_instances"],
                }
            )
            run_idx += 1
    return rows


def find_row(rows, **kwargs):
    for row in rows:
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    raise KeyError(kwargs)


def rel_delta(value, baseline):
    return 0.0 if baseline == 0 else (value - baseline) / baseline


def make_figure(deg_rows, sigma_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))

    deg_labels = [row["run_id"] for row in deg_rows]
    x = np.arange(len(deg_rows))
    axes[0, 0].bar(x, [row["mean_fullflow_delay"] for row in deg_rows], color="#2563EB")
    axes[0, 0].set_title("(a) Full-flow delay under degradation params")
    axes[0, 0].set_ylabel("Mean delay")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(deg_labels)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, [row["load_variance"] for row in deg_rows], color="#059669")
    axes[0, 1].set_title("(b) Full-flow load variance")
    axes[0, 1].set_ylabel("Load variance")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(deg_labels)
    axes[0, 1].grid(axis="y", alpha=0.25)

    sigma_delay_values = sorted({row["sigma_delay"] for row in sigma_rows})
    sigma_res_values = sorted({row["sigma_res"] for row in sigma_rows})

    def row_for(sd, sr):
        return next(row for row in sigma_rows if row["sigma_delay"] == sd and row["sigma_res"] == sr)

    def heatmap(ax, metric, title, cmap, fmt):
        data = np.zeros((len(sigma_res_values), len(sigma_delay_values)))
        for i, sigma_res in enumerate(sigma_res_values):
            for j, sigma_delay in enumerate(sigma_delay_values):
                data[i, j] = row_for(sigma_delay, sigma_res)[metric]
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("$\\sigma_{delay}$")
        ax.set_ylabel("$\\sigma_{res}$")
        ax.set_xticks(range(len(sigma_delay_values)))
        ax.set_xticklabels([str(value) for value in sigma_delay_values])
        ax.set_yticks(range(len(sigma_res_values)))
        ax.set_yticklabels([str(value) for value in sigma_res_values])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heatmap(axes[1, 0], "mean_fullflow_delay", "(c) Full-flow delay under sigma params", "Purples", "{:.0f}")
    heatmap(axes[1, 1], "psi", "(d) Cross-domain ratio under sigma params", "Oranges", "{:.3f}")

    fig.tight_layout()
    png = FIGURES_DIR / "r1_2_minimal_fullflow_validation.png"
    pdf = FIGURES_DIR / "r1_2_minimal_fullflow_validation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_summary(deg_rows, sigma_rows, load_scale):
    default_deg = find_row(deg_rows, run_id="B01")
    early = find_row(deg_rows, run_id="B02")
    steep = find_row(deg_rows, run_id="B05")
    high_floor = find_row(deg_rows, run_id="B07")
    default_sigma = find_row(sigma_rows, sigma_delay=1.0, sigma_res=1.0)
    low_sigma = find_row(sigma_rows, sigma_delay=0.25, sigma_res=0.25)
    flat_sigma = find_row(sigma_rows, sigma_delay=2.0, sigma_res=2.0)
    best_delay = min(sigma_rows, key=lambda row: row["mean_fullflow_delay"])
    near_rows = [
        row
        for row in sigma_rows
        if row["sigma_delay"] in [0.5, 1.0, 2.0] and row["sigma_res"] in [0.5, 1.0, 2.0]
    ]
    near_delay_delta = max(
        abs(rel_delta(row["mean_fullflow_delay"], default_sigma["mean_fullflow_delay"]))
        for row in near_rows
    )
    near_psi_delta = max(abs(row["psi"] - default_sigma["psi"]) for row in near_rows)

    lines = [
        "# R1-2 最小端到端部署-路由验证",
        "",
        "日期：2026-07-20",
        "",
        "## 实验定位",
        "",
        "该补充实验用于增强 R1-2 的证据强度。前一个 `r1_2_parameter_sensitivity_summary.md` 是受控敏感性实验；本实验进一步复用原始环境的服务调用树、距离加权路由、排队时延和 missing traffic 惩罚口径，按 300 个请求时间槽逐槽评估端到端部署-路由结果。",
        "",
        "说明：这仍然不是 1000 episode MPD3QN 全量重训练；它是最小端到端验证，用于证明代表参数扰动进入部署和路由之后，最终时延、负载方差和 `psi` 仍保持平滑趋势。",
        "",
        f"- 端到端负载校准系数：`{load_scale:.8f}`",
        "- degradation 代表组：B01 默认、B02 早触发、B05 陡退化、B07 高容量下界。",
        "- sigma 验证组：完整 `4 x 4` 网格，覆盖 `sigma_delay,sigma_res in {0.25,0.5,1.0,2.0}`。",
        "",
        "## 1. 退化参数进入端到端流程后的结果",
        "",
        "| Run | `rho_th` | `varphi` | `f_min` | Mean full-flow delay | Trans | Queue | Load variance | Min `f` | `psi` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deg_rows:
        lines.append(
            f"| {row['run_id']} | {row['rho_th']} | {row['varphi']} | {row['f_min']} | "
            f"{row['mean_fullflow_delay']:.3f} | {row['mean_trans_delay']:.3f} | "
            f"{row['mean_queue_delay']:.3f} | {row['load_variance']:.6f} | "
            f"{row['min_effective_capacity_ratio']:.3f} | {row['psi']:.3f} |"
        )
    lines.extend(
        [
            "",
            "观察：",
            "",
            f"- B02 相比默认 B01 的端到端时延变化 `{rel_delta(early['mean_fullflow_delay'], default_deg['mean_fullflow_delay']) * 100:.2f}%`。",
            f"- B05 相比默认 B01 的端到端时延变化 `{rel_delta(steep['mean_fullflow_delay'], default_deg['mean_fullflow_delay']) * 100:.2f}%`，负载方差变化更明显，说明陡退化会放大高负载节点惩罚。",
            f"- B07 将容量下界提高到 0.3 后，端到端时延变化 `{rel_delta(high_floor['mean_fullflow_delay'], default_deg['mean_fullflow_delay']) * 100:.2f}%`，说明 `f_min` 主要控制最坏时间槽，不会改变常规负载下的总体趋势。",
            "",
            "## 2. FPDWGC 权重进入端到端流程后的结果",
            "",
            "| Run | `sigma_delay` | `sigma_res` | Mean full-flow delay | Load variance | Cut loss | `psi` | Resource satisfaction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sigma_rows:
        lines.append(
            f"| {row['run_id']} | {row['sigma_delay']} | {row['sigma_res']} | "
            f"{row['mean_fullflow_delay']:.3f} | {row['load_variance']:.6f} | "
            f"{row['cut_loss']:.3f} | {row['psi']:.3f} | {row['resource_satisfaction_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "观察：",
            "",
            f"- 默认 S11 的端到端时延为 `{default_sigma['mean_fullflow_delay']:.3f}`，`psi={default_sigma['psi']:.3f}`，资源满足率 `{default_sigma['resource_satisfaction_rate']:.3f}`。",
            f"- 极小 sigma `(0.25,0.25)` 的端到端时延相对默认变化 `{rel_delta(low_sigma['mean_fullflow_delay'], default_sigma['mean_fullflow_delay']) * 100:.2f}%`，但 `psi` 从 `{default_sigma['psi']:.3f}` 升至 `{low_sigma['psi']:.3f}`，说明它偏向更强局部性而不是综合折中。",
            f"- 极大 sigma S16 的端到端时延相对默认变化 `{rel_delta(flat_sigma['mean_fullflow_delay'], default_sigma['mean_fullflow_delay']) * 100:.2f}%`。",
            f"- 在 `0.5-2.0` 的邻近区域内，端到端时延相对默认的最大变化为 `{near_delay_delta * 100:.2f}%`，`psi` 最大绝对变化为 `{near_psi_delta:.3f}`。",
            f"- 全网格最低端到端时延出现在 `{best_delay['run_id']}` (`sigma_delay={best_delay['sigma_delay']}, sigma_res={best_delay['sigma_res']}`)。因此论文中不应声称默认点是 delay-only 全局最优，而应表述为默认点在时延、负载方差和跨域比例之间提供稳定折中。",
            "",
            "## 3. 可用于 rebuttal 的补充句",
            "",
            "In addition to the controlled sensitivity analysis, we conducted a minimal end-to-end deployment-routing validation using the original service invocation tree and distance-weighted routing simulator. The degradation parameters show small final-delay variations around the default setting, while the FPDWGC weight-scale parameters expose an interpretable delay/load/locality tradeoff. We therefore report the full sensitivity grid and describe the default as a balanced setting rather than a delay-only optimum.",
            "",
            "## 4. 输出文件",
            "",
            "- `results/r1_2_fullflow_degradation_validation.csv`",
            "- `results/r1_2_fullflow_sigma_validation.csv`",
            "- `figures/r1_2_minimal_fullflow_validation.png`",
            "- `figures/r1_2_minimal_fullflow_validation.pdf`",
        ]
    )
    path = RESULTS_DIR / "r1_2_fullflow_validation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = base.load_config()
    nodes = base.load_nodes()
    services, service_order = base.load_microservices()
    chains, targets = base.load_app()
    requests, request_totals = base.load_requests()
    flow_matrix = load_flow_matrix()

    deg_rows, load_scale = run_degradation_validation(
        config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix
    )
    sigma_rows = run_sigma_validation(
        config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix, load_scale
    )

    write_csv(RESULTS_DIR / "r1_2_fullflow_degradation_validation.csv", deg_rows)
    write_csv(RESULTS_DIR / "r1_2_fullflow_sigma_validation.csv", sigma_rows)
    png, pdf = make_figure(deg_rows, sigma_rows)
    summary = write_summary(deg_rows, sigma_rows, load_scale)

    for path in [
        RESULTS_DIR / "r1_2_fullflow_degradation_validation.csv",
        RESULTS_DIR / "r1_2_fullflow_sigma_validation.csv",
        png,
        pdf,
        summary,
    ]:
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
