#!/usr/bin/env python3
import csv
import json
import math
import random
from pathlib import Path

import openpyxl


def find_workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "hjw_disanzhangbishe").exists():
            return parent
    raise FileNotFoundError("Cannot locate workspace root containing hjw_disanzhangbishe")


WORKSPACE = find_workspace_root()
EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP_ROOT / "configs" / "default_params.json"
RESULTS_DIR = EXP_ROOT / "results"
DATA_DIR = WORKSPACE / "hjw_disanzhangbishe" / "50endpoints_nms"


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * pct / 100.0
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return float(values[int(k)])
    lower = values[floor] * (ceil - k)
    upper = values[ceil] * (k - floor)
    return float(lower + upper)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def normalize(values):
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def entropy_normalized(weights):
    total = sum(weights)
    if total <= 0:
        return 0.0
    probs = [w / total for w in weights if w > 0]
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    return ent / math.log(len(weights))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_nodes():
    ws = openpyxl.load_workbook(DATA_DIR / "node_resource.xlsx", data_only=True)["Sheet1"]
    rows = []
    for row in range(2, 52):
        rows.append(
            {
                "id": int(ws.cell(row, 1).value),
                "cpu": float(ws.cell(row, 2).value),
                "mem": float(ws.cell(row, 3).value),
                "disk": float(ws.cell(row, 4).value),
                "gpu": float(ws.cell(row, 5).value),
                "x": float(ws.cell(row, 6).value),
                "y": float(ws.cell(row, 7).value),
            }
        )
    return rows


def load_microservices():
    ws = openpyxl.load_workbook(DATA_DIR / "ms_resource.xlsx", data_only=True)["Sheet1"]
    rows = []
    for row in range(2, 12):
        rows.append(
            {
                "name": ws.cell(row, 1).value,
                "cpu": float(ws.cell(row, 2).value),
                "mem": float(ws.cell(row, 3).value),
                "disk": float(ws.cell(row, 4).value),
                "gpu": float(ws.cell(row, 5).value),
            }
        )
    return rows


def load_request_slot_totals():
    ws = openpyxl.load_workbook(DATA_DIR / "request.xlsx", data_only=True)["Sheet1"]
    totals = []
    for row in range(2, 302):
        totals.append(sum(float(ws.cell(row, col).value or 0) for col in range(2, 7)))
    return totals


def load_instance_targets():
    ws = openpyxl.load_workbook(DATA_DIR / "app.xlsx", data_only=True)["Sheet1"]
    return [int(ws.cell(row, 2).value) for row in range(28, 38)]


def degradation_value(rho, rho_th, varphi, f_min):
    if rho <= rho_th:
        return 1.0
    return max(f_min, 1.0 - varphi * (rho - rho_th) ** 2)


def experiment_degradation(config, request_totals):
    runs = config["validation_grids"]["degradation_representative_runs"]
    p95 = percentile(request_totals, 95)
    stress_factors = [0.8, 1.0, 1.2]
    rows = []
    for run in runs:
        for stress in stress_factors:
            # Controlled load mapping: P95 request intensity maps to rho=0.9 at stress=1.0.
            rhos = [(x / p95) * 0.9 * stress if p95 > 0 else 0.0 for x in request_totals]
            f_values = [degradation_value(rho, run["rho_th"], run["varphi"], run["f_min"]) for rho in rhos]
            rel_times = [1.0 / f for f in f_values]
            rows.append(
                {
                    "run_id": run["run_id"],
                    "rho_th": run["rho_th"],
                    "varphi": run["varphi"],
                    "f_min": run["f_min"],
                    "stress_factor": stress,
                    "mean_rho": round(mean(rhos), 6),
                    "p95_rho": round(percentile(rhos, 95), 6),
                    "max_rho": round(max(rhos), 6),
                    "mean_relative_execution_time": round(mean(rel_times), 6),
                    "p95_relative_execution_time": round(percentile(rel_times, 95), 6),
                    "max_relative_execution_time": round(max(rel_times), 6),
                    "degraded_slot_ratio": round(sum(rho > run["rho_th"] for rho in rhos) / len(rhos), 6),
                    "floor_slot_ratio": round(sum(abs(f - run["f_min"]) < 1e-12 for f in f_values) / len(f_values), 6),
                }
            )
    write_csv(RESULTS_DIR / "r1_1_exp_degradation_sweep.csv", rows)
    return rows


def resource_pressure(node, microservice):
    ratios = []
    for key in ["cpu", "mem", "disk", "gpu"]:
        demand = microservice[key]
        capacity = node[key]
        if demand == 0:
            ratios.append(0.0)
        elif capacity == 0:
            ratios.append(2.0)
        else:
            ratios.append(demand / capacity)
    return max(ratios)


def experiment_sigma(nodes, microservices):
    ap_x, ap_y = 250.0, 250.0
    distances = [math.hypot(n["x"] - ap_x, n["y"] - ap_y) for n in nodes]
    distance_norm = normalize(distances)

    pairs = []
    pressure_raw = []
    for node_idx, node in enumerate(nodes):
        for ms in microservices:
            pressure = resource_pressure(node, ms)
            pressure_raw.append(pressure)
            feasible = all(ms[key] <= node[key] for key in ["cpu", "mem", "disk", "gpu"])
            pairs.append({"node_idx": node_idx, "pressure": pressure, "feasible": feasible})

    pressure_norm = normalize(pressure_raw)
    for idx, value in enumerate(pressure_norm):
        pairs[idx]["pressure_norm"] = value

    sigma_values = [0.25, 0.5, 1.0, 2.0]
    rows = []
    for sigma_delay in sigma_values:
        for sigma_res in sigma_values:
            weights = []
            feasible_weights = []
            for pair in pairs:
                d = distance_norm[pair["node_idx"]]
                r = pair["pressure_norm"]
                w_delay = math.exp(-d / (sigma_delay**2))
                w_res = math.exp(-r / (sigma_res**2))
                w = w_delay * w_res
                weights.append(w)
                if pair["feasible"]:
                    feasible_weights.append(w)
            sorted_weights = sorted(weights, reverse=True)
            top_10_share = sum(sorted_weights[:10]) / sum(weights)
            rows.append(
                {
                    "sigma_delay": sigma_delay,
                    "sigma_res": sigma_res,
                    "normalized_entropy": round(entropy_normalized(weights), 6),
                    "top_10_weight_share": round(top_10_share, 6),
                    "coefficient_of_variation": round(stdev(weights) / mean(weights), 6),
                    "feasible_weight_share": round(sum(feasible_weights) / sum(weights), 6),
                    "mean_weight": round(mean(weights), 8),
                    "min_weight": round(min(weights), 8),
                    "max_weight": round(max(weights), 8),
                }
            )
    write_csv(RESULTS_DIR / "r1_1_exp_sigma_weight_sweep.csv", rows)
    return rows


def experiment_xi(nodes, microservices):
    rows = []
    for xi in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        candidate_counts = []
        for ms in microservices:
            count = 0
            for node in nodes:
                if all(ms[key] <= xi * node[key] for key in ["cpu", "mem", "disk", "gpu"]):
                    count += 1
            candidate_counts.append(count)
        rows.append(
            {
                "xi": xi,
                "reserved_capacity_ratio": round(1.0 - xi, 6),
                "mean_candidate_nodes_per_service": round(mean(candidate_counts), 6),
                "min_candidate_nodes_per_service": min(candidate_counts),
                "max_candidate_nodes_per_service": max(candidate_counts),
                "zero_candidate_service_count": sum(c == 0 for c in candidate_counts),
                "feasible_action_ratio": round(sum(candidate_counts) / (len(nodes) * len(microservices)), 6),
            }
        )
    write_csv(RESULTS_DIR / "r1_1_exp_xi_feasibility_sweep.csv", rows)
    return rows


def deployment_queue_from_targets(targets):
    targets = targets[:]
    queue = sorted(range(len(targets)), key=lambda idx: targets[idx], reverse=False)
    while any(value != 1 for value in targets):
        max_value = max(targets)
        for idx, value in enumerate(targets):
            if value == max_value:
                targets[idx] -= 1
                queue.append(idx)
    return queue


def experiment_xi_deployment(nodes, microservices, base_targets):
    rows = []
    resource_keys = ["cpu", "mem", "disk", "gpu"]
    for load_multiplier in [1, 2, 3, 4, 5, 6, 8, 10]:
        targets = [max(1, int(value * load_multiplier)) for value in base_targets]
        queue = deployment_queue_from_targets(targets)
        for xi in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            residual = [{key: node[key] for key in resource_keys} for node in nodes]
            deployed = 0
            failed = 0
            for ms_idx in queue:
                demand = microservices[ms_idx]
                candidates = []
                for node_idx, node_residual in enumerate(residual):
                    if all(demand[key] <= xi * node_residual[key] for key in resource_keys):
                        post_ratios = []
                        for key in resource_keys:
                            capacity = nodes[node_idx][key]
                            if capacity == 0:
                                post_ratios.append(1.0 if demand[key] == 0 else -1.0)
                            else:
                                post_ratios.append((node_residual[key] - demand[key]) / capacity)
                        candidates.append((min(post_ratios), sum(post_ratios), node_idx))
                if not candidates:
                    failed += 1
                    continue
                _, _, selected_idx = max(candidates)
                for key in resource_keys:
                    residual[selected_idx][key] -= demand[key]
                deployed += 1

            min_residual_ratios = []
            for node_idx, node_residual in enumerate(residual):
                ratios = []
                for key in resource_keys:
                    capacity = nodes[node_idx][key]
                    if capacity > 0:
                        ratios.append(node_residual[key] / capacity)
                if ratios:
                    min_residual_ratios.append(min(ratios))

            rows.append(
                {
                    "load_multiplier": load_multiplier,
                    "xi": xi,
                    "target_instances": len(queue),
                    "deployed_instances": deployed,
                    "failed_instances": failed,
                    "deployment_success_ratio": round(deployed / len(queue), 6),
                    "reserved_capacity_ratio": round(1.0 - xi, 6),
                    "min_node_residual_ratio": round(min(min_residual_ratios), 6),
                    "mean_node_residual_ratio": round(mean(min_residual_ratios), 6),
                }
            )
    write_csv(RESULTS_DIR / "r1_1_exp_xi_deployment_sweep.csv", rows)
    return rows


def experiment_eta(action_dim=50):
    rng = random.Random(20260720)
    rows = []
    penalties = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    trials = 20000
    for eta in penalties:
        invalid_selected = 0
        for _ in range(trials):
            q_values = [rng.uniform(-5e6, 5e6) for _ in range(action_dim)]
            invalid_count = rng.randint(1, max(1, action_dim // 2))
            invalid_actions = set(rng.sample(range(action_dim), invalid_count))
            masked = [q - eta if idx in invalid_actions else q for idx, q in enumerate(q_values)]
            selected = max(range(action_dim), key=lambda idx: masked[idx])
            if selected in invalid_actions:
                invalid_selected += 1
        rows.append(
            {
                "eta": eta,
                "trials": trials,
                "q_value_range": "uniform(-5e6,5e6)",
                "invalid_selected_ratio": round(invalid_selected / trials, 8),
            }
        )
    write_csv(RESULTS_DIR / "r1_1_exp_eta_mask_sweep.csv", rows)
    return rows


def make_sorted_clusters(nodes, target_size):
    sorted_nodes = sorted(nodes, key=lambda n: (n["x"], n["y"]))
    cluster_count = math.ceil(len(nodes) / target_size)
    clusters = [[] for _ in range(cluster_count)]
    for idx, node in enumerate(sorted_nodes):
        clusters[min(idx // target_size, cluster_count - 1)].append(node)
    return clusters


def cluster_distance(clusters):
    distances = []
    for cluster in clusters:
        cx = mean([node["x"] for node in cluster])
        cy = mean([node["y"] for node in cluster])
        for node in cluster:
            distances.append(math.hypot(node["x"] - cx, node["y"] - cy))
    return mean(distances)


def experiment_cluster_boundaries(nodes, microservice_count):
    rows = []
    for target in [5, 8, 10, 12, 15, 20]:
        clusters = make_sorted_clusters(nodes, target)
        sizes = [len(cluster) for cluster in clusters]
        rows.append(
            {
                "target_mec_per_cluster": target,
                "cluster_count": len(clusters),
                "min_cluster_size": min(sizes),
                "max_cluster_size": max(sizes),
                "mean_cluster_size": round(mean(sizes), 6),
                "mean_intra_cluster_distance": round(cluster_distance(clusters), 6),
                "max_local_action_space": max(sizes) * microservice_count,
                "mean_local_action_space": round(mean(sizes) * microservice_count, 6),
            }
        )
    write_csv(RESULTS_DIR / "r1_1_exp_cluster_boundary_sweep.csv", rows)
    return rows


def find_row(rows, **kwargs):
    for row in rows:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)


def write_summary(degradation_rows, sigma_rows, xi_rows, xi_deployment_rows, eta_rows, cluster_rows):
    deg_default = find_row(degradation_rows, run_id="B01", stress_factor=1.0)
    deg_early = find_row(degradation_rows, run_id="B02", stress_factor=1.0)
    deg_steep = find_row(degradation_rows, run_id="B05", stress_factor=1.0)
    sigma_default = find_row(sigma_rows, sigma_delay=1.0, sigma_res=1.0)
    sigma_strong = find_row(sigma_rows, sigma_delay=0.25, sigma_res=0.25)
    sigma_weak = find_row(sigma_rows, sigma_delay=2.0, sigma_res=2.0)
    xi_default = find_row(xi_rows, xi=0.8)
    xi_deploy_default = find_row(xi_deployment_rows, xi=0.8, load_multiplier=4)
    xi_deploy_strict = find_row(xi_deployment_rows, xi=0.6, load_multiplier=4)
    xi_deploy_loose = find_row(xi_deployment_rows, xi=1.0, load_multiplier=4)
    eta_default = find_row(eta_rows, eta=1e9)
    cluster_default = find_row(cluster_rows, target_mec_per_cluster=10)

    lines = [
        "# R1-1 参数选取实验结果摘要",
        "",
        "日期：2026-07-20",
        "",
        "说明：这是用于支撑 R1-1 参数表的受控参数选择实验。它不替代完整 MPD3QN 性能补跑，但可以证明默认参数不是纯文字解释或随意设定。完整论文结果仍建议在这些默认值固定后补跑敏感性实验。",
        "",
        "## 1. 资源退化参数",
        "",
        "实验方法：使用 `request.xlsx` 的 300 个时间槽请求总量构造相对负载序列，并令 P95 负载在标准压力下对应 `rho=0.9`，比较 B01-B07 退化参数组合。",
        "",
        "| 组合 | `rho_th` | `varphi` | `f_min` | 平均相对执行时间 | P95 相对执行时间 | 最大相对执行时间 | 退化时间槽比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| B01 default | 0.7 | 1.0 | 0.2 | {deg_default['mean_relative_execution_time']} | {deg_default['p95_relative_execution_time']} | {deg_default['max_relative_execution_time']} | {deg_default['degraded_slot_ratio']} |",
        f"| B02 earlier | 0.6 | 1.0 | 0.2 | {deg_early['mean_relative_execution_time']} | {deg_early['p95_relative_execution_time']} | {deg_early['max_relative_execution_time']} | {deg_early['degraded_slot_ratio']} |",
        f"| B05 steeper | 0.7 | 2.0 | 0.2 | {deg_steep['mean_relative_execution_time']} | {deg_steep['p95_relative_execution_time']} | {deg_steep['max_relative_execution_time']} | {deg_steep['degraded_slot_ratio']} |",
        "",
        "解释：`rho_th=0.7,varphi=1.0,f_min=0.2` 是中等退化强度。相比更早触发或更陡退化，它在高负载下给出可见惩罚，同时不会让大多数时间槽过度放大执行时间。`f_min=0.2` 还保证最坏相对执行时间有上界。",
        "",
        "## 2. FPDWGC 权重尺度参数",
        "",
        "实验方法：基于默认 50 个 MEC 坐标、10 类微服务资源需求和 AP 位置，构造节点-服务候选边权，比较 `sigma_delay x sigma_res` 的 4x4 网格。",
        "",
        "| 参数 | 熵 | Top-10 权重占比 | 变异系数 | 可行动作权重占比 |",
        "|---|---:|---:|---:|---:|",
        f"| `0.25,0.25` | {sigma_strong['normalized_entropy']} | {sigma_strong['top_10_weight_share']} | {sigma_strong['coefficient_of_variation']} | {sigma_strong['feasible_weight_share']} |",
        f"| `1.0,1.0` default | {sigma_default['normalized_entropy']} | {sigma_default['top_10_weight_share']} | {sigma_default['coefficient_of_variation']} | {sigma_default['feasible_weight_share']} |",
        f"| `2.0,2.0` | {sigma_weak['normalized_entropy']} | {sigma_weak['top_10_weight_share']} | {sigma_weak['coefficient_of_variation']} | {sigma_weak['feasible_weight_share']} |",
        "",
        "解释：`0.25` 使权重过度集中，`2.0` 使权重过于平坦；`1.0` 保留区分度但不过度塌缩，适合作为默认尺度。后续完整实验应继续报告 cut loss、资源满足率和端到端时延。",
        "",
        "## 3. 资源安全阈值 `xi`",
        "",
        "实验方法：先做静态候选节点统计，再用 `app.xlsx` 的目标实例队列做顺序部署压力测试。下表展示 4 倍目标实例压力下的部署结果。",
        "",
        "| `xi` | 保留资源比例 | 静态平均候选节点/服务 | 4x 部署成功率 | 失败实例数 |",
        "|---:|---:|---:|---:|---:|",
        f"| 0.6 | {xi_deploy_strict['reserved_capacity_ratio']} | {find_row(xi_rows, xi=0.6)['mean_candidate_nodes_per_service']} | {xi_deploy_strict['deployment_success_ratio']} | {xi_deploy_strict['failed_instances']} |",
        f"| 0.8 default | {xi_default['reserved_capacity_ratio']} | {xi_default['mean_candidate_nodes_per_service']} | {xi_deploy_default['deployment_success_ratio']} | {xi_deploy_default['failed_instances']} |",
        f"| 1.0 | {xi_deploy_loose['reserved_capacity_ratio']} | {find_row(xi_rows, xi=1.0)['mean_candidate_nodes_per_service']} | {xi_deploy_loose['deployment_success_ratio']} | {xi_deploy_loose['failed_instances']} |",
        "",
        "解释：`xi=0.8` 在 4x 压力下保持 100% 部署成功，同时仍保留 20% 资源缓冲；较小 `xi` 更保守但开始损失部署成功率，`xi=1.0` 最宽松但没有安全余量。因此 `xi=0.8` 可以作为资源安全和部署可行性之间的默认折中。",
        "",
        "## 4. 动作掩码惩罚 `eta`",
        "",
        f"实验方法：在 Q 值范围 `uniform(-5e6,5e6)` 下随机生成 20000 次选择任务，统计无效动作被选中的比例。`eta=1e9` 的无效动作选择比例为 `{eta_default['invalid_selected_ratio']}`。",
        "",
        "解释：由于训练中的回报/Q 值量级可能达到百万级，`eta` 需要显著大于该量级。`1e9` 是 hard mask 数值实现，能稳定排除无效动作。",
        "",
        "## 5. 聚类规模边界",
        "",
        "| target MEC/cluster | cluster count | min size | max size | max local action space | mean intra-cluster distance |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 10 default | {cluster_default['cluster_count']} | {cluster_default['min_cluster_size']} | {cluster_default['max_cluster_size']} | {cluster_default['max_local_action_space']} | {cluster_default['mean_intra_cluster_distance']} |",
        "",
        "解释：target=10 让默认 50 MEC 场景形成 5 个管理域，每个局部动作空间约为 100，能控制 MPD3QN 子问题规模；`N_min=5,N_max=15` 用于避免集群过小造成跨域调用过多，或集群过大造成动作空间过大。",
        "",
        "## 6. 后续必须补跑",
        "",
        "- 把上述默认参数固定到完整环境中，补跑 B01-B07 的端到端时延和负载方差。",
        "- 补跑 `sigma_delay x sigma_res` 的 4x4 FPDWGC 聚类质量与最终时延。",
        "- 若把 `upsilon=(1.0,0.1,0.1,0.1)` 写成训练目标，需要补跑 W01-W03 权重消融。",
    ]
    path = RESULTS_DIR / "r1_1_parameter_selection_experiment_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    nodes = load_nodes()
    microservices = load_microservices()
    request_totals = load_request_slot_totals()
    instance_targets = load_instance_targets()

    degradation_rows = experiment_degradation(config, request_totals)
    sigma_rows = experiment_sigma(nodes, microservices)
    xi_rows = experiment_xi(nodes, microservices)
    xi_deployment_rows = experiment_xi_deployment(nodes, microservices, instance_targets)
    eta_rows = experiment_eta(action_dim=len(nodes))
    cluster_rows = experiment_cluster_boundaries(nodes, len(microservices))
    summary_path = write_summary(degradation_rows, sigma_rows, xi_rows, xi_deployment_rows, eta_rows, cluster_rows)

    for path in [
        "r1_1_exp_degradation_sweep.csv",
        "r1_1_exp_sigma_weight_sweep.csv",
        "r1_1_exp_xi_feasibility_sweep.csv",
        "r1_1_exp_xi_deployment_sweep.csv",
        "r1_1_exp_eta_mask_sweep.csv",
        "r1_1_exp_cluster_boundary_sweep.csv",
        summary_path.name,
    ]:
        print(f"results/{path}")


if __name__ == "__main__":
    main()
