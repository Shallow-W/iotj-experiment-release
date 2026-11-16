#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default_params.json"
RESULTS_DIR = ROOT / "results"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def degradation_value(rho, rho_th, varphi, f_min):
    if rho <= rho_th:
        return 1.0
    return max(f_min, 1.0 - varphi * (rho - rho_th) ** 2)


def write_degradation_samples(config):
    rows = []
    rho_values = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    for run in config["validation_grids"]["degradation_representative_runs"]:
        for rho in rho_values:
            f_rho = degradation_value(rho, run["rho_th"], run["varphi"], run["f_min"])
            rows.append(
                {
                    "run_id": run["run_id"],
                    "rho_th": run["rho_th"],
                    "varphi": run["varphi"],
                    "f_min": run["f_min"],
                    "rho": rho,
                    "effective_capacity_ratio": round(f_rho, 6),
                    "relative_execution_time": round(1.0 / f_rho, 6),
                }
            )
    path = RESULTS_DIR / "degradation_function_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def write_communication_samples(config):
    comm = config["communication_model"]
    alpha = comm["alpha"]["value"]
    beta = comm["beta"]["value"]
    d0 = comm["d0"]["value"]
    t_sw = comm["T_sw"]["value"]
    c_sig = comm["c_sig"]["value"]
    rows = []
    for d_km in [0.5, 1.0, 2.0, 5.0, 10.0]:
        hop_count = alpha * (d_km / d0) ** beta
        prop_ms = (d_km * 1000.0 / c_sig) * 1000.0
        switch_ms = hop_count * t_sw
        rows.append(
            {
                "distance_km": d_km,
                "hop_count": round(hop_count, 6),
                "propagation_delay_ms": round(prop_ms, 6),
                "switching_delay_ms": round(switch_ms, 6),
                "total_transmission_delay_ms": round(prop_ms + switch_ms, 6),
            }
        )
    path = RESULTS_DIR / "communication_model_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def sigma_decay_samples():
    rows = []
    for sigma in [0.25, 0.5, 1.0, 2.0]:
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rows.append(
                {
                    "sigma": sigma,
                    "normalized_input": x,
                    "exp_decay": round(math.exp(-x / (sigma**2)), 8),
                }
            )
    path = RESULTS_DIR / "sigma_decay_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def cluster_boundary_samples():
    rows = []
    for m in [50, 75, 100]:
        k_cluster = math.ceil(m / 10)
        rows.append(
            {
                "mec_servers": m,
                "cluster_count_rule": "ceil(M/10)",
                "cluster_count": k_cluster,
                "average_mec_per_cluster": round(m / k_cluster, 3),
                "N_min": 5,
                "N_max": 15,
            }
        )
    path = RESULTS_DIR / "cluster_boundary_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def write_markdown_summary(degradation_rows, communication_rows, sigma_rows, cluster_rows):
    default_rows = [r for r in degradation_rows if r["run_id"] == "B01"]
    lookup = {r["rho"]: r for r in default_rows}
    sigma_at_one = {r["sigma"]: r for r in sigma_rows if r["normalized_input"] == 1.0}

    lines = [
        "# R1-1 默认参数解析性检查",
        "",
        "日期：2026-07-20",
        "",
        "说明：该文件不是完整 RL 仿真结果，而是对 R1-1 参数默认值做的轻量级解析检查，用来支撑参数表后的选取依据。完整性能结论仍需要后续敏感性实验补跑。",
        "",
        "## 1. 资源退化函数",
        "",
        "默认组合 B01 为 `rho_th=0.7,varphi=1.0,f_min=0.2`。在该设置下：",
        "",
        f"- `rho=0.7` 时 `f(rho)={lookup[0.7]['effective_capacity_ratio']}`，刚到阈值仍不退化。",
        f"- `rho=1.0` 时 `f(rho)={lookup[1.0]['effective_capacity_ratio']}`，相对执行时间约为 `{lookup[1.0]['relative_execution_time']}` 倍。",
        f"- `rho=1.5` 时 `f(rho)={lookup[1.5]['effective_capacity_ratio']}`，相对执行时间约为 `{lookup[1.5]['relative_execution_time']}` 倍。",
        f"- `rho=2.0` 时触及下界，`f(rho)={lookup[2.0]['effective_capacity_ratio']}`，相对执行时间被限制在 `{lookup[2.0]['relative_execution_time']}` 倍。",
        "",
        "这说明 `f_min` 的主要作用是控制过载场景下的最坏情况，避免有效容量为 0 或负值；`rho` 接近 1 时曲线仍保持连续和可解释。",
        "",
        "## 2. 通信模型量级",
        "",
        "| 距离 km | 跳数估计 | 传播时延 ms | 交换时延 ms | 总传输时延 ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in communication_rows:
        lines.append(
            f"| {row['distance_km']} | {row['hop_count']} | {row['propagation_delay_ms']} | {row['switching_delay_ms']} | {row['total_transmission_delay_ms']} |"
        )

    lines.extend(
        [
            "",
            "在默认通信参数下，传播时延远小于交换时延，距离对总传输时延的影响主要来自 hop-count 估计。这与论文将 `T_prop` 和 `T_swh` 分开建模的写法一致。",
            "",
            "## 3. FPDWGC sigma 归一化解释",
            "",
            "对归一化后的输入 `x in [0,1]`，`exp(-x/sigma^2)` 在 `x=1` 处的值如下：",
            "",
            "| sigma | `exp(-1/sigma^2)` | 解释 |",
            "|---:|---:|---|",
        ]
    )
    explanations = {
        0.25: "衰减过强，远端/高差异边几乎被压到 0",
        0.5: "偏强衰减，强调局部性或资源匹配",
        1.0: "默认均衡点，完整区间内仍保留有效区分度",
        2.0: "衰减较弱，权重差异被压缩",
    }
    for sigma in [0.25, 0.5, 1.0, 2.0]:
        lines.append(f"| {sigma} | {sigma_at_one[sigma]['exp_decay']} | {explanations[sigma]} |")

    lines.extend(
        [
            "",
            "因此 `sigma_delay=sigma_res=1.0` 的依据不是经验拍脑袋，而是归一化后让指数权重在 `[0.37,1]` 范围内变化，既不过度塌缩，也不过度平坦。后续 4x4 sigma 网格用于验证默认点不是异常点。",
            "",
            "## 4. 聚类规模边界",
            "",
            "| MEC 数量 | 聚类数量规则 | 聚类数量 | 平均 MEC/cluster | `N_min` | `N_max` |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in cluster_rows:
        lines.append(
            f"| {row['mec_servers']} | `{row['cluster_count_rule']}` | {row['cluster_count']} | {row['average_mec_per_cluster']} | {row['N_min']} | {row['N_max']} |"
        )

    lines.extend(
        [
            "",
            "`N_min=5,N_max=15,target=10` 使 50/75/100 个 MEC 的扩展设置都落在相近的单域决策规模上，便于把性能变化归因到系统规模，而不是单个集群过大或过小。",
            "",
            "## 5. 还需要补跑的实验证据",
            "",
            "- 退化参数 B01-B07 的代表性仿真：确认默认点附近时延和负载方差平滑。",
            "- FPDWGC 的 `sigma_delay x sigma_res` 4x4 网格：确认默认点不是异常点。",
            "- 若把 `upsilon` 并入训练 reward，需要补跑 W01-W03；若只用于评价，则可作为评价权重分析报告。",
        ]
    )

    path = RESULTS_DIR / "r1_1_default_parameter_analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    degradation_path, degradation_rows = write_degradation_samples(config)
    communication_path, communication_rows = write_communication_samples(config)
    sigma_path, sigma_rows = sigma_decay_samples()
    cluster_path, cluster_rows = cluster_boundary_samples()
    summary_path = write_markdown_summary(
        degradation_rows, communication_rows, sigma_rows, cluster_rows
    )
    for path in [degradation_path, communication_path, sigma_path, cluster_path, summary_path]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
