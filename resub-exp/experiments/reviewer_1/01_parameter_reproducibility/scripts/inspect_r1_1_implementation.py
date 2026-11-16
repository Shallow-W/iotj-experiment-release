#!/usr/bin/env python3
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

import openpyxl


def find_workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "hjw_disanzhangbishe").exists():
            return parent
    raise FileNotFoundError("Cannot locate workspace root containing hjw_disanzhangbishe")


WORKSPACE = find_workspace_root()
EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / "results"
DATA_DIR = WORKSPACE / "hjw_disanzhangbishe" / "50endpoints_nms"
SOURCE_DIR = WORKSPACE / "hjw_disanzhangbishe"


def find_source(name):
    matches = list(SOURCE_DIR.glob(f"**/{name}"))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


MASK_SOURCE = find_source("D3QN_STATEMASK_PRI_YANMA.py")
STATE_MASK_SOURCE = find_source("D3QN_STATEMASK_PRI.py")


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def find_line(path, pattern):
    rx = re.compile(pattern)
    for idx, line in enumerate(read_lines(path), start=1):
        if rx.search(line):
            return {"file": str(path.relative_to(WORKSPACE)), "line": idx, "text": line.strip()}
    return {"file": str(path.relative_to(WORKSPACE)), "line": None, "text": None}


def literal_default(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return ast.unparse(node)


def extract_agent_defaults(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "D3QNAgent":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = [arg.arg for arg in item.args.args]
                    defaults = [literal_default(d) for d in item.args.defaults]
                    names_with_defaults = args[-len(defaults) :]
                    return dict(zip(names_with_defaults, defaults))
    raise ValueError("D3QNAgent.__init__ not found")


def read_node_resources():
    ws = openpyxl.load_workbook(DATA_DIR / "node_resource.xlsx", data_only=True)["Sheet1"]
    rows = []
    for row in range(2, 52):
        rows.append(
            {
                "mec_id": ws.cell(row, 1).value,
                "cpu": ws.cell(row, 2).value,
                "mem": ws.cell(row, 3).value,
                "disk": ws.cell(row, 4).value,
                "gpu": ws.cell(row, 5).value,
                "x": ws.cell(row, 6).value,
                "y": ws.cell(row, 7).value,
            }
        )
    return rows


def read_microservice_demands():
    ws = openpyxl.load_workbook(DATA_DIR / "ms_resource.xlsx", data_only=True)["Sheet1"]
    rows = []
    for row in range(2, 12):
        rows.append(
            {
                "microservice": ws.cell(row, 1).value,
                "cpu": ws.cell(row, 2).value,
                "mem": ws.cell(row, 3).value,
                "disk": ws.cell(row, 4).value,
                "gpu": ws.cell(row, 5).value,
            }
        )
    return rows


def read_app_and_requests():
    app_ws = openpyxl.load_workbook(DATA_DIR / "app.xlsx", data_only=True)["Sheet1"]
    chains = []
    for row in range(2, 8):
        chain = []
        for col in range(2, 6):
            value = app_ws.cell(row, col).value
            if value is not None:
                chain.append(value)
        if chain:
            chains.append({"chain_id": f"r{len(chains) + 1}", "microservices": chain, "length": len(chain)})

    ms_accounts = []
    for row in range(28, 38):
        ms_accounts.append(
            {
                "microservice": app_ws.cell(row, 1).value,
                "max_instances": app_ws.cell(row, 2).value,
            }
        )

    request_ws = openpyxl.load_workbook(DATA_DIR / "request.xlsx", data_only=True)["Sheet1"]
    request_headers = [request_ws.cell(1, col).value for col in range(2, 7)]
    request_totals = [request_ws.cell(302, col).value for col in range(2, 7)]
    return {
        "chains": chains,
        "ms_accounts": ms_accounts,
        "request_headers": request_headers,
        "request_totals": request_totals,
        "request_total_sum": sum(request_totals),
        "time_slots": request_ws.max_row - 2,
        "ap_account": app_ws.cell(27, 2).value,
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    node_rows = read_node_resources()
    ms_rows = read_microservice_demands()
    app = read_app_and_requests()
    agent_defaults = extract_agent_defaults(MASK_SOURCE)

    type_counter = Counter((r["cpu"], r["mem"], r["disk"], r["gpu"]) for r in node_rows)
    type_rows = [
        {"cpu": k[0], "mem": k[1], "disk": k[2], "gpu": k[3], "count": count}
        for k, count in sorted(type_counter.items())
    ]

    write_csv(RESULTS_DIR / "implementation_mec_resources.csv", node_rows)
    write_csv(RESULTS_DIR / "implementation_mec_type_counts.csv", type_rows)
    write_csv(RESULTS_DIR / "implementation_microservice_demands.csv", ms_rows)
    write_csv(RESULTS_DIR / "implementation_service_chains.csv", app["chains"])
    write_csv(
        RESULTS_DIR / "implementation_request_totals.csv",
        [
            {"request_class": name, "total": total}
            for name, total in zip(app["request_headers"], app["request_totals"])
        ],
    )
    write_csv(RESULTS_DIR / "implementation_ms_instance_targets.csv", app["ms_accounts"])

    evidence = {
        "source_files": {
            "state_action_mask_script": str(MASK_SOURCE.relative_to(WORKSPACE)),
            "state_mask_only_script": str(STATE_MASK_SOURCE.relative_to(WORKSPACE)),
            "data_dir": str(DATA_DIR.relative_to(WORKSPACE)),
        },
        "line_evidence": {
            "data_prefix": find_line(MASK_SOURCE, r"file_prefix\s*=\s*'50endpoints_nms/'"),
            "node_rows": find_line(MASK_SOURCE, r"for i in range\(2, 52\):"),
            "microservice_rows": find_line(MASK_SOURCE, r"for i in range\(2, 12\):"),
            "request_totals_row": find_line(MASK_SOURCE, r"request_sheet\.cell\(row=302"),
            "missing_traffic_delay": find_line(MASK_SOURCE, r"missing_traffic_delay\s*=\s*400"),
            "agent_defaults": find_line(MASK_SOURCE, r"def __init__\(self, state_dim, action_dim, lr=0\.001"),
            "hard_mask_random_branch": find_line(MASK_SOURCE, r"while 1:"),
            "hard_mask_q_penalty": find_line(MASK_SOURCE, r"q_values\[node\]\s*=\s*-1e9"),
            "episodes": find_line(MASK_SOURCE, r"episodes\s*=\s*1000"),
            "state_mask_only_random_action": find_line(STATE_MASK_SOURCE, r"return random\.randint\(0, self\.action_dim - 1\)"),
        },
        "extracted_parameters": {
            "agent_defaults": agent_defaults,
            "mec_nodes_used_by_script": len(node_rows),
            "mec_type_counts": type_rows,
            "microservice_types": len(ms_rows),
            "microservice_demand_ranges": {
                resource: {
                    "min": min(row[resource] for row in ms_rows),
                    "max": max(row[resource] for row in ms_rows),
                }
                for resource in ["cpu", "mem", "disk", "gpu"]
            },
            "service_chain_templates": len(app["chains"]),
            "service_chain_lengths": [row["length"] for row in app["chains"]],
            "request_classes": app["request_headers"],
            "request_totals": app["request_totals"],
            "request_total_sum": app["request_total_sum"],
            "request_time_slots": app["time_slots"],
            "ap_account": app["ap_account"],
            "microservice_instance_target_sum": sum(row["max_instances"] for row in app["ms_accounts"]),
        },
        "integrity_notes": [
            "The implementation reads exactly 50 MEC rows from node_resource.xlsx even though the workbook contains an aggregate row later.",
            "The state-action mask variant implements a hard feasibility mask by rejecting invalid random actions and assigning -1e9 to invalid greedy-action Q values.",
            "The high-end MEC GPU value in the implementation data is 64, while the current manuscript table states 96; R1-1 should report the actual implementation value or explicitly correct the dataset.",
            "The traffic-to-request scaling factor kappa is still not recoverable from these implementation files; only the generated request totals are visible.",
        ],
    }

    (RESULTS_DIR / "r1_1_implementation_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md = build_markdown(evidence)
    (RESULTS_DIR / "r1_1_implementation_evidence.md").write_text(md, encoding="utf-8")

    for path in [
        "implementation_mec_resources.csv",
        "implementation_mec_type_counts.csv",
        "implementation_microservice_demands.csv",
        "implementation_service_chains.csv",
        "implementation_request_totals.csv",
        "implementation_ms_instance_targets.csv",
        "r1_1_implementation_evidence.json",
        "r1_1_implementation_evidence.md",
    ]:
        print(f"results/{path}")


def evidence_ref(item):
    if item["line"] is None:
        return f"`{item['file']}`"
    return f"`{item['file']}:{item['line']}`"


def build_markdown(evidence):
    p = evidence["extracted_parameters"]
    le = evidence["line_evidence"]
    agent = p["agent_defaults"]
    type_rows = p["mec_type_counts"]
    demand = p["microservice_demand_ranges"]

    lines = [
        "# R1-1 实现证据检查",
        "",
        "日期：2026-07-20",
        "",
        "目的：把 R1-1 参数表的选取依据尽量落到可复跑的实现证据上，而不是只用文字解释。该检查读取了 MPD3QN state-action mask 脚本和 `50endpoints_nms` 下的 xlsx 数据。",
        "",
        "## 1. 关键结论",
        "",
        "- MPD3QN/PER 训练超参可以从实现中直接确认。",
        "- `eta=1e9` 有直接实现证据：state-action mask 版本将资源不可行动作的 Q 值置为 `-1e9`。",
        "- 默认实验脚本实际读取 50 个 MEC、10 类微服务、5 个服务链模板、5 类请求统计列。",
        "- 当前实现数据中的高端 MEC GPU 是 64，不是论文表里的 96；正式 R1-1 表应按实际使用值修正，或先统一代码数据和论文表。",
        "- `kappa` 仍无法从现有实现恢复，只能看到已经生成好的请求总量。",
        "",
        "## 2. 源码证据",
        "",
        "| 参数/机制 | 实现证据 | 说明 |",
        "|---|---|---|",
        f"| 默认数据目录 | {evidence_ref(le['data_prefix'])} | state-action mask 版本读取 `50endpoints_nms/` |",
        f"| MEC 行数 | {evidence_ref(le['node_rows'])} | 读取 xlsx 第 2-51 行，共 50 个 MEC |",
        f"| 微服务需求 | {evidence_ref(le['microservice_rows'])} | 读取 `ms_resource.xlsx` 第 2-11 行，共 10 类微服务 |",
        f"| 请求统计 | {evidence_ref(le['request_totals_row'])} | 读取 `request.xlsx` 第 302 行第 2-6 列作为请求总量 |",
        f"| 缺失服务惩罚 | {evidence_ref(le['missing_traffic_delay'])} | `T_missing=400 ms` |",
        f"| MPD3QN/PER 默认参数 | {evidence_ref(le['agent_defaults'])} | `lr,gamma,epsilon,batch_size,PER alpha/beta` 均在构造函数默认值中 |",
        f"| hard mask Q 惩罚 | {evidence_ref(le['hard_mask_q_penalty'])} | 无效动作 Q 值置为 `-1e9`，对应 `eta=1e9` |",
        f"| 训练轮数 | {evidence_ref(le['episodes'])} | `episodes=1000` |",
        f"| 普通 state-mask 版本差异 | {evidence_ref(le['state_mask_only_random_action'])} | 普通版本随机分支仍可能选任意动作，不能作为 hard mask 证据 |",
        "",
        "## 3. 从实现抽取的训练参数",
        "",
        "| 参数 | 实现值 |",
        "|---|---:|",
    ]
    for key in ["lr", "gamma", "epsilon", "epsilon_decay", "min_epsilon", "max_epsilon", "buffer_size", "batch_size", "alpha", "beta", "beta_increment", "eps"]:
        lines.append(f"| `{key}` | {agent[key]} |")

    lines.extend(
        [
            "",
            "这些值应直接进入 R1-1 参数表。这里不需要额外论证“为什么选择”，因为它们是当前实验已经使用的实现值；补充说明只需要说这些设置与已有 PER/mask 消融保持一致。",
            "",
            "## 4. 从 xlsx 抽取的实验规模",
            "",
            "| 项 | 实现值 | 选取依据/解释 |",
            "|---|---:|---|",
            f"| MEC 数量 | {p['mec_nodes_used_by_script']} | 由脚本读取第 2-51 行得到 |",
            f"| 微服务种类 | {p['microservice_types']} | `ms_resource.xlsx` 第 2-11 行 |",
            f"| 服务链模板数 | {p['service_chain_templates']} | `app.xlsx` 第 2-6 行非空链 |",
            f"| 服务链长度 | {p['service_chain_lengths']} | 覆盖 2-4 个微服务的 SFC |",
            f"| 请求统计列 | {len(p['request_classes'])} | request headers: {p['request_classes']} |",
            f"| 请求总量 | {p['request_total_sum']} | 第 302 行合计，单列 totals={p['request_totals']} |",
            f"| 时间槽 | {p['request_time_slots']} | `request.xlsx` 中 300 个时间槽 |",
            f"| 部署队列目标实例数 | {p['microservice_instance_target_sum']} | `app.xlsx` 第 28-37 行合计 |",
            "",
            "这个证据可以支撑参数表中的系统规模选择：默认规模不是任意写的，而是由当前实现实际读取的数据表决定。",
            "",
            "## 5. MEC 资源配置检查",
            "",
            "| CPU | Mem | Disk | GPU | 数量 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in type_rows:
        lines.append(f"| {row['cpu']} | {row['mem']} | {row['disk']} | {row['gpu']} | {row['count']} |")

    lines.extend(
        [
            "",
            "实现数据近似保持 1:1:1 异构比例：低端 17 个、中端 17 个、高端 16 个。需要注意，高端节点在 xlsx 中的 GPU 为 64；当前论文表是 96。若目标是报告“实际使用参数”，应将表格改为 64，或重新生成 xlsx 并补跑实验。",
            "",
            "## 6. 微服务资源需求异构性",
            "",
            "| 资源 | 最小值 | 最大值 | 说明 |",
            "|---|---:|---:|---|",
            f"| CPU | {demand['cpu']['min']} | {demand['cpu']['max']} | 存在轻量服务和高 CPU 服务 |",
            f"| Mem | {demand['mem']['min']} | {demand['mem']['max']} | 内存需求跨度较大 |",
            f"| Disk | {demand['disk']['min']} | {demand['disk']['max']} | 存储需求跨度较大 |",
            f"| GPU | {demand['gpu']['min']} | {demand['gpu']['max']} | 部分服务需要 GPU，验证多资源掩码必要性 |",
            "",
            "这部分可以作为 `xi=0.8` 和 action mask 的实现依据：由于微服务需求跨 CPU/Mem/Disk/GPU 四维且部分服务有 GPU 需求，部署动作必须做逐维资源可行性检查，并保留资源缓冲。",
            "",
            "## 7. 对 R1-1 文档的修正建议",
            "",
            "1. 参数表中增加 `confirmed_from_data` 状态，用于 xlsx 抽取值。",
            "2. 高端 MEC GPU 从 `96` 改成 `64 (implementation)`，并在待确认项中说明论文原表为 96。",
            "3. 请求规模写成“5 个请求统计列 / 5 个服务链模板”，不要只写模糊的“请求类别”。",
            "4. 动作掩码依据引用 `D3QN_STATEMASK_PRI_YANMA.py`，同时注明普通 state-mask 版本不是 hard mask。",
            "5. `kappa` 保持 `to_confirm`，因为现有实现只保存了生成后的请求序列，没有保留 PeMS flow 到请求数的映射系数。",
        ]
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    summarize()
