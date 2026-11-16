#!/usr/bin/env python3
"""Shared evaluator for the manuscript-aligned Reviewer 3 reruns.

The reruns use the 100-endpoint data as the paper reference configuration:
100 MEC servers, request level 50, 10 microservice types, and one BS.  The
four scale sweeps modify only the named paper variable while every deployment
policy uses the same resource-feasibility check and service-chain router.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl


def find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "hjw_disanzhangbishe").exists():
            return parent
    raise FileNotFoundError("Cannot locate the workspace root")


ROOT = find_workspace_root()
DATA_DIR = ROOT / "hjw_disanzhangbishe" / "100endpoints_nms"
R1_SCRIPT_DIR = ROOT / "resub-exp" / "experiments" / "reviewer_1" / "02_parameter_sensitivity" / "scripts"
R2_SCRIPT_DIR = ROOT / "resub-exp" / "experiments" / "reviewer_2" / "01_hierarchical_contribution_ablation" / "scripts"
R3_2_SCRIPT_DIR = ROOT / "resub-exp" / "experiments" / "reviewer_3" / "02_clustering_baselines" / "scripts"
R3_3_SCRIPT_DIR = ROOT / "resub-exp" / "experiments" / "reviewer_3" / "03_non_rl_baselines" / "scripts"
CONFIG_PATH = ROOT / "resub-exp" / "experiments" / "reviewer_1" / "02_parameter_sensitivity" / "configs" / "default_params.json"

for script_dir in (R1_SCRIPT_DIR, R2_SCRIPT_DIR, R3_2_SCRIPT_DIR, R3_3_SCRIPT_DIR):
    sys.path.insert(0, str(script_dir))

import run_r1_2_parameter_sensitivity as base  # noqa: E402
import run_r2_1_hierarchical_ablation as r2  # noqa: E402
import run_r3_3_non_rl_baselines as policies  # noqa: E402


RESOURCE_KEYS = ["cpu", "mem", "disk", "gpu"]
SERVICE_COUNT = 10
PAPER_REQUEST_REFERENCE = 50
PAPER_MPD3QN_PROFILE = {
    "profile_id": "rl_delay_base",
    "queue_mode": "base",
    "preference_scale": 1.2,
    "affinity_scale": 1.0,
    "static_scale": 0.2,
    "min_residual_scale": 0.8,
    "avg_residual_scale": 0.05,
    "pressure_scale": 0.0,
    "comm_penalty": 0.2,
    "load_var_penalty": 100.0,
}
SELECTED_R3_3_METHOD_IDS = ("first_fit", "load_greedy", "dqn", "sac", "gddpg", "mpd3qn")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_nodes(server_count: int) -> list[dict[str, float]]:
    workbook = openpyxl.load_workbook(DATA_DIR / "node_resource.xlsx", data_only=True, read_only=True)
    sheet = workbook["Sheet1"]
    nodes = []
    for row in range(2, server_count + 2):
        nodes.append(
            {
                "idx": row - 2,
                "cpu": float(sheet.cell(row, 2).value),
                "mem": float(sheet.cell(row, 3).value),
                "disk": float(sheet.cell(row, 4).value),
                "gpu": float(sheet.cell(row, 5).value),
                "x": float(sheet.cell(row, 6).value),
                "y": float(sheet.cell(row, 7).value),
            }
        )
    workbook.close()
    return nodes


def load_services(service_count: int) -> tuple[dict[str, dict[str, float]], list[str]]:
    workbook = openpyxl.load_workbook(DATA_DIR / "ms_resource.xlsx", data_only=True, read_only=True)
    sheet = workbook["Sheet1"]
    services = {}
    order = []
    for row in range(2, service_count + 2):
        name = str(sheet.cell(row, 1).value)
        services[name] = {
            "cpu": float(sheet.cell(row, 2).value),
            "mem": float(sheet.cell(row, 3).value),
            "disk": float(sheet.cell(row, 4).value),
            "gpu": float(sheet.cell(row, 5).value),
        }
        order.append(name)
    workbook.close()
    return services, order


def load_chains_and_targets(service_order: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    active = set(service_order)
    workbook = openpyxl.load_workbook(DATA_DIR / "app.xlsx", data_only=True, read_only=True)
    sheet = workbook["Sheet1"]
    chains = []
    for index, row in enumerate(range(2, 7)):
        members = [str(sheet.cell(row, col).value) for col in range(2, 6) if sheet.cell(row, col).value is not None]
        members = [member for member in members if member in active]
        if len(members) < 2:
            continue
        chains.append(
            {
                "chain_id": f"r{index + 1}",
                "request_class": chr(ord("a") + index),
                "base_delay_ms": float(sheet.cell(row, 1).value),
                "services": members,
                "source_flow": 8.0,
            }
        )
    targets = {str(sheet.cell(row, 1).value): int(sheet.cell(row, 2).value) for row in range(28, 38)}
    workbook.close()
    return chains, {name: targets[name] for name in service_order}


def load_requests(active_classes: set[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    workbook = openpyxl.load_workbook(DATA_DIR / "request.xlsx", data_only=True, read_only=False)
    sheet = workbook["Sheet1"]
    classes = [str(sheet.cell(1, col).value) for col in range(2, 7)]
    slots = []
    for row in range(2, 302):
        slots.append({name: float(sheet.cell(row, col).value or 0.0) for name, col in zip(classes, range(2, 7)) if name in active_classes})
    totals = {name: float(sheet.cell(302, col).value or 0.0) for name, col in zip(classes, range(2, 7)) if name in active_classes}
    workbook.close()
    return slots, totals


def r3_3_method_specs() -> list[dict[str, str]]:
    """Return the R3-3 methods retained for the non-RL reviewer request.

    Delay-aware greedy is intentionally omitted because the current rebuttal
    comparison only needs representative non-RL baselines and the user asked
    to remove that policy from the comparison.
    """
    return [method for method in policies.METHODS if method["method_id"] in SELECTED_R3_3_METHOD_IDS]


def context_with_request_window(context: dict[str, Any], start: int, stop: int | None) -> dict[str, Any]:
    """Reuse the same deployment context but evaluate only a time-slot window."""
    subset = copy.copy(context)
    subset["requests"] = context["requests"][start:stop]
    return subset


def greedy_align_assignment(nodes: list[dict[str, float]], assignment: list[int], cluster_count: int) -> list[int]:
    """Map labels to domain indices without factorial-time permutation search.

    The legacy ablation helper exhaustively tries every label permutation, which
    is fine for five clusters but infeasible for the ten clusters used by the
    manuscript's 100-MEC reference setting.  This deterministic minimum-cost
    greedy assignment only gives labels a stable spatial meaning; it does not
    change cluster membership or any clustering objective.
    """
    raw_xy = r2.node_xy(nodes)
    reference = np.asarray(base.build_domain_centers(nodes, cluster_count), dtype=float)
    labels = np.asarray(assignment, dtype=int)
    centroids = []
    for cluster_idx in range(cluster_count):
        members = raw_xy[labels == cluster_idx]
        centroids.append(members.mean(axis=0) if len(members) else raw_xy.mean(axis=0))
    cost = r2.pairwise_sqdist(np.asarray(centroids, dtype=float), reference)
    candidates = sorted(
        (float(cost[src, dst]), src, dst)
        for src in range(cluster_count)
        for dst in range(cluster_count)
    )
    remap: dict[int, int] = {}
    used_destinations: set[int] = set()
    for _, src, dst in candidates:
        if src not in remap and dst not in used_destinations:
            remap[src] = dst
            used_destinations.add(dst)
    return [remap[int(label)] for label in labels]


def build_kmeans_manuscript_aligned(nodes: list[dict[str, float]], cluster_count: int, n_min: int, n_max: int) -> dict[str, Any]:
    start = time.perf_counter()
    labels, _ = r2.deterministic_kmeans(r2.normalize_columns(r2.node_xy(nodes)), cluster_count)
    labels = greedy_align_assignment(nodes, labels, cluster_count)
    weights = r2.weights_from_assignment(nodes, labels, cluster_count, sigma=0.42)
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    assignment = greedy_align_assignment(nodes, assignment, cluster_count)
    return {
        "method": "K-means",
        "method_id": "kmeans",
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": r2.weights_from_assignment(nodes, assignment, cluster_count, sigma=0.42),
        "cluster_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def build_spectral_manuscript_aligned(nodes: list[dict[str, float]], cluster_count: int, n_min: int, n_max: int) -> dict[str, Any]:
    start = time.perf_counter()
    coords = r2.normalize_columns(r2.node_xy(nodes))
    resources = r2.normalize_columns(r2.node_resource_features(nodes))
    geo_sq = r2.pairwise_sqdist(coords, coords)
    res_sq = r2.pairwise_sqdist(resources, resources)
    adjacency = np.exp(-geo_sq / (2.0 * 0.32**2)) * np.exp(-res_sq / (2.0 * 0.58**2))
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(len(nodes)) - (inv_sqrt_degree[:, None] * adjacency) * inv_sqrt_degree[None, :]
    if r2.eigh is not None:
        _, eigvecs = r2.eigh(laplacian)
    else:
        _, eigvecs = np.linalg.eigh(laplacian)
    embedding = eigvecs[:, 1 : cluster_count + 1]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    labels, _ = r2.deterministic_kmeans(embedding, cluster_count)
    labels = greedy_align_assignment(nodes, labels, cluster_count)
    weights = r2.weights_from_assignment(nodes, labels, cluster_count, sigma=0.42)
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    assignment = greedy_align_assignment(nodes, assignment, cluster_count)
    return {
        "method": "Spectral clustering",
        "method_id": "spectral",
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": r2.weights_from_assignment(nodes, assignment, cluster_count, sigma=0.42),
        "cluster_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def build_cluster_spec(
    method_id: str,
    config: dict[str, Any],
    nodes: list[dict[str, float]],
    services: dict[str, dict[str, float]],
    chains: list[dict[str, Any]],
    request_totals: dict[str, float],
) -> dict[str, Any]:
    target_size = int(config["fpdwgc_clustering"]["target_mec_per_cluster"]["value"])
    cluster_count = math.ceil(len(nodes) / target_size)
    n_min = int(config["fpdwgc_clustering"]["cluster_size_min_mec"]["value"])
    n_max = int(config["fpdwgc_clustering"]["cluster_size_max_mec"]["value"])
    if method_id == "global":
        return r2.build_global(nodes)
    if method_id == "static_geo":
        return r2.build_static_geographic(nodes, cluster_count, n_min, n_max)
    if method_id == "kmeans":
        return build_kmeans_manuscript_aligned(nodes, cluster_count, n_min, n_max)
    if method_id == "spectral":
        return build_spectral_manuscript_aligned(nodes, cluster_count, n_min, n_max)
    if method_id == "fpdwgc":
        return r2.build_fpdwgc(config, nodes, services, chains, request_totals, cluster_count, n_min, n_max)
    raise ValueError(f"Unknown clustering method: {method_id}")


def build_context(server_count: int, service_count: int, clustering_method: str = "fpdwgc") -> dict[str, Any]:
    config = copy.deepcopy(load_config())
    nodes = load_nodes(server_count)
    services, service_order = load_services(service_count)
    chains, targets = load_chains_and_targets(service_order)
    active_classes = {chain["request_class"] for chain in chains}
    requests, request_totals = load_requests(active_classes)
    spec = build_cluster_spec(clustering_method, config, nodes, services, chains, request_totals)
    return {
        "config": config,
        "nodes": nodes,
        "services": services,
        "service_order": service_order,
        "chains": chains,
        "targets": targets,
        "requests": requests,
        "request_totals": request_totals,
        "tuned_spec": spec,
        "xi": float(config["fpdwgc_clustering"]["xi"]["value"]),
        "pressures": r2.service_pressure(services, service_order, chains, request_totals),
        "mpd3qn_profile": dict(PAPER_MPD3QN_PROFILE),
        "degradation": {
            "rho_th": float(config["resource_degradation_model"]["rho_th"]["value"]),
            "varphi": float(config["resource_degradation_model"]["varphi"]["value"]),
            "f_min": float(config["resource_degradation_model"]["f_min"]["value"]),
        },
    }


def state_from_placements(placements: dict[str, list[int]], node_count: int) -> list[list[int]]:
    state = [[0 for _ in range(node_count)] for _ in range(SERVICE_COUNT)]
    for service_name, node_indices in placements.items():
        service_idx = int(service_name[1:]) - 1
        for node_idx in node_indices:
            state[service_idx][node_idx] += 1
    return state


def bs_positions(bs_count: int) -> list[tuple[float, float]]:
    if bs_count == 1:
        return [(250.0, 250.0)]
    if bs_count == 2:
        return [(125.0, 250.0), (375.0, 250.0)]
    if bs_count == 3:
        return [(100.0, 250.0), (250.0, 250.0), (400.0, 250.0)]
    raise ValueError("The manuscript-aligned BS sweep supports 1, 2, or 3 BSs")


def distance(node: dict[str, float], origin: tuple[float, float]) -> float:
    return max(1.0, math.hypot(node["x"] - origin[0], node["y"] - origin[1]))


def distribute_flow(
    nodes: list[dict[str, float]],
    instances: list[int],
    flow: float,
    origin: tuple[float, float],
) -> tuple[list[tuple[int, float]], float, float]:
    active = [idx for idx, count in enumerate(instances) if count > 0]
    if not active:
        return [], 0.0, flow
    weights = [instances[idx] / distance(nodes[idx], origin) for idx in active]
    weight_total = sum(weights)
    if weight_total <= 0:
        return [], 0.0, flow
    assigned = [(idx, flow * weight / weight_total) for idx, weight in zip(active, weights)]
    trans_delay = sum(amount * distance(nodes[idx], origin) for idx, amount in assigned)
    return assigned, trans_delay, 0.0


def route_slot(
    context: dict[str, Any],
    state: list[list[int]],
    slot: dict[str, float],
    request_level: int,
    bs_count: int,
) -> tuple[list[list[float]], float, float]:
    nodes = context["nodes"]
    traffic = [[0.0 for _ in nodes] for _ in range(SERVICE_COUNT)]
    trans_delay = 0.0
    missing_flow = 0.0
    intensity = request_level / PAPER_REQUEST_REFERENCE
    totals = context["request_totals"]

    for chain in context["chains"]:
        request_class = chain["request_class"]
        average = totals.get(request_class, 0.0) / len(context["requests"])
        multiplier = slot.get(request_class, 0.0) / average if average > 0 else 0.0
        source_flow = chain["source_flow"] * multiplier * intensity
        first = int(chain["services"][0][1:]) - 1
        current: list[tuple[int, float]] = []
        for bs in bs_positions(bs_count):
            assigned, delay, missing = distribute_flow(nodes, state[first], source_flow / bs_count, bs)
            current.extend(assigned)
            trans_delay += delay
            missing_flow += missing
        for node_idx, amount in current:
            traffic[first][node_idx] += amount

        for parent_name, child_name in zip(chain["services"], chain["services"][1:]):
            parent_idx = int(parent_name[1:]) - 1
            child_idx = int(child_name[1:]) - 1
            next_current: list[tuple[int, float]] = []
            for parent_node, parent_flow in current:
                origin = (nodes[parent_node]["x"], nodes[parent_node]["y"])
                assigned, delay, missing = distribute_flow(nodes, state[child_idx], parent_flow, origin)
                next_current.extend(assigned)
                trans_delay += delay
                missing_flow += missing
            current = next_current
            for node_idx, amount in current:
                traffic[child_idx][node_idx] += amount
            _ = parent_idx
    return traffic, trans_delay, missing_flow


def raw_utilization(context: dict[str, Any], traffic: list[list[float]], load_scale: float) -> list[float]:
    values = []
    for node_idx, node in enumerate(context["nodes"]):
        cpu_used = 0.0
        gpu_used = 0.0
        for service_name, service in context["services"].items():
            service_idx = int(service_name[1:]) - 1
            cpu_used += traffic[service_idx][node_idx] * service["cpu"]
            gpu_used += traffic[service_idx][node_idx] * service["gpu"]
        cpu_rho = cpu_used / node["cpu"] if node["cpu"] > 0 else 0.0
        gpu_rho = gpu_used / node["gpu"] if node["gpu"] > 0 else 0.0
        values.append(max(cpu_rho, gpu_rho) * load_scale)
    return values


def calibrate_reference_load_scale() -> float:
    context = build_context(100, 10, "fpdwgc")
    method = next(item for item in r3_3_method_specs() if item["method_id"] == "mpd3qn")
    deployment = policies.deploy_method(method, context)
    state = state_from_placements(deployment["placements"], len(context["nodes"]))
    slot_peaks = []
    for slot in context["requests"]:
        traffic, _, _ = route_slot(context, state, slot, PAPER_REQUEST_REFERENCE, 1)
        slot_peaks.append(max(raw_utilization(context, traffic, 1.0)))
    p95 = percentile(slot_peaks, 95)
    return 0.9 / p95 if p95 > 0 else 1.0


def select_mpd3qn_profile(
    context: dict[str, Any],
    request_level: int,
    bs_count: int,
    load_scale: float,
    validation_slots: int = 150,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the MPD3QN-compatible profile on validation slots only."""
    method = next(item for item in r3_3_method_specs() if item["method_id"] == "mpd3qn")
    validation_context = context_with_request_window(context, 0, validation_slots)
    rows: list[dict[str, Any]] = []
    best_profile: dict[str, Any] | None = None
    best_delay = float("inf")
    for profile in policies.MPD3QN_CANDIDATE_PROFILES:
        trial_context = copy.copy(context)
        trial_context["mpd3qn_profile"] = dict(profile)
        deployment = policies.deploy_method(method, trial_context)
        metrics = evaluate_deployment(validation_context, deployment, request_level, bs_count, load_scale)
        row = {
            "profile_id": profile["profile_id"],
            "queue_mode": profile["queue_mode"],
            "preference_scale": profile["preference_scale"],
            "affinity_scale": profile["affinity_scale"],
            "static_scale": profile["static_scale"],
            "min_residual_scale": profile["min_residual_scale"],
            "avg_residual_scale": profile["avg_residual_scale"],
            "pressure_scale": profile["pressure_scale"],
            "comm_penalty": profile["comm_penalty"],
            "load_var_penalty": profile["load_var_penalty"],
            "validation_mean_end_to_end_delay_ms": metrics["mean_end_to_end_delay_ms"],
            "validation_load_variance": metrics["load_variance"],
            "validation_psi": metrics["psi"],
            "validation_resource_satisfaction_rate": metrics["resource_satisfaction_rate"],
        }
        rows.append(row)
        if metrics["mean_end_to_end_delay_ms"] < best_delay:
            best_delay = metrics["mean_end_to_end_delay_ms"]
            best_profile = dict(profile)
    if best_profile is None:
        raise RuntimeError("No MPD3QN profile was evaluated")
    return best_profile, rows


def evaluate_deployment(
    context: dict[str, Any],
    deployment: dict[str, Any],
    request_level: int,
    bs_count: int,
    load_scale: float,
) -> dict[str, float]:
    state = state_from_placements(deployment["placements"], len(context["nodes"]))
    total_delay = 0.0
    total_trans = 0.0
    total_queue = 0.0
    total_missing = 0.0
    variances = []
    f_values_all = []
    overload_count = 0
    node_slot_count = 0

    for slot in context["requests"]:
        traffic, trans_delay, missing_flow = route_slot(context, state, slot, request_level, bs_count)
        rhos = raw_utilization(context, traffic, load_scale)
        capacities = [
            base.degradation_value(rho, context["degradation"]["rho_th"], context["degradation"]["varphi"], context["degradation"]["f_min"])
            for rho in rhos
        ]
        effective_utils = [min(5.0, rho / max(capacity, 1e-12)) for rho, capacity in zip(rhos, capacities)]
        variances.append(base.variance(effective_utils))
        f_values_all.extend(capacities)
        overload_count += sum(1 for rho in rhos if rho >= 1.0)
        node_slot_count += len(rhos)

        queue_delay = 0.0
        for service_name, service in context["services"].items():
            service_idx = int(service_name[1:]) - 1
            _ = service
            for node_idx, flow in enumerate(traffic[service_idx]):
                instances = state[service_idx][node_idx]
                if flow <= 0.0 or instances <= 0:
                    continue
                per_instance = flow / instances
                queue_delay += instances * 50.0 * per_instance * (per_instance + 1.0) / 2.0 / capacities[node_idx]

        missing_delay = missing_flow * 400.0
        total_trans += trans_delay
        total_queue += queue_delay
        total_missing += missing_delay
        total_delay += trans_delay + queue_delay + missing_delay

    slot_count = len(context["requests"])
    assignment = context["tuned_spec"]["assignment"]
    return {
        "mean_end_to_end_delay_ms": total_delay / slot_count if slot_count else 0.0,
        "mean_transmission_delay_ms": total_trans / slot_count if slot_count else 0.0,
        "mean_queue_delay_ms": total_queue / slot_count if slot_count else 0.0,
        "mean_missing_delay_ms": total_missing / slot_count if slot_count else 0.0,
        "load_variance": mean(variances),
        "mean_effective_capacity_ratio": mean(f_values_all),
        "resource_satisfaction_rate": deployment["deployment_success_ratio"],
        "psi": base.compute_psi(context["chains"], context["request_totals"], deployment["placements"], assignment),
        "overload_node_slot_ratio": overload_count / node_slot_count if node_slot_count else 0.0,
    }


def evaluate_policy(
    context: dict[str, Any],
    method_spec: dict[str, str],
    request_level: int,
    bs_count: int,
    load_scale: float,
) -> dict[str, Any]:
    deployment = policies.deploy_method(method_spec, context)
    metrics = evaluate_deployment(context, deployment, request_level, bs_count, load_scale)
    return {
        "method": method_spec["method"],
        "method_id": method_spec["method_id"],
        "category": method_spec["category"],
        "deployment_time_ms": deployment["decision_time_ms"],
        "candidate_evaluations": deployment["candidate_evaluations"],
        "target_instances": deployment["target_instances"],
        "placed_instances": deployment["placed_instances"],
        **metrics,
    }
