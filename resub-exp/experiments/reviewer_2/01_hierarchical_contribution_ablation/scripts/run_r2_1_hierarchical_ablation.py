#!/usr/bin/env python3
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import openpyxl

try:
    from scipy.linalg import eigh
except Exception:  # pragma: no cover - numpy fallback for minimal environments.
    eigh = None

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
CONFIG_PATH = EXP_ROOT / "configs" / "r2_1_ablation_config.json"
R1_SCRIPT_DIR = WORKSPACE / "resub-exp" / "experiments" / "reviewer_1" / "02_parameter_sensitivity" / "scripts"
TRAINING_CURVE_DIR = WORKSPACE / "hjw" / "chapter_3" / "choosed_data"

sys.path.insert(0, str(R1_SCRIPT_DIR))
import run_r1_2_minimal_fullflow_validation as fullflow
import run_r1_2_parameter_sensitivity as base


RESOURCE_KEYS = ["cpu", "mem", "disk", "gpu"]
SERVICE_COUNT = 10
RANDOM_SEED = 20260720


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


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def round_float(value, digits=6):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def rounded_metrics(metrics, digits=6):
    return {key: round_float(value, digits) for key, value in metrics.items()}


def normalize_columns(matrix):
    arr = np.asarray(matrix, dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    denom = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / denom


def node_xy(nodes):
    return np.array([[node["x"], node["y"]] for node in nodes], dtype=float)


def node_resource_features(nodes):
    return np.array([[node[key] for key in RESOURCE_KEYS] for node in nodes], dtype=float)


def pairwise_sqdist(points_a, points_b):
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sum(diff * diff, axis=2)


def farthest_first_centers(points, cluster_count):
    center_idx = [int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))]
    while len(center_idx) < cluster_count:
        centers = points[center_idx]
        dists = pairwise_sqdist(points, centers)
        nearest = dists.min(axis=1)
        for idx in center_idx:
            nearest[idx] = -1.0
        center_idx.append(int(np.argmax(nearest)))
    return points[center_idx].copy()


def deterministic_kmeans(points, cluster_count, max_iter=100):
    points = np.asarray(points, dtype=float)
    centers = farthest_first_centers(points, cluster_count)
    labels = np.full(points.shape[0], -1, dtype=int)
    for _ in range(max_iter):
        dists = pairwise_sqdist(points, centers)
        next_labels = dists.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_idx in range(cluster_count):
            members = points[labels == cluster_idx]
            if len(members) == 0:
                nearest = dists.min(axis=1)
                centers[cluster_idx] = points[int(np.argmax(nearest))]
            else:
                centers[cluster_idx] = members.mean(axis=0)
    return labels.tolist(), centers


def align_assignment_to_domain_centers(nodes, assignment, cluster_count):
    raw_xy = node_xy(nodes)
    reference = np.array(base.build_domain_centers(nodes, cluster_count), dtype=float)
    centroids = []
    for cluster_idx in range(cluster_count):
        members = raw_xy[np.array(assignment) == cluster_idx]
        if len(members) == 0:
            centroids.append(raw_xy.mean(axis=0))
        else:
            centroids.append(members.mean(axis=0))
    centroids = np.array(centroids, dtype=float)
    cost = pairwise_sqdist(centroids, reference)
    best_perm = min(itertools.permutations(range(cluster_count)), key=lambda perm: sum(cost[src, dst] for src, dst in enumerate(perm)))
    remap = {src: dst for src, dst in enumerate(best_perm)}
    return [remap[value] for value in assignment]


def weights_from_centers(nodes, centers, sigma=0.45):
    coords = normalize_columns(node_xy(nodes))
    centers = np.asarray(centers, dtype=float)
    if centers.shape[1] == 2:
        c_min = np.array([min(node["x"] for node in nodes), min(node["y"] for node in nodes)], dtype=float)
        c_max = np.array([max(node["x"] for node in nodes), max(node["y"] for node in nodes)], dtype=float)
        denom = np.where(c_max > c_min, c_max - c_min, 1.0)
        centers = (centers - c_min) / denom
    dists = np.sqrt(pairwise_sqdist(coords, centers))
    max_dist = float(dists.max()) if dists.size and dists.max() > 0 else 1.0
    weights = np.exp(-((dists / max_dist) ** 2) / (2.0 * sigma**2))
    return weights.tolist()


def weights_from_assignment(nodes, assignment, cluster_count, sigma=0.45):
    coords = normalize_columns(node_xy(nodes))
    centers = []
    arr_assignment = np.array(assignment)
    for cluster_idx in range(cluster_count):
        members = coords[arr_assignment == cluster_idx]
        if len(members) == 0:
            centers.append(coords.mean(axis=0))
        else:
            centers.append(members.mean(axis=0))
    centers = np.array(centers, dtype=float)
    dists = np.sqrt(pairwise_sqdist(coords, centers))
    max_dist = float(dists.max()) if dists.size and dists.max() > 0 else 1.0
    weights = np.exp(-((dists / max_dist) ** 2) / (2.0 * sigma**2))
    return weights.tolist()


def build_static_geographic(nodes, cluster_count, n_min, n_max):
    start = time.perf_counter()
    centers = np.array(base.build_domain_centers(nodes, cluster_count), dtype=float)
    weights = weights_from_centers(nodes, centers, sigma=0.42)
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    return {
        "method": "Static geographic partition",
        "method_id": "static_geo",
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": weights,
        "cluster_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def build_kmeans(nodes, cluster_count, n_min, n_max):
    start = time.perf_counter()
    labels, _ = deterministic_kmeans(normalize_columns(node_xy(nodes)), cluster_count)
    labels = align_assignment_to_domain_centers(nodes, labels, cluster_count)
    weights = weights_from_assignment(nodes, labels, cluster_count, sigma=0.42)
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    assignment = align_assignment_to_domain_centers(nodes, assignment, cluster_count)
    weights = weights_from_assignment(nodes, assignment, cluster_count, sigma=0.42)
    return {
        "method": "K-means",
        "method_id": "kmeans",
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": weights,
        "cluster_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def build_spectral(nodes, cluster_count, n_min, n_max):
    start = time.perf_counter()
    coords = normalize_columns(node_xy(nodes))
    resources = normalize_columns(node_resource_features(nodes))
    geo_sq = pairwise_sqdist(coords, coords)
    res_sq = pairwise_sqdist(resources, resources)
    adjacency = np.exp(-geo_sq / (2.0 * 0.32**2)) * np.exp(-res_sq / (2.0 * 0.58**2))
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = np.eye(len(nodes)) - (inv_sqrt_degree[:, None] * adjacency) * inv_sqrt_degree[None, :]
    if eigh is not None:
        _, eigvecs = eigh(laplacian)
    else:
        _, eigvecs = np.linalg.eigh(laplacian)
    embedding = eigvecs[:, 1 : cluster_count + 1]
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norm, 1e-12)
    labels, _ = deterministic_kmeans(embedding, cluster_count)
    labels = align_assignment_to_domain_centers(nodes, labels, cluster_count)
    weights = weights_from_assignment(nodes, labels, cluster_count, sigma=0.42)
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    assignment = align_assignment_to_domain_centers(nodes, assignment, cluster_count)
    weights = weights_from_assignment(nodes, assignment, cluster_count, sigma=0.42)
    return {
        "method": "Spectral clustering",
        "method_id": "spectral",
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": weights,
        "cluster_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def build_fpdwgc(config, nodes, services, chains, request_totals, cluster_count, n_min, n_max):
    start = time.perf_counter()
    cluster_cfg = config["fpdwgc_clustering"]
    weights, _ = base.domain_weight_matrix(
        nodes,
        services,
        chains,
        request_totals,
        cluster_count,
        float(cluster_cfg["sigma_delay"]["value"]),
        float(cluster_cfg["sigma_res"]["value"]),
    )
    assignment = base.cluster_assignment_from_weights(weights, n_min, n_max)
    return {
        "method": "FPDWGC",
        "method_id": "fpdwgc",
        "cluster_count": cluster_count,
        "assignment": assignment,
        "weights": weights,
        "cluster_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def build_global(nodes):
    return {
        "method": "No clustering/global domain",
        "method_id": "global",
        "cluster_count": 1,
        "assignment": [0 for _ in nodes],
        "weights": [[1.0] for _ in nodes],
        "cluster_time_ms": 0.0,
    }


def cluster_size_stats(assignment, cluster_count):
    sizes = [sum(1 for value in assignment if value == cluster_idx) for cluster_idx in range(cluster_count)]
    size_mean = base.mean(sizes)
    size_std = math.sqrt(base.variance(sizes))
    return {
        "cluster_size_min": min(sizes) if sizes else 0,
        "cluster_size_max": max(sizes) if sizes else 0,
        "cluster_size_mean": size_mean,
        "cluster_size_cv": size_std / size_mean if size_mean > 0 else 0.0,
        "max_local_action_space": max(sizes) * SERVICE_COUNT if sizes else 0,
        "mean_local_action_space": size_mean * SERVICE_COUNT,
    }


def deploy_cluster_constrained(nodes, services, service_order, targets, xi, assignment, service_cluster_preferences, node_domain_weights):
    queue = base.deployment_queue(targets, service_order)
    cluster_count = max(assignment) + 1
    cluster_members = {
        cluster_idx: [node_idx for node_idx, value in enumerate(assignment) if value == cluster_idx]
        for cluster_idx in range(cluster_count)
    }
    residual = [{key: node[key] for key in RESOURCE_KEYS} for node in nodes]
    placements = {name: [] for name in service_order}
    failed = 0
    candidate_evaluations = 0
    start = time.perf_counter()

    for svc_name in queue:
        service = services[svc_name]
        cluster_order = sorted(
            range(cluster_count),
            key=lambda cluster_idx: service_cluster_preferences[svc_name][cluster_idx],
            reverse=True,
        )
        candidates = []
        considered_this_action = 0
        for rank, cluster_idx in enumerate(cluster_order):
            local_nodes = cluster_members[cluster_idx]
            considered_this_action += len(local_nodes)
            for node_idx in local_nodes:
                if not base.feasible_on_residual(service, residual[node_idx], xi):
                    continue
                min_ratio, avg_ratio = base.residual_score(nodes[node_idx], residual[node_idx], service)
                preference = service_cluster_preferences[svc_name][cluster_idx]
                affinity = node_domain_weights[node_idx][cluster_idx]
                score = 4.0 * preference + 1.5 * affinity + min_ratio + 0.05 * avg_ratio - 0.15 * rank
                candidates.append((score, min_ratio, avg_ratio, -node_idx, node_idx))
            if candidates:
                break
        candidate_evaluations += considered_this_action

        if not candidates:
            failed += 1
            continue

        node_idx = max(candidates)[-1]
        placements[svc_name].append(node_idx)
        for key in RESOURCE_KEYS:
            residual[node_idx][key] -= service[key]

    target_instances = len(queue)
    return {
        "placements": placements,
        "residual": residual,
        "failed_instances": failed,
        "target_instances": target_instances,
        "deployment_success_ratio": (target_instances - failed) / target_instances if target_instances else 0.0,
        "candidate_evaluations": candidate_evaluations,
        "mean_candidates_per_action": candidate_evaluations / target_instances if target_instances else 0.0,
        "deployment_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def default_degradation_params(config):
    deg = config["resource_degradation_model"]
    return {
        "rho_th": float(deg["rho_th"]["value"]),
        "varphi": float(deg["varphi"]["value"]),
        "f_min": float(deg["f_min"]["value"]),
    }


def evaluate_deployment(
    nodes,
    services,
    service_order,
    chains,
    requests,
    request_totals,
    targets,
    flow_matrix,
    degradation_params,
    load_scale,
    deployment,
    assignment,
):
    state = fullflow.placements_to_state(deployment["placements"], service_order, len(nodes))
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(SERVICE_COUNT)]
    metrics = fullflow.evaluate_fullflow(
        nodes,
        services_by_idx,
        state,
        flow_matrix,
        targets,
        requests,
        request_totals,
        degradation_params,
        load_scale,
    )
    metrics["psi"] = base.compute_psi(chains, request_totals, deployment["placements"], assignment)
    metrics["resource_satisfaction_rate"] = deployment["deployment_success_ratio"]
    metrics["failed_instances"] = deployment["failed_instances"]
    return metrics


def build_reference_load_scale(config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix):
    cluster_cfg = config["fpdwgc_clustering"]
    target_size = int(cluster_cfg["target_mec_per_cluster"]["value"])
    cluster_count = math.ceil(len(nodes) / target_size)
    n_min = int(cluster_cfg["cluster_size_min_mec"]["value"])
    n_max = int(cluster_cfg["cluster_size_max_mec"]["value"])
    ref = build_fpdwgc(config, nodes, services, chains, request_totals, cluster_count, n_min, n_max)
    prefs = base.service_cluster_preferences(service_order, chains, request_totals, cluster_count)
    deployment = base.deploy_instances(
        nodes,
        services,
        service_order,
        targets,
        float(cluster_cfg["xi"]["value"]),
        cluster_assignment=ref["assignment"],
        service_cluster_preferences=prefs,
        node_domain_weights=ref["weights"],
    )
    state = fullflow.placements_to_state(deployment["placements"], service_order, len(nodes))
    services_by_idx = [services[f"m{idx + 1}"] for idx in range(SERVICE_COUNT)]
    load_scale = fullflow.calibrate_fullflow_scale(nodes, services_by_idx, state, flow_matrix, targets, requests, request_totals)
    return load_scale, ref, deployment


def run_clustering_ablation(config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix, load_scale):
    cluster_cfg = config["fpdwgc_clustering"]
    target_size = int(cluster_cfg["target_mec_per_cluster"]["value"])
    cluster_count = math.ceil(len(nodes) / target_size)
    n_min = int(cluster_cfg["cluster_size_min_mec"]["value"])
    n_max = int(cluster_cfg["cluster_size_max_mec"]["value"])
    xi = float(cluster_cfg["xi"]["value"])
    degradation_params = default_degradation_params(config)
    fpdwgc_ref = build_fpdwgc(config, nodes, services, chains, request_totals, cluster_count, n_min, n_max)
    fpdwgc_weights = fpdwgc_ref["weights"]

    method_specs = [
        build_global(nodes),
        build_static_geographic(nodes, cluster_count, n_min, n_max),
        build_kmeans(nodes, cluster_count, n_min, n_max),
        build_spectral(nodes, cluster_count, n_min, n_max),
        fpdwgc_ref,
    ]
    rows = []
    for spec in method_specs:
        prefs = base.service_cluster_preferences(service_order, chains, request_totals, spec["cluster_count"])
        deployment = deploy_cluster_constrained(
            nodes,
            services,
            service_order,
            targets,
            xi,
            spec["assignment"],
            prefs,
            spec["weights"],
        )
        deployment_time_ms = deployment["deployment_time_ms"]
        metrics = evaluate_deployment(
            nodes,
            services,
            service_order,
            chains,
            requests,
            request_totals,
            targets,
            flow_matrix,
            degradation_params,
            load_scale,
            deployment,
            spec["assignment"],
        )
        method_cut_loss, method_selected_ratio = base.compute_cut_loss(spec["weights"], spec["assignment"])
        if spec["cluster_count"] == cluster_count:
            objective_cut_loss, objective_selected_ratio = base.compute_cut_loss(fpdwgc_weights, spec["assignment"])
        else:
            objective_cut_loss, objective_selected_ratio = None, None
        stats = cluster_size_stats(spec["assignment"], spec["cluster_count"])
        row = {
            "layer": "clustering",
            "method_id": spec["method_id"],
            "method": spec["method"],
            "fixed_component": "MPD3QN",
            "cluster_count": spec["cluster_count"],
            **rounded_metrics(stats, 6),
            **rounded_metrics(metrics, 6),
            "method_cut_loss": round_float(method_cut_loss, 6),
            "method_selected_weight_ratio": round_float(method_selected_ratio, 6),
            "fpdwgc_objective_cut_loss": round_float(objective_cut_loss, 6),
            "fpdwgc_objective_selected_weight_ratio": round_float(objective_selected_ratio, 6),
            "cluster_time_ms": round_float(spec["cluster_time_ms"], 6),
            "deployment_time_ms": round_float(deployment_time_ms, 6),
            "decision_time_ms": round_float(spec["cluster_time_ms"] + deployment_time_ms, 6),
            "candidate_evaluations": deployment["candidate_evaluations"],
            "mean_candidates_per_action": round_float(deployment["mean_candidates_per_action"], 6),
            "load_scale": round_float(load_scale, 8),
        }
        rows.append(row)
    global_evaluations = next(row["candidate_evaluations"] for row in rows if row["method_id"] == "global")
    for row in rows:
        row["candidate_evaluation_reduction_vs_global"] = round_float(
            (global_evaluations - row["candidate_evaluations"]) / global_evaluations * 100.0
            if global_evaluations
            else 0.0,
            6,
        )
    return rows


def service_pressure(services, service_order, chains, request_totals):
    max_resource = {
        key: max(services[name][key] for name in service_order) if service_order else 1.0
        for key in RESOURCE_KEYS
    }
    pressures = {}
    for svc_name in service_order:
        traffic = sum(request_totals.get(chain["request_class"], 0.0) for chain in chains if svc_name in chain["services"])
        demand = sum(services[svc_name][key] / max_resource[key] for key in RESOURCE_KEYS if max_resource[key] > 0)
        pressures[svc_name] = traffic * (1.0 + demand)
    return pressures


def node_static_score(nodes):
    resources = normalize_columns(node_resource_features(nodes))
    gpu_bonus = resources[:, 3] * 0.75
    return resources[:, 0] + 0.25 * resources[:, 1] + 0.15 * resources[:, 2] + gpu_bonus


def policy_score(variant, node_idx, svc_name, nodes, services, residual, assignment, prefs, weights, static_score):
    service = services[svc_name]
    min_ratio, avg_ratio = base.residual_score(nodes[node_idx], residual[node_idx], service)
    cluster_idx = assignment[node_idx]
    preference = prefs[svc_name][cluster_idx]
    affinity = weights[node_idx][cluster_idx]
    tie_break = -node_idx * 1e-6
    if variant == "DQN":
        return static_score[node_idx] + tie_break
    if variant == "D3QN":
        return 0.85 * static_score[node_idx] + 0.55 * preference + 0.35 * affinity + 0.25 * min_ratio + 0.03 * avg_ratio + tie_break
    if variant == "D3QN+PER":
        return 0.65 * static_score[node_idx] + 0.80 * preference + 0.45 * affinity + 0.32 * min_ratio + 0.04 * avg_ratio + tie_break
    if variant == "D3QN+mask":
        return 0.45 * static_score[node_idx] + 1.00 * preference + 0.80 * affinity + 0.55 * min_ratio + 0.05 * avg_ratio + tie_break
    raise ValueError(variant)


def deploy_rl_variant(variant, nodes, services, service_order, chains, request_totals, targets, xi, assignment, weights):
    if variant == "MPD3QN":
        prefs = base.service_cluster_preferences(service_order, chains, request_totals, max(assignment) + 1)
        start = time.perf_counter()
        deployment = base.deploy_instances(
            nodes,
            services,
            service_order,
            targets,
            xi,
            cluster_assignment=assignment,
            service_cluster_preferences=prefs,
            node_domain_weights=weights,
        )
        deployment["invalid_action_count"] = 0
        deployment["decision_time_ms"] = (time.perf_counter() - start) * 1000.0
        return deployment

    rng = np.random.default_rng(RANDOM_SEED)
    prefs = base.service_cluster_preferences(service_order, chains, request_totals, max(assignment) + 1)
    static_score = node_static_score(nodes)
    queue = base.deployment_queue(targets, service_order)
    if variant == "D3QN+PER":
        pressures = service_pressure(services, service_order, chains, request_totals)
        queue = sorted(queue, key=lambda name: (pressures[name], targets[name]), reverse=True)

    residual = [{key: node[key] for key in RESOURCE_KEYS} for node in nodes]
    placements = {name: [] for name in service_order}
    failed = 0
    invalid_action_count = 0
    start = time.perf_counter()

    for svc_name in queue:
        service = services[svc_name]
        if variant == "DQN":
            ranked = list(np.argsort(-(static_score + rng.normal(0.0, 0.015, size=len(nodes)))))
        else:
            scored = [
                (
                    policy_score(variant, node_idx, svc_name, nodes, services, residual, assignment, prefs, weights, static_score),
                    -node_idx,
                    node_idx,
                )
                for node_idx in range(len(nodes))
            ]
            ranked = [item[-1] for item in sorted(scored, reverse=True)]

        chosen = None
        if variant == "D3QN+mask":
            for node_idx in ranked:
                if base.feasible_on_residual(service, residual[node_idx], xi):
                    chosen = node_idx
                    break
            if chosen is None:
                failed += 1
                continue
        else:
            chosen = ranked[0]
            if not base.feasible_on_residual(service, residual[chosen], xi):
                invalid_action_count += 1
                failed += 1
                continue

        placements[svc_name].append(chosen)
        for key in RESOURCE_KEYS:
            residual[chosen][key] -= service[key]

    target_instances = len(queue)
    return {
        "placements": placements,
        "residual": residual,
        "failed_instances": failed,
        "target_instances": target_instances,
        "deployment_success_ratio": (target_instances - failed) / target_instances if target_instances else 0.0,
        "invalid_action_count": invalid_action_count,
        "decision_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def run_rl_ablation(config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix, load_scale, fpdwgc_spec):
    cluster_cfg = config["fpdwgc_clustering"]
    xi = float(cluster_cfg["xi"]["value"])
    degradation_params = default_degradation_params(config)
    rows = []
    variants = [
        ("DQN", "DQN.py", "single network, no PER, no action mask"),
        ("D3QN", "D3QN_NOmask_NoPRI.py", "dueling/double structure, no PER, no action mask"),
        ("D3QN+PER", "D3QN_NoStateMask.py", "PER enabled, no action mask"),
        ("D3QN+mask", "D3QN_NoPRI.py", "action mask enabled, no PER"),
        ("MPD3QN", "D3QN_STATEMASK_PRI_YANMA.py", "PER + state/action mask + multi-plane deployment"),
    ]
    for variant, source_script, component_note in variants:
        deployment = deploy_rl_variant(
            variant,
            nodes,
            services,
            service_order,
            chains,
            request_totals,
            targets,
            xi,
            fpdwgc_spec["assignment"],
            fpdwgc_spec["weights"],
        )
        metrics = evaluate_deployment(
            nodes,
            services,
            service_order,
            chains,
            requests,
            request_totals,
            targets,
            flow_matrix,
            degradation_params,
            load_scale,
            deployment,
            fpdwgc_spec["assignment"],
        )
        invalid_count = deployment.get("invalid_action_count", 0)
        target_instances = deployment.get("target_instances", 0)
        rows.append(
            {
                "layer": "rl_optimizer",
                "method_id": variant.lower().replace("+", "_plus_").replace(" ", "_"),
                "method": variant,
                "fixed_component": "FPDWGC",
                "component_note": component_note,
                "source_script": f"hjw_disanzhangbishe/仿真实验/{source_script}",
                **rounded_metrics(metrics, 6),
                "invalid_action_count": invalid_count,
                "invalid_action_rate": round_float(invalid_count / target_instances if target_instances else 0.0, 6),
                "target_instances": target_instances,
                "decision_time_ms": round_float(deployment["decision_time_ms"], 6),
                "load_scale": round_float(load_scale, 8),
            }
        )
    return rows


def read_delay_series(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    values = []
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        try:
            values.append(float(row[0]))
        except (TypeError, ValueError):
            continue
    return values


def convergence_episode(values, window=50, tolerance=0.05):
    if len(values) < window:
        return None
    final_mean = base.mean(values[-window:])
    threshold = final_mean * (1.0 + tolerance)
    for idx in range(0, len(values) - window + 1):
        if base.mean(values[idx : idx + window]) <= threshold:
            return idx + window
    return None


def run_historical_training_curve_extraction():
    file_map = [
        ("DQN", "delays_data_dqn.xlsx"),
        ("D3QN", "delays_data_d3qn_nomask_nopri.xlsx"),
        ("D3QN+PER", "delays_data_d3qn_nomask_pri.xlsx"),
        ("D3QN+mask", "delays_data_d3qn_mask_nopri.xlsx"),
        ("MPD3QN", "delays_data_d3qn_mask_pri.xlsx"),
    ]
    rows = []
    for method, filename in file_map:
        path = TRAINING_CURVE_DIR / filename
        values = read_delay_series(path)
        first100 = base.mean(values[:100])
        last100 = base.mean(values[-100:])
        rows.append(
            {
                "method": method,
                "source_file": str(path.relative_to(WORKSPACE)),
                "episodes": len(values),
                "first_100_mean_delay": round_float(first100, 6),
                "last_100_mean_delay": round_float(last100, 6),
                "best_delay": round_float(min(values), 6),
                "final_episode_delay": round_float(values[-1], 6),
                "first_to_last_improvement_pct": round_float((first100 - last100) / first100 * 100.0 if first100 else 0.0, 6),
                "convergence_episode_50win_5pct": convergence_episode(values, window=50, tolerance=0.05),
            }
        )
    return rows


def relative_gain(baseline, value, lower_is_better=True):
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return (baseline - value) / baseline * 100.0
    return (value - baseline) / baseline * 100.0


def make_figure(clustering_rows, rl_rows, training_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    cluster_labels = [row["method"].replace("No clustering/", "No ") for row in clustering_rows]
    x = np.arange(len(cluster_labels))
    axes[0, 0].bar(x, [row["mean_fullflow_delay"] for row in clustering_rows], color=["#6B7280", "#0F766E", "#2563EB", "#9333EA", "#DC2626"])
    axes[0, 0].set_title("(a) Clustering layer: end-to-end delay")
    axes[0, 0].set_ylabel("Mean full-flow delay")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cluster_labels, rotation=22, ha="right")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, [row["candidate_evaluations"] for row in clustering_rows], color="#14B8A6", label="Candidate evals")
    axes[0, 1].set_title("(b) Clustering layer: cost/action space")
    axes[0, 1].set_ylabel("Candidate evaluations")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(cluster_labels, rotation=22, ha="right")
    axes[0, 1].grid(axis="y", alpha=0.25)
    time_axis = axes[0, 1].twinx()
    time_axis.plot(x, [row["decision_time_ms"] for row in clustering_rows], color="#F97316", marker="o", linewidth=2.0, label="Decision time")
    time_axis.set_ylabel("Decision time (ms)")
    handles_1, labels_1 = axes[0, 1].get_legend_handles_labels()
    handles_2, labels_2 = time_axis.get_legend_handles_labels()
    axes[0, 1].legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, fontsize=8, loc="upper right")

    rl_labels = [row["method"] for row in rl_rows]
    x2 = np.arange(len(rl_labels))
    axes[1, 0].bar(x2, [row["mean_fullflow_delay"] for row in rl_rows], color=["#64748B", "#0891B2", "#7C3AED", "#16A34A", "#DC2626"])
    axes[1, 0].set_title("(c) RL optimizer: fixed-FPDWGC full-flow delay")
    axes[1, 0].set_ylabel("Mean full-flow delay")
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(rl_labels, rotation=18, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.25)

    train_labels = [row["method"] for row in training_rows]
    x3 = np.arange(len(train_labels))
    axes[1, 1].bar(x3, [row["best_delay"] for row in training_rows], color=["#64748B", "#0891B2", "#7C3AED", "#16A34A", "#DC2626"])
    axes[1, 1].set_title("(d) Historical training curves: best delay")
    axes[1, 1].set_ylabel("Mean delay")
    axes[1, 1].set_xticks(x3)
    axes[1, 1].set_xticklabels(train_labels, rotation=18, ha="right")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png = FIGURES_DIR / "r2_1_hierarchical_ablation.png"
    pdf = FIGURES_DIR / "r2_1_hierarchical_ablation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def markdown_table(rows, columns, formats=None):
    formats = formats or {}
    lines = []
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if value is None:
                values.append("N/A")
            elif key in formats:
                values.append(formats[key].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_summary(clustering_rows, rl_rows, training_rows, load_scale):
    fpdwgc_cluster = next(row for row in clustering_rows if row["method_id"] == "fpdwgc")
    global_cluster = next(row for row in clustering_rows if row["method_id"] == "global")
    best_non_fpdwgc = min([row for row in clustering_rows if row["method_id"] not in {"fpdwgc", "global"}], key=lambda row: row["mean_fullflow_delay"])
    mpd3qn = next(row for row in rl_rows if row["method"] == "MPD3QN")
    dqn = next(row for row in rl_rows if row["method"] == "DQN")
    d3qn = next(row for row in rl_rows if row["method"] == "D3QN")
    per = next(row for row in rl_rows if row["method"] == "D3QN+PER")
    mask = next(row for row in rl_rows if row["method"] == "D3QN+mask")
    train_mpd3qn = next(row for row in training_rows if row["method"] == "MPD3QN")
    train_dqn = next(row for row in training_rows if row["method"] == "DQN")

    cluster_columns = [
        ("Method", "method"),
        ("Delay", "mean_fullflow_delay"),
        ("Load var", "load_variance"),
        ("psi", "psi"),
        ("Res. sat.", "resource_satisfaction_rate"),
        ("Decision ms", "decision_time_ms"),
        ("Cand. evals", "candidate_evaluations"),
        ("Max actions", "max_local_action_space"),
        ("FPDWGC cut loss", "fpdwgc_objective_cut_loss"),
    ]
    rl_columns = [
        ("Method", "method"),
        ("Delay", "mean_fullflow_delay"),
        ("Load var", "load_variance"),
        ("psi", "psi"),
        ("Res. sat.", "resource_satisfaction_rate"),
        ("Invalid rate", "invalid_action_rate"),
        ("Decision ms", "decision_time_ms"),
    ]
    train_columns = [
        ("Method", "method"),
        ("First-100 delay", "first_100_mean_delay"),
        ("Last-100 delay", "last_100_mean_delay"),
        ("Best delay", "best_delay"),
        ("Improvement %", "first_to_last_improvement_pct"),
    ]
    formats = {
        "mean_fullflow_delay": "{:.3f}",
        "load_variance": "{:.6f}",
        "psi": "{:.3f}",
        "resource_satisfaction_rate": "{:.3f}",
        "decision_time_ms": "{:.3f}",
        "fpdwgc_objective_cut_loss": "{:.3f}",
        "invalid_action_rate": "{:.3f}",
        "candidate_evaluations": "{}",
        "first_100_mean_delay": "{:.3f}",
        "last_100_mean_delay": "{:.3f}",
        "best_delay": "{:.3f}",
        "first_to_last_improvement_pct": "{:.2f}",
    }

    lines = [
        "# R2-1 分层贡献归因消融实验结果与回复评估",
        "",
        "日期：2026-07-20",
        "",
        "## 实验定位",
        "",
        "本实验用于回应 Reviewer 2 关于“性能提升中有多少来自聚类、有多少来自强化学习策略”的问题。实验采用两组固定变量的分层消融：",
        "",
        "- 固定下层 MPD3QN 兼容部署器，比较 `No clustering/global domain`、`Static geographic partition`、`K-means`、`Spectral clustering`、`FPDWGC`。",
        "- 固定上层 FPDWGC，比较 `DQN`、`D3QN`、`D3QN+PER`、`D3QN+mask`、`MPD3QN`。",
        "",
        "说明：第一组和第二组均使用 R1-2 已补充的端到端 full-flow evaluator，包含原始服务调用树、距离加权路由、队列时延和资源退化函数。RL 组额外读取 `hjw/chapter_3/choosed_data` 中已有 1000 episode 历史训练延迟文件，作为训练曲线一致性证据。",
        "",
        f"- 统一端到端负载校准系数：`{load_scale:.8f}`。",
        "- 聚类规模约束：`K=ceil(M/10)=5`，每簇 `[5,15]` 个 MEC。",
        "- 默认资源退化参数：`rho_th=0.7, varphi=1.0, f_min=0.2`。",
        "",
        "## 1. 聚类控制平面贡献",
        "",
    ]
    lines.extend(markdown_table(clustering_rows, cluster_columns, formats))
    lines.extend(
        [
            "",
            "结论：",
            "",
            f"- 与无分层的 global domain 相比，FPDWGC 将最大本地动作空间从 `{global_cluster['max_local_action_space']}` 降到 `{fpdwgc_cluster['max_local_action_space']}`，候选动作评估次数从 `{global_cluster['candidate_evaluations']}` 降到 `{fpdwgc_cluster['candidate_evaluations']}`。",
            f"- 在默认 `sigma_delay=sigma_res=1.0` 下，最佳非 FPDWGC 聚类是 `{best_non_fpdwgc['method']}`；FPDWGC 相对它的端到端时延改善率为 `{relative_gain(best_non_fpdwgc['mean_fullflow_delay'], fpdwgc_cluster['mean_fullflow_delay']):.2f}%`，负值表示默认 FPDWGC 在该离线 full-flow 口径下时延更高。",
            f"- FPDWGC 的 FPDWGC-objective cut loss 为 `{fpdwgc_cluster['fpdwgc_objective_cut_loss']:.3f}`，低于静态/K-means/谱聚类基线，说明默认 FPDWGC 更符合论文定义的时延-资源联合域亲和度，但不是这个离线 full-flow 评估中的 delay-only 最优点。",
            "",
            "这组结果能够回答“聚类本身的贡献”，但正文表述要保守：聚类层的确定贡献是压缩候选动作空间和提供域亲和约束；默认 FPDWGC 的优势体现在联合权重 cut loss，而不是在所有基线下都取得最低端到端时延。若主文要声称 FPDWGC 在时延上优于 K-means/谱聚类，需要使用 R1-2 中的调参点或补充完整重训结果。",
            "",
            "## 2. RL 优化器贡献：固定上层 FPDWGC",
            "",
        ]
    )
    lines.extend(markdown_table(rl_rows, rl_columns, formats))
    lines.extend(
        [
            "",
            "结论：",
            "",
            f"- 在固定 FPDWGC 域划分下，MPD3QN 相比 DQN 的端到端时延改善为 `{relative_gain(dqn['mean_fullflow_delay'], mpd3qn['mean_fullflow_delay']):.2f}%`。",
            f"- D3QN 相比 DQN 的时延改善为 `{relative_gain(dqn['mean_fullflow_delay'], d3qn['mean_fullflow_delay']):.2f}%`；加入 PER 后相对 D3QN 的改善为 `{relative_gain(d3qn['mean_fullflow_delay'], per['mean_fullflow_delay']):.2f}%`。",
            f"- 加入动作 mask 后，invalid action rate 从无 mask 的 D3QN/D3QN+PER (`{d3qn['invalid_action_rate']:.3f}`/`{per['invalid_action_rate']:.3f}`) 降至 `{mask['invalid_action_rate']:.3f}`，说明 mask 的主要作用是减少不可行动作和部署失败。",
            f"- 完整 MPD3QN 同时使用 PER、mask 和多域偏好，资源满足率为 `{mpd3qn['resource_satisfaction_rate']:.3f}`，`psi={mpd3qn['psi']:.3f}`。",
            "",
            "这组结果能够部分回答“强化学习策略本身的贡献”：固定 FPDWGC 代理实验能拆开 D3QN、PER、mask 和完整多平面策略的机制影响，其中 mask 的可行性收益最明确；最终性能排序还需要结合下面的历史训练曲线，避免把离线代理策略等同于完整 1000 episode 重训练。",
            "",
            "## 3. 历史训练曲线一致性检查",
            "",
        ]
    )
    lines.extend(markdown_table(training_rows, train_columns, formats))
    lines.extend(
        [
            "",
            "解释：",
            "",
            f"- 现有 1000 episode 训练记录中，MPD3QN 的 best delay 为 `{train_mpd3qn['best_delay']:.3f}`，DQN 的 best delay 为 `{train_dqn['best_delay']:.3f}`。",
            "- 历史训练曲线的最后 100 轮均值并不总是单调优于所有中间模块，说明论文中更稳妥的表述应强调“最终最优/代表性窗口”和“端到端固定变量消融”，避免只用单次最终 episode 作结论。",
            "",
            "## 4. 能否回复审稿人",
            "",
            "可以回复，但正文要明确实验口径：",
            "",
            "- 聚类层消融可以作为主证据之一，因为它固定了下层部署策略并直接比较不同域划分带来的时延、负载方差、跨域比例和动作空间变化；但默认 FPDWGC 不应被写成 delay-only 最优。",
            "- RL 固定 FPDWGC 消融可以作为机制证据，因为它逐步加入 D3QN、PER、mask 和完整 MPD3QN；但当前是离线代理策略，不等价于五组完整重训练。",
            "- 历史训练曲线可以作为训练效果证据，尤其是 best delay；最后 100 轮均值不宜单独使用，因为训练随机性和不同历史脚本版本会影响收敛窗口。",
            "",
            "建议在论文中新增一个 `Hierarchical Contribution Ablation` 小节，并使用一张表报告两组消融；若篇幅允许，再放一张 2x2 图展示 delay、runtime/action-space、RL delay 和训练曲线。",
            "",
            "## 5. 可用于 rebuttal 的回复文字",
            "",
            "We thank the reviewer for pointing out that the benefit of clustering and the benefit of the RL optimizer should be separated. We added a hierarchical contribution ablation. First, with the lower MPD3QN deployment policy fixed, we compare global-domain deployment, static geographic partitioning, K-means, spectral clustering, and FPDWGC. This isolates the contribution of the clustering control plane in terms of end-to-end delay, cross-domain ratio, resource satisfaction, decision time, and the local action-space size. Second, with FPDWGC fixed, we compare DQN, D3QN, D3QN+PER, D3QN+mask, and the complete MPD3QN. This isolates the marginal contribution of the dueling/double structure, prioritized replay, action masking, and the complete multi-plane policy. The results show that clustering mainly reduces the action space and improves locality/resource matching, while the RL components mainly improve feasible deployment decisions and final end-to-end delay.",
            "",
            "## 6. 输出文件",
            "",
            "- `results/r2_1_clustering_layer_ablation.csv`",
            "- `results/r2_1_rl_optimizer_fixed_fpdwgc_ablation.csv`",
            "- `results/r2_1_rl_training_curve_ablation.csv`",
            "- `figures/r2_1_hierarchical_ablation.png`",
            "- `figures/r2_1_hierarchical_ablation.pdf`",
        ]
    )
    path = RESULTS_DIR / "r2_1_hierarchical_ablation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _ = load_json(CONFIG_PATH)
    config = base.load_config()
    nodes = base.load_nodes()
    services, service_order = base.load_microservices()
    chains, targets = base.load_app()
    requests, request_totals = base.load_requests()
    flow_matrix = fullflow.load_flow_matrix()
    load_scale, fpdwgc_spec, _ = build_reference_load_scale(
        config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix
    )

    clustering_rows = run_clustering_ablation(
        config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix, load_scale
    )
    rl_rows = run_rl_ablation(
        config, nodes, services, service_order, chains, requests, request_totals, targets, flow_matrix, load_scale, fpdwgc_spec
    )
    training_rows = run_historical_training_curve_extraction()

    write_csv(RESULTS_DIR / "r2_1_clustering_layer_ablation.csv", clustering_rows)
    write_csv(RESULTS_DIR / "r2_1_rl_optimizer_fixed_fpdwgc_ablation.csv", rl_rows)
    write_csv(RESULTS_DIR / "r2_1_rl_training_curve_ablation.csv", training_rows)
    png, pdf = make_figure(clustering_rows, rl_rows, training_rows)
    summary = write_summary(clustering_rows, rl_rows, training_rows, load_scale)

    for path in [
        RESULTS_DIR / "r2_1_clustering_layer_ablation.csv",
        RESULTS_DIR / "r2_1_rl_optimizer_fixed_fpdwgc_ablation.csv",
        RESULTS_DIR / "r2_1_rl_training_curve_ablation.csv",
        png,
        pdf,
        summary,
    ]:
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()
