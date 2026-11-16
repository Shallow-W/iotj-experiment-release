#!/usr/bin/env python3
"""Paper-axis R3-3 rerun under one shared evaluator.

This runner uses the manuscript scale variables:

- number of MEC servers: 50, 75, 100
- number of requests: 30, 50, 70, 100
- number of microservice types: 5, 7, 10
- number of base stations: 1, 2, 3

All compared methods use the same 100-endpoint source data, the same
resource-feasibility rule, the same FPDWGC domain context, and the same
service-chain routing/evaluation code.  The retained non-RL baselines are
First-fit feasible placement and Load-aware greedy placement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "resub-exp").exists())
sys.path.insert(0, str(ROOT / "resub-exp" / "experiments" / "reviewer_3"))

import manuscript_aligned_support as support  # noqa: E402


EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_ROOT / "results"
METRICS_CSV = RESULTS_DIR / "r3_3_paper_aligned_unified_metrics.csv"
PROFILE_CSV = RESULTS_DIR / "r3_3_paper_aligned_mpd3qn_profile_diagnostic.csv"

VALIDATION_SLOTS = 150
FIXED_MPD3QN_PROFILE_ID = support.PAPER_MPD3QN_PROFILE["profile_id"]
SWEEPS = {
    "server_count": [(50, 50, 10, 1), (75, 50, 10, 1), (100, 50, 10, 1)],
    "request_level": [(100, 30, 10, 1), (100, 50, 10, 1), (100, 70, 10, 1), (100, 100, 10, 1)],
    "microservice_types": [(100, 50, 5, 1), (100, 50, 7, 1), (100, 50, 10, 1)],
    "base_station_count": [(100, 50, 10, 1), (100, 50, 10, 2), (100, 50, 10, 3)],
}


def x_value_for(sweep: str, server_count: int, request_level: int, service_count: int, bs_count: int) -> int:
    return {
        "server_count": server_count,
        "request_level": request_level,
        "microservice_types": service_count,
        "base_station_count": bs_count,
    }[sweep]


def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: round(value, 8) if isinstance(value, float) else value for key, value in row.items()}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_scale = support.calibrate_reference_load_scale()
    method_specs = support.r3_3_method_specs()
    metric_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    context_cache: dict[tuple[int, int], dict[str, Any]] = {}

    for sweep, configurations in SWEEPS.items():
        for server_count, request_level, service_count, bs_count in configurations:
            print(f"[R3-3] {sweep}: servers={server_count}, requests={request_level}, services={service_count}, BS={bs_count}", flush=True)
            cache_key = (server_count, service_count)
            if cache_key not in context_cache:
                context_cache[cache_key] = support.build_context(server_count, service_count, "fpdwgc")
            context = dict(context_cache[cache_key])
            diagnostic_profile, validation_rows = support.select_mpd3qn_profile(
                context,
                request_level,
                bs_count,
                load_scale,
                validation_slots=VALIDATION_SLOTS,
            )
            fixed_profile = dict(support.PAPER_MPD3QN_PROFILE)
            context["mpd3qn_profile"] = fixed_profile
            test_context = support.context_with_request_window(context, VALIDATION_SLOTS, None)
            common = {
                "sweep": sweep,
                "x_value": x_value_for(sweep, server_count, request_level, service_count, bs_count),
                "server_count": server_count,
                "request_level": request_level,
                "microservice_types": service_count,
                "base_station_count": bs_count,
                "split": f"test_slots_{VALIDATION_SLOTS + 1}_300",
                "load_scale_reference": load_scale,
                "state_action_feasibility": "same_fpdwgc_domain_and_same_hard_feasible_node_actions",
                "routing_policy": "same_distance_weighted_service_chain_routing",
                "data_source": "hjw_disanzhangbishe/100endpoints_nms",
            }
            for profile_row in validation_rows:
                profile_rows.append(
                    rounded_row(
                        {
                            **common,
                            **profile_row,
                            "validation_best_profile": diagnostic_profile["profile_id"],
                            "fixed_profile_for_reporting": FIXED_MPD3QN_PROFILE_ID,
                            "selected_for_reporting": int(profile_row["profile_id"] == FIXED_MPD3QN_PROFILE_ID),
                        }
                    )
                )
            for method_spec in method_specs:
                result = support.evaluate_policy(test_context, method_spec, request_level, bs_count, load_scale)
                metric_rows.append(
                    rounded_row(
                        {
                            **common,
                            "selected_mpd3qn_profile": fixed_profile["profile_id"],
                            **result,
                        }
                    )
                )

    support.write_csv(METRICS_CSV, metric_rows)
    support.write_csv(PROFILE_CSV, profile_rows)
    print(METRICS_CSV)
    print(PROFILE_CSV)
    print(f"reference_load_scale={load_scale:.8f}")


if __name__ == "__main__":
    main()
