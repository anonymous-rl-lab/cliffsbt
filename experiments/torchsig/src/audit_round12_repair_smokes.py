#!/usr/bin/env python3
"""Read-only audit of the stopped Round 12 repair exploration."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    specs = {
        "round12a_v1": (
            "results/round12a_blind_flux_repair_smoke/summary.json",
            12,
            13,
            "SMOKE_STOP_DIAGNOSE",
        ),
        "round12a_v2": (
            "results/round12a_blind_flux_repair_smoke_v2_stratified/summary.json",
            11,
            14,
            "SMOKE_STOP_DIAGNOSE",
        ),
        "round12a_v4": (
            "results/round12a_common_query_repair_smoke_v4/summary.json",
            13,
            16,
            "SMOKE_STOP_DIAGNOSE",
        ),
        "round12b": (
            "results/round12b_influence_repair_smoke/summary.json",
            13,
            16,
            "SMOKE_STOP_DIAGNOSE",
        ),
    }
    checks = {}
    for name, (relative, passed, total, decision) in specs.items():
        summary = load(relative)
        checks[name] = {
            "passed": summary["checks"]["passed"] == passed,
            "total": summary["checks"]["total"] == total,
            "decision": summary["decision"] == decision,
            "smoke_only": summary["claim_status"] == "single-seed smoke only; no manuscript claim",
        }
    abort = load("results/round12a_queried_flux_repair_smoke_v3/ABORT.json")
    checks["round12a_v3"] = {
        "decision": abort["decision"] == "PRETARGET_ACQUISITION_ABORT",
        "no_evaluation_panel": abort["evaluation_panel_generated"] is False,
        "no_target_metrics": abort["target_metrics_computed"] is False,
    }
    v1 = load("results/round12a_blind_flux_repair_smoke/summary.json")["checks"]["values"]
    v4 = load("results/round12a_common_query_repair_smoke_v4/summary.json")["checks"]["values"]
    r12b = load("results/round12b_influence_repair_smoke/summary.json")["checks"]["values"]
    scientific = {
        "v1_high_incident_precision": v1["blind_flux_selector_incident_precision"] == 0.9375,
        "v1_random_still_better": v1["blind_flux_end_risk_advantage_over_random"] < 0,
        "v4_signed_flux_repairs": v4["queried_flux_mean_end_risk_reduction"] > 0,
        "v4_random_and_uncertainty_better": (
            v4["queried_flux_end_risk_advantage_over_random"] < 0
            and v4["queried_flux_end_risk_advantage_over_uncertainty"] < 0
        ),
        "round12b_influence_repairs": r12b["influence_flux_mean_end_risk_reduction"] > 0,
        "round12b_no_ranking_advantage": (
            r12b["influence_flux_end_risk_advantage_over_random"] < 0
            and r12b["influence_flux_end_risk_advantage_over_uncertainty"] < 0
            and r12b["influence_flux_end_risk_advantage_over_queried_flux"] < 0
        ),
    }
    all_pass = all(all(item.values()) for item in checks.values()) and all(scientific.values())
    report = {
        "audit": "round12_control_free_repair_exploration",
        "read_only": True,
        "stage_checks": checks,
        "scientific_consistency": scientific,
        "all_pass": all_pass,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

