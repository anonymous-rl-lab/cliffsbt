#!/usr/bin/env python3
"""Phase 2: reveal confirmation labels and evaluate every frozen gate once."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import (
    DATA, MODEL_SEEDS, PACKAGE, SCHEDULE_IDS, TARGET_FAMILIES, exact_flow,
    persistent_cliff, read_json, seed_cluster_interval, sha256, write_json,
)


def audit_package() -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "code" / "audit_package.py"), "--strict"],
        check=False,
    )
    if result.returncode:
        raise RuntimeError("registration package audit failed")


def verify_blind(workspace: Path) -> None:
    commitment = workspace / "BLIND_OUTPUTS.sha256"
    if not commitment.exists():
        raise RuntimeError("blind commitment is missing")
    for line in commitment.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        path = workspace / filename
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"blind commitment mismatch: {filename}")
    metadata = read_json(workspace / "run_metadata.json")
    provenance = read_json(PACKAGE / "PROVENANCE.json")
    accepted_manifests = {
        sha256(PACKAGE / "PACKAGE_MANIFEST.sha256"),
        provenance["osf_registration_manifest_sha256"],
    }
    if metadata["package_manifest_sha256"] not in accepted_manifests:
        raise RuntimeError("registered package changed between phases")
    if metadata.get("confirmation_labels_scored") is not False:
        raise RuntimeError("phase-1 outcome-blind declaration is invalid")


def confirmation_truth() -> dict[int, np.ndarray]:
    streams = read_json(DATA / "TARGET_STREAMS_FROZEN.json")["confirmation"]
    grouped = {family: [] for family in TARGET_FAMILIES}
    for stream in streams:
        grouped[int(stream["family"])].append(stream)
    result = {}
    base_order = None
    for family in TARGET_FAMILIES:
        grouped[family].sort(key=lambda stream: tuple(stream["base"]))
        bases = [tuple(stream["base"]) for stream in grouped[family]]
        if base_order is None:
            base_order = bases
        elif bases != base_order:
            raise RuntimeError("confirmation identity order differs between families")
        result[family] = np.asarray([int(stream["class_index"]) for stream in grouped[family]], dtype=np.int8)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def event_time(risk: np.ndarray) -> int:
    event = persistent_cliff(risk.tolist())
    return 13 if event is None else event


def phase2(workspace: Path) -> dict:
    verify_blind(workspace)
    gates = read_json(PACKAGE / "CLAIMS_AND_GATES.json")
    assessment = read_json(workspace / "calibration_assessment.json")
    alarms = {
        (int(row["seed"]), int(row["schedule_id"]), int(row["family"])): row
        for row in read_json(workspace / "blind_alarm_rows.json")
    }
    arrays = np.load(workspace / "blind_predictions.npz")
    if arrays["seeds"].tolist() != MODEL_SEEDS:
        raise RuntimeError("blind model-seed order mismatch")
    if arrays["schedule_ids"].tolist() != SCHEDULE_IDS:
        raise RuntimeError("blind schedule order mismatch")
    if arrays["families"].tolist() != TARGET_FAMILIES:
        raise RuntimeError("blind family order mismatch")
    truth = confirmation_truth()
    baseline = arrays["baseline_predicted"]
    deployed = arrays["deployed_repair_predicted"]

    path_rows, repair_path_rows, closure_errors = [], [], []
    seed_warning = {}
    seed_repair_risk_gain, seed_repair_event_gain = {}, {}
    model_family = {}
    for seed_index, seed in enumerate(MODEL_SEEDS):
        cliffs = noncliffs = timely = false = 0
        leads, seed_risk_gains, seed_event_gains = [], [], []
        for schedule_index, schedule_id in enumerate(SCHEDULE_IDS):
            for family_index, family in enumerate(TARGET_FAMILIES):
                target = truth[family]
                baseline_status = baseline[seed_index, schedule_index, family_index] != target[None, :]
                repaired_status = deployed[seed_index, schedule_index, family_index] != target[None, :]
                baseline_risk = baseline_status.mean(axis=1)
                repaired_risk = repaired_status.mean(axis=1)
                forward, recovery, residual, closure = exact_flow(baseline_status)
                closure_errors.append(closure)
                eligible = bool(baseline_risk[0] < gates["h1_formation"]["eligible_baseline_risk_strictly_below"])
                event = persistent_cliff(baseline_risk.tolist()) if eligible else None
                repaired_event = persistent_cliff(repaired_risk.tolist())
                alarm = alarms[(seed, schedule_id, family)]["hybrid25_alarm"]
                timely_path = bool(event is not None and alarm is not None and int(alarm) < event)
                false_path = bool(eligible and event is None and alarm is not None)
                if eligible and event is not None:
                    cliffs += 1
                    if timely_path:
                        timely += 1
                        leads.append(event - int(alarm))
                elif eligible:
                    noncliffs += 1
                    false += int(false_path)
                path_rows.append({
                    "seed": seed,
                    "schedule_id": schedule_id,
                    "family": family,
                    "eligible": eligible,
                    "baseline_risk": float(baseline_risk[0]),
                    "endpoint_risk": float(baseline_risk[-1]),
                    "risk": baseline_risk.tolist(),
                    "event": event,
                    "hybrid25_alarm": alarm,
                    "timely": timely_path,
                    "false_alarm": false_path,
                    "forward": forward.tolist(),
                    "recovery": recovery.tolist(),
                    "flow_residual": residual.tolist(),
                    "max_closure_error": closure,
                })
                mean_risk_gain = float(np.mean(baseline_risk) - np.mean(repaired_risk))
                time_gain = event_time(repaired_risk) - event_time(baseline_risk)
                seed_risk_gains.append(mean_risk_gain)
                seed_event_gains.append(float(time_gain))
                repair_path_rows.append({
                    "seed": seed,
                    "schedule_id": schedule_id,
                    "family": family,
                    "repair_eligible": bool(assessment["repair_by_model"][str(seed)]["eligible"]),
                    "baseline_event": event,
                    "repaired_event": repaired_event,
                    "mean_risk_gain": mean_risk_gain,
                    "endpoint_risk_gain": float(baseline_risk[-1] - repaired_risk[-1]),
                    "event_time_gain": float(time_gain),
                    "baseline_risk_series": baseline_risk.tolist(),
                    "repaired_risk_series": repaired_risk.tolist(),
                })
                key = (seed, family)
                item = model_family.setdefault(key, {"eligible": set(), "baseline": set(), "repaired": set()})
                item["eligible"].add(eligible)
                item["baseline"].add(event is not None)
                item["repaired"].add(repaired_event is not None)
        seed_warning[seed] = {
            "cliffs": cliffs,
            "noncliffs": noncliffs,
            "timely": timely,
            "false": false,
            "timely_rate": rate(timely, cliffs),
            "false_rate": rate(false, noncliffs),
            "median_lead": float(np.median(leads)) if leads else None,
        }
        seed_repair_risk_gain[seed] = float(np.mean(seed_risk_gains))
        seed_repair_event_gain[seed] = float(np.mean(seed_event_gains))

    for key, item in model_family.items():
        if len(item["eligible"]) != 1 or len(item["baseline"]) != 1 or len(item["repaired"]) != 1:
            raise RuntimeError(f"schedule variants disagree on cliff status for {key}")
        item["eligible"] = item["eligible"].pop()
        item["baseline"] = item["baseline"].pop()
        item["repaired"] = item["repaired"].pop()

    eligible_rows = [row for row in path_rows if row["eligible"]]
    cliff_rows = [row for row in eligible_rows if row["event"] is not None]
    noncliff_rows = [row for row in eligible_rows if row["event"] is None]
    h1_spec = gates["h1_formation"]
    h1_checks = {
        "persistent_cliffs": len(cliff_rows) >= h1_spec["persistent_cliffs_min"],
        "noncliff_controls": len(noncliff_rows) >= h1_spec["noncliff_controls_min"],
        "seeds_with_cliff": len({row["seed"] for row in cliff_rows}) >= h1_spec["seeds_with_cliff_min"],
        "flow_closure": max(closure_errors, default=float("inf")) <= h1_spec["max_flux_closure_error"],
        "positive_cliff_endpoint_delta": bool(cliff_rows) and float(np.median([
            row["endpoint_risk"] - row["baseline_risk"] for row in cliff_rows
        ])) > 0,
    }
    h1_qualified = h1_checks["persistent_cliffs"] and h1_checks["noncliff_controls"]
    h1 = {
        "pass": all(h1_checks.values()),
        "field_qualified": h1_qualified,
        "checks": h1_checks,
        "eligible_paths": len(eligible_rows),
        "persistent_cliffs": len(cliff_rows),
        "noncliff_controls": len(noncliff_rows),
        "maximum_flow_closure_error": max(closure_errors, default=None),
        "median_cliff_endpoint_delta": float(np.median([
            row["endpoint_risk"] - row["baseline_risk"] for row in cliff_rows
        ])) if cliff_rows else None,
    }

    total_cliffs = len(cliff_rows)
    total_noncliffs = len(noncliff_rows)
    total_timely = sum(row["timely"] for row in cliff_rows)
    total_false = sum(row["false_alarm"] for row in noncliff_rows)
    leads = [row["event"] - int(row["hybrid25_alarm"]) for row in cliff_rows if row["timely"]]
    timely_rate = rate(total_timely, total_cliffs)
    false_rate = rate(total_false, total_noncliffs)
    median_lead = float(np.median(leads)) if leads else None
    h2_spec = gates["h2_warning"]
    seed_timely_passes = sum(
        value["timely_rate"] is not None and value["timely_rate"] >= h2_spec["per_seed_timely_floor"]
        for value in seed_warning.values()
    )
    seed_false_passes = sum(
        value["false_rate"] is not None and value["false_rate"] <= h2_spec["per_seed_false_ceiling"]
        for value in seed_warning.values()
    )
    h2_checks = {
        "timely_rate": timely_rate is not None and timely_rate >= h2_spec["timely_rate_min"],
        "false_alarm_rate": false_rate is not None and false_rate <= h2_spec["false_alarm_rate_max"],
        "median_lead": median_lead is not None and median_lead >= h2_spec["median_lead_windows_min"],
        "seed_timely_floor": seed_timely_passes >= h2_spec["seeds_meeting_timely_floor_min"],
        "seed_false_ceiling": seed_false_passes >= h2_spec["seeds_meeting_false_ceiling_min"],
        "blind_commit": True,
    }
    h2 = {
        "pass": all(h2_checks.values()),
        "checks": h2_checks,
        "cliffs": total_cliffs,
        "noncliffs": total_noncliffs,
        "timely": total_timely,
        "false": total_false,
        "timely_rate": timely_rate,
        "false_alarm_rate": false_rate,
        "median_lead": median_lead,
        "per_seed": {str(key): value for key, value in seed_warning.items()},
        "calibration_diagnostic": assessment["warning"],
    }

    removed = {
        key for key, value in model_family.items()
        if value["eligible"] and value["baseline"] and not value["repaired"]
    }
    introduced = {
        key for key, value in model_family.items()
        if value["eligible"] and not value["baseline"] and value["repaired"]
    }
    models_with_removed = len({seed for seed, family in removed})
    eligible_models = sum(
        bool(assessment["repair_by_model"][str(seed)]["eligible"])
        for seed in MODEL_SEEDS
    )
    risk_interval = seed_cluster_interval(
        seed_repair_risk_gain, gates["bootstrap"]["replicates"], gates["bootstrap"]["rng_seed"],
    )
    event_interval = seed_cluster_interval(
        seed_repair_event_gain, gates["bootstrap"]["replicates"], gates["bootstrap"]["rng_seed"],
    )
    h3_spec = gates["h3_repair"]
    h3_checks = {
        "eligible_models": eligible_models >= h3_spec["eligible_models_min"],
        "cliffs_removed": len(removed) >= h3_spec["unique_model_family_cliffs_removed_min"],
        "no_new_cliffs": len(introduced) <= h3_spec["new_model_family_cliffs_max"],
        "models_with_removed_cliff": models_with_removed >= h3_spec["models_with_a_removed_cliff_min"],
        "mean_risk_gain_ci_positive": risk_interval["ci95"][0] > 0,
        "mean_event_time_gain_positive": event_interval["estimate"] > 0,
        "blind_commit": True,
    }
    h3 = {
        "pass": all(h3_checks.values()),
        "checks": h3_checks,
        "eligible_models": eligible_models,
        "unique_model_family_cliffs_removed": len(removed),
        "new_model_family_cliffs": len(introduced),
        "models_with_removed_cliff": models_with_removed,
        "removed_model_families": [f"{seed}:{family}" for seed, family in sorted(removed)],
        "introduced_model_families": [f"{seed}:{family}" for seed, family in sorted(introduced)],
        "mean_risk_gain": risk_interval,
        "mean_event_time_gain": event_interval,
        "per_seed": {
            str(seed): {
                "eligible": bool(assessment["repair_by_model"][str(seed)]["eligible"]),
                "mean_risk_gain": seed_repair_risk_gain[seed],
                "mean_event_time_gain": seed_repair_event_gain[seed],
            }
            for seed in MODEL_SEEDS
        },
    }

    if not h1_qualified:
        overall = "FIELD_NOT_QUALIFIED"
    elif h1["pass"] and h2["pass"] and h3["pass"]:
        overall = "FORMATION_WARNING_REPAIR_CONFIRMED"
    elif h1["pass"] and h2["pass"]:
        overall = "FORMATION_AND_WARNING_ONLY"
    elif h1["pass"] and h3["pass"]:
        overall = "FORMATION_AND_REPAIR_ONLY"
    elif h1["pass"]:
        overall = "FORMATION_ONLY"
    else:
        overall = "FORMATION_ONLY"

    seed_rows = []
    for seed in MODEL_SEEDS:
        seed_rows.append({
            "seed": seed,
            **seed_warning[seed],
            "repair_eligible": bool(assessment["repair_by_model"][str(seed)]["eligible"]),
            "repair_mean_risk_gain": seed_repair_risk_gain[seed],
            "repair_mean_event_time_gain": seed_repair_event_gain[seed],
        })
    write_csv(workspace / "seed_level_results.csv", seed_rows)
    write_csv(workspace / "path_level_results.csv", path_rows)
    write_csv(workspace / "repair_path_results.csv", repair_path_rows)
    result = {
        "integrity": {"package_audit": True, "blind_commit": True, "phase_order": True},
        "h1_formation": h1,
        "h2_warning": h2,
        "h3_repair": h3,
        "overall_decision": overall,
        "claim_boundary": (
            "Prospective confirmation across classifier-head model seeds and frozen schedule variants on the "
            "same public CURE-OR benchmark; not a fresh-identity, cross-dataset, or cross-domain replication."
        ),
    }
    write_json(workspace / "results.json", result)
    post_files = [
        workspace / "results.json",
        workspace / "seed_level_results.csv",
        workspace / "path_level_results.csv",
        workspace / "repair_path_results.csv",
    ]
    (workspace / "POSTREVEAL.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in post_files), encoding="utf-8",
    )
    evidence_files = [
        workspace / line.split("  ", 1)[1]
        for line in (workspace / "BLIND_OUTPUTS.sha256").read_text(encoding="utf-8").splitlines()
    ]
    evidence_files += [workspace / "BLIND_OUTPUTS.sha256", *post_files, workspace / "POSTREVEAL.sha256"]
    (workspace / "EVIDENCE.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in evidence_files), encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    audit_package()
    result = phase2(args.workspace.resolve())
    print(json.dumps({
        "overall_decision": result["overall_decision"],
        "h1_pass": result["h1_formation"]["pass"],
        "h2_pass": result["h2_warning"]["pass"],
        "h3_pass": result["h3_repair"]["pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
