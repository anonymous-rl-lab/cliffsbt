#!/usr/bin/env python3
"""Phase 1: outcome-blind prediction, warning, and guarded repair commitment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    CONFIG, DATA, MODEL_SEEDS, PACKAGE, SCHEDULE_IDS, TARGET_FAMILIES, WINDOWS,
    alarm_time, anchored_update, assigned_levels, fit_ridge, flux25, head_scores,
    hybrid25, moments25, persistent_cliff, read_json, score_hybrid, sha256,
    softmax, write_json,
)
from data_access import WEIGHTS_SHA256, fetch_all, generate_features, required_image_ids


def typed_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integers = {
        "image_id", "class_index", "class", "background", "perspective",
        "family", "level", "candidate_rank", "dense_floor",
    }
    return [{key: int(value) if key in integers else value for key, value in row.items()} for row in rows]


def grouped_streams(role: str) -> dict[int, list[dict]]:
    streams = read_json(DATA / "TARGET_STREAMS_FROZEN.json")[role]
    result = {family: [] for family in TARGET_FAMILIES}
    for stream in streams:
        result[int(stream["family"])].append(stream)
    for family in result:
        result[family].sort(key=lambda stream: tuple(stream["base"]))
        if len(result[family]) != 50:
            raise RuntimeError(f"{role} family {family} has {len(result[family])} identities")
    return result


def build_trajectory(
    streams: list[dict], probabilities: dict[int, np.ndarray], norms: dict[int, float],
    schedule_id: int, reveal: bool,
) -> dict:
    family = int(streams[0]["family"])
    bases = [tuple(stream["base"]) for stream in streams]
    baseline_probabilities = np.asarray([
        probabilities[int(stream["baseline_image_id"])] for stream in streams
    ])
    previous = None
    flux_rows, moment_rows, hybrid_rows, predictions, risk = [], [], [], [], []
    for window in range(WINDOWS):
        levels = assigned_levels(bases, family, window, schedule_id)
        image_ids = [
            int(stream["baseline_image_id"]) if levels[tuple(stream["base"])] == 0
            else int(stream["levels"][str(levels[tuple(stream["base"])])])
            for stream in streams
        ]
        batch = np.asarray([probabilities[image_id] for image_id in image_ids])
        batch_norms = np.asarray([norms[image_id] for image_id in image_ids])
        flux = flux25(batch, baseline_probabilities, previous)
        moments = moments25(batch, batch_norms)
        predicted = np.argmax(batch, axis=1)
        flux_rows.append(flux)
        moment_rows.append(moments)
        hybrid_rows.append(hybrid25(flux, moments))
        predictions.append(predicted)
        if reveal:
            truth = np.asarray([int(stream["class_index"]) for stream in streams])
            risk.append(float(np.mean(predicted != truth)))
        previous = batch
    result = {
        "family": family,
        "schedule_id": schedule_id,
        "flux25": np.asarray(flux_rows),
        "moments25": np.asarray(moment_rows),
        "hybrid25": np.asarray(hybrid_rows),
        "predicted": np.asarray(predictions, dtype=np.int8),
    }
    if reveal:
        result["risk"] = risk
    return result


def model_outputs(head: dict, ids: np.ndarray, features: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    probabilities = softmax(head_scores(head, features))
    return (
        {int(image_id): probabilities[index] for index, image_id in enumerate(ids)},
        {int(image_id): float(np.linalg.norm(features[index])) for index, image_id in enumerate(ids)},
    )


def warning_calibration_metrics(items: list[dict], threshold: float) -> dict:
    cliffs = noncliffs = timely = false = 0
    leads = []
    for item in items:
        event = persistent_cliff(item["risk"])
        alarm = alarm_time(item["warning_score"], threshold)
        if event is None:
            noncliffs += 1
            false += int(alarm is not None)
        else:
            cliffs += 1
            if alarm is not None and alarm < event:
                timely += 1
                leads.append(event - alarm)
    return {
        "cliffs": cliffs,
        "noncliffs": noncliffs,
        "timely": timely,
        "false": false,
        "timely_rate": timely / cliffs if cliffs else None,
        "false_alarm_rate": false / noncliffs if noncliffs else None,
        "median_lead": float(np.median(leads)) if leads else None,
    }


def repair_calibration_guard(baseline: list[dict], repaired: list[dict], spec: dict) -> dict:
    paired = {}
    for base, repair in zip(baseline, repaired):
        key = (int(base["schedule_id"]), int(base["family"]))
        if key != (int(repair["schedule_id"]), int(repair["family"])):
            raise RuntimeError("repair calibration path order mismatch")
        paired[key] = {
            "baseline_event": persistent_cliff(base["risk"]),
            "repaired_event": persistent_cliff(repair["risk"]),
            "mean_risk_gain": float(np.mean(base["risk"]) - np.mean(repair["risk"])),
        }
    family_status = {}
    for family in TARGET_FAMILIES:
        rows = [value for (schedule, candidate_family), value in paired.items() if candidate_family == family]
        baseline_cliff = {row["baseline_event"] is not None for row in rows}
        repaired_cliff = {row["repaired_event"] is not None for row in rows}
        if len(baseline_cliff) != 1 or len(repaired_cliff) != 1:
            raise RuntimeError("schedule variants disagree on endpoint-defined cliff status")
        family_status[family] = {
            "baseline_cliff": baseline_cliff.pop(),
            "repaired_cliff": repaired_cliff.pop(),
        }
    removed = sum(value["baseline_cliff"] and not value["repaired_cliff"] for value in family_status.values())
    introduced = sum(not value["baseline_cliff"] and value["repaired_cliff"] for value in family_status.values())
    path_gains = [value["mean_risk_gain"] for value in paired.values()]
    guards = {
        "removes_at_least_one_unique_family_cliff": removed >= int(spec["unique_families_with_persistent_cliff_removed_min"]),
        "introduces_no_family_cliff": introduced <= int(spec["new_family_cliffs_max"]),
        "no_path_mean_risk_increase": min(path_gains) >= float(spec["minimum_path_mean_risk_gain"]),
        "aggregate_mean_risk_margin": float(np.mean(path_gains)) >= float(spec["aggregate_mean_risk_gain_min"]),
    }
    return {
        "eligible": all(guards.values()),
        "guards": guards,
        "unique_family_cliffs_removed": int(removed),
        "new_family_cliffs": int(introduced),
        "mean_risk_gain": float(np.mean(path_gains)),
        "minimum_path_mean_risk_gain": float(min(path_gains)),
        "family_status": {str(key): value for key, value in family_status.items()},
    }


def fixed_repair_rows(candidates: list[dict], per_fragment: int) -> list[dict]:
    grouped = defaultdict(list)
    for item in candidates:
        grouped[str(item["fragment"])].append(item)
    for fragment in grouped:
        grouped[fragment].sort(key=lambda item: (int(item["candidate_rank"]), int(item["image_id"])))
    if len(grouped) != 100 or min(len(items) for items in grouped.values()) < per_fragment:
        raise RuntimeError("frozen repair pool does not contain the required 100 fragments")
    selected = [item for fragment in sorted(grouped) for item in grouped[fragment][:per_fragment]]
    if len(selected) != 100 * per_fragment:
        raise RuntimeError("repair budget cardinality mismatch")
    return selected


def audit() -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "code" / "audit_package.py"), "--strict"],
        check=False,
    )
    if result.returncode:
        raise RuntimeError("registration package audit failed")


def commit(workspace: Path, files: list[Path]) -> None:
    target = workspace / "BLIND_OUTPUTS.sha256"
    if target.exists():
        raise RuntimeError("blind commitment already exists")
    target.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")


def phase1(workspace: Path, feature_path: Path, weights: Path) -> None:
    if (workspace / "BLIND_OUTPUTS.sha256").exists():
        raise RuntimeError("phase 1 already committed; use a fresh workspace")
    experiment = read_json(CONFIG / "experiment.json")
    warning_protocol = read_json(CONFIG / "warning_protocol.json")
    repair_protocol = read_json(CONFIG / "repair_protocol.json")
    warning_model = read_json(CONFIG / "hybrid25_warning_model_frozen.json")
    if float(warning_model["alarm_threshold"]) != float(warning_protocol["readout"]["threshold"]):
        raise RuntimeError("warning threshold differs between frozen model and protocol")
    feature_cache = np.load(feature_path)
    train_ids, test_ids = feature_cache["train_ids"], feature_cache["test_ids"]
    expected_ids = required_image_ids()
    if train_ids.tolist() != expected_ids["train"] or test_ids.tolist() != expected_ids["test"]:
        raise RuntimeError("feature cache image IDs differ from frozen acquisition manifest")
    if feature_cache["test_features"].shape != (len(test_ids), 768):
        raise RuntimeError("test feature cache shape mismatch")
    for seed in MODEL_SEEDS:
        if feature_cache[f"train_features_seed{seed}"].shape != (len(train_ids), 768):
            raise RuntimeError(f"train feature cache shape mismatch for seed {seed}")
    if sha256(weights) != WEIGHTS_SHA256:
        raise RuntimeError("ConvNeXt-Tiny weight SHA-256 mismatch")

    train_lookup = {int(image_id): index for index, image_id in enumerate(train_ids)}
    clean = typed_rows(DATA / "TRAINING_BASELINE_FROZEN.csv")
    candidates = typed_rows(DATA / "REPAIR_CANDIDATES_FROZEN.csv")
    clean_indices = [train_lookup[item["image_id"]] for item in clean]
    clean_labels = np.asarray([item["class_index"] for item in clean])
    repair_rows = fixed_repair_rows(candidates, int(repair_protocol["samples_per_fragment"]))
    repair_indices = [train_lookup[item["image_id"]] for item in repair_rows]
    repair_labels = np.asarray([item["class_index"] for item in repair_rows])
    calibration = grouped_streams("calibration")
    confirmation = grouped_streams("confirmation")

    shape = (len(MODEL_SEEDS), len(SCHEDULE_IDS), len(TARGET_FAMILIES), WINDOWS, 50)
    baseline_predicted = np.empty(shape, dtype=np.int8)
    candidate_repair_predicted = np.empty(shape, dtype=np.int8)
    deployed_repair_predicted = np.empty(shape, dtype=np.int8)
    hybrid_sensors = np.empty(shape[:-1] + (25,), dtype=np.float64)
    warning_scores = np.empty(shape[:-1], dtype=np.float64)
    alarm_rows, calibration_items, repair_decisions = [], [], {}

    for seed_index, seed in enumerate(MODEL_SEEDS):
        train_features = feature_cache[f"train_features_seed{seed}"]
        base_head = fit_ridge(train_features[clean_indices], clean_labels, 1.0)
        candidate_head = anchored_update(
            base_head, train_features[repair_indices], repair_labels,
            float(repair_protocol["trust_lambda"]),
        )
        base_probabilities, base_norms = model_outputs(base_head, test_ids, feature_cache["test_features"])
        repair_probabilities, repair_norms = model_outputs(candidate_head, test_ids, feature_cache["test_features"])

        base_calibration, repair_calibration = [], []
        for schedule_id in SCHEDULE_IDS:
            for family in TARGET_FAMILIES:
                base_item = build_trajectory(
                    calibration[family], base_probabilities, base_norms, schedule_id, True,
                )
                repaired_item = build_trajectory(
                    calibration[family], repair_probabilities, repair_norms, schedule_id, True,
                )
                base_item.update({"seed": seed})
                base_item["warning_score"] = score_hybrid(base_item["hybrid25"], warning_model)
                calibration_items.append(base_item)
                base_calibration.append(base_item)
                repair_calibration.append(repaired_item)
        decision = repair_calibration_guard(
            base_calibration, repair_calibration, repair_protocol["calibration_guard_per_model"],
        )
        repair_decisions[str(seed)] = decision

        for schedule_index, schedule_id in enumerate(SCHEDULE_IDS):
            for family_index, family in enumerate(TARGET_FAMILIES):
                base_item = build_trajectory(
                    confirmation[family], base_probabilities, base_norms, schedule_id, False,
                )
                repaired_item = build_trajectory(
                    confirmation[family], repair_probabilities, repair_norms, schedule_id, False,
                )
                score = score_hybrid(base_item["hybrid25"], warning_model)
                alarm = alarm_time(
                    score,
                    float(warning_protocol["readout"]["threshold"]),
                    int(warning_protocol["readout"]["persistence_windows"]),
                    int(warning_protocol["readout"]["earliest_alarm_window"]),
                )
                baseline_predicted[seed_index, schedule_index, family_index] = base_item["predicted"]
                candidate_repair_predicted[seed_index, schedule_index, family_index] = repaired_item["predicted"]
                deployed_repair_predicted[seed_index, schedule_index, family_index] = (
                    repaired_item["predicted"] if decision["eligible"] else base_item["predicted"]
                )
                hybrid_sensors[seed_index, schedule_index, family_index] = base_item["hybrid25"]
                warning_scores[seed_index, schedule_index, family_index] = score
                alarm_rows.append({
                    "seed": seed,
                    "schedule_id": schedule_id,
                    "family": family,
                    "hybrid25_scores": score.tolist(),
                    "hybrid25_alarm": alarm,
                })

    warning_calibration = warning_calibration_metrics(
        calibration_items, float(warning_protocol["readout"]["threshold"]),
    )
    workspace.mkdir(parents=True, exist_ok=True)
    assessment_file = workspace / "calibration_assessment.json"
    alarm_file = workspace / "blind_alarm_rows.json"
    prediction_file = workspace / "blind_predictions.npz"
    metadata_file = workspace / "run_metadata.json"
    write_json(assessment_file, {
        "warning": warning_calibration,
        "repair_by_model": repair_decisions,
        "repair_selected_image_ids": [int(item["image_id"]) for item in repair_rows],
        "confirmation_outcomes_used": False,
    })
    write_json(alarm_file, alarm_rows)
    np.savez_compressed(
        prediction_file,
        baseline_predicted=baseline_predicted,
        candidate_repair_predicted=candidate_repair_predicted,
        deployed_repair_predicted=deployed_repair_predicted,
        hybrid25_sensors=hybrid_sensors,
        warning_scores=warning_scores,
        seeds=np.asarray(MODEL_SEEDS),
        schedule_ids=np.asarray(SCHEDULE_IDS),
        families=np.asarray(TARGET_FAMILIES),
    )
    write_json(metadata_file, {
        "phase": "PHASE1_OUTCOME_BLIND_COMMITTED",
        "package_manifest_sha256": sha256(PACKAGE / "PACKAGE_MANIFEST.sha256"),
        "phase1_source_sha256": sha256(Path(__file__)),
        "feature_cache_sha256": sha256(feature_path),
        "weights_sha256": sha256(weights),
        "weights_expected_sha256": WEIGHTS_SHA256,
        "array_order": {
            "seeds": MODEL_SEEDS,
            "schedule_ids": SCHEDULE_IDS,
            "families": TARGET_FAMILIES,
            "confirmation_identities": "sorted base tuple within family",
        },
        "prospective_unit": experiment["prospective_unit"],
        "confirmation_labels_scored": False,
    })
    commit(workspace, [metadata_file, assessment_file, alarm_file, prediction_file])
    print(json.dumps({
        "status": "PHASE1_COMMITTED",
        "workspace": str(workspace),
        "commit": sha256(workspace / "BLIND_OUTPUTS.sha256"),
        "warning_calibration": warning_calibration,
        "repair_eligible_models": sum(value["eligible"] for value in repair_decisions.values()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--action", choices=["all", "fetch", "features", "run"], default="all")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    audit()
    workspace = args.workspace.resolve()
    cache = (args.data_cache or (workspace / "image_cache")).resolve()
    feature_path = workspace / "features.npz"
    if args.action in ("all", "fetch"):
        workspace.mkdir(parents=True, exist_ok=True)
        write_json(workspace / "fetch_manifest.json", fetch_all(cache, args.workers))
        if args.action == "fetch":
            return
    if args.action in ("all", "features"):
        if args.weights is None:
            raise SystemExit("--weights is required for feature extraction")
        generate_features(cache, args.weights.resolve(), feature_path)
        if args.action == "features":
            return
    if args.action in ("all", "run"):
        if args.weights is None:
            raise SystemExit("--weights is required to record the frozen weight digest")
        if not feature_path.exists():
            raise SystemExit(f"feature cache missing: {feature_path}")
        phase1(workspace, feature_path, args.weights.resolve())


if __name__ == "__main__":
    main()
