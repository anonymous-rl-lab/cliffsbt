#!/usr/bin/env python3
"""Frozen batch-16 resolution probe using the v2 revealed trajectory panel."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from measurement_design import MEASUREMENT_GROUPS
from probe_cliff_early_warning import (
    build_design,
    generate_online_means,
    risk_path,
    score_predictions,
    summarize_scores,
    trajectory_scalars,
)
from probe_cliff_early_warning_v2 import scalar_predictions
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import calibration_panel, environment_record, train_deployment_model


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "early_warning_batch16_probe.json").read_text())
    output = ROOT / "results" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    model, training = train_deployment_model(cfg)
    environment, store = calibration_panel(cfg, model)
    fitted = fit_screening_bridge(environment, store, cfg)
    designs = {
        "compressed": build_design(tuple(cfg["early_warning"]["fixed_groups"]), fitted, cfg),
        "full": build_design(tuple(MEASUREMENT_GROUPS), fitted, cfg),
    }
    observation_columns = [column for column in environment if column.startswith("rich_")]
    entropy_model = Ridge(alpha=1e-4).fit(
        environment[observation_columns].to_numpy()[:, [4, 5]], environment["risk"]
    )
    boundary = float(fitted["risk_intercept"] + cfg["target"]["risk_margin_gamma"])
    spec = cfg["early_warning"]
    trajectories = []
    for item in spec["trajectories"]:
        direction = np.asarray(item["direction"], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        for planned_type, end in (
            ("event", float(item["event_end_scalar"])),
            ("stationary_safe", float(item["start_scalar"])),
        ):
            scalars = trajectory_scalars(float(item["start_scalar"]), end, spec)
            offsets = scalars[:, None] * direction[None, :]
            trajectories.append(
                {
                    "id": f"{item['name']}:{planned_type}",
                    "direction_name": item["name"],
                    "planned_type": planned_type,
                    "direction": direction,
                    "scalars": scalars,
                    "offsets": offsets,
                    "predicted_risks": risk_path(offsets, fitted),
                }
            )
    revealed = pd.read_csv(ROOT / cfg["source_reveal"])
    expected_ids = sorted(item["id"] for item in trajectories)
    reveal_ids = sorted(revealed["trajectory_id"].unique())
    gates = cfg["pilot_gates"]
    metrics = {
        "calibration_risk_range": float(environment["risk"].max() - environment["risk"].min()),
        "risk_surface_r2": fitted["risk_surface_r2"],
        "risk_surface_cv_r2": fitted["risk_surface_cv_r2"],
        "relevant_score_linear_r2": fitted["relevant_score_linear_r2"],
        "compressed_risk_null_ratio": designs["compressed"]["risk_null_ratio"],
        "full_risk_null_ratio": designs["full"]["risk_null_ratio"],
        "relative_warning_boundary": boundary,
    }
    pretarget_gates = {
        "risk_range": metrics["calibration_risk_range"] >= gates["minimum_calibration_risk_range"],
        "risk_surface_fit": metrics["risk_surface_r2"] >= gates["minimum_risk_surface_r2"],
        "risk_surface_crossfit": metrics["risk_surface_cv_r2"] >= gates["minimum_risk_surface_cv_r2"],
        "relevant_score": metrics["relevant_score_linear_r2"] >= gates["minimum_relevant_score_linear_r2"],
        "compressed_identified": metrics["compressed_risk_null_ratio"] <= gates["maximum_risk_null_ratio"],
        "full_identified": metrics["full_risk_null_ratio"] <= gates["maximum_risk_null_ratio"],
        "reveal_panel_matches_frozen_trajectories": expected_ids == reveal_ids,
    }
    pretarget = {
        "metrics": metrics,
        "gates": pretarget_gates,
        "all_passed": bool(all(pretarget_gates.values())),
        "sequence_seed": cfg["sequence_seed"],
        "source_reveal": cfg["source_reveal"],
        "frozen_protocol": {
            key: spec[key] for key in (
                "batch_size", "replicates", "history_windows", "forecast_horizon", "forecast_buffer"
            )
        },
    }
    (output / "pretarget_batch16_gate.json").write_text(
        json.dumps(serialize(pretarget), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment.to_csv(output / "calibration_environment_panel.csv", index=False)
    (output / "training_summary.json").write_text(
        json.dumps(serialize(training), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(environment_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not pretarget["all_passed"]:
        print(json.dumps({"aborted_before_online": True, "pretarget": pretarget}, indent=2))
        return
    sequence_cfg = copy.deepcopy(cfg)
    sequence_cfg["master_seed"] = int(cfg["sequence_seed"])
    means = generate_online_means(
        sequence_cfg, model, np.asarray(cfg["calibration"]["theta_center"]), trajectories
    )
    predictions = scalar_predictions(means, trajectories, designs, fitted, entropy_model, cfg)
    predictions.to_csv(output / "sealed_online_predictions.csv", index=False)
    outcomes, records = score_predictions(predictions, revealed, boundary, cfg)
    outcomes.to_csv(output / "trajectory_outcomes.csv", index=False)
    records.to_csv(output / "replicate_warning_ledger.csv", index=False)
    summary = summarize_scores(records)
    summary.to_csv(output / "warning_method_summary.csv", index=False)
    indexed = summary.set_index("method")
    compressed = indexed.loc["compressed_forecast"]
    full = indexed.loc["full_forecast"]
    final_gates = {
        "compressed_timely_warning": compressed["timely_warning_rate"] >= gates["minimum_compressed_timely_warning_rate"],
        "compressed_false_alarm": compressed["non_event_false_alarm_rate"] <= gates["maximum_compressed_non_event_false_alarm_rate"],
        "compressed_premature_warning": compressed["premature_warning_rate"] <= gates["maximum_compressed_premature_warning_rate"],
        "compressed_lead": compressed["median_timely_lead"] >= gates["minimum_compressed_median_timely_lead"],
        "compressed_near_full": full["timely_warning_rate"] - compressed["timely_warning_rate"] <= gates["maximum_timely_rate_gap_from_full"],
    }
    checks = {
        "pretarget_all_passed": True,
        "metrics": {
            "compressed_timely_warning_rate": compressed["timely_warning_rate"],
            "compressed_non_event_false_alarm_rate": compressed["non_event_false_alarm_rate"],
            "compressed_premature_warning_rate": compressed["premature_warning_rate"],
            "compressed_median_timely_lead": compressed["median_timely_lead"],
            "full_timely_warning_rate": full["timely_warning_rate"],
        },
        "gates": final_gates,
        "all_passed": bool(all(final_gates.values())),
        "semantics": "exploratory batch-resolution probe; revealed outcomes reused from v2",
    }
    (output / "checks.json").write_text(
        json.dumps(serialize(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(serialize({"checks": checks, "summary": summary.to_dict("records")}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
