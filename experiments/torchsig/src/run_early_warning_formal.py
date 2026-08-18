#!/usr/bin/env python3
"""Formal sealed validation of unlabeled Cliff early warning."""

from __future__ import annotations

import argparse
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
    reveal_trajectory_risks,
    risk_path,
    score_predictions,
    summarize_scores,
    trajectory_scalars,
)
from probe_cliff_early_warning_v2 import plot_results, scalar_predictions
from probe_early_warning_geometry import direction_record, oriented_axis
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import calibration_panel, environment_record, train_deployment_model


ROOT = Path(__file__).resolve().parents[1]


def all_time_risk_coordinates(means: dict, trajectories: list[dict], designs: dict,
                              fitted: dict) -> pd.DataFrame:
    """Expose the memoryless risk coordinate for every online window.

    The forecasting routine historically wrote rows only after a complete
    history was available.  The matched-order knockout needs the earlier
    coordinates to reconstruct those histories without imputation.  This
    function performs no sampling and does not alter the frozen predictions.
    """
    rows = []
    for trajectory in trajectories:
        rich_means = means[trajectory["id"]]
        for design_name, design in designs.items():
            selected = rich_means[:, :, design["indices"]]
            centered = selected - fitted["observation_intercept"][design["indices"]]
            states = centered @ design["estimator"].T
            estimated_risk = risk_path(states.reshape(-1, 3), fitted).reshape(
                states.shape[:2]
            )
            for replicate in range(estimated_risk.shape[0]):
                for time_index in range(estimated_risk.shape[1]):
                    rows.append(
                        {
                            "trajectory_id": trajectory["id"],
                            "direction": trajectory["direction_name"],
                            "planned_type": trajectory["planned_type"],
                            "replicate": replicate,
                            "time_index": time_index,
                            "method": f"{design_name}_current",
                            "estimated_current_risk": float(
                                estimated_risk[replicate, time_index]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/formal_early_warning_v1.json",
        help="configuration path relative to the repository root",
    )
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    output = ROOT / "results" / cfg["output_tag"]
    figures = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    model, training = train_deployment_model(cfg)
    environment, store = calibration_panel(cfg, model)
    fitted = fit_screening_bridge(environment, store, cfg)
    protocol_cfg = copy.deepcopy(cfg)
    buffer_spec = cfg["early_warning"]["forecast_buffer"]
    if isinstance(buffer_spec, str) and buffer_spec.startswith("calibration_"):
        quantile = buffer_spec.removeprefix("calibration_")
        forecast_buffer = float(fitted["risk_cv_abs_residual_quantiles"][quantile])
        protocol_cfg["early_warning"]["forecast_buffer"] = forecast_buffer
    else:
        forecast_buffer = float(buffer_spec)
    designs = {
        "compressed": build_design(tuple(cfg["early_warning"]["fixed_groups"]), fitted, protocol_cfg),
        "full": build_design(tuple(MEASUREMENT_GROUPS), fitted, protocol_cfg),
    }
    observation_columns = [column for column in environment if column.startswith("rich_")]
    entropy_model = Ridge(alpha=1e-4).fit(
        environment[observation_columns].to_numpy()[:, [4, 5]], environment["risk"]
    )
    tau = float(fitted["risk_intercept"])
    b = np.asarray(fitted["b"], dtype=float)
    hessian = np.asarray(fitted["H"], dtype=float)
    boundary = tau + float(cfg["target"]["risk_margin_gamma"])
    limit = float(cfg["early_warning"]["trajectory_offset_limit"])
    candidate_directions = {
        "noise": oriented_axis("noise", 0, tau, b, hessian, limit),
        "phase": oriented_axis("phase", 1, tau, b, hessian, limit),
        "mixed_gradient": b / max(float(np.linalg.norm(b)), 1e-12),
    }
    geometry = [
        direction_record(name, candidate_directions[name], tau, b, hessian, protocol_cfg)
        for name in cfg["early_warning"]["directions"]
    ]
    trajectories = []
    for item in geometry:
        if not item["feasible"]:
            continue
        direction = np.asarray(item["direction"], dtype=float)
        for planned_type, end in (
            ("event", float(item["end_scalar"])),
            ("stationary_safe", float(item["start_scalar"])),
        ):
            scalars = trajectory_scalars(float(item["start_scalar"]), end, protocol_cfg["early_warning"])
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
    gates = cfg["pilot_gates"]
    metrics = {
        "calibration_risk_range": float(environment["risk"].max() - environment["risk"].min()),
        "risk_surface_r2": fitted["risk_surface_r2"],
        "risk_surface_cv_r2": fitted["risk_surface_cv_r2"],
        "relevant_score_linear_r2": fitted["relevant_score_linear_r2"],
        "compressed_risk_null_ratio": designs["compressed"]["risk_null_ratio"],
        "full_risk_null_ratio": designs["full"]["risk_null_ratio"],
        "relative_warning_boundary": boundary,
        "forecast_buffer_frozen_from_calibration": forecast_buffer,
        "feasible_directions": int(sum(item["feasible"] for item in geometry)),
    }
    pretarget_gates = {
        "risk_range": metrics["calibration_risk_range"] >= gates["minimum_calibration_risk_range"],
        "risk_surface_fit": metrics["risk_surface_r2"] >= gates["minimum_risk_surface_r2"],
        "risk_surface_crossfit": metrics["risk_surface_cv_r2"] >= gates["minimum_risk_surface_cv_r2"],
        "relevant_score": metrics["relevant_score_linear_r2"] >= gates["minimum_relevant_score_linear_r2"],
        "compressed_identified": metrics["compressed_risk_null_ratio"] <= gates["maximum_risk_null_ratio"],
        "full_identified": metrics["full_risk_null_ratio"] <= gates["maximum_risk_null_ratio"],
        "all_frozen_directions_feasible": metrics["feasible_directions"] == len(cfg["early_warning"]["directions"]),
        "event_paths_cross_fitted_boundary": bool(trajectories) and all(
            float(item["predicted_risks"][-1]) >= boundary for item in trajectories
            if item["planned_type"] == "event"
        ),
        "stationary_controls_below_fitted_boundary": bool(trajectories) and all(
            float(np.max(item["predicted_risks"])) < boundary for item in trajectories
            if item["planned_type"] == "stationary_safe"
        ),
    }
    pretarget = {
        "metrics": metrics,
        "gates": pretarget_gates,
        "all_passed": bool(all(pretarget_gates.values())),
        "geometry": geometry,
        "frozen_protocol": {
            key: protocol_cfg["early_warning"][key] for key in (
                "fixed_groups", "batch_size", "replicates", "history_windows",
                "forecast_horizon", "forecast_buffer", "crossing_confirmation_windows"
            )
        },
        "frozen_trajectories": [
            {
                "trajectory_id": item["id"],
                "direction": item["direction"],
                "start_scalar": float(item["scalars"][0]),
                "end_scalar": float(item["scalars"][-1]),
                "predicted_start_risk": float(item["predicted_risks"][0]),
                "predicted_end_risk": float(item["predicted_risks"][-1]),
            }
            for item in trajectories
        ],
    }
    (output / "pretarget_formal_early_warning_gate.json").write_text(
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
        print(json.dumps(serialize({"aborted_before_target": True, "pretarget": pretarget}), indent=2))
        return
    center = np.asarray(cfg["calibration"]["theta_center"])
    means = generate_online_means(protocol_cfg, model, center, trajectories)
    all_time = all_time_risk_coordinates(means, trajectories, designs, fitted)
    all_time.to_csv(output / "sealed_all_time_risk_coordinates.csv", index=False)
    predictions = scalar_predictions(
        means, trajectories, designs, fitted, entropy_model, protocol_cfg
    )
    predictions.to_csv(output / "sealed_online_predictions.csv", index=False)
    revealed = reveal_trajectory_risks(protocol_cfg, model, center, trajectories)
    revealed.to_csv(output / "revealed_trajectory_risks.csv", index=False)
    outcomes, records = score_predictions(predictions, revealed, boundary, protocol_cfg)
    outcomes.to_csv(output / "trajectory_outcomes.csv", index=False)
    records.to_csv(output / "replicate_warning_ledger.csv", index=False)
    summary = summarize_scores(records)
    summary.to_csv(output / "warning_method_summary.csv", index=False)
    annotated = records.merge(outcomes[["trajectory_id", "direction"]], on="trajectory_id")
    direction_rows = []
    for (method, direction), frame in annotated.groupby(["method", "direction"]):
        events = frame[frame["actual_event"]]
        non_events = frame[~frame["actual_event"]]
        direction_rows.append(
            {
                "method": method,
                "direction": direction,
                "timely_warning_rate": float(events["timely_warning"].mean()) if len(events) else None,
                "premature_warning_rate": float(events["premature_warning"].mean()) if len(events) else None,
                "false_alarm_rate": float(non_events["false_alarm"].mean()) if len(non_events) else None,
            }
        )
    direction_summary = pd.DataFrame(direction_rows)
    direction_summary.to_csv(output / "direction_warning_summary.csv", index=False)
    indexed = summary.set_index("method")
    compressed = indexed.loc["compressed_forecast"]
    full = indexed.loc["full_forecast"]
    current = indexed.loc["compressed_current"]
    compressed_directions = direction_summary[
        direction_summary["method"] == "compressed_forecast"
    ]
    actual_events = int(outcomes["actual_event"].sum())
    actual_non_events = int((~outcomes["actual_event"]).sum())
    forecast_gain = float(compressed["timely_warning_rate"] - current["timely_warning_rate"])
    final_gates = {
        "actual_event_trajectories": actual_events >= gates["minimum_actual_event_trajectories"],
        "actual_non_event_trajectories": actual_non_events >= gates["minimum_actual_non_event_trajectories"],
        "compressed_timely_warning": compressed["timely_warning_rate"] >= gates["minimum_compressed_timely_warning_rate"],
        "compressed_false_alarm": compressed["non_event_false_alarm_rate"] <= gates["maximum_compressed_non_event_false_alarm_rate"],
        "compressed_premature_warning": compressed["premature_warning_rate"] <= gates["maximum_compressed_premature_warning_rate"],
        "compressed_lead": compressed["median_timely_lead"] >= gates["minimum_compressed_median_timely_lead"],
        "compressed_near_full": full["timely_warning_rate"] - compressed["timely_warning_rate"] <= gates["maximum_timely_rate_gap_from_full"],
        "forecast_gain_over_current": forecast_gain >= gates["minimum_forecast_gain_over_current"],
        "direction_timely_warning": compressed_directions["timely_warning_rate"].min() >= gates["minimum_direction_timely_warning_rate"],
        "direction_false_alarm": compressed_directions["false_alarm_rate"].max() <= gates["maximum_direction_false_alarm_rate"],
    }
    checks = {
        "pretarget_all_passed": True,
        "metrics": {
            "actual_event_trajectories": actual_events,
            "actual_non_event_trajectories": actual_non_events,
            "compressed_timely_warning_rate": compressed["timely_warning_rate"],
            "compressed_non_event_false_alarm_rate": compressed["non_event_false_alarm_rate"],
            "compressed_premature_warning_rate": compressed["premature_warning_rate"],
            "compressed_median_timely_lead": compressed["median_timely_lead"],
            "full_timely_warning_rate": full["timely_warning_rate"],
            "compressed_current_timely_warning_rate": current["timely_warning_rate"],
            "forecast_gain_over_current": forecast_gain,
            "minimum_direction_timely_warning_rate": compressed_directions["timely_warning_rate"].min(),
            "maximum_direction_false_alarm_rate": compressed_directions["false_alarm_rate"].max(),
        },
        "gates": final_gates,
        "all_passed": bool(all(final_gates.values())),
        "semantics": "formal warning of a frozen relative-risk boundary under persistent local drift; not absolute safety prediction",
    }
    (output / "checks.json").write_text(
        json.dumps(serialize(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(
        revealed,
        predictions,
        boundary,
        figures / f"{cfg['output_tag']}.png",
        alarm_threshold=boundary + forecast_buffer,
    )
    print(json.dumps(serialize({"checks": checks, "outcomes": outcomes.to_dict("records"),
                                "summary": summary.to_dict("records"),
                                "direction_summary": direction_rows}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
