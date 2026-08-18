#!/usr/bin/env python3
"""Held-out-stream probe of a frozen scalar-risk Cliff warning rule."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import calibration_panel, environment_record, train_deployment_model


ROOT = Path(__file__).resolve().parents[1]


def scalar_predictions(means: dict, trajectories: list[dict], designs: dict,
                       fitted: dict, entropy_model: Ridge, cfg: dict) -> pd.DataFrame:
    spec = cfg["early_warning"]
    history_windows = int(spec["history_windows"])
    horizon = int(spec["forecast_horizon"])
    buffer = float(spec["forecast_buffer"])
    boundary = float(fitted["risk_intercept"] + cfg["target"]["risk_margin_gamma"])
    rows = []
    time_axis = np.arange(history_windows, dtype=float)
    centered_time = time_axis - time_axis.mean()
    denominator = float(np.sum(centered_time**2))
    for trajectory in trajectories:
        rich_means = means[trajectory["id"]]
        for design_name, design in designs.items():
            selected = rich_means[:, :, design["indices"]]
            centered = selected - fitted["observation_intercept"][design["indices"]]
            states = centered @ design["estimator"].T
            estimated_risk = risk_path(states.reshape(-1, 3), fitted).reshape(states.shape[:2])
            for replicate in range(estimated_risk.shape[0]):
                for time_index in range(history_windows - 1, estimated_risk.shape[1]):
                    history = estimated_risk[
                        replicate, time_index - history_windows + 1:time_index + 1
                    ]
                    slope = float(np.sum(centered_time * history) / denominator)
                    forecast_risk = float(history[-1] + horizon * max(slope, 0.0))
                    rows.extend(
                        [
                            {
                                "trajectory_id": trajectory["id"],
                                "direction": trajectory["direction_name"],
                                "planned_type": trajectory["planned_type"],
                                "replicate": replicate,
                                "time_index": time_index,
                                "method": f"{design_name}_forecast",
                                "alarm": forecast_risk >= boundary + buffer,
                                "alarm_score": forecast_risk,
                                "estimated_current_risk": float(history[-1]),
                            },
                            {
                                "trajectory_id": trajectory["id"],
                                "direction": trajectory["direction_name"],
                                "planned_type": trajectory["planned_type"],
                                "replicate": replicate,
                                "time_index": time_index,
                                "method": f"{design_name}_current",
                                "alarm": float(history[-1]) >= boundary + buffer,
                                "alarm_score": float(history[-1]),
                                "estimated_current_risk": float(history[-1]),
                            },
                        ]
                    )
        entropy_margin = rich_means[:, :, [4, 5]]
        proxy = entropy_model.predict(entropy_margin.reshape(-1, 2)).reshape(
            entropy_margin.shape[:2]
        )
        for replicate in range(proxy.shape[0]):
            for time_index in range(history_windows - 1, proxy.shape[1]):
                history = proxy[replicate, time_index - history_windows + 1:time_index + 1]
                slope = float(np.sum(centered_time * history) / denominator)
                forecast_risk = float(history[-1] + horizon * max(slope, 0.0))
                rows.append(
                    {
                        "trajectory_id": trajectory["id"],
                        "direction": trajectory["direction_name"],
                        "planned_type": trajectory["planned_type"],
                        "replicate": replicate,
                        "time_index": time_index,
                        "method": "entropy_margin_trend",
                        "alarm": forecast_risk >= boundary + buffer,
                        "alarm_score": forecast_risk,
                        "estimated_current_risk": float(history[-1]),
                    }
                )
    return pd.DataFrame(rows)


def plot_results(revealed: pd.DataFrame, predictions: pd.DataFrame, boundary: float,
                 output: Path, alarm_threshold: float | None = None) -> None:
    event_ids = [item for item in revealed["trajectory_id"].unique() if item.endswith(":event")]
    fig, axes = plt.subplots(len(event_ids), 1, figsize=(9.8, 3.0 * len(event_ids)), constrained_layout=True)
    if len(event_ids) == 1:
        axes = [axes]
    for axis, trajectory_id in zip(axes, event_ids):
        risk = revealed[revealed["trajectory_id"] == trajectory_id].sort_values("time_index")
        axis.plot(risk["time_index"], risk["risk_revealed"], "-o", color="#333333", label="revealed risk")
        axis.axhline(boundary, color="#c00000", linestyle="--", label="relative boundary")
        if alarm_threshold is not None:
            axis.axhline(
                alarm_threshold, color="#7030a0", linestyle=":", label="forecast alarm threshold"
            )
        for method, color in (("compressed_forecast", "#2f5597"), ("full_forecast", "#c55a11")):
            frame = predictions[
                (predictions["trajectory_id"] == trajectory_id) & (predictions["method"] == method)
            ]
            mean_score = frame.groupby("time_index")["alarm_score"].mean()
            axis.plot(mean_score.index, mean_score.values, color=color, label=method)
        axis.set_ylabel("Risk / forecast")
        axis.set_title(trajectory_id)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Sequential window")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "early_warning_sequence_probe_v2.json").read_text())
    output = ROOT / "results" / cfg["output_tag"]
    figures = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
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
        "event_paths_cross_fitted_boundary": all(
            float(item["predicted_risks"][-1]) >= boundary for item in trajectories
            if item["planned_type"] == "event"
        ),
        "stationary_controls_below_fitted_boundary": all(
            float(np.max(item["predicted_risks"])) < boundary for item in trajectories
            if item["planned_type"] == "stationary_safe"
        ),
    }
    pretarget = {
        "metrics": metrics,
        "gates": pretarget_gates,
        "all_passed": bool(all(pretarget_gates.values())),
        "sequence_seed": cfg["sequence_seed"],
        "frozen_protocol": {
            key: spec[key] for key in (
                "batch_size", "replicates", "history_windows", "forecast_horizon",
                "forecast_buffer", "crossing_confirmation_windows"
            )
        },
        "frozen_trajectories": [
            {
                "trajectory_id": item["id"],
                "direction": item["direction"],
                "predicted_start_risk": float(item["predicted_risks"][0]),
                "predicted_end_risk": float(item["predicted_risks"][-1]),
            }
            for item in trajectories
        ],
    }
    (output / "pretarget_early_warning_gate.json").write_text(
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
        print(json.dumps({"aborted_before_sequence": True, "pretarget": pretarget}, indent=2))
        return
    sequence_cfg = copy.deepcopy(cfg)
    sequence_cfg["master_seed"] = int(cfg["sequence_seed"])
    center = np.asarray(cfg["calibration"]["theta_center"])
    means = generate_online_means(sequence_cfg, model, center, trajectories)
    predictions = scalar_predictions(means, trajectories, designs, fitted, entropy_model, cfg)
    predictions.to_csv(output / "sealed_online_predictions.csv", index=False)
    revealed = reveal_trajectory_risks(sequence_cfg, model, center, trajectories)
    revealed.to_csv(output / "revealed_trajectory_risks.csv", index=False)
    outcomes, records = score_predictions(predictions, revealed, boundary, cfg)
    outcomes.to_csv(output / "trajectory_outcomes.csv", index=False)
    records.to_csv(output / "replicate_warning_ledger.csv", index=False)
    summary = summarize_scores(records)
    summary.to_csv(output / "warning_method_summary.csv", index=False)
    indexed = summary.set_index("method")
    compressed = indexed.loc["compressed_forecast"]
    full = indexed.loc["full_forecast"]
    actual_events = int(outcomes["actual_event"].sum())
    actual_non_events = int((~outcomes["actual_event"]).sum())
    compressed_false_alarm = compressed["non_event_false_alarm_rate"]
    compressed_lead = compressed["median_timely_lead"]
    final_gates = {
        "actual_event_trajectories": actual_events >= gates["minimum_actual_event_trajectories"],
        "actual_non_event_trajectories": actual_non_events >= gates["minimum_actual_non_event_trajectories"],
        "compressed_timely_warning": compressed["timely_warning_rate"] >= gates["minimum_compressed_timely_warning_rate"],
        "compressed_false_alarm": bool(
            pd.notna(compressed_false_alarm)
            and compressed_false_alarm <= gates["maximum_compressed_non_event_false_alarm_rate"]
        ),
        "compressed_premature_warning": compressed["premature_warning_rate"] <= gates["maximum_compressed_premature_warning_rate"],
        "compressed_lead": bool(
            pd.notna(compressed_lead)
            and compressed_lead >= gates["minimum_compressed_median_timely_lead"]
        ),
        "compressed_near_full": full["timely_warning_rate"] - compressed["timely_warning_rate"] <= gates["maximum_timely_rate_gap_from_full"],
    }
    checks = {
        "pretarget_all_passed": True,
        "metrics": {
            "actual_event_trajectories": actual_events,
            "actual_non_event_trajectories": actual_non_events,
            "compressed_timely_warning_rate": compressed["timely_warning_rate"],
            "compressed_non_event_false_alarm_rate": compressed_false_alarm,
            "compressed_premature_warning_rate": compressed["premature_warning_rate"],
            "compressed_median_timely_lead": compressed_lead,
            "full_timely_warning_rate": full["timely_warning_rate"],
        },
        "gates": final_gates,
        "all_passed": bool(all(final_gates.values())),
        "semantics": "held-out-stream probe of relative-risk warning; calibration/model retained from v1 exploration",
    }
    (output / "checks.json").write_text(
        json.dumps(serialize(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(revealed, predictions, boundary, figures / "early_warning_sequence_probe_v2.png")
    print(json.dumps(serialize({"checks": checks, "outcomes": outcomes.to_dict("records"),
                                "summary": summary.to_dict("records")}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
