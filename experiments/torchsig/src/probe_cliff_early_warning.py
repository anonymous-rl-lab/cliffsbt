#!/usr/bin/env python3
"""Minimal sealed sequential probe for unlabeled Cliff early warning."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from measurement_design import MEASUREMENT_GROUPS, group_indices, subset_information
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import (
    calibration_panel,
    environment_record,
    generate_samples,
    quadratic_risk,
    rich_observation,
    stable_seed,
    train_deployment_model,
)


ROOT = Path(__file__).resolve().parents[1]


def build_design(groups: tuple[str, ...], fitted: dict, cfg: dict) -> dict:
    indices = group_indices(groups)
    info = subset_information(fitted, indices, cfg["target"]["rank_relative_tolerance"])
    estimator = info["Q_pinv_per_sample"] @ info["A"].T @ info["Sigma_inverse_per_sample"]
    return {
        "groups": groups,
        "indices": indices,
        "cost": int(len(indices)),
        "risk_null_ratio": float(info["risk_null_ratio"]),
        "effective_rank": int(info["effective_rank"]),
        "estimator": estimator,
        "state_covariance_per_sample": info["Q_pinv_per_sample"],
    }


def scalar_for_risk(direction: np.ndarray, start: float, end: float, target: float,
                    fitted: dict) -> float:
    grid = np.linspace(start, end, 4001)
    offsets = grid[:, None] * direction[None, :]
    risk = quadratic_risk(offsets, fitted["risk_intercept"], fitted["b"], fitted["H"])
    return float(grid[int(np.argmin(np.abs(risk - target)))])


def trajectory_scalars(start: float, end: float, spec: dict) -> np.ndarray:
    return np.r_[
        np.repeat(start, int(spec["pre_plateau_windows"])),
        np.linspace(start, end, int(spec["drift_windows"])),
        np.repeat(end, int(spec["post_plateau_windows"])),
    ]


def crossing_time(risk: np.ndarray, boundary: float, confirmation: int) -> int | None:
    above = np.asarray(risk >= boundary, dtype=bool)
    for index in range(0, len(above) - confirmation + 1):
        if bool(np.all(above[index:index + confirmation])):
            return int(index)
    return None


def fit_velocity(history: np.ndarray) -> np.ndarray:
    time = np.arange(len(history), dtype=float)
    centered = time - time.mean()
    return np.sum(centered[:, None] * history, axis=0) / float(np.sum(centered**2))


def risk_path(states: np.ndarray, fitted: dict) -> np.ndarray:
    return np.asarray(
        quadratic_risk(states, fitted["risk_intercept"], fitted["b"], fitted["H"]),
        dtype=float,
    )


def warning_probabilities(history: np.ndarray, covariance: np.ndarray, fitted: dict,
                          cfg: dict, seed_parts: tuple) -> tuple[float, float, float, int | None]:
    spec = cfg["early_warning"]
    draws = int(spec["posterior_draws"])
    horizon = int(spec["forecast_horizon"])
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], *seed_parts))
    sampled = rng.multivariate_normal(
        np.zeros(3), covariance, size=(draws, len(history)), check_valid="ignore"
    ) + history[None, :, :]
    velocities = np.asarray([fit_velocity(item) for item in sampled])
    future_h = np.arange(1, horizon + 1, dtype=float)
    future = sampled[:, -1:, :] + future_h[None, :, None] * velocities[:, None, :]
    future = np.clip(future, -float(spec["state_clip"]), float(spec["state_clip"]))
    future_risk = risk_path(future.reshape(-1, 3), fitted).reshape(draws, horizon)
    current_risk = risk_path(sampled[:, -1, :], fitted)
    boundary = float(fitted["risk_intercept"] + cfg["target"]["risk_margin_gamma"])
    forecast_probability = float(np.mean(np.any(future_risk >= boundary, axis=1)))
    current_probability = float(np.mean(current_risk >= boundary))
    point_velocity = fit_velocity(history)
    point_future = history[-1][None, :] + future_h[:, None] * point_velocity[None, :]
    point_future = np.clip(point_future, -float(spec["state_clip"]), float(spec["state_clip"]))
    point_risk = risk_path(point_future, fitted)
    crossing = np.flatnonzero(point_risk >= boundary)
    point_horizon = int(crossing[0] + 1) if len(crossing) else None
    return forecast_probability, current_probability, float(risk_path(history[-1:], fitted)[0]), point_horizon


def generate_online_means(cfg: dict, model, center: np.ndarray, trajectories: list[dict]) -> dict:
    spec = cfg["early_warning"]
    batch = int(spec["batch_size"])
    replicates = int(spec["replicates"])
    means = {}
    for trajectory in trajectories:
        values = []
        for time_index, offset in enumerate(trajectory["offsets"]):
            rng = np.random.default_rng(
                stable_seed(cfg["master_seed"], "early_warning_online", trajectory["id"], time_index)
            )
            features, _ = generate_samples(
                cfg, center + offset, batch * replicates, rng, balanced=False
            )
            probabilities = model.predict_proba(features)
            rich = rich_observation(features, probabilities)
            values.append(rich.reshape(replicates, batch, -1).mean(axis=1))
        means[trajectory["id"]] = np.stack(values, axis=1)
    return means


def online_predictions(means: dict, trajectories: list[dict], designs: dict,
                       fitted: dict, entropy_model: Ridge, cfg: dict) -> pd.DataFrame:
    spec = cfg["early_warning"]
    history_windows = int(spec["history_windows"])
    threshold = float(spec["alarm_probability"])
    batch = int(spec["batch_size"])
    rows = []
    for trajectory in trajectories:
        rich_means = means[trajectory["id"]]
        for design_name, design in designs.items():
            selected = rich_means[:, :, design["indices"]]
            centered = selected - fitted["observation_intercept"][design["indices"]]
            states = centered @ design["estimator"].T
            covariance = design["state_covariance_per_sample"] / batch
            for replicate in range(states.shape[0]):
                for time_index in range(history_windows - 1, states.shape[1]):
                    history = states[replicate, time_index - history_windows + 1:time_index + 1]
                    forecast_p, current_p, current_risk, point_h = warning_probabilities(
                        history, covariance, fitted, cfg,
                        ("warning_mc", trajectory["id"], design_name, replicate, time_index),
                    )
                    rows.extend(
                        [
                            {
                                "trajectory_id": trajectory["id"],
                                "direction": trajectory["direction_name"],
                                "planned_type": trajectory["planned_type"],
                                "replicate": replicate,
                                "time_index": time_index,
                                "method": f"{design_name}_forecast",
                                "alarm": forecast_p >= threshold,
                                "alarm_score": forecast_p,
                                "estimated_current_risk": current_risk,
                                "point_crossing_horizon": point_h,
                            },
                            {
                                "trajectory_id": trajectory["id"],
                                "direction": trajectory["direction_name"],
                                "planned_type": trajectory["planned_type"],
                                "replicate": replicate,
                                "time_index": time_index,
                                "method": f"{design_name}_current",
                                "alarm": current_p >= threshold,
                                "alarm_score": current_p,
                                "estimated_current_risk": current_risk,
                                "point_crossing_horizon": 0,
                            },
                        ]
                    )
        entropy_margin = rich_means[:, :, [4, 5]]
        proxy_risk = entropy_model.predict(entropy_margin.reshape(-1, 2)).reshape(
            entropy_margin.shape[:2]
        )
        for replicate in range(proxy_risk.shape[0]):
            for time_index in range(history_windows - 1, proxy_risk.shape[1]):
                history = proxy_risk[replicate, time_index - history_windows + 1:time_index + 1]
                slope = float(fit_velocity(history[:, None])[0])
                future = history[-1] + np.arange(1, int(spec["forecast_horizon"]) + 1) * slope
                score = float(np.max(future) - (fitted["risk_intercept"] + cfg["target"]["risk_margin_gamma"]))
                rows.append(
                    {
                        "trajectory_id": trajectory["id"],
                        "direction": trajectory["direction_name"],
                        "planned_type": trajectory["planned_type"],
                        "replicate": replicate,
                        "time_index": time_index,
                        "method": "entropy_margin_trend",
                        "alarm": score >= 0,
                        "alarm_score": score,
                        "estimated_current_risk": float(history[-1]),
                        "point_crossing_horizon": None,
                    }
                )
    return pd.DataFrame(rows)


def reveal_trajectory_risks(cfg: dict, model, center: np.ndarray,
                            trajectories: list[dict]) -> pd.DataFrame:
    count = int(cfg["early_warning"]["reveal_samples_per_unique_state"])
    cache = {}
    rows = []
    for trajectory in trajectories:
        for time_index, offset in enumerate(trajectory["offsets"]):
            key = tuple(np.round(offset, 12))
            if key not in cache:
                rng = np.random.default_rng(stable_seed(cfg["master_seed"], "warning_reveal", *key))
                features, labels = generate_samples(cfg, center + offset, count, rng, balanced=True)
                cache[key] = float(np.mean(model.predict(features) != labels))
            rows.append(
                {
                    "trajectory_id": trajectory["id"],
                    "direction": trajectory["direction_name"],
                    "planned_type": trajectory["planned_type"],
                    "time_index": time_index,
                    "scalar": float(trajectory["scalars"][time_index]),
                    "risk_revealed": cache[key],
                }
            )
    return pd.DataFrame(rows)


def score_predictions(predictions: pd.DataFrame, revealed: pd.DataFrame,
                      boundary: float, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    confirmation = int(cfg["early_warning"]["crossing_confirmation_windows"])
    horizon = int(cfg["early_warning"]["forecast_horizon"])
    outcomes = []
    crossing_by_trajectory = {}
    for trajectory_id, frame in revealed.groupby("trajectory_id"):
        frame = frame.sort_values("time_index")
        crossing = crossing_time(frame["risk_revealed"].to_numpy(), boundary, confirmation)
        crossing_by_trajectory[trajectory_id] = crossing
        outcomes.append(
            {
                "trajectory_id": trajectory_id,
                "direction": frame["direction"].iloc[0],
                "planned_type": frame["planned_type"].iloc[0],
                "actual_event": crossing is not None,
                "actual_crossing_time": crossing,
                "maximum_revealed_risk": float(frame["risk_revealed"].max()),
            }
        )
    outcomes_frame = pd.DataFrame(outcomes)
    records = []
    for (trajectory_id, method, replicate), frame in predictions.groupby(
        ["trajectory_id", "method", "replicate"]
    ):
        alarms = frame.loc[frame["alarm"], "time_index"]
        alarm_time = int(alarms.min()) if len(alarms) else None
        crossing = crossing_by_trajectory[trajectory_id]
        lead = int(crossing - alarm_time) if crossing is not None and alarm_time is not None else None
        records.append(
            {
                "trajectory_id": trajectory_id,
                "method": method,
                "replicate": int(replicate),
                "actual_event": crossing is not None,
                "crossing_time": crossing,
                "alarm_time": alarm_time,
                "lead_time": lead,
                "timely_warning": bool(lead is not None and 1 <= lead <= horizon),
                "premature_warning": bool(lead is not None and lead > horizon),
                "late_or_no_warning": bool(crossing is not None and (lead is None or lead <= 0)),
                "false_alarm": bool(crossing is None and alarm_time is not None),
            }
        )
    return outcomes_frame, pd.DataFrame(records)


def summarize_scores(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in records.groupby("method"):
        events = frame[frame["actual_event"]]
        non_events = frame[~frame["actual_event"]]
        timely_leads = events.loc[events["timely_warning"], "lead_time"]
        rows.append(
            {
                "method": method,
                "event_replicates": int(len(events)),
                "non_event_replicates": int(len(non_events)),
                "timely_warning_rate": float(events["timely_warning"].mean()) if len(events) else None,
                "premature_warning_rate": float(events["premature_warning"].mean()) if len(events) else None,
                "late_or_no_warning_rate": float(events["late_or_no_warning"].mean()) if len(events) else None,
                "non_event_false_alarm_rate": float(non_events["false_alarm"].mean()) if len(non_events) else None,
                "median_timely_lead": float(timely_leads.median()) if len(timely_leads) else None,
            }
        )
    return pd.DataFrame(rows)


def plot_results(revealed: pd.DataFrame, predictions: pd.DataFrame, boundary: float,
                 output: Path) -> None:
    event_ids = [item for item in revealed["trajectory_id"].unique() if item.endswith(":event")]
    fig, axes = plt.subplots(len(event_ids), 1, figsize=(9.8, 3.0 * len(event_ids)), constrained_layout=True)
    if len(event_ids) == 1:
        axes = [axes]
    for axis, trajectory_id in zip(axes, event_ids):
        risk = revealed[revealed["trajectory_id"] == trajectory_id].sort_values("time_index")
        axis.plot(risk["time_index"], risk["risk_revealed"], "-o", color="#333333", label="revealed risk")
        axis.axhline(boundary, color="#c00000", linestyle="--", label="relative boundary")
        for method, color in (("compressed_forecast", "#2f5597"), ("full_forecast", "#c55a11")):
            frame = predictions[
                (predictions["trajectory_id"] == trajectory_id) & (predictions["method"] == method)
            ]
            mean_score = frame.groupby("time_index")["alarm_score"].mean()
            axis.plot(mean_score.index, mean_score.values, color=color, alpha=0.75,
                      label=f"{method} probability")
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Risk / warning probability")
        axis.set_title(trajectory_id)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Sequential window")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "early_warning_sequence_probe.json").read_text())
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
    observation_means = environment[observation_columns].to_numpy()
    entropy_model = Ridge(alpha=1e-4).fit(observation_means[:, [4, 5]], environment["risk"])
    boundary = float(fitted["risk_intercept"] + cfg["target"]["risk_margin_gamma"])
    spec = cfg["early_warning"]
    trajectories = []
    for item in spec["trajectories"]:
        direction = np.asarray(item["direction"], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        near_end = scalar_for_risk(
            direction, float(item["start_scalar"]), float(item["event_end_scalar"]),
            boundary - float(spec["near_miss_margin"]), fitted
        )
        for planned_type, end in (("event", float(item["event_end_scalar"])), ("near_miss", near_end)):
            scalars = trajectory_scalars(float(item["start_scalar"]), end, spec)
            trajectories.append(
                {
                    "id": f"{item['name']}:{planned_type}",
                    "direction_name": item["name"],
                    "planned_type": planned_type,
                    "direction": direction,
                    "start_scalar": float(item["start_scalar"]),
                    "end_scalar": float(end),
                    "scalars": scalars,
                    "offsets": scalars[:, None] * direction[None, :],
                    "predicted_risks": risk_path(scalars[:, None] * direction[None, :], fitted),
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
        "near_miss_paths_stay_below_fitted_boundary": all(
            float(np.max(item["predicted_risks"])) < boundary for item in trajectories
            if item["planned_type"] == "near_miss"
        ),
    }
    frozen_public = [
        {
            "trajectory_id": item["id"],
            "direction": item["direction"],
            "start_scalar": item["start_scalar"],
            "end_scalar": item["end_scalar"],
            "predicted_start_risk": float(item["predicted_risks"][0]),
            "predicted_end_risk": float(item["predicted_risks"][-1]),
            "windows": int(len(item["scalars"])),
        }
        for item in trajectories
    ]
    pretarget = {
        "metrics": metrics,
        "gates": pretarget_gates,
        "all_passed": bool(all(pretarget_gates.values())),
        "frozen_trajectories": frozen_public,
        "online_protocol": {
            key: spec[key] for key in (
                "batch_size", "replicates", "history_windows", "forecast_horizon",
                "posterior_draws", "alarm_probability", "crossing_confirmation_windows"
            )
        },
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
    means = generate_online_means(cfg, model, np.asarray(cfg["calibration"]["theta_center"]), trajectories)
    predictions = online_predictions(means, trajectories, designs, fitted, entropy_model, cfg)
    predictions.to_csv(output / "sealed_online_predictions.csv", index=False)
    revealed = reveal_trajectory_risks(
        cfg, model, np.asarray(cfg["calibration"]["theta_center"]), trajectories
    )
    revealed.to_csv(output / "revealed_trajectory_risks.csv", index=False)
    outcomes, records = score_predictions(predictions, revealed, boundary, cfg)
    outcomes.to_csv(output / "trajectory_outcomes.csv", index=False)
    records.to_csv(output / "replicate_warning_ledger.csv", index=False)
    summary = summarize_scores(records)
    summary.to_csv(output / "warning_method_summary.csv", index=False)
    compressed = summary.set_index("method").loc["compressed_forecast"]
    full = summary.set_index("method").loc["full_forecast"]
    actual_events = int(outcomes["actual_event"].sum())
    compressed_false_alarm = compressed["non_event_false_alarm_rate"]
    false_alarm_evaluable = bool(pd.notna(compressed_false_alarm))
    final_gates = {
        "actual_event_trajectories": actual_events >= gates["minimum_actual_event_trajectories"],
        "negative_control_available": false_alarm_evaluable,
        "compressed_timely_warning": compressed["timely_warning_rate"] >= gates["minimum_compressed_timely_warning_rate"],
        "compressed_false_alarm": bool(
            false_alarm_evaluable
            and compressed_false_alarm <= gates["maximum_compressed_non_event_false_alarm_rate"]
        ),
        "compressed_lead": compressed["median_timely_lead"] >= gates["minimum_compressed_median_timely_lead"],
        "compressed_near_full": (
            full["timely_warning_rate"] - compressed["timely_warning_rate"]
            <= gates["maximum_timely_rate_gap_from_full"]
        ),
    }
    checks = {
        "pretarget_all_passed": True,
        "metrics": {
            "actual_event_trajectories": actual_events,
            "actual_non_event_trajectories": int((~outcomes["actual_event"]).sum()),
            "compressed_timely_warning_rate": compressed["timely_warning_rate"],
            "compressed_non_event_false_alarm_rate": compressed_false_alarm,
            "compressed_median_timely_lead": compressed["median_timely_lead"],
            "full_timely_warning_rate": full["timely_warning_rate"],
        },
        "gates": final_gates,
        "all_passed": bool(all(final_gates.values())),
        "semantics": "probe of relative-risk boundary warning under persistent local drift; not absolute safety prediction",
    }
    (output / "checks.json").write_text(
        json.dumps(serialize(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(revealed, predictions, boundary, figures / "early_warning_sequence_probe.png")
    print(json.dumps(serialize({"checks": checks, "outcomes": outcomes.to_dict("records"),
                                "summary": summary.to_dict("records")}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
