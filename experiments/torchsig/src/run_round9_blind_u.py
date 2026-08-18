#!/usr/bin/env python3
"""Round 9: blind-u, training-referenced Cliff warning.

The warning models never receive the three TorchSig impairment coordinates.
Those coordinates remain available only to the experiment generator, the
oracle comparator, and sealed evaluation.  The experiment separates two
questions:

1. can risk position and velocity be learned directly from outcome-blind
   telemetry without a named physical mechanism chart; and
2. do sample-level statistics relative to the realized training set add value
   beyond the frozen 25-dimensional telemetry mean?
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from measurement_design import MEASUREMENT_GROUPS, group_indices
from probe_cliff_early_warning import (
    crossing_time,
    reveal_trajectory_risks,
    risk_path,
    score_predictions,
    summarize_scores,
    trajectory_scalars,
)
from probe_cliff_early_warning import build_design as build_oracle_design
from probe_early_warning_geometry import direction_record, oriented_axis
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import (
    FEATURE_NAMES,
    calibration_panel,
    environment_record,
    extract_features,
    generate_iq,
    generate_samples,
    rich_observation,
    stable_seed,
)


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_manifest(root: Path) -> None:
    output = root / "SHA256SUMS.txt"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output)
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        for path in files
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_with_reference(cfg: dict) -> tuple[ExtraTreesClassifier, np.ndarray, dict]:
    """Reproduce the deployment fit while retaining its realized training set."""
    train_cfg = cfg["training"]
    total = int(train_cfg["samples_per_class"]) * len(cfg["classes"])
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], "training"))
    labels = np.arange(total) % len(cfg["classes"])
    rng.shuffle(labels)
    features = np.empty((total, len(FEATURE_NAMES)), dtype=float)
    theta_values = np.empty((total, 3), dtype=float)
    for index, label in enumerate(labels):
        theta = rng.uniform(train_cfg["theta_low"], train_cfg["theta_high"], size=3)
        theta_values[index] = theta
        iq = generate_iq(cfg["classes"][int(label)], theta, rng, cfg["signal"])
        features[index] = extract_features(iq)
    model = ExtraTreesClassifier(
        n_estimators=int(train_cfg["n_estimators"]),
        min_samples_leaf=int(train_cfg["min_samples_leaf"]),
        max_features="sqrt",
        random_state=stable_seed(cfg["master_seed"], "model") % (2**32 - 1),
        n_jobs=int(train_cfg.get("n_jobs", 1)),
    )
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    reference = rich_observation(features, probabilities)
    summary = {
        "n_training_samples": total,
        "resubstitution_accuracy": float(accuracy_score(labels, model.predict(features))),
        "theta_min": theta_values.min(axis=0),
        "theta_max": theta_values.max(axis=0),
        "reference_semantics": (
            "realized training inputs with post-fit model outputs; raw-feature components are "
            "not outcome labels, while model-output components are resubstitution summaries"
        ),
    }
    return model, reference, summary


def shifted_reference(cfg: dict, model: ExtraTreesClassifier, count: int) -> np.ndarray:
    """A deliberately mismatched reference used only in the swap placebo."""
    low, high = map(float, cfg["round9"]["wrong_reference_theta_range"])
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], "wrong_reference"))
    labels = np.arange(count) % len(cfg["classes"])
    rng.shuffle(labels)
    features = np.empty((count, len(FEATURE_NAMES)), dtype=float)
    for index, label in enumerate(labels):
        theta = rng.uniform(low, high, size=3)
        iq = generate_iq(cfg["classes"][int(label)], theta, rng, cfg["signal"])
        features[index] = extract_features(iq)
    return rich_observation(features, model.predict_proba(features))


def reference_parameters(samples: np.ndarray, indices: np.ndarray) -> dict:
    selected = np.asarray(samples[:, indices], dtype=float)
    variance = selected.var(axis=0)
    floor = max(float(np.median(variance)) * 1e-6, 1e-10)
    variance = np.maximum(variance, floor)
    return {
        "mean": selected.mean(axis=0),
        "variance": variance,
        "scale": np.sqrt(variance),
        "indices": np.asarray(indices, dtype=int),
    }


def batch_features(samples: np.ndarray, method: str, reference: dict | None) -> np.ndarray:
    """Map (..., batch, 54) telemetry into a frozen blind-u feature chart."""
    samples = np.asarray(samples, dtype=float)
    if method == "blind25_mean":
        return samples[..., reference["indices"]].mean(axis=-2)
    if method == "blind25_moments":
        selected = samples[..., reference["indices"]]
        return np.concatenate(
            [selected.mean(axis=-2), np.log(selected.var(axis=-2) + 1e-10)],
            axis=-1,
        )
    if method == "blind54_mean":
        return samples.mean(axis=-2)
    if method not in {"train25_relative", "train54_relative"}:
        raise ValueError(f"unknown blind-u method: {method}")
    selected = samples[..., reference["indices"]]
    mean = selected.mean(axis=-2)
    variance = selected.var(axis=-2)
    standardized_mean = (mean - reference["mean"]) / reference["scale"]
    log_variance_ratio = np.log((variance + reference["variance"] * 1e-3) / reference["variance"])
    standardized = (selected - reference["mean"]) / reference["scale"]
    radius = np.mean(standardized**2, axis=-1)
    radius_mean = radius.mean(axis=-1, keepdims=True)
    radius_q90 = np.quantile(radius, 0.90, axis=-1)[..., None]
    return np.concatenate(
        [standardized_mean, log_variance_ratio, radius_mean, radius_q90], axis=-1
    )


def new_risk_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def calibration_batches(store: dict, count: int, batch_size: int) -> np.ndarray:
    rich = np.asarray(store["rich"], dtype=float)
    environment_count = rich.shape[0] // count
    if count % batch_size:
        raise ValueError("calibration samples_per_environment must be divisible by batch_size")
    return rich.reshape(environment_count, count // batch_size, batch_size, rich.shape[1])


def fit_blind_model(
    method: str,
    environment_samples: np.ndarray,
    batch_samples: np.ndarray,
    risk: np.ndarray,
    reference: dict,
    cfg: dict,
) -> dict:
    aggregate = batch_features(environment_samples, method, reference)
    batch_matrix = batch_features(batch_samples, method, reference)
    splitter = KFold(
        n_splits=int(cfg["round9"]["cv_folds"]),
        shuffle=True,
        random_state=stable_seed(cfg["master_seed"], "round9_cv", method) % (2**32 - 1),
    )
    aggregate_prediction = np.full(len(risk), np.nan)
    batch_prediction = np.full(batch_matrix.shape[:2], np.nan)
    for train, test in splitter.split(aggregate):
        train_shape = batch_matrix[train].shape
        fold_model = new_risk_model().fit(
            batch_matrix[train].reshape(-1, train_shape[-1]),
            np.repeat(risk[train], train_shape[1]),
        )
        test_shape = batch_matrix[test].shape
        batch_prediction[test] = np.clip(
            fold_model.predict(batch_matrix[test].reshape(-1, test_shape[-1])).reshape(test_shape[:2]),
            0.0,
            1.0,
        )
        aggregate_prediction[test] = batch_prediction[test].mean(axis=1)
    repeated_risk = np.repeat(risk[:, None], batch_prediction.shape[1], axis=1)
    batch_residual = np.abs(batch_prediction - repeated_risk)
    model = new_risk_model().fit(
        batch_matrix.reshape(-1, batch_matrix.shape[-1]),
        np.repeat(risk, batch_matrix.shape[1]),
    )
    return {
        "method": method,
        "model": model,
        "feature_dimension": int(aggregate.shape[1]),
        "aggregate_cv_r2": float(r2_score(risk, aggregate_prediction)),
        "aggregate_cv_spearman": float(spearmanr(risk, aggregate_prediction).statistic),
        "aggregate_cv_mae": float(np.mean(np.abs(aggregate_prediction - risk))),
        "batch_cv_r2": float(r2_score(repeated_risk.ravel(), batch_prediction.ravel())),
        "batch_cv_mae": float(batch_residual.mean()),
        "batch_cv_q90_absolute_error": float(np.quantile(batch_residual, 0.90)),
        "selected_alpha": float(model.named_steps["ridgecv"].alpha_),
        "aggregate_cv_prediction": aggregate_prediction,
        "batch_cv_prediction": batch_prediction,
    }


def generate_online_batches(
    cfg: dict,
    model: ExtraTreesClassifier,
    center: np.ndarray,
    trajectories: list[dict],
) -> dict[str, np.ndarray]:
    spec = cfg["early_warning"]
    batch_size = int(spec["batch_size"])
    replicates = int(spec["replicates"])
    result = {}
    for trajectory in trajectories:
        windows = []
        for time_index, offset in enumerate(trajectory["offsets"]):
            rng = np.random.default_rng(
                stable_seed(cfg["master_seed"], "round9_online", trajectory["id"], time_index)
            )
            features, _ = generate_samples(
                cfg, center + offset, batch_size * replicates, rng, balanced=False
            )
            rich = rich_observation(features, model.predict_proba(features))
            windows.append(rich.reshape(replicates, batch_size, rich.shape[1]))
        result[trajectory["id"]] = np.stack(windows, axis=1)
    return result


def risk_coordinates(
    online: dict[str, np.ndarray],
    trajectories: list[dict],
    blind_models: dict,
    actual_references: dict,
    wrong_references: dict,
    oracle_design: dict,
    fitted: dict,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for trajectory in trajectories:
        batches = online[trajectory["id"]]
        coordinates = {}
        for method, item in blind_models.items():
            features = batch_features(batches, method, actual_references[method])
            shape = features.shape
            coordinates[method] = np.clip(
                item["model"].predict(features.reshape(-1, shape[-1])).reshape(shape[:2]),
                0.0,
                1.0,
            )
        wrong_features = batch_features(
            batches, "train25_relative", wrong_references["train25_relative"]
        )
        wrong_shape = wrong_features.shape
        coordinates["train25_wrong_swap"] = np.clip(
            blind_models["train25_relative"]["model"]
            .predict(wrong_features.reshape(-1, wrong_shape[-1]))
            .reshape(wrong_shape[:2]),
            0.0,
            1.0,
        )
        means = batches.mean(axis=2)
        selected = means[:, :, oracle_design["indices"]]
        centered = selected - fitted["observation_intercept"][oracle_design["indices"]]
        states = centered @ oracle_design["estimator"].T
        coordinates["oracle25_u"] = risk_path(
            states.reshape(-1, 3), fitted
        ).reshape(states.shape[:2])
        result[trajectory["id"]] = coordinates
    return result


def forecast_rows(
    coordinates: dict[str, dict[str, np.ndarray]],
    trajectories: list[dict],
    buffers: dict[str, float],
    boundary: float,
    cfg: dict,
) -> pd.DataFrame:
    history = int(cfg["early_warning"]["history_windows"])
    horizon = int(cfg["early_warning"]["forecast_horizon"])
    time = np.arange(history, dtype=float)
    centered_time = time - time.mean()
    denominator = float(np.sum(centered_time**2))
    rows = []
    for trajectory in trajectories:
        for method, values in coordinates[trajectory["id"]].items():
            buffer = float(buffers[method])
            for replicate in range(values.shape[0]):
                for time_index in range(history - 1, values.shape[1]):
                    window = values[replicate, time_index - history + 1 : time_index + 1]
                    slope = float(np.sum(centered_time * window) / denominator)
                    current = float(window[-1])
                    forecast = current + horizon * max(slope, 0.0)
                    for suffix, score in (("forecast", forecast), ("current", current)):
                        rows.append(
                            {
                                "trajectory_id": trajectory["id"],
                                "direction": trajectory["direction_name"],
                                "planned_type": trajectory["planned_type"],
                                "replicate": replicate,
                                "time_index": time_index,
                                "method": f"{method}_{suffix}",
                                "alarm": bool(score >= boundary + buffer),
                                "alarm_score": score,
                                "estimated_current_risk": current,
                                "estimated_slope": slope,
                                "forecast_buffer": buffer,
                            }
                        )
    return pd.DataFrame(rows)


def position_metrics(predictions: pd.DataFrame, revealed: pd.DataFrame) -> pd.DataFrame:
    truth = revealed[["trajectory_id", "time_index", "risk_revealed"]]
    current = predictions[predictions["method"].str.endswith("_current")].merge(
        truth, on=["trajectory_id", "time_index"], how="left"
    )
    rows = []
    for method, frame in current.groupby("method"):
        error = frame["estimated_current_risk"] - frame["risk_revealed"]
        estimated = frame["estimated_current_risk"].to_numpy()
        observed = frame["risk_revealed"].to_numpy()
        rank = (
            float(spearmanr(estimated, observed).statistic)
            if np.std(estimated) > 1e-12 and np.std(observed) > 1e-12
            else None
        )
        rows.append(
            {
                "method": method,
                "position_mae": float(np.mean(np.abs(error))),
                "position_rmse": float(np.sqrt(np.mean(error**2))),
                "position_bias": float(np.mean(error)),
                "position_spearman": rank,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_interval(
    values: np.ndarray, seed: int, draws: int = 10000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_summary(
    calibration: pd.DataFrame,
    warning: pd.DataFrame,
    position: pd.DataFrame,
    revealed: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
) -> None:
    methods = [
        "blind25_mean", "blind25_moments", "blind54_mean",
        "train25_relative", "train54_relative"
    ]
    colors = {
        "blind25_mean": "#7f8c8d",
        "blind25_moments": "#6c5b7b",
        "blind54_mean": "#34495e",
        "train25_relative": "#2f5597",
        "train54_relative": "#00a6a6",
        "train25_wrong_swap": "#c00000",
        "oracle25_u": "#c55a11",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    calib = calibration.set_index("method").loc[methods]
    axes[0, 0].bar(
        np.arange(len(methods)), calib["aggregate_cv_r2"],
        color=[colors[item] for item in methods]
    )
    axes[0, 0].set_xticks(np.arange(len(methods)), methods, rotation=25, ha="right")
    axes[0, 0].set_ylabel("Calibration CV $R^2$")
    axes[0, 0].set_title("Blind-u risk-position calibration")
    pos = position.copy()
    pos["base"] = pos["method"].str.removesuffix("_current")
    pos = pos.set_index("base")
    shown = [item for item in colors if item in pos.index]
    axes[0, 1].bar(
        np.arange(len(shown)), pos.loc[shown, "position_mae"],
        color=[colors[item] for item in shown]
    )
    axes[0, 1].set_xticks(np.arange(len(shown)), shown, rotation=25, ha="right")
    axes[0, 1].set_ylabel("Sealed position MAE")
    axes[0, 1].set_title("Current-risk coordinate error")
    warn = warning[warning["method"].str.endswith("_forecast")].copy()
    warn["base"] = warn["method"].str.removesuffix("_forecast")
    for _, row in warn.iterrows():
        axes[1, 0].scatter(
            row["non_event_false_alarm_rate"], row["timely_warning_rate"],
            s=75, color=colors.get(row["base"], "#555555"), label=row["base"]
        )
    axes[1, 0].set_xlabel("Stationary false-alarm rate")
    axes[1, 0].set_ylabel("Timely-warning rate")
    axes[1, 0].set_title("Operational warning trade-off")
    axes[1, 0].legend(fontsize=7)
    event_id = next(item for item in revealed["trajectory_id"].unique() if item.endswith(":event"))
    truth = revealed[revealed["trajectory_id"] == event_id].sort_values("time_index")
    axes[1, 1].plot(truth["time_index"], truth["risk_revealed"], "-o", color="#222222", label="revealed")
    for base in ("blind25_mean", "train25_relative", "train25_wrong_swap", "oracle25_u"):
        frame = predictions[
            (predictions["trajectory_id"] == event_id)
            & (predictions["method"] == f"{base}_forecast")
        ]
        mean_score = frame.groupby("time_index")["alarm_score"].mean()
        axes[1, 1].plot(mean_score.index, mean_score.values, color=colors[base], label=base)
    axes[1, 1].set_xlabel("Sequential window")
    axes[1, 1].set_ylabel("Risk / forecast score")
    axes[1, 1].set_title(event_id)
    axes[1, 1].legend(fontsize=7)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round9_blind_u_probe.json")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / "results" / cfg["output_tag"]
    figure_dir = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", cfg)

    model, training_reference, training = train_with_reference(cfg)
    wrong = shifted_reference(cfg, model, len(training_reference))
    environment, store = calibration_panel(cfg, model)
    fitted = fit_screening_bridge(environment, store, cfg)
    count = int(cfg["calibration"]["samples_per_environment"])
    batch_size = int(cfg["early_warning"]["batch_size"])
    environment_samples = np.asarray(store["rich"]).reshape(len(environment), count, -1)
    batch_samples = calibration_batches(store, count, batch_size)
    fixed_indices = group_indices(tuple(cfg["early_warning"]["fixed_groups"]))
    all_indices = np.arange(environment_samples.shape[-1])
    actual_references = {
        "blind25_mean": reference_parameters(training_reference, fixed_indices),
        "blind25_moments": reference_parameters(training_reference, fixed_indices),
        "blind54_mean": reference_parameters(training_reference, all_indices),
        "train25_relative": reference_parameters(training_reference, fixed_indices),
        "train54_relative": reference_parameters(training_reference, all_indices),
    }
    wrong_references = {
        "train25_relative": reference_parameters(wrong, fixed_indices),
        "train54_relative": reference_parameters(wrong, all_indices),
    }
    risk = environment["risk"].to_numpy(dtype=float)
    blind_models = {
        method: fit_blind_model(
            method, environment_samples, batch_samples, risk,
            actual_references[method], cfg
        )
        for method in actual_references
    }
    calibration_rows = [
        {key: value for key, value in item.items() if not key.endswith("prediction") and key != "model"}
        for item in blind_models.values()
    ]
    calibration_summary = pd.DataFrame(calibration_rows)
    calibration_summary.to_csv(output / "blind_u_calibration_summary.csv", index=False)

    oracle_design = build_oracle_design(
        tuple(cfg["early_warning"]["fixed_groups"]), fitted, cfg
    )
    tau = float(fitted["risk_intercept"])
    b = np.asarray(fitted["b"], dtype=float)
    hessian = np.asarray(fitted["H"], dtype=float)
    limit = float(cfg["early_warning"]["trajectory_offset_limit"])
    directions = {
        "noise": oriented_axis("noise", 0, tau, b, hessian, limit),
        "phase": oriented_axis("phase", 1, tau, b, hessian, limit),
        "mixed_gradient": b / max(float(np.linalg.norm(b)), 1e-12),
    }
    geometry = [
        direction_record(name, directions[name], tau, b, hessian, cfg)
        for name in cfg["early_warning"]["directions"]
    ]
    boundary = tau + float(cfg["target"]["risk_margin_gamma"])
    pretarget_metrics = {
        "calibration_risk_range": float(risk.max() - risk.min()),
        "oracle_risk_surface_r2": float(fitted["risk_surface_r2"]),
        "oracle_risk_surface_cv_r2": float(fitted["risk_surface_cv_r2"]),
        "oracle25_risk_null_ratio": float(oracle_design["risk_null_ratio"]),
        "relative_warning_boundary": boundary,
        "feasible_directions": int(sum(item["feasible"] for item in geometry)),
        "blind_models": {
            item["method"]: {key: value for key, value in item.items() if key not in {"method"}}
            for item in calibration_rows
        },
    }
    probe_gates = cfg["round9"]["pretarget_gates"]
    train25 = blind_models["train25_relative"]
    blind25 = blind_models["blind25_moments"]
    pretarget_gates = {
        "risk_range": pretarget_metrics["calibration_risk_range"] >= probe_gates["minimum_risk_range"],
        "oracle_surface": pretarget_metrics["oracle_risk_surface_cv_r2"] >= probe_gates["minimum_oracle_cv_r2"],
        "trajectory_geometry": pretarget_metrics["feasible_directions"] == len(geometry),
        "blind_chart_r2": train25["aggregate_cv_r2"] >= probe_gates["minimum_train25_cv_r2"],
        "blind_chart_rank": train25["aggregate_cv_spearman"] >= probe_gates["minimum_train25_cv_spearman"],
        "blind_chart_batch_error": train25["batch_cv_mae"] <= probe_gates["maximum_train25_batch_mae"],
        "training_reference_calibration_noninferiority": (
            blind25["batch_cv_mae"] - train25["batch_cv_mae"]
            >= probe_gates["minimum_train25_batch_mae_gain"]
        ),
    }
    pretarget = {
        "metrics": pretarget_metrics,
        "gates": pretarget_gates,
        "all_passed": bool(all(pretarget_gates.values())),
        "target_release_rule": cfg["round9"]["target_release_rule"],
        "physical_u_policy": (
            "hidden from every blind model; used only for trajectory generation, oracle comparator, "
            "and sealed evaluation"
        ),
        "geometry": geometry,
    }
    write_json(output / "pretarget_round9_gate.json", pretarget)
    environment.to_csv(output / "calibration_environment_panel.csv", index=False)
    training["wrong_reference_theta_range"] = cfg["round9"]["wrong_reference_theta_range"]
    write_json(output / "training_summary.json", training)
    write_json(output / "environment.json", environment_record())
    reference_rows = []
    names = ["prob_bpsk", "prob_qpsk", "prob_8psk", "prob_16qam", "entropy", "margin"] + FEATURE_NAMES
    actual_full = reference_parameters(training_reference, all_indices)
    wrong_full = reference_parameters(wrong, all_indices)
    for index, name in enumerate(names):
        reference_rows.append(
            {
                "dimension": index,
                "name": name,
                "training_mean": actual_full["mean"][index],
                "training_variance": actual_full["variance"][index],
                "wrong_mean": wrong_full["mean"][index],
                "wrong_variance": wrong_full["variance"][index],
            }
        )
    pd.DataFrame(reference_rows).to_csv(output / "training_reference_summary.csv", index=False)

    release = cfg["round9"]["target_release_rule"]
    release_pass = (
        pretarget_gates["blind_chart_r2"]
        and pretarget_gates["blind_chart_rank"]
        and pretarget_gates["blind_chart_batch_error"]
        and pretarget_gates["trajectory_geometry"]
    )
    if args.calibration_only or (release == "require_blind_chart" and not release_pass):
        sha256_manifest(output)
        print(json.dumps(serialize({"target_generated": False, "pretarget": pretarget}), indent=2))
        return

    trajectories = []
    for item in geometry:
        if not item["feasible"]:
            continue
        direction = np.asarray(item["direction"], dtype=float)
        for planned_type, end in (
            ("event", float(item["end_scalar"])),
            ("stationary_safe", float(item["start_scalar"])),
        ):
            scalars = trajectory_scalars(
                float(item["start_scalar"]), end, cfg["early_warning"]
            )
            trajectories.append(
                {
                    "id": f"{item['name']}:{planned_type}",
                    "direction_name": item["name"],
                    "planned_type": planned_type,
                    "direction": direction,
                    "scalars": scalars,
                    "offsets": scalars[:, None] * direction[None, :],
                }
            )
    target_cfg = copy.deepcopy(cfg)
    target_cfg["master_seed"] = int(cfg["target_seed"])
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    online = generate_online_batches(target_cfg, model, center, trajectories)
    coordinates = risk_coordinates(
        online, trajectories, blind_models, actual_references, wrong_references,
        oracle_design, fitted
    )
    buffers = {
        method: float(item["batch_cv_q90_absolute_error"])
        for method, item in blind_models.items()
    }
    buffers["train25_wrong_swap"] = buffers["train25_relative"]
    buffers["oracle25_u"] = float(fitted["risk_cv_abs_residual_quantiles"]["q90"])
    predictions = forecast_rows(coordinates, trajectories, buffers, boundary, cfg)
    predictions.to_csv(output / "sealed_online_predictions.csv", index=False)
    revealed = reveal_trajectory_risks(target_cfg, model, center, trajectories)
    revealed.to_csv(output / "revealed_trajectory_risks.csv", index=False)
    outcomes, records = score_predictions(predictions, revealed, boundary, cfg)
    outcomes.to_csv(output / "trajectory_outcomes.csv", index=False)
    records.to_csv(output / "replicate_warning_ledger.csv", index=False)
    warning = summarize_scores(records)
    warning.to_csv(output / "warning_method_summary.csv", index=False)
    position = position_metrics(predictions, revealed)
    position.to_csv(output / "position_metric_summary.csv", index=False)

    warning_index = warning.set_index("method")
    position_index = position.set_index("method")
    train_forecast = warning_index.loc["train25_relative_forecast"]
    blind_forecast = warning_index.loc["blind25_moments_forecast"]
    oracle_forecast = warning_index.loc["oracle25_u_forecast"]
    wrong_forecast = warning_index.loc["train25_wrong_swap_forecast"]
    train_position = position_index.loc["train25_relative_current"]
    blind_position = position_index.loc["blind25_moments_current"]
    wrong_position = position_index.loc["train25_wrong_swap_current"]
    train_current = warning_index.loc["train25_relative_current"]
    paired_warning = (
        records[
            records["method"].isin(
                ["train25_relative_forecast", "blind25_moments_forecast"]
            )
            & records["actual_event"]
        ][["trajectory_id", "replicate", "method", "timely_warning"]]
        .pivot(index=["trajectory_id", "replicate"], columns="method", values="timely_warning")
        .dropna()
    )
    timely_pair_difference = (
        paired_warning["train25_relative_forecast"].astype(float)
        - paired_warning["blind25_moments_forecast"].astype(float)
    ).to_numpy()
    timely_interval = paired_bootstrap_interval(
        timely_pair_difference,
        stable_seed(cfg["target_seed"], "round9_timely_bootstrap"),
    )
    current_truth = revealed[["trajectory_id", "time_index", "risk_revealed"]]
    paired_position = predictions[
        predictions["method"].isin(
            ["train25_relative_current", "blind25_moments_current"]
        )
    ].merge(current_truth, on=["trajectory_id", "time_index"], how="left")
    paired_position["absolute_error"] = np.abs(
        paired_position["estimated_current_risk"] - paired_position["risk_revealed"]
    )
    paired_position = (
        paired_position.groupby(["trajectory_id", "replicate", "method"])["absolute_error"]
        .mean()
        .unstack("method")
        .dropna()
    )
    position_pair_difference = (
        paired_position["blind25_moments_current"]
        - paired_position["train25_relative_current"]
    ).to_numpy()
    position_interval = paired_bootstrap_interval(
        position_pair_difference,
        stable_seed(cfg["target_seed"], "round9_position_bootstrap"),
    )
    final_metrics = {
        "actual_event_trajectories": int(outcomes["actual_event"].sum()),
        "actual_non_event_trajectories": int((~outcomes["actual_event"]).sum()),
        "train25_timely_warning_rate": float(train_forecast["timely_warning_rate"]),
        "blind25_timely_warning_rate": float(blind_forecast["timely_warning_rate"]),
        "oracle25_timely_warning_rate": float(oracle_forecast["timely_warning_rate"]),
        "wrong_swap_timely_warning_rate": float(wrong_forecast["timely_warning_rate"]),
        "train25_false_alarm_rate": float(train_forecast["non_event_false_alarm_rate"]),
        "blind25_false_alarm_rate": float(blind_forecast["non_event_false_alarm_rate"]),
        "oracle25_false_alarm_rate": float(oracle_forecast["non_event_false_alarm_rate"]),
        "wrong_swap_false_alarm_rate": float(wrong_forecast["non_event_false_alarm_rate"]),
        "train25_median_lead": float(train_forecast["median_timely_lead"]),
        "train25_premature_warning_rate": float(train_forecast["premature_warning_rate"]),
        "train25_position_mae": float(train_position["position_mae"]),
        "blind25_position_mae": float(blind_position["position_mae"]),
        "wrong_swap_position_mae": float(wrong_position["position_mae"]),
        "training_reference_timely_gain": float(
            train_forecast["timely_warning_rate"] - blind_forecast["timely_warning_rate"]
        ),
        "training_reference_timely_gain_bootstrap_interval": timely_interval,
        "training_reference_position_mae_gain": float(
            blind_position["position_mae"] - train_position["position_mae"]
        ),
        "training_reference_position_mae_gain_bootstrap_interval": position_interval,
        "blind_u_timely_gap_to_oracle": float(
            oracle_forecast["timely_warning_rate"] - train_forecast["timely_warning_rate"]
        ),
        "train25_forecast_gain_over_current": float(
            train_forecast["timely_warning_rate"] - train_current["timely_warning_rate"]
        ),
        "wrong_swap_timely_drop": float(
            train_forecast["timely_warning_rate"] - wrong_forecast["timely_warning_rate"]
        ),
        "wrong_swap_position_mae_increase": float(
            wrong_position["position_mae"] - train_position["position_mae"]
        ),
    }
    annotated = records.merge(
        outcomes[["trajectory_id", "direction"]], on="trajectory_id", how="left"
    )
    direction_rows = []
    for (method, direction), frame in annotated.groupby(["method", "direction"]):
        events = frame[frame["actual_event"]]
        non_events = frame[~frame["actual_event"]]
        direction_rows.append(
            {
                "method": method,
                "direction": direction,
                "timely_warning_rate": float(events["timely_warning"].mean()) if len(events) else None,
                "false_alarm_rate": float(non_events["false_alarm"].mean()) if len(non_events) else None,
            }
        )
    direction_summary = pd.DataFrame(direction_rows)
    direction_summary.to_csv(output / "direction_warning_summary.csv", index=False)
    train_directions = direction_summary[
        direction_summary["method"] == "train25_relative_forecast"
    ]
    final_metrics["minimum_direction_timely_warning_rate"] = float(
        train_directions["timely_warning_rate"].min()
    )
    final_metrics["maximum_direction_false_alarm_rate"] = float(
        train_directions["false_alarm_rate"].max()
    )
    final_gate_spec = cfg["round9"]["final_gates"]
    final_gates = {
        "event_paths_revealed": final_metrics["actual_event_trajectories"] >= len(cfg["early_warning"]["directions"]),
        "stationary_paths_revealed": final_metrics["actual_non_event_trajectories"] >= len(cfg["early_warning"]["directions"]),
        "blind_u_timely_warning": final_metrics["train25_timely_warning_rate"] >= final_gate_spec["minimum_train25_timely_warning_rate"],
        "blind_u_false_alarm": final_metrics["train25_false_alarm_rate"] <= final_gate_spec["maximum_train25_false_alarm_rate"],
        "blind_u_oracle_gap": final_metrics["blind_u_timely_gap_to_oracle"] <= final_gate_spec["maximum_timely_gap_to_oracle"],
        "blind_u_position": final_metrics["train25_position_mae"] <= final_gate_spec["maximum_train25_position_mae"],
        "blind_u_premature_warning": final_metrics["train25_premature_warning_rate"] <= final_gate_spec["maximum_train25_premature_warning_rate"],
        "blind_u_lead": final_metrics["train25_median_lead"] >= final_gate_spec["minimum_train25_median_lead"],
        "forecast_gain_over_current": final_metrics["train25_forecast_gain_over_current"] >= final_gate_spec["minimum_forecast_gain_over_current"],
        "direction_replication": final_metrics["minimum_direction_timely_warning_rate"] >= final_gate_spec["minimum_direction_timely_warning_rate"],
        "direction_false_alarm": final_metrics["maximum_direction_false_alarm_rate"] <= final_gate_spec["maximum_direction_false_alarm_rate"],
        "training_reference_position_noninferiority": final_metrics["training_reference_position_mae_gain"] >= final_gate_spec["minimum_position_mae_gain"],
        "training_reference_warning_gain": final_metrics["training_reference_timely_gain"] >= final_gate_spec["minimum_timely_warning_gain"],
        "training_reference_warning_gain_interval": final_metrics["training_reference_timely_gain_bootstrap_interval"][0] >= final_gate_spec["minimum_timely_gain_ci_lower"],
        "wrong_reference_position_placebo": final_metrics["wrong_swap_position_mae_increase"] >= final_gate_spec["minimum_wrong_swap_position_mae_increase"],
        "wrong_reference_warning_placebo": final_metrics["wrong_swap_timely_drop"] >= final_gate_spec["minimum_wrong_swap_timely_drop"],
    }
    checks = {
        "pretarget_all_passed": bool(pretarget["all_passed"]),
        "pretarget": pretarget,
        "metrics": final_metrics,
        "gates": final_gates,
        "all_passed": bool(all(final_gates.values())),
        "interpretation_policy": {
            "blind_u": "supported only if blind-u operational gates pass",
            "training_reference_increment": (
                "supported only if position is noninferior and timely warning improves over the "
                "matched moment control; otherwise the "
                "training set is a coordinate reference without demonstrated incremental value"
            ),
            "wrong_reference": "negative-control sensitivity, not a mechanism benefit",
        },
    }
    write_json(output / "checks.json", checks)
    plot_summary(
        calibration_summary, warning, position, revealed, predictions,
        figure_dir / f"{cfg['output_tag']}.png"
    )
    sha256_manifest(output)
    print(json.dumps(serialize({"checks": checks, "warning": warning.to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()
