#!/usr/bin/env python3
"""Round 10: intervene on D_train, retrain, and replay one shared deployment stream.

The unit of intervention is the training distribution.  Within every paired
replicate, all arms have the same sample count, class balance, learner
hyperparameters, and model random state.  Calibration examples and deployment
examples are generated once and reused by every trained model.  Consequently,
between-arm differences are attributable to the training-distribution/model
path rather than to different evaluation streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

from measurement_design import group_indices, subset_information
from probe_cliff_early_warning import crossing_time, trajectory_scalars
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import (
    FEATURE_NAMES,
    environment_record,
    extract_features,
    generate_iq,
    rich_observation,
    stable_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_manifest(root: Path) -> None:
    output = root / "SHA256SUMS.txt"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output)
    output.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )


def write_release_manifest(root: Path, names: list[str]) -> str:
    lines = []
    for name in names:
        path = root / name
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    content = "\n".join(lines) + "\n"
    (root / "PRETARGET_RELEASE_SHA256.txt").write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_arrays(items: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def path_specs(cfg: dict) -> list[dict]:
    return [
        {
            "name": item["name"],
            "direction": np.asarray(item["direction"], dtype=float),
            "start_scalar": float(item["start_scalar"]),
            "end_scalar": float(item["end_scalar"]),
        }
        for item in cfg["deployment"]["paths"]
    ]


def nearest_path_distance(theta: np.ndarray, cfg: dict) -> np.ndarray:
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    distances = []
    for item in path_specs(cfg):
        direction = item["direction"] / max(np.linalg.norm(item["direction"]), 1e-12)
        scalar = (theta - center) @ direction
        scalar = np.clip(scalar, item["start_scalar"], item["end_scalar"])
        projection = center + scalar[:, None] * direction[None, :]
        distances.append(np.linalg.norm(theta - projection, axis=1))
    return np.min(np.column_stack(distances), axis=1)


def sample_training_theta(regime: str, count: int, rng: np.random.Generator,
                          cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    spec = cfg["round10"]["regimes"][regime]
    baseline = cfg["round10"]["regimes"]["baseline"]
    if spec["type"] == "uniform":
        theta = rng.uniform(float(spec["low"]), float(spec["high"]), size=(count, 3))
        source = np.repeat(spec.get("source", regime), count).astype(object)
        return theta, source

    fraction = float(spec["enrichment_fraction"])
    enriched_count = int(round(count * fraction))
    theta = rng.uniform(float(baseline["low"]), float(baseline["high"]), size=(count, 3))
    source = np.repeat("baseline", count).astype(object)
    indices = rng.permutation(count)[:enriched_count]
    if spec["type"] == "random_broad_mixture":
        theta[indices] = rng.uniform(
            float(spec["broad_low"]), float(spec["broad_high"]), size=(enriched_count, 3)
        )
        source[indices] = "random_broad"
    elif spec["type"] == "cliff_tube_mixture":
        center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
        paths = path_specs(cfg)
        assignments = rng.integers(0, len(paths), size=enriched_count)
        jitter = float(spec["tube_jitter_std"])
        for path_index, item in enumerate(paths):
            selected = indices[assignments == path_index]
            if len(selected) == 0:
                continue
            scalar = rng.uniform(item["start_scalar"], item["end_scalar"], size=len(selected))
            direction = item["direction"] / max(np.linalg.norm(item["direction"]), 1e-12)
            theta[selected] = (
                center[None, :]
                + scalar[:, None] * direction[None, :]
                + rng.normal(0.0, jitter, size=(len(selected), 3))
            )
            source[selected] = f"cliff_tube_{item['name']}"
        theta[indices] = np.clip(theta[indices], float(spec["clip_low"]), float(spec["clip_high"]))
    else:
        raise ValueError(f"unknown training regime type: {spec['type']}")
    return theta, source


def train_model(cfg: dict, regime: str, replicate_seed: int) -> tuple[ExtraTreesClassifier, dict, pd.DataFrame]:
    train_cfg = cfg["training"]
    count = int(train_cfg["samples_per_class"]) * len(cfg["classes"])
    rng = np.random.default_rng(stable_seed(replicate_seed, "training_samples", regime))
    labels = np.arange(count) % len(cfg["classes"])
    rng.shuffle(labels)
    theta, source = sample_training_theta(regime, count, rng, cfg)
    features = np.empty((count, len(FEATURE_NAMES)), dtype=float)
    for index, label in enumerate(labels):
        iq = generate_iq(cfg["classes"][int(label)], theta[index], rng, cfg["signal"])
        features[index] = extract_features(iq)
    model_seed = stable_seed(replicate_seed, "paired_model") % (2**32 - 1)
    model = ExtraTreesClassifier(
        n_estimators=int(train_cfg["n_estimators"]),
        min_samples_leaf=int(train_cfg["min_samples_leaf"]),
        max_features="sqrt",
        random_state=model_seed,
        n_jobs=int(train_cfg.get("n_jobs", min(4, os.cpu_count() or 1))),
    )
    model.fit(features, labels)
    distance = nearest_path_distance(theta, cfg)
    ledger = pd.DataFrame(
        {
            "replicate_seed": replicate_seed,
            "regime": regime,
            "sample_index": np.arange(count),
            "class_index": labels,
            "source": source,
            "theta_noise": theta[:, 0],
            "theta_phase": theta[:, 1],
            "theta_nonlinearity": theta[:, 2],
            "nearest_frozen_path_distance": distance,
        }
    )
    summary = {
        "replicate_seed": int(replicate_seed),
        "regime": regime,
        "n_training_samples": count,
        "samples_per_class": int(train_cfg["samples_per_class"]),
        "model_random_state": int(model_seed),
        "resubstitution_accuracy": float(accuracy_score(labels, model.predict(features))),
        "theta_mean": theta.mean(axis=0),
        "theta_std": theta.std(axis=0),
        "theta_min": theta.min(axis=0),
        "theta_max": theta.max(axis=0),
        "enriched_fraction_realized": float(np.mean(source != "baseline")),
        "mean_nearest_path_distance": float(distance.mean()),
        "fraction_within_path_tube_005": float(np.mean(distance <= 0.05)),
        "training_cache_sha256": hash_arrays(
            [("theta", theta), ("labels", labels), ("features", features)]
        ),
    }
    return model, summary, ledger


def generate_shared_calibration(cfg: dict) -> tuple[list[dict], str]:
    levels = cfg["calibration"]["theta_offset_levels"]
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    count = int(cfg["calibration"]["samples_per_environment"])
    cache = []
    hash_items = []
    for environment, values in enumerate(product(levels, repeat=3)):
        offset = np.asarray(values, dtype=float)
        theta = center + offset
        rng = np.random.default_rng(stable_seed(cfg["evaluation_seed"], "calibration", environment))
        labels = np.arange(count) % len(cfg["classes"])
        rng.shuffle(labels)
        features = np.empty((count, len(FEATURE_NAMES)), dtype=float)
        for index, label in enumerate(labels):
            iq = generate_iq(cfg["classes"][int(label)], theta, rng, cfg["signal"])
            features[index] = extract_features(iq)
        cache.append({"environment": environment, "offset": offset, "theta": theta,
                      "features": features, "labels": labels})
        hash_items.extend([(f"features_{environment}", features), (f"labels_{environment}", labels)])
    return cache, hash_arrays(hash_items)


def evaluate_calibration(model, cache: list[dict], cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    rich_samples = []
    theta_samples = []
    for item in cache:
        probabilities = model.predict_proba(item["features"])
        rich = rich_observation(item["features"], probabilities)
        predicted = np.argmax(probabilities, axis=1)
        row = {
            "environment": item["environment"],
            "theta_noise": item["offset"][0],
            "theta_phase": item["offset"][1],
            "theta_nonlinearity": item["offset"][2],
            "theta_absolute_noise": item["theta"][0],
            "theta_absolute_phase": item["theta"][1],
            "theta_absolute_nonlinearity": item["theta"][2],
            "risk": float(np.mean(predicted != item["labels"])),
        }
        for index, value in enumerate(rich.mean(axis=0)):
            row[f"rich_{index}"] = float(value)
        rows.append(row)
        rich_samples.append(rich)
        theta_samples.append(np.repeat(item["offset"][None, :], len(rich), axis=0))
    environment = pd.DataFrame(rows)
    store = {"rich": np.vstack(rich_samples), "theta": np.vstack(theta_samples)}
    fitted = fit_screening_bridge(environment, store, cfg)
    return environment, store, fitted


def generate_shared_deployment(cfg: dict) -> tuple[list[dict], str]:
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    count = int(cfg["deployment"]["samples_per_window"])
    cache = []
    hash_items = []
    for path in path_specs(cfg):
        scalars = trajectory_scalars(path["start_scalar"], path["end_scalar"], cfg["deployment"])
        direction = path["direction"] / max(np.linalg.norm(path["direction"]), 1e-12)
        for time_index, scalar in enumerate(scalars):
            offset = float(scalar) * direction
            theta = center + offset
            rng = np.random.default_rng(
                stable_seed(cfg["evaluation_seed"], "deployment", path["name"], time_index)
            )
            labels = np.arange(count) % len(cfg["classes"])
            rng.shuffle(labels)
            features = np.empty((count, len(FEATURE_NAMES)), dtype=float)
            for index, label in enumerate(labels):
                iq = generate_iq(cfg["classes"][int(label)], theta, rng, cfg["signal"])
                features[index] = extract_features(iq)
            cache.append(
                {
                    "path": path["name"], "time_index": time_index,
                    "scalar": float(scalar), "offset": offset, "theta": theta,
                    "features": features, "labels": labels,
                }
            )
            hash_items.extend(
                [(f"features_{path['name']}_{time_index}", features),
                 (f"labels_{path['name']}_{time_index}", labels)]
            )
    return cache, hash_arrays(hash_items)


def geometry_record(fitted: dict, regime: str, replicate_seed: int, cfg: dict) -> dict:
    groups = tuple(cfg["round10"]["fixed_groups"])
    indices = group_indices(groups)
    info = subset_information(fitted, indices, cfg["target"]["rank_relative_tolerance"])
    b_vector = np.asarray(fitted["b"], dtype=float)
    hessian = np.asarray(fitted["H"], dtype=float)
    variance = float(b_vector @ info["Q_pinv_per_sample"] @ b_vector)
    eigenvalues = np.asarray(info["eigenvalues_Q"], dtype=float)
    row = {
        "replicate_seed": int(replicate_seed), "regime": regime,
        "tau": float(fitted["risk_intercept"]),
        "b_noise": b_vector[0], "b_phase": b_vector[1],
        "b_nonlinearity": b_vector[2], "b_norm": float(np.linalg.norm(b_vector)),
        "H_frobenius": float(np.linalg.norm(hessian, ord="fro")),
        "H_spectral": float(np.linalg.norm(hessian, ord=2)),
        "risk_surface_r2": float(fitted["risk_surface_r2"]),
        "risk_surface_cv_r2": float(fitted["risk_surface_cv_r2"]),
        "calibration_risk_range": float(np.ptp(fitted["risk_values"])),
        "Q_trace": float(info["trace_Q"]),
        "Q_effective_rank": int(info["effective_rank"]),
        "Q_risk_null_ratio": float(info["risk_null_ratio"]),
        "Q_risk_coordinate_variance_per_sample": variance,
        "Q_risk_directed_information": float(1.0 / variance) if variance > 0 else 0.0,
        "Q_min_retained_eigenvalue": float(eigenvalues[eigenvalues > 0].min())
        if np.any(eigenvalues > 0) else 0.0,
    }
    row["risk_geometry_active"] = bool(
        row["calibration_risk_range"] >= float(cfg["round10"].get("minimum_active_risk_range", 0.04))
        and row["b_norm"] >= float(cfg["round10"].get("minimum_active_b_norm", 0.05))
    )
    q_matrix = np.asarray(info["Q_per_sample"], dtype=float)
    for i in range(3):
        row[f"Q_eigenvalue_{i}"] = float(eigenvalues[i])
        for j in range(i, 3):
            row[f"Q_{i}{j}"] = float(q_matrix[i, j])
    for i in range(3):
        for j in range(i, 3):
            row[f"H_{i}{j}"] = float(hessian[i, j])
    for path in path_specs(cfg):
        direction = path["direction"] / max(np.linalg.norm(path["direction"]), 1e-12)
        row[f"b_direction_{path['name']}"] = float(direction @ b_vector)
        row[f"H_direction_{path['name']}"] = float(direction @ hessian @ direction)
    return row


def evaluate_deployment(model, cache: list[dict], fitted: dict, regime: str,
                        replicate_seed: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_boundary = float(cfg["deployment"]["shared_risk_boundary"])
    relative_boundary = float(fitted["risk_intercept"] + cfg["target"]["risk_margin_gamma"])
    rows = []
    for item in cache:
        risk = float(np.mean(model.predict(item["features"]) != item["labels"]))
        rows.append(
            {
                "replicate_seed": replicate_seed, "regime": regime,
                "path": item["path"], "time_index": item["time_index"],
                "scalar": item["scalar"],
                "theta_noise": item["theta"][0], "theta_phase": item["theta"][1],
                "theta_nonlinearity": item["theta"][2], "risk": risk,
                "shared_boundary": shared_boundary, "relative_boundary": relative_boundary,
                "above_shared_boundary": risk >= shared_boundary,
                "above_relative_boundary": risk >= relative_boundary,
            }
        )
    ledger = pd.DataFrame(rows)
    summaries = []
    confirmation = int(cfg["deployment"]["crossing_confirmation_windows"])
    for path, frame in ledger.groupby("path", sort=False):
        frame = frame.sort_values("time_index")
        risk = frame["risk"].to_numpy()
        shared_crossing = crossing_time(risk, shared_boundary, confirmation)
        relative_crossing = crossing_time(risk, relative_boundary, confirmation)
        terminal = len(risk)
        summaries.append(
            {
                "replicate_seed": replicate_seed, "regime": regime, "path": path,
                "start_risk": float(risk[0]), "end_risk": float(risk[-1]),
                "max_risk": float(risk.max()), "min_risk": float(risk.min()),
                "risk_auc": float(np.trapezoid(risk) / max(len(risk) - 1, 1)),
                "shared_excess_auc": float(np.mean(np.maximum(risk - shared_boundary, 0.0))),
                "shared_cliff_crossed": shared_crossing is not None,
                "shared_cliff_time": shared_crossing,
                "shared_cliff_time_censored": terminal if shared_crossing is None else shared_crossing,
                "relative_cliff_crossed": relative_crossing is not None,
                "relative_cliff_time": relative_crossing,
                "relative_cliff_time_censored": terminal if relative_crossing is None else relative_crossing,
                "shared_risk_boundary": shared_boundary,
                "model_relative_boundary": relative_boundary,
            }
        )
    return ledger, pd.DataFrame(summaries)


def bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def paired_effects(trajectories: pd.DataFrame, geometry: pd.DataFrame,
                   cfg: dict) -> pd.DataFrame:
    rows = []
    baseline = "baseline"
    draws = int(cfg["round10"]["bootstrap_draws"])
    outcomes = ["start_risk", "end_risk", "max_risk", "risk_auc", "shared_excess_auc",
                "shared_cliff_time_censored", "shared_cliff_crossed"]
    for comparator in cfg["round10"]["regime_order"]:
        if comparator == baseline:
            continue
        for outcome in outcomes:
            pivot = trajectories.pivot_table(
                index=["replicate_seed", "path"], columns="regime", values=outcome
            ).dropna()
            pair_difference = (
                pivot[comparator].astype(float) - pivot[baseline].astype(float)
            )
            difference = pair_difference.groupby(level="replicate_seed").mean().to_numpy()
            interval = bootstrap_interval(
                difference, stable_seed(cfg["evaluation_seed"], comparator, outcome), draws
            )
            rows.append(
                {
                    "level": "trajectory", "contrast": f"{comparator}-baseline",
                    "outcome": outcome, "n_pairs": len(pair_difference),
                    "n_clusters": len(difference),
                    "mean_difference": float(difference.mean()),
                    "sd_difference": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                    "fraction_negative": float(np.mean(pair_difference.to_numpy() < 0)),
                    "fraction_positive": float(np.mean(pair_difference.to_numpy() > 0)),
                    "bootstrap_95_low": interval[0],
                    "bootstrap_95_high": interval[1],
                }
            )
    for outcome in outcomes:
        pivot = trajectories.pivot_table(
            index=["replicate_seed", "path"], columns="regime", values=outcome
        ).dropna()
        pair_difference = (
            pivot["cliff_aware"].astype(float) - pivot["random_broad"].astype(float)
        )
        difference = pair_difference.groupby(level="replicate_seed").mean().to_numpy()
        interval = bootstrap_interval(
            difference, stable_seed(cfg["evaluation_seed"], "specificity", outcome), draws
        )
        rows.append(
            {
                "level": "trajectory", "contrast": "cliff_aware-random_broad",
                "outcome": outcome, "n_pairs": len(pair_difference),
                "n_clusters": len(difference),
                "mean_difference": float(difference.mean()),
                "sd_difference": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                "fraction_negative": float(np.mean(pair_difference.to_numpy() < 0)),
                "fraction_positive": float(np.mean(pair_difference.to_numpy() > 0)),
                "bootstrap_95_low": interval[0], "bootstrap_95_high": interval[1],
            }
        )
    geometry_outcomes = [
        "tau", "b_norm", "H_frobenius", "risk_surface_cv_r2", "Q_trace",
        "Q_risk_null_ratio", "Q_risk_coordinate_variance_per_sample",
        "Q_risk_directed_information",
    ]
    for comparator in cfg["round10"]["regime_order"]:
        if comparator == baseline:
            continue
        for outcome in geometry_outcomes:
            pivot = geometry.pivot(index="replicate_seed", columns="regime", values=outcome).dropna()
            difference = (pivot[comparator] - pivot[baseline]).to_numpy(dtype=float)
            interval = bootstrap_interval(
                difference, stable_seed(cfg["evaluation_seed"], "geometry", comparator, outcome), draws
            )
            rows.append(
                {
                    "level": "model_geometry", "contrast": f"{comparator}-baseline",
                    "outcome": outcome, "n_pairs": len(difference),
                    "n_clusters": len(difference),
                    "mean_difference": float(difference.mean()),
                    "sd_difference": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                    "fraction_negative": float(np.mean(difference < 0)),
                    "fraction_positive": float(np.mean(difference > 0)),
                    "bootstrap_95_low": interval[0], "bootstrap_95_high": interval[1],
                }
            )
    for outcome in geometry_outcomes:
        pivot = geometry.pivot(index="replicate_seed", columns="regime", values=outcome).dropna()
        difference = (pivot["cliff_aware"] - pivot["random_broad"]).to_numpy(dtype=float)
        interval = bootstrap_interval(
            difference, stable_seed(cfg["evaluation_seed"], "geometry_specificity", outcome), draws
        )
        rows.append(
            {
                "level": "model_geometry", "contrast": "cliff_aware-random_broad",
                "outcome": outcome, "n_pairs": len(difference),
                "n_clusters": len(difference),
                "mean_difference": float(difference.mean()),
                "sd_difference": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                "fraction_negative": float(np.mean(difference < 0)),
                "fraction_positive": float(np.mean(difference > 0)),
                "bootstrap_95_low": interval[0], "bootstrap_95_high": interval[1],
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, keys: list[str], outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for group, values in frame.groupby(keys, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(keys, group))
        row["n"] = len(values)
        for outcome in outcomes:
            numeric = values[outcome].astype(float)
            row[f"{outcome}_mean"] = float(numeric.mean())
            row[f"{outcome}_sd"] = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_pretarget(training: pd.DataFrame, geometry: pd.DataFrame,
                    calibration_sha: str, cfg: dict) -> dict:
    active = geometry[geometry["risk_geometry_active"].astype(bool)]
    baseline = geometry[geometry["regime"] == "baseline"]
    distances = training.groupby("regime")["mean_nearest_path_distance"].mean()
    expected = len(cfg["round10"]["replicate_seeds"]) * len(cfg["round10"]["regime_order"])
    metrics = {
        "expected_models": expected,
        "fitted_models": int(len(geometry)),
        "shared_calibration_sha256": calibration_sha,
        "active_risk_geometries": int(len(active)),
        "vanished_or_weak_risk_geometries": int(len(geometry) - len(active)),
        "minimum_active_risk_surface_cv_r2": float(active["risk_surface_cv_r2"].min())
        if len(active) else None,
        "maximum_active_Q_risk_null_ratio": float(active["Q_risk_null_ratio"].max())
        if len(active) else None,
        "minimum_baseline_risk_range": float(baseline["calibration_risk_range"].min()),
        "cliff_aware_mean_path_distance": float(distances["cliff_aware"]),
        "random_broad_mean_path_distance": float(distances["random_broad"]),
    }
    gates = {
        "all_models_fitted": len(geometry) == expected,
        "equal_training_counts": training["n_training_samples"].nunique() == 1,
        "paired_model_random_states": all(
            group["model_random_state"].nunique() == 1
            for _, group in training.groupby("replicate_seed")
        ),
        "shared_calibration_stream": len(calibration_sha) == 64,
        "baseline_has_nontrivial_risk_geometry": metrics["minimum_baseline_risk_range"] >= 0.20,
        "active_geometry_estimable": len(active) > 0 and bool(
            metrics["minimum_active_risk_surface_cv_r2"]
            >= float(cfg["round10"]["minimum_risk_surface_cv_r2"])
        ),
        "active_geometry_observable": len(active) > 0 and bool(
            metrics["maximum_active_Q_risk_null_ratio"]
            <= float(cfg["round10"]["maximum_Q_risk_null_ratio"])
        ),
        "cliff_aware_support_is_targeted": distances["cliff_aware"] < distances["random_broad"],
    }
    return {
        "metrics": metrics, "gates": gates, "all_passed": bool(all(gates.values())),
        "target_generated": False,
        "release_rule": "all pretarget integrity, active-geometry, and support-targeting gates",
        "weak_geometry_policy": (
            "models below the frozen risk-range and b-norm activity thresholds are recorded as "
            "vanished/weak risk geometry, not as observation-channel failures"
        ),
    }


def build_checks(training: pd.DataFrame, geometry: pd.DataFrame,
                 trajectories: pd.DataFrame, effects: pd.DataFrame,
                 calibration_sha: str, deployment_sha: str, cfg: dict) -> dict:
    order = cfg["round10"]["regime_order"]
    seeds = cfg["round10"]["replicate_seeds"]
    expected_models = len(order) * len(seeds)
    distance = training.groupby("regime")["mean_nearest_path_distance"].mean()
    effect_index = effects.set_index(["contrast", "outcome"])
    target = effect_index.loc[("cliff_aware-baseline", "end_risk")]
    random_target = effect_index.loc[("random_broad-baseline", "end_risk")]
    auc = effect_index.loc[("cliff_aware-baseline", "risk_auc")]
    minimum = float(cfg["round10"]["pilot_upgrade_rule"]["minimum_mean_end_risk_reduction"])
    fraction = float(cfg["round10"]["pilot_upgrade_rule"]["minimum_pair_fraction_improved"])
    baseline_by_seed = (
        trajectories[trajectories["regime"] == "baseline"]
        .groupby("replicate_seed")["end_risk"].mean()
    )
    baseline_seed_noise = float(baseline_by_seed.std(ddof=1)) if len(seeds) > 1 else 0.0
    active_geometry = geometry[geometry["risk_geometry_active"].astype(bool)]
    inactive_geometry = geometry[~geometry["risk_geometry_active"].astype(bool)]
    metrics = {
        "expected_models": expected_models,
        "fitted_models": int(len(geometry)),
        "shared_calibration_sha256": calibration_sha,
        "shared_deployment_sha256": deployment_sha,
        "active_risk_geometries": int(len(active_geometry)),
        "vanished_risk_geometries": int(len(inactive_geometry)),
        "minimum_active_risk_surface_cv_r2": float(active_geometry["risk_surface_cv_r2"].min())
        if len(active_geometry) else None,
        "maximum_active_Q_risk_null_ratio": float(active_geometry["Q_risk_null_ratio"].max())
        if len(active_geometry) else None,
        "cliff_aware_minus_baseline_end_risk": float(target["mean_difference"]),
        "cliff_aware_end_risk_fraction_improved": float(target["fraction_negative"]),
        "cliff_aware_minus_baseline_risk_auc": float(auc["mean_difference"]),
        "random_broad_minus_baseline_end_risk": float(random_target["mean_difference"]),
        "baseline_end_risk_seed_noise": baseline_seed_noise,
        "cliff_aware_mean_path_distance": float(distance["cliff_aware"]),
        "random_broad_mean_path_distance": float(distance["random_broad"]),
    }
    gates = {
        "all_models_fitted": len(geometry) == expected_models,
        "equal_training_counts": training["n_training_samples"].nunique() == 1,
        "paired_model_random_states": all(
            group["model_random_state"].nunique() == 1
            for _, group in training.groupby("replicate_seed")
        ),
        "shared_calibration_stream": len(calibration_sha) == 64,
        "shared_deployment_stream": len(deployment_sha) == 64,
        "quadratic_geometry_estimable": len(active_geometry) > 0 and bool(
            metrics["minimum_active_risk_surface_cv_r2"]
            >= float(cfg["round10"]["minimum_risk_surface_cv_r2"])
        ),
        "fixed_channel_observes_risk": len(active_geometry) > 0 and bool(
            metrics["maximum_active_Q_risk_null_ratio"]
            <= float(cfg["round10"]["maximum_Q_risk_null_ratio"])
        ),
        "cliff_tube_is_targeted": distance["cliff_aware"] < distance["random_broad"],
        "cliff_aware_end_risk_effect": -metrics["cliff_aware_minus_baseline_end_risk"] >= minimum,
        "cliff_aware_pair_consistency": metrics["cliff_aware_end_risk_fraction_improved"] >= fraction,
        "cliff_aware_auc_direction": metrics["cliff_aware_minus_baseline_risk_auc"] < 0,
        "cliff_specificity_over_random": metrics["cliff_aware_minus_baseline_end_risk"]
        < metrics["random_broad_minus_baseline_end_risk"],
        "effect_exceeds_seed_noise": -metrics["cliff_aware_minus_baseline_end_risk"]
        > baseline_seed_noise,
    }
    integrity_names = [
        "all_models_fitted", "equal_training_counts", "paired_model_random_states",
        "shared_calibration_stream", "shared_deployment_stream",
    ]
    upgrade_names = [
        "cliff_aware_end_risk_effect", "cliff_aware_pair_consistency",
        "cliff_aware_auc_direction", "cliff_specificity_over_random",
        "effect_exceeds_seed_noise",
    ]
    result = {
        "metrics": metrics, "gates": gates,
        "integrity_all_passed": bool(all(gates[name] for name in integrity_names)),
        "geometry_all_passed": bool(
            gates["quadratic_geometry_estimable"] and gates["fixed_channel_observes_risk"]
        ),
        "pilot_upgrade_passed": bool(all(gates[name] for name in upgrade_names)),
        "all_passed": bool(all(gates.values())),
        "interpretation_policy": {
            "causal_unit": "training distribution -> retrained model under paired evaluation streams",
            "primary_boundary": "shared Round 9 risk boundary, fixed across models",
            "relative_boundary": "model-specific tau+gamma is supplementary because it moves with training",
            "pilot": "exploratory scale-up decision only; formal claims require fresh training seeds",
        },
    }
    if "formal_gates" in cfg["round10"]:
        spec = cfg["round10"]["formal_gates"]
        specificity_end = effect_index.loc[("cliff_aware-random_broad", "end_risk")]
        specificity_auc = effect_index.loc[("cliff_aware-random_broad", "risk_auc")]
        cliff_b = effect_index.loc[("cliff_aware-baseline", "b_norm")]
        cliff_h = effect_index.loc[("cliff_aware-baseline", "H_frobenius")]
        cliff_q = effect_index.loc[("cliff_aware-baseline", "Q_trace")]
        baseline_mean_b = float(geometry[geometry["regime"] == "baseline"]["b_norm"].mean())
        baseline_mean_h = float(
            geometry[geometry["regime"] == "baseline"]["H_frobenius"].mean()
        )
        cliff_mean_b = float(geometry[geometry["regime"] == "cliff_aware"]["b_norm"].mean())
        cliff_mean_h = float(
            geometry[geometry["regime"] == "cliff_aware"]["H_frobenius"].mean()
        )
        baseline_cross = float(
            trajectories[trajectories["regime"] == "baseline"]["shared_cliff_crossed"].mean()
        )
        cliff_cross = float(
            trajectories[trajectories["regime"] == "cliff_aware"]["shared_cliff_crossed"].mean()
        )
        support_start = float(
            trajectories[trajectories["regime"] == "support_depleted"]["start_risk"].mean()
        )
        baseline_start = float(
            trajectories[trajectories["regime"] == "baseline"]["start_risk"].mean()
        )
        formal_metrics = {
            "cliff_aware_vs_baseline_end_risk": metrics["cliff_aware_minus_baseline_end_risk"],
            "cliff_aware_vs_baseline_end_risk_ci_high": float(target["bootstrap_95_high"]),
            "cliff_aware_vs_random_end_risk": float(specificity_end["mean_difference"]),
            "cliff_aware_vs_random_end_risk_ci_high": float(specificity_end["bootstrap_95_high"]),
            "cliff_aware_vs_random_end_fraction_improved": float(
                specificity_end["fraction_negative"]
            ),
            "cliff_aware_vs_random_auc": float(specificity_auc["mean_difference"]),
            "cliff_aware_vs_random_auc_ci_high": float(specificity_auc["bootstrap_95_high"]),
            "baseline_crossing_fraction": baseline_cross,
            "cliff_aware_crossing_fraction": cliff_cross,
            "support_depletion_start_risk_increase": support_start - baseline_start,
            "cliff_aware_b_norm_ratio": cliff_mean_b / max(baseline_mean_b, 1e-12),
            "cliff_aware_H_norm_ratio": cliff_mean_h / max(baseline_mean_h, 1e-12),
            "cliff_aware_b_norm_difference": float(cliff_b["mean_difference"]),
            "cliff_aware_H_norm_difference": float(cliff_h["mean_difference"]),
            "cliff_aware_Q_trace_difference": float(cliff_q["mean_difference"]),
            "cliff_aware_Q_trace_ci": [
                float(cliff_q["bootstrap_95_low"]), float(cliff_q["bootstrap_95_high"])
            ],
        }
        formal_gates = {
            "integrity": result["integrity_all_passed"],
            "baseline_reproduces_cliff": baseline_cross >= spec["minimum_baseline_crossing_fraction"],
            "support_depletion_worsens_start": formal_metrics["support_depletion_start_risk_increase"]
            >= spec["minimum_support_depletion_start_risk_increase"],
            "cliff_aware_reduces_end_risk": -formal_metrics["cliff_aware_vs_baseline_end_risk"]
            >= spec["minimum_cliff_aware_end_risk_reduction"],
            "cliff_aware_end_risk_interval": formal_metrics[
                "cliff_aware_vs_baseline_end_risk_ci_high"
            ] < 0,
            "cliff_aware_prevents_crossing": cliff_cross
            <= spec["maximum_cliff_aware_crossing_fraction"],
            "targeting_beats_random_end": -formal_metrics["cliff_aware_vs_random_end_risk"]
            >= spec["minimum_targeting_end_risk_advantage"],
            "targeting_end_interval": formal_metrics["cliff_aware_vs_random_end_risk_ci_high"] < 0,
            "targeting_pair_consistency": formal_metrics[
                "cliff_aware_vs_random_end_fraction_improved"
            ] >= spec["minimum_targeting_pair_fraction"],
            "targeting_beats_random_auc": -formal_metrics["cliff_aware_vs_random_auc"]
            >= spec["minimum_targeting_auc_advantage"],
            "targeting_auc_interval": formal_metrics["cliff_aware_vs_random_auc_ci_high"] < 0,
            "linear_geometry_contracts": formal_metrics["cliff_aware_b_norm_ratio"]
            <= spec["maximum_cliff_aware_b_norm_ratio"],
            "quadratic_geometry_contracts": formal_metrics["cliff_aware_H_norm_ratio"]
            <= spec["maximum_cliff_aware_H_norm_ratio"],
            "Q_changes": not (
                formal_metrics["cliff_aware_Q_trace_ci"][0] <= 0
                <= formal_metrics["cliff_aware_Q_trace_ci"][1]
            ),
            "active_geometry_fit": gates["quadratic_geometry_estimable"],
            "active_geometry_observable": gates["fixed_channel_observes_risk"],
        }
        result["formal_metrics"] = formal_metrics
        result["formal_gates"] = formal_gates
        result["formal_all_passed"] = bool(all(formal_gates.values()))
    return result


def plot_summary(training_ledger: pd.DataFrame, risk_ledger: pd.DataFrame,
                 geometry: pd.DataFrame, trajectories: pd.DataFrame,
                 output: Path, cfg: dict) -> None:
    order = cfg["round10"]["regime_order"]
    colors = dict(zip(order, ["#777777", "#2b6cb0", "#d97706", "#0f766e"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    sampled = training_ledger.groupby(["regime", "replicate_seed"], group_keys=False).head(350)
    for regime in order:
        frame = sampled[sampled["regime"] == regime]
        axes[0, 0].scatter(frame["theta_noise"], frame["theta_phase"], s=5, alpha=0.25,
                           color=colors[regime], label=regime)
    axes[0, 0].set_title("Intervened training support")
    axes[0, 0].set_xlabel("noise coordinate")
    axes[0, 0].set_ylabel("phase coordinate")
    axes[0, 0].legend(fontsize=7)

    averaged = risk_ledger.groupby(["regime", "path", "time_index"], as_index=False)["risk"].mean()
    styles = {"noise": "-", "mixed_gradient": "--"}
    for regime in order:
        for path in averaged["path"].unique():
            frame = averaged[(averaged["regime"] == regime) & (averaged["path"] == path)]
            axes[0, 1].plot(frame["time_index"], frame["risk"], styles.get(path, "-"),
                            color=colors[regime], label=f"{regime}: {path}")
    axes[0, 1].axhline(float(cfg["deployment"]["shared_risk_boundary"]), color="black",
                       linewidth=1, linestyle=":", label="shared boundary")
    axes[0, 1].set_title("Exact same deployment streams")
    axes[0, 1].set_xlabel("window")
    axes[0, 1].set_ylabel("revealed error rate")
    axes[0, 1].legend(fontsize=6, ncol=2)

    positions = np.arange(len(order))
    end = trajectories.groupby("regime")["end_risk"].agg(["mean", "std"]).reindex(order)
    axes[1, 0].bar(positions, end["mean"], yerr=end["std"].fillna(0),
                   color=[colors[item] for item in order], alpha=0.85)
    axes[1, 0].axhline(float(cfg["deployment"]["shared_risk_boundary"]), color="black",
                       linewidth=1, linestyle=":")
    axes[1, 0].set_xticks(positions, order, rotation=18, ha="right")
    axes[1, 0].set_ylabel("end risk")
    axes[1, 0].set_title("Terminal deployment risk")

    width = 0.25
    metrics = [("b_norm", "||b||"), ("H_frobenius", "||H||F"),
               ("Q_trace", "tr(Q)")]
    normalized = geometry.copy()
    for metric, _ in metrics:
        baseline = normalized[normalized["regime"] == "baseline"][metric].mean()
        normalized[metric] = normalized[metric] / max(abs(baseline), 1e-12)
    for index, (metric, label) in enumerate(metrics):
        means = normalized.groupby("regime")[metric].mean().reindex(order)
        axes[1, 1].bar(positions + (index - 1) * width, means, width=width, label=label,
                       alpha=0.8)
    axes[1, 1].axhline(1.0, color="black", linewidth=1, linestyle=":")
    axes[1, 1].set_xticks(positions, order, rotation=18, ha="right")
    axes[1, 1].set_ylabel("ratio to baseline")
    axes[1, 1].set_title("Training reshapes risk and observability")
    axes[1, 1].legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round10_training_intervention_probe.json")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    cfg.setdefault("master_seed", int(cfg["evaluation_seed"]))
    output = ROOT / "results" / cfg["output_tag"]
    figure_dir = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", cfg)

    calibration_cache, calibration_sha = generate_shared_calibration(cfg)
    if args.calibration_only:
        deployment_cache, deployment_sha = [], "not_generated"
    else:
        deployment_cache, deployment_sha = generate_shared_deployment(cfg)
    all_training = []
    all_training_ledger = []
    all_geometry = []
    all_environment = []
    all_risk = []
    all_trajectories = []
    for replicate_seed in cfg["round10"]["replicate_seeds"]:
        for regime in cfg["round10"]["regime_order"]:
            print(f"training seed={replicate_seed} regime={regime}", flush=True)
            model, training, training_ledger = train_model(cfg, regime, int(replicate_seed))
            environment, _, fitted = evaluate_calibration(model, calibration_cache, cfg)
            geometry = geometry_record(fitted, regime, int(replicate_seed), cfg)
            environment.insert(0, "regime", regime)
            environment.insert(0, "replicate_seed", int(replicate_seed))
            all_training.append(training)
            all_training_ledger.append(training_ledger)
            all_geometry.append(geometry)
            all_environment.append(environment)
            if not args.calibration_only:
                risk, trajectories = evaluate_deployment(
                    model, deployment_cache, fitted, regime, int(replicate_seed), cfg
                )
                all_risk.append(risk)
                all_trajectories.append(trajectories)

    training = pd.DataFrame(all_training)
    training_ledger = pd.concat(all_training_ledger, ignore_index=True)
    geometry = pd.DataFrame(all_geometry)
    environment = pd.concat(all_environment, ignore_index=True)
    training.to_csv(output / "training_distribution_summary.csv", index=False)
    training_ledger.to_csv(output / "training_theta_ledger.csv", index=False)
    geometry.to_csv(output / "model_geometry.csv", index=False)
    environment.to_csv(output / "calibration_environment_panel.csv", index=False)
    write_json(
        output / "shared_calibration.json",
        {
            "calibration_sha256": calibration_sha,
            "calibration_examples": int(sum(len(item["labels"]) for item in calibration_cache)),
            "reuse_policy": "the cache is generated once and passed unchanged to every model",
        },
    )
    write_json(output / "environment.json", environment_record())
    pretarget = build_pretarget(training, geometry, calibration_sha, cfg)
    write_json(output / "pretarget_round10_gate.json", pretarget)
    release_hash = write_release_manifest(
        output,
        [
            "frozen_config.json", "training_distribution_summary.csv",
            "training_theta_ledger.csv", "model_geometry.csv",
            "calibration_environment_panel.csv", "shared_calibration.json",
            "pretarget_round10_gate.json", "environment.json",
        ],
    )
    if args.calibration_only:
        sha256_manifest(output)
        print(json.dumps(serialize({"target_generated": False, "pretarget": pretarget,
                                    "release_sha256": release_hash}), indent=2))
        return

    write_json(
        output / "shared_deployment.json",
        {
            "deployment_sha256": deployment_sha,
            "deployment_examples": int(sum(len(item["labels"]) for item in deployment_cache)),
            "reuse_policy": "the cache is generated once and passed unchanged to every model",
        },
    )

    risk_ledger = pd.concat(all_risk, ignore_index=True)
    trajectories = pd.concat(all_trajectories, ignore_index=True)
    effects = paired_effects(trajectories, geometry, cfg)
    regime_summary = summarize(
        trajectories, ["regime", "path"],
        ["start_risk", "end_risk", "max_risk", "risk_auc", "shared_excess_auc",
         "shared_cliff_time_censored", "shared_cliff_crossed"],
    )
    geometry_summary = summarize(
        geometry, ["regime"],
        ["tau", "b_norm", "H_frobenius", "risk_surface_cv_r2", "Q_trace",
         "Q_risk_null_ratio", "Q_risk_coordinate_variance_per_sample",
         "Q_risk_directed_information"],
    )
    checks = build_checks(
        training, geometry, trajectories, effects, calibration_sha, deployment_sha, cfg
    )
    checks["pretarget_all_passed"] = bool(pretarget["all_passed"])
    risk_ledger.to_csv(output / "shared_deployment_risk_ledger.csv", index=False)
    trajectories.to_csv(output / "trajectory_summary.csv", index=False)
    regime_summary.to_csv(output / "regime_path_summary.csv", index=False)
    geometry_summary.to_csv(output / "geometry_summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    write_json(output / "checks.json", checks)
    plot_summary(
        training_ledger, risk_ledger, geometry, trajectories,
        figure_dir / f"{cfg['output_tag']}.png", cfg,
    )
    sha256_manifest(output)
    print(json.dumps(serialize(checks), indent=2))


if __name__ == "__main__":
    main()
