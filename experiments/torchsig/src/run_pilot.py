#!/usr/bin/env python3
"""Stepwise TorchSig operational pilot for the Generalization Cliff project.

The target-state labels and deployment errors are generated in separate RNG
streams and are not used to estimate A, Sigma, b, gamma, or the matching
auditor.  They are revealed only for final scoring inside this reproducible
pseudo-prospective replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew, spearmanr
import sklearn
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torchsig
from torchsig.signals.builders.constellation import constellation_modulator
from torchsig.transforms import functional as tsf


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def mechanism_values(theta: np.ndarray) -> dict[str, float]:
    return {
        "snr_db": float(18.0 - 8.0 * theta[0]),
        "phase_noise_degrees": float(6.0 + 6.0 * theta[1]),
        "amplifier_psat_backoff": float(4.0 * np.exp(-0.8 * theta[2])),
    }


def generate_iq(
    class_name: str, theta: np.ndarray, rng: np.random.Generator, signal_cfg: dict
) -> np.ndarray:
    x = constellation_modulator(
        constellation_name=class_name,
        pulse_shape_name=signal_cfg["pulse_shape"],
        bandwidth=float(signal_cfg["bandwidth"]),
        sample_rate=float(signal_cfg["sample_rate"]),
        num_samples=int(signal_cfg["num_iq_samples"]),
        alpha_rolloff=float(signal_cfg["alpha_rolloff"]),
        rng=rng,
    )
    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    mech = mechanism_values(theta)
    x = tsf.nonlinear_amplifier(
        x,
        gain=1.0,
        psat_backoff=mech["amplifier_psat_backoff"],
        phi_max=0.20,
        phi_slope=0.02,
        auto_scale=True,
    )
    x = tsf.carrier_phase_noise(
        x, phase_noise_degrees=mech["phase_noise_degrees"], rng=rng
    )
    x = tsf.awgn(x, noise_power_db=-mech["snr_db"], rng=rng)
    return np.asarray(x, dtype=np.complex64)


def feature_names() -> list[str]:
    names = ["log_power"]
    names += [f"amp_q{q}" for q in (10, 25, 50, 75, 90)]
    names += ["amp_std", "amp_skew", "amp_kurtosis"]
    names += [f"moment_abs_{k}" for k in (2, 4, 6, 8)]
    for lag in (1, 2, 4, 8):
        names += [f"acf_abs_{lag}", f"acf_real_{lag}", f"acf_imag_{lag}"]
    names += [f"phase_increment_bin_{i}" for i in range(12)]
    names += [f"amplitude_bin_{i}" for i in range(8)]
    names += ["spectral_entropy", "spectral_flatness", "spectral_peak_ratio"]
    return names


FEATURE_NAMES = feature_names()
AMP_FEATURE_INDICES = list(range(9))


def extract_features(x: np.ndarray) -> np.ndarray:
    power = float(np.mean(np.abs(x) ** 2))
    xn = x / np.sqrt(power + 1e-12)
    amp = np.abs(xn)
    values: list[float] = [float(np.log(power + 1e-12))]
    values.extend(np.quantile(amp, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist())
    values.extend(
        [
            float(np.std(amp)),
            float(skew(amp, bias=False)),
            float(kurtosis(amp, fisher=True, bias=False)),
        ]
    )
    for order in (2, 4, 6, 8):
        values.append(float(np.abs(np.mean(xn**order))))
    for lag in (1, 2, 4, 8):
        acf = np.mean(xn[lag:] * np.conj(xn[:-lag]))
        values.extend([float(np.abs(acf)), float(np.real(acf)), float(np.imag(acf))])
    phase_increment = np.angle(xn[1:] * np.conj(xn[:-1]))
    phase_hist, _ = np.histogram(
        phase_increment, bins=np.linspace(-np.pi, np.pi, 13), density=False
    )
    values.extend((phase_hist / max(phase_hist.sum(), 1)).astype(float).tolist())
    amp_hist, _ = np.histogram(
        amp, bins=np.array([0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf])
    )
    values.extend((amp_hist / max(amp_hist.sum(), 1)).astype(float).tolist())
    spectrum = np.abs(np.fft.fft(xn)) ** 2
    spectrum = spectrum / (spectrum.sum() + 1e-12)
    spectral_entropy = -np.sum(spectrum * np.log(spectrum + 1e-12)) / np.log(len(spectrum))
    spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12)
    spectral_peak_ratio = np.max(spectrum) / (np.mean(spectrum) + 1e-12)
    values.extend(map(float, [spectral_entropy, spectral_flatness, spectral_peak_ratio]))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(f"feature shape {result.shape} != {(len(FEATURE_NAMES),)}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite feature encountered")
    return result


def rich_observation(features: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return np.column_stack([probabilities, entropy, margin, features])


def generate_samples(
    cfg: dict,
    theta: np.ndarray,
    count: int,
    rng: np.random.Generator,
    balanced: bool,
) -> tuple[np.ndarray, np.ndarray]:
    classes = cfg["classes"]
    if balanced:
        labels = np.arange(count) % len(classes)
        rng.shuffle(labels)
    else:
        labels = rng.integers(0, len(classes), size=count)
    features = np.empty((count, len(FEATURE_NAMES)), dtype=np.float64)
    for index, label in enumerate(labels):
        iq = generate_iq(classes[int(label)], theta, rng, cfg["signal"])
        features[index] = extract_features(iq)
    return features, labels.astype(int)


def train_deployment_model(cfg: dict) -> tuple[ExtraTreesClassifier, dict]:
    train_cfg = cfg["training"]
    total = int(train_cfg["samples_per_class"]) * len(cfg["classes"])
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], "training"))
    labels = np.arange(total) % len(cfg["classes"])
    rng.shuffle(labels)
    features = np.empty((total, len(FEATURE_NAMES)), dtype=np.float64)
    theta_values = np.empty((total, 3), dtype=np.float64)
    for i, label in enumerate(labels):
        theta = rng.uniform(train_cfg["theta_low"], train_cfg["theta_high"], size=3)
        theta_values[i] = theta
        iq = generate_iq(cfg["classes"][int(label)], theta, rng, cfg["signal"])
        features[i] = extract_features(iq)
    model = ExtraTreesClassifier(
        n_estimators=int(train_cfg["n_estimators"]),
        min_samples_leaf=int(train_cfg["min_samples_leaf"]),
        max_features="sqrt",
        random_state=stable_seed(cfg["master_seed"], "model") % (2**32 - 1),
        n_jobs=int(train_cfg.get("n_jobs", min(4, os.cpu_count() or 1))),
    )
    model.fit(features, labels)
    predicted = model.predict(features)
    return model, {
        "n_training_samples": int(total),
        "resubstitution_accuracy": float(accuracy_score(labels, predicted)),
        "theta_min": theta_values.min(axis=0).tolist(),
        "theta_max": theta_values.max(axis=0).tolist(),
    }


def calibration_panel(cfg: dict, model: ExtraTreesClassifier) -> tuple[pd.DataFrame, dict]:
    levels = cfg["calibration"]["theta_offset_levels"]
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=np.float64)
    count = int(cfg["calibration"]["samples_per_environment"])
    env_rows = []
    sample_store: dict[str, list[np.ndarray]] = {
        "theta": [],
        "rich": [],
        "restricted": [],
    }
    for env_index, theta_tuple in enumerate(product(levels, repeat=3)):
        theta_offset = np.asarray(theta_tuple, dtype=np.float64)
        theta = center + theta_offset
        rng = np.random.default_rng(
            stable_seed(cfg["master_seed"], "calibration", env_index)
        )
        features, labels = generate_samples(cfg, theta, count, rng, balanced=True)
        probabilities = model.predict_proba(features)
        predicted = np.argmax(probabilities, axis=1)
        rich = rich_observation(features, probabilities)
        restricted = features[:, AMP_FEATURE_INDICES]
        risk = float(np.mean(predicted != labels))
        row = {
            "environment": env_index,
            "theta_noise": theta_offset[0],
            "theta_phase": theta_offset[1],
            "theta_nonlinearity": theta_offset[2],
            "theta_absolute_noise": theta[0],
            "theta_absolute_phase": theta[1],
            "theta_absolute_nonlinearity": theta[2],
            "risk": risk,
        }
        for j, value in enumerate(rich.mean(axis=0)):
            row[f"rich_{j}"] = float(value)
        for j, value in enumerate(restricted.mean(axis=0)):
            row[f"restricted_{j}"] = float(value)
        env_rows.append(row)
        sample_store["theta"].append(np.repeat(theta_offset[None, :], count, axis=0))
        sample_store["rich"].append(rich)
        sample_store["restricted"].append(restricted)
    packed = {key: np.vstack(value) for key, value in sample_store.items()}
    return pd.DataFrame(env_rows), packed


def quadratic_design(theta: np.ndarray) -> np.ndarray:
    """Design for b^T u + 1/2 u^T H u with a symmetric H."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    return np.column_stack(
        [
            theta,
            0.5 * theta[:, 0] ** 2,
            0.5 * theta[:, 1] ** 2,
            0.5 * theta[:, 2] ** 2,
            theta[:, 0] * theta[:, 1],
            theta[:, 0] * theta[:, 2],
            theta[:, 1] * theta[:, 2],
        ]
    )


def unpack_quadratic_model(model: Ridge) -> tuple[float, np.ndarray, np.ndarray]:
    coefficients = np.asarray(model.coef_, dtype=float)
    b_vector = coefficients[:3]
    hessian = np.array(
        [
            [coefficients[3], coefficients[6], coefficients[7]],
            [coefficients[6], coefficients[4], coefficients[8]],
            [coefficients[7], coefficients[8], coefficients[5]],
        ],
        dtype=float,
    )
    return float(model.intercept_), b_vector, hessian


def quadratic_risk(
    u: np.ndarray, tau: float, b_vector: np.ndarray, hessian: np.ndarray
) -> np.ndarray | float:
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        return float(tau + b_vector @ u + 0.5 * u @ hessian @ u)
    return tau + u @ b_vector + 0.5 * np.einsum("...i,ij,...j->...", u, hessian, u)


def fit_risk_surface(theta_env: np.ndarray, risk: np.ndarray, cfg: dict) -> dict:
    linear_model = Ridge(alpha=1e-8).fit(theta_env, risk)
    surface_cfg = cfg.get("risk_surface", {"type": "linear"})
    surface_type = surface_cfg.get("type", "linear")
    if surface_type == "linear":
        tau = float(linear_model.intercept_)
        b_vector = np.asarray(linear_model.coef_, dtype=float)
        hessian = np.zeros((3, 3), dtype=float)
        prediction = linear_model.predict(theta_env)
        cv_r2 = float("nan")
        cv_abs_residual_quantiles = {}
        model = linear_model
    elif surface_type == "quadratic":
        design = quadratic_design(theta_env)
        alpha = float(surface_cfg.get("ridge_alpha", 1e-4))
        model = Ridge(alpha=alpha).fit(design, risk)
        tau, b_vector, hessian = unpack_quadratic_model(model)
        prediction = model.predict(design)
        folds = int(surface_cfg.get("cv_folds", 5))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=stable_seed(cfg["master_seed"], "risk_surface_cv") % (2**32 - 1),
        )
        cv_prediction = cross_val_predict(Ridge(alpha=alpha), design, risk, cv=splitter)
        cv_r2 = float(r2_score(risk, cv_prediction))
        abs_residual = np.abs(risk - cv_prediction)
        cv_abs_residual_quantiles = {
            "q50": float(np.quantile(abs_residual, 0.50)),
            "q80": float(np.quantile(abs_residual, 0.80)),
            "q90": float(np.quantile(abs_residual, 0.90)),
            "q95": float(np.quantile(abs_residual, 0.95)),
        }
    else:
        raise ValueError(f"unsupported risk surface type: {surface_type}")
    return {
        "type": surface_type,
        "model": model,
        "tau": tau,
        "b": b_vector,
        "H": hessian,
        "prediction": prediction,
        "r2": float(r2_score(risk, prediction)),
        "cv_r2": cv_r2,
        "cv_abs_residual_quantiles": cv_abs_residual_quantiles,
        "linear_r2": float(r2_score(risk, linear_model.predict(theta_env))),
        "linear_intercept": float(linear_model.intercept_),
    }


def optimize_supporting_pair(
    tau: float,
    b_vector: np.ndarray,
    hessian: np.ndarray,
    q_matrix: np.ndarray,
    q_pinv: np.ndarray,
    cfg: dict,
) -> dict:
    surface_cfg = cfg["risk_surface"]
    gamma = float(cfg["target"]["risk_margin_gamma"])
    remainder_buffer = float(surface_cfg.get("remainder_buffer", 0.0))
    design_margin = gamma + remainder_buffer
    rho = float(surface_cfg["mechanism_ball_radius"])

    def risk_one(u: np.ndarray) -> float:
        return float(quadratic_risk(u, tau, b_vector, hessian))

    def objective(x: np.ndarray) -> float:
        difference = x[3:] - x[:3]
        return float(difference @ q_matrix @ difference)

    q_r = float(b_vector @ q_pinv @ b_vector)
    if q_r <= 1e-12:
        raise RuntimeError("quadratic pair initialization failed: no risk-directed information")
    linear_direction = design_margin * (q_pinv @ b_vector) / q_r
    direct_direction = design_margin * b_vector / max(float(b_vector @ b_vector), 1e-12)
    starts = [
        np.r_[-linear_direction, linear_direction],
        np.r_[-direct_direction, direct_direction],
        np.r_[-linear_direction - 0.02 * direct_direction, linear_direction - 0.02 * direct_direction],
        np.r_[-linear_direction + 0.02 * direct_direction, linear_direction + 0.02 * direct_direction],
    ]
    constraints = [
        {"type": "eq", "fun": lambda x: risk_one(x[:3]) - (tau - design_margin)},
        {"type": "eq", "fun": lambda x: risk_one(x[3:]) - (tau + design_margin)},
        {"type": "ineq", "fun": lambda x: rho * rho - x[:3] @ x[:3]},
        {"type": "ineq", "fun": lambda x: rho * rho - x[3:] @ x[3:]},
    ]
    solutions = []
    for start in starts:
        solution = minimize(
            objective,
            np.clip(start, -rho, rho),
            method="SLSQP",
            bounds=[(-rho, rho)] * 6,
            constraints=constraints,
            options={"ftol": 1e-13, "maxiter": 3000},
        )
        constraint_error = max(
            abs(risk_one(solution.x[:3]) - (tau - design_margin)),
            abs(risk_one(solution.x[3:]) - (tau + design_margin)),
            max(0.0, np.linalg.norm(solution.x[:3]) - rho),
            max(0.0, np.linalg.norm(solution.x[3:]) - rho),
        )
        if solution.success and constraint_error <= 1e-6:
            solutions.append((solution, float(constraint_error)))
    if not solutions:
        raise RuntimeError("no feasible asymmetric quadratic boundary pair found")
    solution, constraint_error = min(solutions, key=lambda item: objective(item[0].x))
    u_minus = np.asarray(solution.x[:3], dtype=float)
    u_plus = np.asarray(solution.x[3:], dtype=float)
    difference = u_plus - u_minus
    midpoint = 0.5 * (u_plus + u_minus)
    d_q_d = float(difference @ q_matrix @ difference)

    def score(u: np.ndarray) -> float:
        return float(difference @ q_matrix @ (u - midpoint))

    support_starts = [u_minus, u_plus, np.zeros(3), -linear_direction, linear_direction]
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], "support_optimizer"))
    for _ in range(12):
        direction = rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        support_starts.append(direction * rho * rng.uniform(0.2, 1.0))

    safe_solutions = []
    cliff_solutions = []
    for start in support_starts:
        safe = minimize(
            lambda u: -score(u),
            np.clip(start, -rho, rho),
            method="SLSQP",
            bounds=[(-rho, rho)] * 3,
            constraints=[
                {"type": "ineq", "fun": lambda u: (tau - design_margin) - risk_one(u)},
                {"type": "ineq", "fun": lambda u: rho * rho - u @ u},
            ],
            options={"ftol": 1e-13, "maxiter": 2000},
        )
        if safe.success:
            safe_solutions.append(safe)
        cliff = minimize(
            score,
            np.clip(start, -rho, rho),
            method="SLSQP",
            bounds=[(-rho, rho)] * 3,
            constraints=[
                {"type": "ineq", "fun": lambda u: risk_one(u) - (tau + design_margin)},
                {"type": "ineq", "fun": lambda u: rho * rho - u @ u},
            ],
            options={"ftol": 1e-13, "maxiter": 2000},
        )
        if cliff.success:
            cliff_solutions.append(cliff)
    if not safe_solutions or not cliff_solutions:
        raise RuntimeError("supporting-hyperplane diagnostic failed to optimize")
    max_safe_score = max(score(item.x) for item in safe_solutions)
    min_cliff_score = min(score(item.x) for item in cliff_solutions)
    half_pair_score = 0.5 * d_q_d
    safe_support_slack = float(-half_pair_score - max_safe_score)
    cliff_support_slack = float(min_cliff_score - half_pair_score)
    return {
        "u_minus": u_minus,
        "u_plus": u_plus,
        "difference": difference,
        "midpoint": midpoint,
        "pair_dQd_per_sample": d_q_d,
        "certification_gamma": gamma,
        "remainder_buffer": remainder_buffer,
        "design_margin": design_margin,
        "mechanism_ball_radius": rho,
        "optimizer_constraint_error": constraint_error,
        "optimizer_successful_starts": len(solutions),
        "safe_support_slack": safe_support_slack,
        "cliff_support_slack": cliff_support_slack,
        "support_slack_min": min(safe_support_slack, cliff_support_slack),
    }


def quadratic_pair_fold_stability(
    theta_env: np.ndarray,
    risk: np.ndarray,
    pair: dict,
    cfg: dict,
) -> dict:
    surface_cfg = cfg["risk_surface"]
    alpha = float(surface_cfg.get("ridge_alpha", 1e-4))
    folds = int(surface_cfg.get("cv_folds", 5))
    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=stable_seed(cfg["master_seed"], "pair_stability_cv") % (2**32 - 1),
    )
    errors = []
    design = quadratic_design(theta_env)
    for train, _ in splitter.split(theta_env):
        model = Ridge(alpha=alpha).fit(design[train], risk[train])
        tau, b_vector, hessian = unpack_quadratic_model(model)
        risk_minus = quadratic_risk(pair["u_minus"], tau, b_vector, hessian)
        risk_plus = quadratic_risk(pair["u_plus"], tau, b_vector, hessian)
        errors.append(
            [
                float(risk_minus - (tau - pair["design_margin"])),
                float(risk_plus - (tau + pair["design_margin"])),
            ]
        )
    errors = np.asarray(errors)
    return {
        "fold_constraint_errors": errors,
        "fold_constraint_max_absolute_error": float(np.max(np.abs(errors))),
        "fold_constraint_rmse": float(np.sqrt(np.mean(errors**2))),
    }


def fit_operational_bridge(
    env: pd.DataFrame, sample_store: dict, channel: str, cfg: dict
) -> dict:
    theta_env = env[["theta_noise", "theta_phase", "theta_nonlinearity"]].to_numpy()
    risk_values = env["risk"].to_numpy()
    obs_cols = [column for column in env.columns if column.startswith(channel + "_")]
    obs_env = env[obs_cols].to_numpy()
    obs_model = LinearRegression().fit(theta_env, obs_env)
    risk_surface = fit_risk_surface(theta_env, risk_values, cfg)
    sample_theta = sample_store["theta"]
    sample_obs = sample_store[channel]
    sample_residuals = sample_obs - obs_model.predict(sample_theta)
    covariance = LedoitWolf().fit(sample_residuals).covariance_
    inverse_covariance = np.linalg.pinv(covariance, rcond=1e-10)
    a_matrix = np.asarray(obs_model.coef_, dtype=float)
    b_vector = np.asarray(risk_surface["b"], dtype=float)
    hessian = np.asarray(risk_surface["H"], dtype=float)
    q_per_sample = a_matrix.T @ inverse_covariance @ a_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(q_per_sample)
    maximum = max(float(eigenvalues.max()), 1e-15)
    keep = eigenvalues > cfg["target"]["rank_relative_tolerance"] * maximum
    q_pinv = (
        eigenvectors[:, keep]
        @ np.diag(1.0 / eigenvalues[keep])
        @ eigenvectors[:, keep].T
        if np.any(keep)
        else np.zeros_like(q_per_sample)
    )
    projection = eigenvectors[:, keep] @ eigenvectors[:, keep].T if np.any(keep) else np.zeros((3, 3))
    b_range = projection @ b_vector
    b_null = b_vector - b_range
    null_ratio = float(np.linalg.norm(b_null) / max(np.linalg.norm(b_vector), 1e-12))
    efficient_weights = b_vector @ q_pinv @ a_matrix.T @ inverse_covariance
    efficient_score = (obs_env - obs_model.intercept_) @ efficient_weights
    fitted_risk_contrast = theta_env @ b_vector
    relevant_score_linear_r2 = float(r2_score(fitted_risk_contrast, efficient_score))
    center_rows = env[
        (env["theta_noise"] == 0)
        & (env["theta_phase"] == 0)
        & (env["theta_nonlinearity"] == 0)
    ]
    if len(center_rows) != 1:
        raise RuntimeError("calibration design must contain exactly one center environment")
    boundary_definition = cfg["calibration"].get(
        "boundary_definition", "center-environment deployment error"
    )
    if boundary_definition in {"local-linear intercept", "risk-surface intercept"}:
        boundary_risk = float(risk_surface["tau"])
    elif boundary_definition == "center-environment deployment error":
        boundary_risk = float(center_rows["risk"].iloc[0])
    else:
        raise ValueError(f"unsupported boundary definition: {boundary_definition}")
    gamma = float(cfg["target"]["risk_margin_gamma"])
    blind = null_ratio > cfg["target"]["null_ratio_threshold"]
    if risk_surface["type"] == "quadratic" and not blind:
        pair = optimize_supporting_pair(
            tau=float(risk_surface["tau"]),
            b_vector=b_vector,
            hessian=hessian,
            q_matrix=q_per_sample,
            q_pinv=q_pinv,
            cfg=cfg,
        )
        pair_stability = quadratic_pair_fold_stability(
            theta_env, risk_values, pair, cfg
        )
        direction = 0.5 * pair["difference"]
        task_type = "quadratic_identified"
    elif blind:
        denominator = float(b_vector @ b_null)
        direction = gamma * b_null / max(denominator, 1e-12)
        task_type = "near_null"
        pair = {}
        pair_stability = {}
    else:
        q_r_per_sample = float(b_vector @ q_pinv @ b_vector)
        direction = gamma * (q_pinv @ b_vector) / max(q_r_per_sample, 1e-12)
        task_type = "identified"
        pair = {}
        pair_stability = {}
    return {
        "channel": channel,
        "task_type": task_type,
        "A": a_matrix,
        "observation_intercept": np.asarray(obs_model.intercept_, dtype=float),
        "Sigma_per_sample": covariance,
        "Sigma_inverse_per_sample": inverse_covariance,
        "b": b_vector,
        "H": hessian,
        "risk_surface_type": risk_surface["type"],
        "risk_intercept": float(risk_surface["tau"]),
        "risk_surface_r2": risk_surface["r2"],
        "risk_surface_cv_r2": risk_surface["cv_r2"],
        "risk_cv_abs_residual_quantiles": risk_surface["cv_abs_residual_quantiles"],
        "risk_linear_r2": risk_surface["linear_r2"],
        "risk_linear_intercept": risk_surface["linear_intercept"],
        "observation_linear_r2": float(r2_score(obs_env, obs_model.predict(theta_env), multioutput="variance_weighted")),
        "relevant_score_linear_r2": relevant_score_linear_r2,
        "boundary_risk_frozen": boundary_risk,
        "boundary_definition": boundary_definition,
        "Q_per_sample": q_per_sample,
        "Q_pinv_per_sample": q_pinv,
        "eigenvalues": eigenvalues,
        "effective_rank": int(np.sum(keep)),
        "b_range": b_range,
        "b_null": b_null,
        "null_ratio": null_ratio,
        "direction": direction,
        "quadratic_pair": pair,
        "quadratic_pair_fold_stability": pair_stability,
    }


def sample_target_batches(
    cfg: dict,
    model: ExtraTreesClassifier,
    theta: np.ndarray,
    batch_size: int,
    replicates: int,
    channel: str,
    state: str,
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed(cfg["master_seed"], "sealed_target", channel, batch_size, state)
    )
    features, _ = generate_samples(
        cfg, theta, batch_size * replicates, rng, balanced=False
    )
    probabilities = model.predict_proba(features)
    observation = (
        rich_observation(features, probabilities)
        if channel == "rich"
        else features[:, AMP_FEATURE_INDICES]
    )
    return observation.reshape(replicates, batch_size, -1).mean(axis=1)


def reveal_risk(
    cfg: dict,
    model: ExtraTreesClassifier,
    theta: np.ndarray,
    channel: str,
    state: str,
) -> float:
    count = int(cfg["target"]["reveal_samples_per_state"])
    rng = np.random.default_rng(
        stable_seed(cfg["master_seed"], "outcome_reveal", channel, state)
    )
    features, labels = generate_samples(cfg, theta, count, rng, balanced=True)
    return float(np.mean(model.predict(features) != labels))


def score_channel(
    cfg: dict, model: ExtraTreesClassifier, fitted: dict
) -> tuple[pd.DataFrame, dict]:
    channel = fitted["channel"]
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    if fitted["task_type"] == "quadratic_identified":
        u_minus = np.asarray(fitted["quadratic_pair"]["u_minus"], dtype=float)
        u_plus = np.asarray(fitted["quadratic_pair"]["u_plus"], dtype=float)
        theta_minus = center + u_minus
        theta_plus = center + u_plus
    else:
        direction = fitted["direction"]
        u_minus = -direction
        u_plus = direction
        theta_minus = center + u_minus
        theta_plus = center + u_plus
    risk_minus = reveal_risk(cfg, model, theta_minus, channel, "minus")
    risk_plus = reveal_risk(cfg, model, theta_plus, channel, "plus")
    actual_half_gap = 0.5 * (risk_plus - risk_minus)
    rows = []
    for batch_size in cfg["target"]["batch_sizes"]:
        batch_size = int(batch_size)
        minus = sample_target_batches(
            cfg, model, theta_minus, batch_size, cfg["target"]["replicates_per_state"], channel, "minus"
        )
        plus = sample_target_batches(
            cfg, model, theta_plus, batch_size, cfg["target"]["replicates_per_state"], channel, "plus"
        )
        x_all = np.vstack([minus, plus])
        y_all = np.concatenate([-np.ones(len(minus), dtype=int), np.ones(len(plus), dtype=int)])
        if fitted["task_type"] == "identified":
            q_n = batch_size * fitted["Q_per_sample"]
            q_n_pinv = fitted["Q_pinv_per_sample"] / batch_size
            sigma_n_inverse = batch_size * fitted["Sigma_inverse_per_sample"]
            weights = fitted["b"] @ q_n_pinv @ fitted["A"].T @ sigma_n_inverse
            centered = x_all - fitted["observation_intercept"]
            scores = centered @ weights
            predictions = np.where(scores >= 0, 1, -1)
            operational_accuracy = float(np.mean(predictions == y_all))
            q_r_n = float(fitted["b"] @ q_n_pinv @ fitted["b"])
            theoretical_accuracy = float(
                norm.cdf(cfg["target"]["risk_margin_gamma"] / np.sqrt(max(q_r_n, 1e-15)))
            )
            theory_actual_gamma = float(norm.cdf(abs(actual_half_gap) / np.sqrt(max(q_r_n, 1e-15))))
        elif fitted["task_type"] == "quadratic_identified":
            difference = u_plus - u_minus
            midpoint = 0.5 * (u_plus + u_minus)
            weights = fitted["Sigma_inverse_per_sample"] @ fitted["A"] @ difference
            observation_midpoint = fitted["observation_intercept"] + fitted["A"] @ midpoint
            scores = (x_all - observation_midpoint) @ weights
            predictions = np.where(scores >= 0, 1, -1)
            operational_accuracy = float(np.mean(predictions == y_all))
            d_q_d = float(fitted["quadratic_pair"]["pair_dQd_per_sample"])
            theoretical_accuracy = float(norm.cdf(0.5 * np.sqrt(batch_size * d_q_d)))
            theory_actual_gamma = theoretical_accuracy
        else:
            operational_accuracy = 0.5
            theoretical_accuracy = 0.5
            theory_actual_gamma = 0.5
        split = len(minus) // 2
        train_indices = np.concatenate([np.arange(split), len(minus) + np.arange(split)])
        test_indices = np.concatenate([
            np.arange(split, len(minus)),
            len(minus) + np.arange(split, len(plus)),
        ])
        oracle = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=cfg["master_seed"] % (2**32 - 1)),
        )
        oracle.fit(x_all[train_indices], y_all[train_indices])
        oracle_accuracy = float(np.mean(oracle.predict(x_all[test_indices]) == y_all[test_indices]))
        rows.append(
            {
                "channel": channel,
                "task_type": fitted["task_type"],
                "batch_size": batch_size,
                "theoretical_accuracy_frozen_gamma": theoretical_accuracy,
                "theoretical_accuracy_actual_gap_posthoc": theory_actual_gamma,
                "operational_accuracy": operational_accuracy,
                "oracle_linear_accuracy": oracle_accuracy,
                "risk_minus_revealed": risk_minus,
                "risk_plus_revealed": risk_plus,
                "actual_half_gap": actual_half_gap,
                "requested_gamma": cfg["target"]["risk_margin_gamma"],
                "boundary_risk_frozen": fitted["boundary_risk_frozen"],
                "pair_design_margin": fitted["quadratic_pair"].get(
                    "design_margin", cfg["target"]["risk_margin_gamma"]
                ),
            }
        )
    return pd.DataFrame(rows), {
        "u_minus": u_minus.tolist(),
        "u_plus": u_plus.tolist(),
        "theta_minus": theta_minus.tolist(),
        "theta_plus": theta_plus.tolist(),
        "risk_minus_revealed": risk_minus,
        "risk_plus_revealed": risk_plus,
        "actual_half_gap": actual_half_gap,
    }


def serializable_fit(value):
    if isinstance(value, dict):
        return {key: serializable_fit(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable_fit(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    z = float(norm.ppf(1 - alpha / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def build_checks(cfg: dict, env: pd.DataFrame, fits: dict, target: pd.DataFrame) -> dict:
    gates = cfg["pilot_gates"]
    rich_rows = target[target["channel"] == "rich"]
    restricted_rows = target[target["channel"] == "restricted"]
    rho = spearmanr(
        rich_rows["theoretical_accuracy_frozen_gamma"],
        rich_rows["operational_accuracy"],
    ).statistic
    if not np.isfinite(rho):
        rho = 0.0
    rich_mae = float(
        np.mean(
            np.abs(
                rich_rows["theoretical_accuracy_frozen_gamma"]
                - rich_rows["operational_accuracy"]
            )
        )
    )
    reveal_n = int(cfg["target"]["reveal_samples_per_state"])
    risk_minus = float(rich_rows["risk_minus_revealed"].iloc[0])
    risk_plus = float(rich_rows["risk_plus_revealed"].iloc[0])
    minus_ci = wilson_interval(int(round(risk_minus * reveal_n)), reveal_n)
    plus_ci = wilson_interval(int(round(risk_plus * reveal_n)), reveal_n)
    boundary = float(rich_rows["boundary_risk_frozen"].iloc[0])
    gamma = float(cfg["target"]["risk_margin_gamma"])
    checks = {
        "calibration_risk_range": float(env["risk"].max() - env["risk"].min()),
        "risk_linear_r2": float(fits["rich"]["risk_linear_r2"]),
        "risk_surface_type": fits["rich"]["risk_surface_type"],
        "risk_surface_r2": float(fits["rich"]["risk_surface_r2"]),
        "risk_surface_cv_r2": float(fits["rich"]["risk_surface_cv_r2"]),
        "rich_null_ratio": float(fits["rich"]["null_ratio"]),
        "rich_relevant_score_linear_r2": float(fits["rich"]["relevant_score_linear_r2"]),
        "rich_theory_empirical_mae": rich_mae,
        "rich_theory_empirical_spearman": float(rho),
        "rich_actual_margin_ratio": float(
            rich_rows["actual_half_gap"].iloc[0] / cfg["target"]["risk_margin_gamma"]
        ),
        "rich_safe_cliff_realized": bool(
            risk_minus <= boundary - gamma and risk_plus >= boundary + gamma
        ),
        "risk_minus_ci95": list(minus_ci),
        "risk_plus_ci95": list(plus_ci),
        "rich_safe_cliff_ci_realized": bool(
            minus_ci[1] <= boundary - gamma and plus_ci[0] >= boundary + gamma
        ),
    }
    passed = {
        "risk_range": checks["calibration_risk_range"] >= gates["minimum_calibration_risk_range"],
        "relevant_score_linearity": checks["rich_relevant_score_linear_r2"] >= gates.get("minimum_relevant_score_linear_r2", 0.0),
        "rich_identifiable": checks["rich_null_ratio"] <= gates["rich_max_null_ratio"],
        "rich_bridge_mae": checks["rich_theory_empirical_mae"] <= gates["rich_max_theory_empirical_mae"],
        "rich_bridge_rank": checks["rich_theory_empirical_spearman"] >= gates["rich_min_theory_empirical_spearman"],
        "rich_margin_realized": checks["rich_actual_margin_ratio"] >= gates["minimum_actual_margin_ratio"],
        "rich_safe_cliff": checks["rich_safe_cliff_realized"],
    }
    if fits["rich"]["risk_surface_type"] == "quadratic":
        pair = fits["rich"]["quadratic_pair"]
        stability = fits["rich"]["quadratic_pair_fold_stability"]
        checks.update(
            {
                "quadratic_H_frobenius_norm": float(np.linalg.norm(fits["rich"]["H"])),
                "quadratic_linear_intercept_shift": float(
                    fits["rich"]["risk_intercept"] - fits["rich"]["risk_linear_intercept"]
                ),
                "quadratic_optimizer_constraint_error": float(pair["optimizer_constraint_error"]),
                "quadratic_support_slack_min": float(pair["support_slack_min"]),
                "quadratic_fold_constraint_max_absolute_error": float(
                    stability["fold_constraint_max_absolute_error"]
                ),
                "quadratic_pair_dQd_per_sample": float(pair["pair_dQd_per_sample"]),
                "quadratic_design_margin": float(pair["design_margin"]),
                "quadratic_remainder_buffer": float(pair["remainder_buffer"]),
            }
        )
        passed.update(
            {
                "risk_surface_fit": checks["risk_surface_r2"] >= gates["minimum_risk_surface_r2"],
                "risk_surface_crossfit": checks["risk_surface_cv_r2"] >= gates["minimum_risk_surface_cv_r2"],
                "quadratic_optimizer": checks["quadratic_optimizer_constraint_error"] <= gates["maximum_optimizer_constraint_error"],
                "quadratic_support": checks["quadratic_support_slack_min"] >= gates["minimum_support_slack"],
                "quadratic_pair_stability": checks["quadratic_fold_constraint_max_absolute_error"] <= gates["maximum_fold_constraint_error"],
                "rich_safe_cliff_ci": checks["rich_safe_cliff_ci_realized"],
            }
        )
    else:
        passed["risk_linearity"] = checks["risk_linear_r2"] >= gates["minimum_risk_linear_r2"]
    if len(restricted_rows):
        largest_n = restricted_rows.loc[restricted_rows["batch_size"].idxmax()]
        checks.update(
            {
                "restricted_null_ratio": float(fits["restricted"]["null_ratio"]),
                "restricted_oracle_accuracy_at_largest_n": float(largest_n["oracle_linear_accuracy"]),
                "restricted_actual_margin_ratio": float(
                    restricted_rows["actual_half_gap"].iloc[0] / cfg["target"]["risk_margin_gamma"]
                ),
            }
        )
        passed.update(
            {
                "restricted_blind": checks["restricted_null_ratio"] >= gates["restricted_min_null_ratio"],
                "restricted_oracle_chance": checks["restricted_oracle_accuracy_at_largest_n"] <= gates["restricted_max_oracle_accuracy_at_largest_n"],
                "restricted_margin_realized": checks["restricted_actual_margin_ratio"] >= gates["minimum_actual_margin_ratio"],
            }
        )
    return {"metrics": checks, "gates": passed, "all_passed": bool(all(passed.values()))}


def save_figure(target: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for channel, marker in [("rich", "o"), ("restricted", "s")]:
        rows = target[target["channel"] == channel]
        if rows.empty:
            continue
        axes[0].plot(
            rows["batch_size"], rows["operational_accuracy"], marker=marker, label=f"{channel}: operational"
        )
        axes[0].plot(
            rows["batch_size"], rows["theoretical_accuracy_frozen_gamma"], marker=marker, linestyle="--", label=f"{channel}: theory"
        )
        axes[1].plot(
            rows["batch_size"], rows["oracle_linear_accuracy"], marker=marker, label=channel
        )
    axes[0].axhline(0.5, color="black", linewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0.45, 1.01)
    axes[0].set_xlabel("Target batch size")
    axes[0].set_ylabel("Balanced state accuracy")
    axes[0].set_title("Frozen operational auditor vs bridge")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.5, color="black", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0.45, 1.01)
    axes[1].set_xlabel("Target batch size")
    axes[1].set_ylabel("Held-out oracle linear accuracy")
    axes[1].set_title("Can target-state labels rescue the channel?")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pretarget_freeze_gate(fitted: dict, cfg: dict) -> dict:
    if fitted["risk_surface_type"] != "quadratic":
        return {"required": False, "all_passed": True, "gates": {}}
    gates = cfg["pilot_gates"]
    pair = fitted["quadratic_pair"]
    stability = fitted["quadratic_pair_fold_stability"]
    checks = {
        "risk_surface_r2": float(fitted["risk_surface_r2"]),
        "risk_surface_cv_r2": float(fitted["risk_surface_cv_r2"]),
        "relevant_score_linear_r2": float(fitted["relevant_score_linear_r2"]),
        "null_ratio": float(fitted["null_ratio"]),
        "optimizer_constraint_error": float(pair["optimizer_constraint_error"]),
        "support_slack_min": float(pair["support_slack_min"]),
        "fold_constraint_max_absolute_error": float(
            stability["fold_constraint_max_absolute_error"]
        ),
    }
    passed = {
        "risk_surface_fit": checks["risk_surface_r2"] >= gates["minimum_risk_surface_r2"],
        "risk_surface_crossfit": checks["risk_surface_cv_r2"] >= gates["minimum_risk_surface_cv_r2"],
        "relevant_score_linearity": checks["relevant_score_linear_r2"] >= gates["minimum_relevant_score_linear_r2"],
        "identified_direction": checks["null_ratio"] <= gates["rich_max_null_ratio"],
        "optimizer": checks["optimizer_constraint_error"] <= gates["maximum_optimizer_constraint_error"],
        "support": checks["support_slack_min"] >= gates["minimum_support_slack"],
        "pair_stability": checks["fold_constraint_max_absolute_error"] <= gates["maximum_fold_constraint_error"],
    }
    return {
        "required": True,
        "metrics": checks,
        "gates": passed,
        "all_passed": bool(all(passed.values())),
    }


def environment_record() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torchsig": getattr(torchsig, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def sha256_manifest(root: Path, output: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != output and ".venv" not in path.parts
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.json")
    args = parser.parse_args()
    cfg = read_json(args.config)
    output_tag = cfg.get("output_tag", "default")
    results_dir = ROOT / "results" / output_tag
    figures_dir = ROOT / "figures" / output_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, training = train_deployment_model(cfg)
    env, sample_store = calibration_panel(cfg, model)
    channels = cfg.get("channels", ["rich", "restricted"])
    fits = {
        channel: fit_operational_bridge(env, sample_store, channel, cfg)
        for channel in channels
    }
    freeze_gate = pretarget_freeze_gate(fits["rich"], cfg)
    write_json(results_dir / "pretarget_freeze_gate.json", freeze_gate)
    if not freeze_gate["all_passed"]:
        env.to_csv(results_dir / "calibration_environment_panel.csv", index=False)
        write_json(results_dir / "training_summary.json", training)
        write_json(
            results_dir / "frozen_estimands.json",
            {key: serializable_fit(value) for key, value in fits.items()},
        )
        write_json(results_dir / "environment.json", environment_record())
        sha256_manifest(results_dir, results_dir / "SHA256SUMS.txt")
        print(json.dumps({"aborted_before_target": True, "freeze_gate": freeze_gate}, indent=2))
        return
    target_parts = []
    reveal = {}
    for channel in channels:
        part, reveal[channel] = score_channel(cfg, model, fits[channel])
        target_parts.append(part)
    target = pd.concat(target_parts, ignore_index=True)
    checks = build_checks(cfg, env, fits, target)

    env.to_csv(results_dir / "calibration_environment_panel.csv", index=False)
    target.to_csv(results_dir / "sealed_target_panel.csv", index=False)
    write_json(results_dir / "training_summary.json", training)
    write_json(results_dir / "frozen_estimands.json", {key: serializable_fit(value) for key, value in fits.items()})
    write_json(results_dir / "outcome_reveal.json", reveal)
    write_json(results_dir / "checks.json", checks)
    write_json(results_dir / "environment.json", environment_record())
    save_figure(target, figures_dir / "operational_pilot.png")
    sha256_manifest(results_dir, results_dir / "SHA256SUMS.txt")
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
