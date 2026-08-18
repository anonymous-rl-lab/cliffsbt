#!/usr/bin/env python3
"""Locked primitives for the CURE-OR confirmatory experiment v2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
CONFIG = PACKAGE / "config"
DESIGN_SEED = "cure-or-osf-reregistration-v2"
SCHEDULE_SEED = "cure-or-osf-reregistration-v2-schedule"
TARGET_FAMILIES = [2, 6, 11, 12, 13, 14, 15, 16, 17, 18]
MODEL_SEEDS = [113, 127, 139, 151, 163]
SCHEDULE_IDS = [211, 223, 227]
WINDOWS = 13
TAU = 0.50
QUANTILES = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_unit(text: str) -> float:
    return int(stable_hash(text)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family_baseline(family: int) -> int:
    return 1 if family in (2, 6) else 10


def assigned_levels(
    bases: list[tuple[int, int, int]], family: int, window: int, schedule_id: int
) -> dict[tuple[int, int, int], int]:
    """Assign paired identities to ordered severities under one frozen schedule."""
    if schedule_id not in SCHEDULE_IDS:
        raise ValueError(f"unregistered schedule id: {schedule_id}")
    if window == WINDOWS - 1:
        return {base: 4 for base in bases}
    lower, phase = divmod(window, 3)
    order = sorted(
        bases,
        key=lambda base: stable_hash(f"{SCHEDULE_SEED}|{schedule_id}|{family}|{base}"),
    )
    advanced = set(order[: round(len(order) * phase / 3)])
    return {base: lower + int(base in advanced) for base in order}


def margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def flux25(
    probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    previous_probabilities: np.ndarray | None,
) -> np.ndarray:
    current_class = np.argmax(probabilities, axis=1)
    baseline_class = np.argmax(baseline_probabilities, axis=1)
    departed = current_class != baseline_class
    class_departure = np.asarray([
        np.mean((baseline_class == class_id) & departed) for class_id in range(10)
    ])
    current_margin = margin(probabilities)
    baseline_margin = margin(baseline_probabilities)
    margin_quantiles = np.quantile(current_margin, QUANTILES)
    delta_quantiles = np.quantile(current_margin - baseline_margin, QUANTILES)
    if previous_probabilities is None:
        new_departure = recovery = net_departure = persistent = 0.0
    else:
        previous_departed = np.argmax(previous_probabilities, axis=1) != baseline_class
        new_departure = float(np.mean((~previous_departed) & departed))
        recovery = float(np.mean(previous_departed & (~departed)))
        net_departure = new_departure - recovery
        persistent = float(np.mean(previous_departed & departed))
    near_boundary = float(np.mean(current_margin < 0.10))
    result = np.concatenate([
        class_departure,
        margin_quantiles,
        delta_quantiles,
        [new_departure, recovery, net_departure, persistent, near_boundary],
    ]).astype(np.float64)
    if result.shape != (25,):
        raise RuntimeError(f"Flux25 shape is {result.shape}, expected (25,)")
    return result


def moments25(probabilities: np.ndarray, norms: np.ndarray) -> np.ndarray:
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    probability_margin = margin(probabilities)
    result = np.concatenate([
        probabilities.mean(axis=0),
        probabilities.var(axis=0),
        [entropy.mean(), entropy.var(), probability_margin.mean(), probability_margin.var(), norms.mean()],
    ]).astype(np.float64)
    if result.shape != (25,):
        raise RuntimeError(f"Moments25 shape is {result.shape}, expected (25,)")
    return result


def hybrid25(flux: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """Return a 25-channel chart with an 11-channel frozen active subchart.

    Channels 0--10 are the frozen mechanism/coherence subchart used by
    the frozen warning readout.  Channels 11--24 are committed diagnostics with
    zero readout weight; they retain the declared 25-channel telemetry budget.
    """
    active = np.asarray([
        np.sum(flux[:10]), flux[10], flux[12], flux[14], flux[15], flux[17],
        flux[19], flux[22], flux[23], flux[24], moments[24],
    ], dtype=np.float64)
    diagnostic = flux[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 20, 21]]
    result = np.concatenate([active, diagnostic]).astype(np.float64)
    if result.shape != (25,):
        raise RuntimeError(f"Hybrid25 shape is {result.shape}, expected (25,)")
    return result


def hybrid_temporal(sensor: np.ndarray, window: int) -> np.ndarray:
    active = np.asarray(sensor, dtype=np.float64)[:, :11]
    current = active[window]
    previous = active[max(0, window - 1)]
    start = max(0, window - 2)
    beginning = active[start]
    slope = np.zeros_like(current) if start == window else (current - beginning) / (window - start)
    result = np.concatenate([current, current - previous, slope, [window / 12.0]])
    if result.shape != (34,):
        raise RuntimeError(f"Hybrid25 temporal readout shape is {result.shape}, expected (34,)")
    return result


def score_hybrid(sensor: np.ndarray, model: dict) -> np.ndarray:
    x = np.asarray([hybrid_temporal(sensor, window) for window in range(WINDOWS)])
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coefficient):
        raise RuntimeError("frozen Hybrid25 model dimensions do not match")
    logits = ((x - mean) / scale) @ coefficient + float(model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def alarm_time(scores: np.ndarray, threshold: float, persistence: int = 1, earliest: int = 1) -> int | None:
    for window in range(earliest, len(scores) - persistence + 1):
        if np.all(scores[window: window + persistence] >= threshold):
            return int(window)
    return None


def persistent_cliff(risk: list[float]) -> int | None:
    values = [float(value) for value in risk]
    return next(
        (index for index in range(len(values)) if values[index] >= TAU and all(value >= TAU for value in values[index:])),
        None,
    )


def exact_flow(status: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    binary = np.asarray(status, dtype=bool)
    forward = np.sum((~binary[:-1]) & binary[1:], axis=1)
    recovery = np.sum(binary[:-1] & (~binary[1:]), axis=1)
    risk = binary.mean(axis=1)
    residual = risk[1:] - risk[:-1] - (forward - recovery) / binary.shape[1]
    return forward, recovery, residual, float(np.max(np.abs(residual)))


def fit_ridge(features: np.ndarray, labels: np.ndarray, regularization: float) -> dict:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (features - mean) / scale
    x = np.column_stack([x, np.ones(len(x))])
    y = np.eye(10)[labels]
    weights = np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def anchored_update(base_head: dict, features: np.ndarray, labels: np.ndarray, trust_lambda: float) -> dict:
    x = (features - base_head["mean"]) / base_head["scale"]
    x = np.column_stack([x, np.ones(len(x))])
    residual = np.eye(10)[labels] - x @ base_head["weights"]
    if len(x) <= x.shape[1]:
        delta = x.T @ np.linalg.solve(x @ x.T + trust_lambda * np.eye(len(x)), residual)
    else:
        delta = np.linalg.solve(x.T @ x + trust_lambda * np.eye(x.shape[1]), x.T @ residual)
    return {
        "mean": base_head["mean"],
        "scale": base_head["scale"],
        "weights": base_head["weights"] + delta,
    }


def head_scores(head: dict, features: np.ndarray) -> np.ndarray:
    x = (features - head["mean"]) / head["scale"]
    x = np.column_stack([x, np.ones(len(x))])
    return x @ head["weights"]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def seed_cluster_interval(values: dict[int, float], replicates: int, rng_seed: int) -> dict:
    seeds = sorted(values)
    array = np.asarray([values[seed] for seed in seeds], dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    samples = array[indices].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }
