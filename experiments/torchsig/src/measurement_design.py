#!/usr/bin/env python3
"""Cost-aware observation-channel design for local cliff auditability."""

from __future__ import annotations

from itertools import combinations
from math import ceil

import numpy as np
from scipy.stats import norm


MEASUREMENT_GROUPS = {
    "model_outputs": tuple(range(0, 6)),
    "amplitude": tuple(range(6, 15)) + tuple(range(43, 51)),
    "complex_moments": tuple(range(15, 19)),
    "autocorrelation": tuple(range(19, 31)),
    "phase_increment": tuple(range(31, 43)),
    "spectral": tuple(range(51, 54)),
}


def validate_groups(observation_dimension: int = 54) -> None:
    flattened = [index for values in MEASUREMENT_GROUPS.values() for index in values]
    if sorted(flattened) != list(range(observation_dimension)):
        raise RuntimeError("measurement groups must partition the rich observation channel")


def group_indices(groups: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        sorted(index for group in groups for index in MEASUREMENT_GROUPS[group]),
        dtype=int,
    )


def all_group_subsets() -> list[tuple[str, ...]]:
    names = tuple(MEASUREMENT_GROUPS)
    return [subset for size in range(1, len(names) + 1) for subset in combinations(names, size)]


def subset_information(fitted: dict, indices: np.ndarray, rank_tolerance: float) -> dict:
    a_matrix = np.asarray(fitted["A"], dtype=float)[indices]
    covariance = np.asarray(fitted["Sigma_per_sample"], dtype=float)[np.ix_(indices, indices)]
    inverse_covariance = np.linalg.pinv(covariance, rcond=1e-10)
    q_matrix = a_matrix.T @ inverse_covariance @ a_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(q_matrix)
    maximum = max(float(eigenvalues.max()), 1e-15)
    keep = eigenvalues > rank_tolerance * maximum
    q_pinv = (
        eigenvectors[:, keep]
        @ np.diag(1.0 / eigenvalues[keep])
        @ eigenvectors[:, keep].T
        if np.any(keep)
        else np.zeros_like(q_matrix)
    )
    projection = (
        eigenvectors[:, keep] @ eigenvectors[:, keep].T
        if np.any(keep)
        else np.zeros_like(q_matrix)
    )
    b_vector = np.asarray(fitted["b"], dtype=float)
    b_null = b_vector - projection @ b_vector
    null_ratio = float(np.linalg.norm(b_null) / max(np.linalg.norm(b_vector), 1e-12))
    return {
        "A": a_matrix,
        "Sigma_per_sample": covariance,
        "Sigma_inverse_per_sample": inverse_covariance,
        "Q_per_sample": q_matrix,
        "Q_pinv_per_sample": q_pinv,
        "effective_rank": int(np.sum(keep)),
        "risk_null_ratio": null_ratio,
        "trace_Q": float(np.trace(q_matrix)),
        "eigenvalues_Q": eigenvalues,
    }


def predicted_batch_size(pair_dqd: float, target_accuracy: float) -> int | None:
    if not np.isfinite(pair_dqd) or pair_dqd <= 0:
        return None
    z_value = float(norm.ppf(target_accuracy))
    return int(ceil(4.0 * z_value * z_value / pair_dqd))


def linear_pair_dqd(fitted: dict, information: dict, design_margin: float) -> float:
    b_vector = np.asarray(fitted["b"], dtype=float)
    variance = float(b_vector @ information["Q_pinv_per_sample"] @ b_vector)
    if variance <= 0:
        return 0.0
    return float(4.0 * design_margin * design_margin / variance)


def evaluate_subsets(fitted: dict, cfg: dict, optimizer) -> list[dict]:
    validate_groups(len(fitted["A"]))
    rows = []
    rank_tolerance = float(cfg["target"]["rank_relative_tolerance"])
    target_accuracy = float(cfg["measurement_design"]["target_accuracy"])
    for groups in all_group_subsets():
        indices = group_indices(groups)
        information = subset_information(fitted, indices, rank_tolerance)
        row = {
            "groups": groups,
            "group_key": "+".join(groups),
            "cost": int(len(indices)),
            "dimensions": indices.tolist(),
            "effective_rank": information["effective_rank"],
            "risk_null_ratio": information["risk_null_ratio"],
            "trace_Q": information["trace_Q"],
            "optimizer_feasible": False,
            "pair_dQd_per_sample": float("nan"),
            "predicted_n_target": None,
            "support_slack_min": float("nan"),
            "optimizer_constraint_error": float("nan"),
        }
        try:
            pair = optimizer(
                tau=float(fitted["risk_intercept"]),
                b_vector=np.asarray(fitted["b"], dtype=float),
                hessian=np.asarray(fitted["H"], dtype=float),
                q_matrix=information["Q_per_sample"],
                q_pinv=information["Q_pinv_per_sample"],
                cfg=cfg,
            )
            dqd = float(pair["pair_dQd_per_sample"])
            row.update(
                {
                    "optimizer_feasible": True,
                    "pair_dQd_per_sample": dqd,
                    "predicted_n_target": predicted_batch_size(dqd, target_accuracy),
                    "support_slack_min": float(pair["support_slack_min"]),
                    "optimizer_constraint_error": float(pair["optimizer_constraint_error"]),
                    "u_minus": np.asarray(pair["u_minus"], dtype=float).tolist(),
                    "u_plus": np.asarray(pair["u_plus"], dtype=float).tolist(),
                }
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            row["optimizer_error"] = str(error)
        rows.append(row)
    return rows


def eligible(row: dict, cfg: dict) -> bool:
    gates = cfg["pilot_gates"]
    return bool(
        row["optimizer_feasible"]
        and row["risk_null_ratio"] <= gates["rich_max_null_ratio"]
        and row["optimizer_constraint_error"] <= gates["maximum_optimizer_constraint_error"]
        and row["support_slack_min"] >= gates["minimum_support_slack"]
    )


def select_for_budget(rows: list[dict], budget: int, cfg: dict, objective: str) -> dict | None:
    candidates = [row for row in rows if row["cost"] <= budget and eligible(row, cfg)]
    if not candidates:
        return None
    if objective == "risk_directed":
        key = lambda row: (row["pair_dQd_per_sample"], -row["cost"], row["group_key"])
    elif objective == "trace":
        key = lambda row: (row["trace_Q"], -row["cost"], row["group_key"])
    else:
        raise ValueError(f"unknown objective: {objective}")
    return max(candidates, key=key)


def frontier(rows: list[dict], budgets: list[int], cfg: dict) -> list[dict]:
    selected = []
    for budget in budgets:
        for objective in ("risk_directed", "trace"):
            row = select_for_budget(rows, int(budget), cfg, objective)
            if row is not None:
                selected.append({"budget": int(budget), "objective": objective, **row})
    return selected
