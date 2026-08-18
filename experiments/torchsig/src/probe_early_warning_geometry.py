#!/usr/bin/env python3
"""Calibration-only geometry gate for the Cliff early-warning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from measurement_design import group_indices, subset_information
from run_measurement_design_formal import fit_screening_bridge, serialize
from run_pilot import calibration_panel, environment_record, quadratic_risk, train_deployment_model


ROOT = Path(__file__).resolve().parents[1]


def oriented_axis(name: str, index: int, tau: float, b: np.ndarray, h: np.ndarray,
                  limit: float) -> np.ndarray:
    direction = np.zeros(3, dtype=float)
    direction[index] = 1.0
    high = quadratic_risk(limit * direction, tau, b, h)
    low = quadratic_risk(-limit * direction, tau, b, h)
    return direction if high >= low else -direction


def direction_record(name: str, direction: np.ndarray, tau: float, b: np.ndarray,
                     h: np.ndarray, cfg: dict) -> dict:
    spec = cfg["early_warning"]
    limit = float(spec["trajectory_offset_limit"])
    scalar = np.linspace(-limit, limit, int(spec["dense_grid_points"]))
    offsets = scalar[:, None] * direction[None, :]
    risk = np.asarray(quadratic_risk(offsets, tau, b, h), dtype=float)
    rho = float(spearmanr(scalar, risk).statistic)
    boundary = tau + float(cfg["target"]["risk_margin_gamma"])
    start_target = boundary - float(spec["start_margin_below_boundary"])
    end_target = boundary + float(spec["end_margin_above_boundary"])
    start_candidates = np.flatnonzero(risk <= start_target)
    end_candidates = np.flatnonzero(risk >= end_target)
    feasible = bool(
        len(start_candidates)
        and len(end_candidates)
        and start_candidates[0] < end_candidates[-1]
        and float(risk.max() - risk.min()) >= spec["minimum_predicted_risk_span"]
        and rho >= spec["minimum_monotone_spearman"]
    )
    start_index = int(start_candidates[-1]) if len(start_candidates) else None
    end_index = int(end_candidates[0]) if len(end_candidates) else None
    if start_index is not None and end_index is not None and start_index >= end_index:
        feasible = False
    return {
        "name": name,
        "direction": direction,
        "feasible": feasible,
        "spearman_scalar_risk": rho,
        "predicted_risk_min": float(risk.min()),
        "predicted_risk_max": float(risk.max()),
        "predicted_risk_span": float(risk.max() - risk.min()),
        "start_scalar": float(scalar[start_index]) if start_index is not None else None,
        "end_scalar": float(scalar[end_index]) if end_index is not None else None,
        "predicted_start_risk": float(risk[start_index]) if start_index is not None else None,
        "predicted_end_risk": float(risk[end_index]) if end_index is not None else None,
        "boundary": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/early_warning_geometry_probe.json",
        help="configuration path relative to the repository root",
    )
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    output = ROOT / "results" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    model, training = train_deployment_model(cfg)
    environment, store = calibration_panel(cfg, model)
    fitted = fit_screening_bridge(environment, store, cfg)
    indices = group_indices(tuple(cfg["early_warning"]["fixed_groups"]))
    information = subset_information(fitted, indices, cfg["target"]["rank_relative_tolerance"])
    tau = float(fitted["risk_intercept"])
    b = np.asarray(fitted["b"], dtype=float)
    h = np.asarray(fitted["H"], dtype=float)
    limit = float(cfg["early_warning"]["trajectory_offset_limit"])
    directions = {
        "noise": oriented_axis("noise", 0, tau, b, h, limit),
        "phase": oriented_axis("phase", 1, tau, b, h, limit),
        "nonlinearity": oriented_axis("nonlinearity", 2, tau, b, h, limit),
        "mixed_gradient": b / max(float(np.linalg.norm(b)), 1e-12),
    }
    records = [direction_record(name, direction, tau, b, h, cfg)
               for name, direction in directions.items()]
    frame = pd.DataFrame([serialize(item) for item in records])
    frame.to_csv(output / "trajectory_geometry.csv", index=False)
    gates = cfg["pilot_gates"]
    metrics = {
        "calibration_risk_range": float(environment["risk"].max() - environment["risk"].min()),
        "risk_surface_r2": fitted["risk_surface_r2"],
        "risk_surface_cv_r2": fitted["risk_surface_cv_r2"],
        "relevant_score_linear_r2": fitted["relevant_score_linear_r2"],
        "fixed_channel_risk_null_ratio": information["risk_null_ratio"],
        "feasible_directions": int(frame["feasible"].sum()),
        "tau": tau,
        "relative_warning_boundary": tau + cfg["target"]["risk_margin_gamma"],
    }
    passed = {
        "risk_range": metrics["calibration_risk_range"] >= gates["minimum_calibration_risk_range"],
        "risk_surface_fit": metrics["risk_surface_r2"] >= gates["minimum_risk_surface_r2"],
        "risk_surface_crossfit": metrics["risk_surface_cv_r2"] >= gates["minimum_risk_surface_cv_r2"],
        "relevant_score": metrics["relevant_score_linear_r2"] >= gates["minimum_relevant_score_linear_r2"],
        "fixed_channel_identified": metrics["fixed_channel_risk_null_ratio"] <= gates["maximum_risk_null_ratio"],
        "trajectory_geometry": metrics["feasible_directions"] >= gates["minimum_feasible_directions"],
    }
    result = {
        "metrics": metrics,
        "gates": passed,
        "all_passed": bool(all(passed.values())),
        "directions": records,
        "semantics": "calibration-only geometry; no sequential target outcomes generated",
    }
    (output / "geometry_gate.json").write_text(
        json.dumps(serialize(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment.to_csv(output / "calibration_environment_panel.csv", index=False)
    (output / "training_summary.json").write_text(
        json.dumps(serialize(training), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(environment_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(serialize(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
