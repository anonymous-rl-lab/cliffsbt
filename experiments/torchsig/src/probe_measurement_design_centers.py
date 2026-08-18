#!/usr/bin/env python3
"""Screen whether risk-directed and generic information design diverge by center."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression

from measurement_design import (
    all_group_subsets,
    group_indices,
    linear_pair_dqd,
    subset_information,
)
from run_pilot import calibration_panel, fit_risk_surface, train_deployment_model


ROOT = Path(__file__).resolve().parents[1]


def fit_screening_bridge(environment: pd.DataFrame, store: dict, cfg: dict) -> dict:
    theta_env = environment[["theta_noise", "theta_phase", "theta_nonlinearity"]].to_numpy()
    risk_surface = fit_risk_surface(theta_env, environment["risk"].to_numpy(), cfg)
    observation_columns = [column for column in environment if column.startswith("rich_")]
    observation_means = environment[observation_columns].to_numpy()
    observation_model = LinearRegression().fit(theta_env, observation_means)
    residuals = store["rich"] - observation_model.predict(store["theta"])
    covariance = LedoitWolf().fit(residuals).covariance_
    return {
        "A": np.asarray(observation_model.coef_, dtype=float),
        "Sigma_per_sample": covariance,
        "b": np.asarray(risk_surface["b"], dtype=float),
        "H": np.asarray(risk_surface["H"], dtype=float),
        "risk_intercept": float(risk_surface["tau"]),
        "risk_surface_r2": float(risk_surface["r2"]),
        "risk_surface_cv_r2": float(risk_surface["cv_r2"]),
    }


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "measurement_design_center_probe.json").read_text())
    model, training = train_deployment_model(cfg)
    rows = []
    selections = []
    design_margin = float(cfg["target"]["risk_margin_gamma"] + cfg["risk_surface"]["remainder_buffer"])
    for center_spec in cfg["measurement_design"]["centers"]:
        local_cfg = copy.deepcopy(cfg)
        local_cfg["calibration"]["theta_center"] = center_spec["theta_center"]
        environment, store = calibration_panel(local_cfg, model)
        fitted = fit_screening_bridge(environment, store, local_cfg)
        center_rows = []
        for groups in all_group_subsets():
            indices = group_indices(groups)
            information = subset_information(
                fitted, indices, float(cfg["target"]["rank_relative_tolerance"])
            )
            record = {
                "center": center_spec["name"],
                "theta_center": center_spec["theta_center"],
                "groups": groups,
                "group_key": "+".join(groups),
                "cost": int(len(indices)),
                "risk_surface_r2": float(fitted["risk_surface_r2"]),
                "risk_surface_cv_r2": float(fitted["risk_surface_cv_r2"]),
                "calibration_risk_range": float(environment["risk"].max() - environment["risk"].min()),
                "risk_null_ratio": information["risk_null_ratio"],
                "trace_Q": information["trace_Q"],
                "linear_pair_dQd": linear_pair_dqd(fitted, information, design_margin),
            }
            rows.append(record)
            center_rows.append(record)
        for budget in cfg["measurement_design"]["budgets"]:
            eligible = [
                row for row in center_rows
                if row["cost"] <= budget
                and row["risk_null_ratio"] <= cfg["pilot_gates"]["rich_max_null_ratio"]
            ]
            if not eligible:
                continue
            risk = max(eligible, key=lambda row: (row["linear_pair_dQd"], -row["cost"]))
            trace = max(eligible, key=lambda row: (row["trace_Q"], -row["cost"]))
            selections.append(
                {
                    "center": center_spec["name"],
                    "budget": budget,
                    "risk_group_key": risk["group_key"],
                    "risk_linear_pair_dQd": risk["linear_pair_dQd"],
                    "trace_group_key": trace["group_key"],
                    "trace_selected_linear_pair_dQd": trace["linear_pair_dQd"],
                    "selection_differs": risk["group_key"] != trace["group_key"],
                    "risk_surface_r2": risk["risk_surface_r2"],
                    "risk_surface_cv_r2": risk["risk_surface_cv_r2"],
                    "calibration_risk_range": risk["calibration_risk_range"],
                }
            )
    output = ROOT / "results" / "measurement_design_center_probe"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "all_center_subsets.csv", index=False)
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(output / "selections.csv", index=False)
    summary = {
        "semantics": "calibration-only center screening; no target outcomes generated",
        "training": training,
        "centers": cfg["measurement_design"]["centers"],
        "number_selection_comparisons": int(len(selection_frame)),
        "number_different_selections": int(selection_frame["selection_differs"].sum()),
        "selections": selection_frame.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
