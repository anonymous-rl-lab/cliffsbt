#!/usr/bin/env python3
"""Independent confirmation of a frozen 25/54-dimensional measurement channel."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from measurement_design import MEASUREMENT_GROUPS
from measurement_design import group_indices, predicted_batch_size, subset_information
from run_measurement_design_formal import (
    fit_screening_bridge,
    score_design,
    serialize,
)
from run_pilot import (
    calibration_panel,
    environment_record,
    optimize_supporting_pair,
    quadratic_pair_fold_stability,
    train_deployment_model,
)


ROOT = Path(__file__).resolve().parents[1]


def fit_design(label: str, groups: tuple[str, ...], fitted: dict, cfg: dict) -> dict:
    indices = group_indices(groups)
    information = subset_information(
        fitted, indices, cfg["target"]["rank_relative_tolerance"]
    )
    pair = optimize_supporting_pair(
        tau=float(fitted["risk_intercept"]),
        b_vector=np.asarray(fitted["b"], dtype=float),
        hessian=np.asarray(fitted["H"], dtype=float),
        q_matrix=information["Q_per_sample"],
        q_pinv=information["Q_pinv_per_sample"],
        cfg=cfg,
    )
    return {
        "label": label,
        "groups": groups,
        "group_key": "+".join(groups),
        "cost": int(len(indices)),
        "indices": indices,
        "pair_dQd_per_sample": float(pair["pair_dQd_per_sample"]),
        "predicted_n_target": predicted_batch_size(
            float(pair["pair_dQd_per_sample"]), cfg["measurement_compression"]["target_accuracy"]
        ),
        "risk_null_ratio": float(information["risk_null_ratio"]),
        "support_slack_min": float(pair["support_slack_min"]),
        "optimizer_constraint_error": float(pair["optimizer_constraint_error"]),
        "u_minus": np.asarray(pair["u_minus"], dtype=float),
        "u_plus": np.asarray(pair["u_plus"], dtype=float),
        **information,
    }


def stability(design: dict, fitted: dict, cfg: dict) -> float:
    pair = {
        "u_minus": design["u_minus"],
        "u_plus": design["u_plus"],
        "design_margin": float(
            cfg["target"]["risk_margin_gamma"] + cfg["risk_surface"]["remainder_buffer"]
        ),
    }
    return float(
        quadratic_pair_fold_stability(
            fitted["theta_env"], fitted["risk_values"], pair, cfg
        )["fold_constraint_max_absolute_error"]
    )


def plot(target: pd.DataFrame, output: Path) -> None:
    centers = list(target["center"].unique())
    fig, axes = plt.subplots(1, len(centers), figsize=(10.8, 4.2), constrained_layout=True)
    if len(centers) == 1:
        axes = [axes]
    colors = {"compressed": "#2f5597", "full": "#c55a11"}
    for axis, center in zip(axes, centers):
        for design in ("compressed", "full"):
            curve = target[(target["center"] == center) & (target["design"] == design)]
            axis.plot(
                curve["batch_size"], curve["operational_accuracy"], "-o",
                color=colors[design], label=f"{design}: auditor"
            )
            axis.plot(
                curve["batch_size"], curve["theoretical_accuracy"], "--",
                color=colors[design], alpha=0.6, label=f"{design}: theory"
            )
        axis.set_xscale("log", base=2)
        axis.set_ylim(0.48, 1.01)
        axis.set_xlabel("Unlabelled batch size")
        axis.set_ylabel("Balanced state accuracy")
        axis.set_title(center)
        axis.legend(fontsize=8)
    fig.suptitle("Frozen 25-dimensional channel versus full 54-dimensional channel")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "formal_measurement_compression_v1.json").read_text())
    output = ROOT / "results" / cfg["output_tag"]
    figures = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    model, training = train_deployment_model(cfg)
    fixed_groups = tuple(cfg["measurement_compression"]["fixed_groups"])
    full_groups = tuple(MEASUREMENT_GROUPS)
    fixed_cost = len(group_indices(fixed_groups))
    if fixed_cost != int(cfg["measurement_compression"]["fixed_cost"]):
        raise RuntimeError("frozen measurement groups do not match fixed_cost")
    if len(group_indices(full_groups)) != int(cfg["measurement_compression"]["full_cost"]):
        raise RuntimeError("full measurement groups do not match full_cost")
    environments = []
    frozen = {}
    frozen_public = {}
    center_gates = {}
    thresholds = cfg["pilot_gates"]
    for center_spec in cfg["measurement_compression"]["centers"]:
        name = center_spec["name"]
        center = np.asarray(center_spec["theta_center"], dtype=float)
        local_cfg = copy.deepcopy(cfg)
        local_cfg["calibration"]["theta_center"] = center.tolist()
        environment, store = calibration_panel(local_cfg, model)
        environment.insert(0, "center", name)
        environments.append(environment)
        fitted = fit_screening_bridge(environment, store, local_cfg)
        compressed = fit_design("compressed", fixed_groups, fitted, local_cfg)
        full = fit_design("full", full_groups, fitted, local_cfg)
        compressed_stability = stability(compressed, fitted, local_cfg)
        full_stability = stability(full, fitted, local_cfg)
        retention = compressed["pair_dQd_per_sample"] / full["pair_dQd_per_sample"]
        cost_fraction = compressed["cost"] / full["cost"]
        metrics = {
            "calibration_risk_range": float(environment["risk"].max() - environment["risk"].min()),
            "risk_surface_r2": fitted["risk_surface_r2"],
            "risk_surface_cv_r2": fitted["risk_surface_cv_r2"],
            "relevant_score_linear_r2": fitted["relevant_score_linear_r2"],
            "compressed_risk_null_ratio": compressed["risk_null_ratio"],
            "full_risk_null_ratio": full["risk_null_ratio"],
            "compressed_dQd": compressed["pair_dQd_per_sample"],
            "full_dQd": full["pair_dQd_per_sample"],
            "information_retention": retention,
            "cost_fraction": cost_fraction,
            "compressed_predicted_n90": compressed["predicted_n_target"],
            "full_predicted_n90": full["predicted_n_target"],
            "compressed_pair_stability": compressed_stability,
            "full_pair_stability": full_stability,
        }
        passed = {
            "risk_range": metrics["calibration_risk_range"] >= thresholds["minimum_calibration_risk_range"],
            "risk_surface_fit": metrics["risk_surface_r2"] >= thresholds["minimum_risk_surface_r2"],
            "risk_surface_crossfit": metrics["risk_surface_cv_r2"] >= thresholds["minimum_risk_surface_cv_r2"],
            "relevant_score": metrics["relevant_score_linear_r2"] >= thresholds["minimum_relevant_score_linear_r2"],
            "compressed_identified": metrics["compressed_risk_null_ratio"] <= thresholds["rich_max_null_ratio"],
            "compressed_optimizer": compressed["optimizer_constraint_error"] <= thresholds["maximum_optimizer_constraint_error"],
            "compressed_support": compressed["support_slack_min"] >= thresholds["minimum_support_slack"],
            "compressed_pair_stability": compressed_stability <= thresholds["maximum_fold_constraint_error"],
            "full_identified": metrics["full_risk_null_ratio"] <= thresholds["rich_max_null_ratio"],
            "full_optimizer": full["optimizer_constraint_error"] <= thresholds["maximum_optimizer_constraint_error"],
            "full_support": full["support_slack_min"] >= thresholds["minimum_support_slack"],
            "full_pair_stability": full_stability <= thresholds["maximum_fold_constraint_error"],
            "information_retention": retention >= thresholds["minimum_information_retention"],
            "cost_fraction": cost_fraction <= thresholds["maximum_cost_fraction"],
        }
        center_gates[name] = {"metrics": metrics, "gates": passed, "all_passed": bool(all(passed.values()))}
        frozen[name] = {"center": center, "fitted": fitted, "designs": {"compressed": compressed, "full": full}}
        frozen_public[name] = {
            "theta_center": center,
            "designs": {
                label: {
                    "groups": design["groups"],
                    "indices": design["indices"],
                    "cost": design["cost"],
                    "pair_dQd_per_sample": design["pair_dQd_per_sample"],
                    "predicted_n90": design["predicted_n_target"],
                    "risk_null_ratio": design["risk_null_ratio"],
                    "support_slack_min": design["support_slack_min"],
                    "optimizer_constraint_error": design["optimizer_constraint_error"],
                    "u_minus": design["u_minus"],
                    "u_plus": design["u_plus"],
                }
                for label, design in (("compressed", compressed), ("full", full))
            },
        }
    pretarget = {
        "centers": center_gates,
        "frozen_measurement_designs": frozen_public,
        "all_passed": all(item["all_passed"] for item in center_gates.values()),
    }
    (output / "pretarget_measurement_compression_gate.json").write_text(
        json.dumps(serialize(pretarget), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.concat(environments, ignore_index=True).to_csv(output / "calibration_environment_panel.csv", index=False)
    (output / "training_summary.json").write_text(
        json.dumps(serialize(training), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(environment_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not pretarget["all_passed"]:
        print(json.dumps({"aborted_before_target": True, "pretarget": pretarget}, indent=2))
        return
    target_rows = []
    reveal = {}
    for name, item in frozen.items():
        reveal[name] = {}
        for label, design in item["designs"].items():
            rows, outcomes = score_design(cfg, model, name, item["center"], item["fitted"], design)
            target_rows.extend(rows)
            reveal[name][label] = outcomes
    target = pd.DataFrame(target_rows)
    target.to_csv(output / "sealed_target_compression_curves.csv", index=False)
    (output / "outcome_reveal.json").write_text(
        json.dumps(serialize(reveal), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = []
    curve_gates = {}
    for (center, design), curve in target.groupby(["center", "design"]):
        curve = curve.sort_values("batch_size")
        mae = float(np.mean(np.abs(curve["theoretical_accuracy"] - curve["operational_accuracy"])))
        rho = float(spearmanr(curve["theoretical_accuracy"], curve["operational_accuracy"]).statistic)
        if not np.isfinite(rho):
            rho = 0.0
        candidates = curve[curve["operational_accuracy"] >= cfg["measurement_compression"]["target_accuracy"]]
        empirical_n90 = int(candidates["batch_size"].min()) if len(candidates) else None
        first = curve.iloc[0]
        ledger.append(
            {
                "center": center,
                "design": design,
                "groups": first["groups"],
                "cost": int(first["cost"]),
                "pair_dQd_per_sample": float(first["pair_dQd_per_sample"]),
                "predicted_n90": int(np.ceil(4 * norm.ppf(0.9) ** 2 / first["pair_dQd_per_sample"])),
                "empirical_n90": empirical_n90,
                "curve_mae": mae,
                "curve_spearman": rho,
                "risk_minus": float(first["risk_minus_revealed"]),
                "risk_plus": float(first["risk_plus_revealed"]),
            }
        )
        curve_gates[f"{center}:{design}"] = {
            "curve_mae": mae <= thresholds["maximum_curve_mae"],
            "curve_spearman": rho >= thresholds["minimum_curve_spearman"],
        }
    ledger_frame = pd.DataFrame(ledger)
    ledger_frame.to_csv(output / "measurement_compression_ledger.csv", index=False)
    accuracy_loss = {}
    empirical_gates = {}
    for center in target["center"].unique():
        pivot = target[target["center"] == center].pivot(
            index="batch_size", columns="design", values="operational_accuracy"
        )
        loss = float(np.mean(pivot["full"] - pivot["compressed"]))
        accuracy_loss[center] = loss
        empirical_gates[center] = loss <= thresholds["maximum_mean_accuracy_loss"]
    checks = {
        "pretarget_all_passed": True,
        "curve_gates": curve_gates,
        "mean_full_minus_compressed_accuracy": accuracy_loss,
        "mean_accuracy_loss_gates": empirical_gates,
        "all_passed": bool(
            all(all(item.values()) for item in curve_gates.values())
            and all(empirical_gates.values())
        ),
        "semantics": "frozen measurement compression; risk-directed superiority over trace(Q) was rejected in the preceding formal run",
        "ledger": ledger,
    }
    (output / "checks.json").write_text(
        json.dumps(serialize(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot(target, figures / "formal_measurement_compression_v1.png")
    print(json.dumps(serialize(checks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
