#!/usr/bin/env python3
"""Formal outcome-blind validation of cost-aware Cliff measurement design."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from measurement_design import (
    MEASUREMENT_GROUPS,
    eligible,
    evaluate_subsets,
    group_indices,
    select_for_budget,
    subset_information,
)
from run_pilot import (
    calibration_panel,
    environment_record,
    fit_risk_surface,
    generate_samples,
    optimize_supporting_pair,
    quadratic_pair_fold_stability,
    rich_observation,
    stable_seed,
    train_deployment_model,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_GROUPS = tuple(MEASUREMENT_GROUPS)


def fit_screening_bridge(environment: pd.DataFrame, store: dict, cfg: dict) -> dict:
    theta_env = environment[["theta_noise", "theta_phase", "theta_nonlinearity"]].to_numpy()
    risk_values = environment["risk"].to_numpy()
    risk_surface = fit_risk_surface(theta_env, risk_values, cfg)
    observation_columns = [column for column in environment if column.startswith("rich_")]
    observation_means = environment[observation_columns].to_numpy()
    observation_model = LinearRegression().fit(theta_env, observation_means)
    residuals = store["rich"] - observation_model.predict(store["theta"])
    covariance = LedoitWolf().fit(residuals).covariance_
    fitted = {
        "A": np.asarray(observation_model.coef_, dtype=float),
        "observation_intercept": np.asarray(observation_model.intercept_, dtype=float),
        "Sigma_per_sample": covariance,
        "b": np.asarray(risk_surface["b"], dtype=float),
        "H": np.asarray(risk_surface["H"], dtype=float),
        "risk_intercept": float(risk_surface["tau"]),
        "risk_surface_r2": float(risk_surface["r2"]),
        "risk_surface_cv_r2": float(risk_surface["cv_r2"]),
        "risk_cv_abs_residual_quantiles": risk_surface["cv_abs_residual_quantiles"],
        "theta_env": theta_env,
        "risk_values": risk_values,
    }
    full = subset_information(
        fitted, np.arange(observation_means.shape[1]), cfg["target"]["rank_relative_tolerance"]
    )
    efficient_weights = (
        fitted["b"]
        @ full["Q_pinv_per_sample"]
        @ full["A"].T
        @ full["Sigma_inverse_per_sample"]
    )
    efficient_score = (observation_means - fitted["observation_intercept"]) @ efficient_weights
    fitted["relevant_score_linear_r2"] = float(
        r2_score(theta_env @ fitted["b"], efficient_score)
    )
    fitted["full_risk_null_ratio"] = float(full["risk_null_ratio"])
    return fitted


def serialize(value):
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def pair_from_row(row: dict, cfg: dict) -> dict:
    return {
        "u_minus": np.asarray(row["u_minus"], dtype=float),
        "u_plus": np.asarray(row["u_plus"], dtype=float),
        "design_margin": float(
            cfg["target"]["risk_margin_gamma"] + cfg["risk_surface"]["remainder_buffer"]
        ),
    }


def design_record(label: str, row: dict, fitted: dict, cfg: dict) -> dict:
    indices = np.asarray(row["dimensions"], dtype=int)
    information = subset_information(
        fitted, indices, cfg["target"]["rank_relative_tolerance"]
    )
    return {
        "label": label,
        "groups": tuple(row["groups"]),
        "group_key": row["group_key"],
        "cost": int(row["cost"]),
        "indices": indices,
        "pair_dQd_per_sample": float(row["pair_dQd_per_sample"]),
        "predicted_n_target": row["predicted_n_target"],
        "risk_null_ratio": float(row["risk_null_ratio"]),
        "support_slack_min": float(row["support_slack_min"]),
        "optimizer_constraint_error": float(row["optimizer_constraint_error"]),
        "u_minus": np.asarray(row["u_minus"], dtype=float),
        "u_plus": np.asarray(row["u_plus"], dtype=float),
        **information,
    }


def target_batches(cfg: dict, model, theta: np.ndarray, count: int, replicates: int,
                   indices: np.ndarray, seed_parts: tuple) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], *seed_parts))
    features, _ = generate_samples(cfg, theta, count * replicates, rng, balanced=False)
    probabilities = model.predict_proba(features)
    observation = rich_observation(features, probabilities)[:, indices]
    return observation.reshape(replicates, count, -1).mean(axis=1)


def reveal_risk(cfg: dict, model, theta: np.ndarray, seed_parts: tuple) -> float:
    rng = np.random.default_rng(stable_seed(cfg["master_seed"], *seed_parts))
    count = int(cfg["target"]["reveal_samples_per_state"])
    features, labels = generate_samples(cfg, theta, count, rng, balanced=True)
    return float(np.mean(model.predict(features) != labels))


def score_design(cfg: dict, model, center_name: str, center: np.ndarray,
                 fitted: dict, design: dict) -> tuple[list[dict], dict]:
    u_minus = design["u_minus"]
    u_plus = design["u_plus"]
    theta_minus = center + u_minus
    theta_plus = center + u_plus
    risk_minus = reveal_risk(
        cfg, model, theta_minus, ("measurement_outcome", center_name, design["label"], "minus")
    )
    risk_plus = reveal_risk(
        cfg, model, theta_plus, ("measurement_outcome", center_name, design["label"], "plus")
    )
    difference = u_plus - u_minus
    midpoint = 0.5 * (u_plus + u_minus)
    weights = design["Sigma_inverse_per_sample"] @ design["A"] @ difference
    observation_midpoint = (
        fitted["observation_intercept"][design["indices"]] + design["A"] @ midpoint
    )
    rows = []
    for batch_size in cfg["target"]["batch_sizes"]:
        batch_size = int(batch_size)
        replicates = int(cfg["target"]["replicates_per_state"])
        minus = target_batches(
            cfg, model, theta_minus, batch_size, replicates, design["indices"],
            ("measurement_target", center_name, design["label"], batch_size, "minus"),
        )
        plus = target_batches(
            cfg, model, theta_plus, batch_size, replicates, design["indices"],
            ("measurement_target", center_name, design["label"], batch_size, "plus"),
        )
        x_all = np.vstack([minus, plus])
        y_all = np.r_[-np.ones(replicates, dtype=int), np.ones(replicates, dtype=int)]
        predictions = np.where((x_all - observation_midpoint) @ weights >= 0, 1, -1)
        operational = float(np.mean(predictions == y_all))
        split = replicates // 2
        train = np.r_[np.arange(split), replicates + np.arange(split)]
        test = np.r_[np.arange(split, replicates), replicates + np.arange(split, replicates)]
        oracle = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=cfg["master_seed"] % (2**32 - 1)),
        )
        oracle.fit(x_all[train], y_all[train])
        oracle_accuracy = float(np.mean(oracle.predict(x_all[test]) == y_all[test]))
        theoretical = float(
            __import__("scipy").stats.norm.cdf(
                0.5 * np.sqrt(batch_size * design["pair_dQd_per_sample"])
            )
        )
        rows.append(
            {
                "center": center_name,
                "design": design["label"],
                "groups": design["group_key"],
                "cost": design["cost"],
                "batch_size": batch_size,
                "theoretical_accuracy": theoretical,
                "operational_accuracy": operational,
                "oracle_accuracy": oracle_accuracy,
                "pair_dQd_per_sample": design["pair_dQd_per_sample"],
                "risk_minus_revealed": risk_minus,
                "risk_plus_revealed": risk_plus,
            }
        )
    return rows, {
        "theta_minus": theta_minus,
        "theta_plus": theta_plus,
        "risk_minus_revealed": risk_minus,
        "risk_plus_revealed": risk_plus,
    }


def plot_results(frontiers: pd.DataFrame, target: pd.DataFrame, output: Path) -> None:
    centers = list(target["center"].unique())
    fig, axes = plt.subplots(len(centers), 2, figsize=(11.4, 4.1 * len(centers)), constrained_layout=True)
    if len(centers) == 1:
        axes = np.asarray([axes])
    styles = {"risk_directed": "-o", "trace": "--s", "full": ":^"}
    for row_index, center in enumerate(centers):
        subset = frontiers[frontiers["center"] == center]
        axes[row_index, 0].scatter(
            subset["cost"], subset["pair_dQd_per_sample"], alpha=0.48, color="#7f8c8d"
        )
        chosen = target[target["center"] == center].drop_duplicates("design")
        for _, item in chosen.iterrows():
            axes[row_index, 0].scatter(
                item["cost"], item["pair_dQd_per_sample"], s=90, label=item["design"]
            )
        axes[row_index, 0].set_xlabel("Measurement cost (dimensions)")
        axes[row_index, 0].set_ylabel("Candidate worst-pair $D_Q^2$")
        axes[row_index, 0].set_title(f"{center}: calibration design")
        axes[row_index, 0].legend(fontsize=8)
        for design, style in styles.items():
            curve = target[(target["center"] == center) & (target["design"] == design)]
            axes[row_index, 1].plot(
                curve["batch_size"], curve["operational_accuracy"], style, label=f"{design}: auditor"
            )
            axes[row_index, 1].plot(
                curve["batch_size"], curve["theoretical_accuracy"], style,
                alpha=0.45, label=f"{design}: theory"
            )
        axes[row_index, 1].set_xscale("log", base=2)
        axes[row_index, 1].set_ylim(0.48, 1.01)
        axes[row_index, 1].set_xlabel("Unlabelled batch size")
        axes[row_index, 1].set_ylabel("Balanced state accuracy")
        axes[row_index, 1].set_title(f"{center}: sealed target replay")
        axes[row_index, 1].legend(fontsize=7, ncol=2)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_pretarget_frontier(subsets: pd.DataFrame, budget: int, cfg: dict,
                            output: Path) -> None:
    """Plot the calibration-only design frontier even when the target gate aborts."""
    centers = list(subsets["center"].unique())
    fig, axes = plt.subplots(1, len(centers), figsize=(10.8, 4.2), constrained_layout=True)
    if len(centers) == 1:
        axes = [axes]
    gates = cfg["pilot_gates"]
    for axis, center in zip(axes, centers):
        frame = subsets[subsets["center"] == center].copy()
        eligible_mask = (
            frame["optimizer_feasible"].astype(bool)
            & (frame["risk_null_ratio"] <= gates["rich_max_null_ratio"])
            & (frame["optimizer_constraint_error"] <= gates["maximum_optimizer_constraint_error"])
            & (frame["support_slack_min"] >= gates["minimum_support_slack"])
        )
        candidates = frame[eligible_mask & (frame["cost"] <= budget)]
        risk = candidates.sort_values(
            ["pair_dQd_per_sample", "cost"], ascending=[False, True]
        ).iloc[0]
        trace = candidates.sort_values(["trace_Q", "cost"], ascending=[False, True]).iloc[0]
        full = frame[frame["cost"] == 54].iloc[0]
        axis.scatter(frame["cost"], frame["pair_dQd_per_sample"], alpha=0.45,
                     color="#7f8c8d", label="all 63 subsets")
        axis.scatter(risk["cost"], risk["pair_dQd_per_sample"], s=100,
                     color="#2f5597", marker="o", label="Cliff-directed")
        axis.scatter(trace["cost"], trace["pair_dQd_per_sample"], s=100,
                     facecolors="none", edgecolors="#c55a11", linewidths=2,
                     marker="s", label="trace(Q)")
        axis.scatter(full["cost"], full["pair_dQd_per_sample"], s=100,
                     color="#548235", marker="^", label="full 54D")
        axis.axvline(budget, color="#555555", linestyle="--", alpha=0.6)
        axis.set_xlabel("Measurement cost (scalar dimensions)")
        axis.set_ylabel("Optimized worst-pair $D_Q^2$")
        axis.set_title(center)
        axis.legend(fontsize=8)
    fig.suptitle(f"Calibration-only measurement frontier; primary budget = {budget}")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "formal_measurement_design_v1.json").read_text())
    output = ROOT / "results" / cfg["output_tag"]
    figures = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    model, training = train_deployment_model(cfg)
    all_environments = []
    all_subsets = []
    frozen = {}
    gates = {}
    budget = int(cfg["measurement_design"]["budget"])
    full_key = "+".join(FULL_GROUPS)
    for center_spec in cfg["measurement_design"]["centers"]:
        center_name = center_spec["name"]
        center = np.asarray(center_spec["theta_center"], dtype=float)
        local_cfg = copy.deepcopy(cfg)
        local_cfg["calibration"]["theta_center"] = center.tolist()
        environment, store = calibration_panel(local_cfg, model)
        environment.insert(0, "center", center_name)
        all_environments.append(environment)
        fitted = fit_screening_bridge(environment, store, local_cfg)
        rows = evaluate_subsets(fitted, local_cfg, optimize_supporting_pair)
        for row in rows:
            all_subsets.append({"center": center_name, **row})
        risk_row = select_for_budget(rows, budget, local_cfg, "risk_directed")
        trace_row = select_for_budget(rows, budget, local_cfg, "trace")
        full_row = next(row for row in rows if row["group_key"] == full_key)
        if risk_row is None or trace_row is None or not eligible(full_row, local_cfg):
            gates[center_name] = {"all_passed": False, "reason": "no eligible frozen designs"}
            continue
        selected = {
            "risk_directed": design_record("risk_directed", risk_row, fitted, local_cfg),
            "trace": design_record("trace", trace_row, fitted, local_cfg),
            "full": design_record("full", full_row, fitted, local_cfg),
        }
        stability = {}
        for label, design in selected.items():
            stability[label] = quadratic_pair_fold_stability(
                fitted["theta_env"], fitted["risk_values"],
                pair_from_row(design, local_cfg), local_cfg
            )
        gain = float(
            selected["risk_directed"]["pair_dQd_per_sample"]
            / selected["trace"]["pair_dQd_per_sample"]
        )
        metrics = {
            "calibration_risk_range": float(environment["risk"].max() - environment["risk"].min()),
            "risk_surface_r2": fitted["risk_surface_r2"],
            "risk_surface_cv_r2": fitted["risk_surface_cv_r2"],
            "relevant_score_linear_r2": fitted["relevant_score_linear_r2"],
            "full_risk_null_ratio": fitted["full_risk_null_ratio"],
            "dqd_gain_over_trace": gain,
            "risk_directed_pair_stability": stability["risk_directed"]["fold_constraint_max_absolute_error"],
            "trace_pair_stability": stability["trace"]["fold_constraint_max_absolute_error"],
            "full_pair_stability": stability["full"]["fold_constraint_max_absolute_error"],
        }
        thresholds = cfg["pilot_gates"]
        passed = {
            "risk_range": metrics["calibration_risk_range"] >= thresholds["minimum_calibration_risk_range"],
            "risk_surface_fit": metrics["risk_surface_r2"] >= thresholds["minimum_risk_surface_r2"],
            "risk_surface_crossfit": metrics["risk_surface_cv_r2"] >= thresholds["minimum_risk_surface_cv_r2"],
            "relevant_score": metrics["relevant_score_linear_r2"] >= thresholds["minimum_relevant_score_linear_r2"],
            "full_identified": metrics["full_risk_null_ratio"] <= thresholds["rich_max_null_ratio"],
            "risk_gain": gain >= thresholds["minimum_dqd_gain_over_trace"],
            "risk_pair_stability": metrics["risk_directed_pair_stability"] <= thresholds["maximum_fold_constraint_error"],
            "trace_pair_stability": metrics["trace_pair_stability"] <= thresholds["maximum_fold_constraint_error"],
            "full_pair_stability": metrics["full_pair_stability"] <= thresholds["maximum_fold_constraint_error"],
        }
        gates[center_name] = {"metrics": metrics, "gates": passed, "all_passed": bool(all(passed.values()))}
        frozen[center_name] = {"center": center, "fitted": fitted, "selected": selected}
    pretarget = {"centers": gates, "all_passed": bool(gates) and all(item["all_passed"] for item in gates.values())}
    (output / "pretarget_measurement_design_gate.json").write_text(
        json.dumps(serialize(pretarget), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.concat(all_environments, ignore_index=True).to_csv(
        output / "calibration_environment_panel.csv", index=False
    )
    subset_frame = pd.DataFrame([serialize(row) for row in all_subsets])
    subset_frame.to_csv(output / "all_measurement_subsets.csv", index=False)
    plot_pretarget_frontier(
        subset_frame, budget, cfg, figures / "formal_measurement_design_frontier.png"
    )
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
    for center_name, item in frozen.items():
        reveal[center_name] = {}
        for label, design in item["selected"].items():
            rows, outcome = score_design(
                cfg, model, center_name, item["center"], item["fitted"], design
            )
            target_rows.extend(rows)
            reveal[center_name][label] = outcome
    target = pd.DataFrame(target_rows)
    target.to_csv(output / "sealed_target_measurement_curves.csv", index=False)
    (output / "outcome_reveal.json").write_text(
        json.dumps(serialize(reveal), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result_rows = []
    final_gates = {}
    for (center, design), curve in target.groupby(["center", "design"]):
        curve = curve.sort_values("batch_size")
        mae = float(np.mean(np.abs(curve["theoretical_accuracy"] - curve["operational_accuracy"])))
        rho = float(spearmanr(curve["theoretical_accuracy"], curve["operational_accuracy"]).statistic)
        if not np.isfinite(rho):
            rho = 0.0
        empirical_candidates = curve.loc[curve["operational_accuracy"] >= cfg["measurement_design"]["target_accuracy"], "batch_size"]
        empirical_n = int(empirical_candidates.min()) if len(empirical_candidates) else None
        first = curve.iloc[0]
        reveal_n = int(cfg["target"]["reveal_samples_per_state"])
        minus_ci = wilson_interval(int(round(first["risk_minus_revealed"] * reveal_n)), reveal_n)
        plus_ci = wilson_interval(int(round(first["risk_plus_revealed"] * reveal_n)), reveal_n)
        result_rows.append(
            {
                "center": center,
                "design": design,
                "groups": first["groups"],
                "cost": int(first["cost"]),
                "pair_dQd_per_sample": float(first["pair_dQd_per_sample"]),
                "predicted_n90": int(np.ceil(4 * __import__("scipy").stats.norm.ppf(0.9) ** 2 / first["pair_dQd_per_sample"])),
                "empirical_n90": empirical_n,
                "curve_mae": mae,
                "curve_spearman": rho,
                "risk_minus": float(first["risk_minus_revealed"]),
                "risk_plus": float(first["risk_plus_revealed"]),
                "risk_minus_ci95": list(minus_ci),
                "risk_plus_ci95": list(plus_ci),
            }
        )
        final_gates[f"{center}:{design}"] = {
            "curve_mae": mae <= cfg["pilot_gates"]["maximum_curve_mae"],
            "curve_spearman": rho >= cfg["pilot_gates"]["minimum_curve_spearman"],
        }
    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(output / "measurement_design_ledger.csv", index=False)
    checks = {
        "pretarget_all_passed": True,
        "curve_gates": final_gates,
        "all_curve_gates_passed": all(all(item.values()) for item in final_gates.values()),
        "ledger": result_rows,
        "semantics": "measurement-cost and relative-risk auditability; not absolute safety",
    }
    (output / "checks.json").write_text(
        json.dumps(serialize(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(pd.DataFrame([serialize(row) for row in all_subsets]), target,
                 figures / "formal_measurement_design_v1.png")
    print(json.dumps(serialize(checks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
