#!/usr/bin/env python3
"""Round 12A: blind boundary-flux ranking followed by equal-budget repair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsig_official_source_numpy_runtime import (  # noqa: E402
    RUNTIME_KIND,
    TORCHSIG_SOURCE_COMMIT,
    TORCHSIG_TAG,
    install,
)

if "torchsig" not in sys.modules:
    install()

from run_pilot import FEATURE_NAMES, extract_features, generate_iq, stable_seed  # noqa: E402
from run_round10_training_intervention import hash_arrays, path_specs  # noqa: E402
from run_round11_paired_sync import (  # noqa: E402
    confirmed_first_crossing,
    paired_panel,
    serializable,
    true_margin,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze(config_relative: str, output: Path) -> str:
    cfg = json.loads((ROOT / config_relative).read_text(encoding="utf-8"))
    names = [
        config_relative,
        cfg["source_round10_config"],
        "ROUND12A_BLIND_FLUX_REPAIR_PROTOCOL.md",
        "src/run_round12a_blind_flux_repair.py",
        "src/torchsig_official_source_numpy_runtime.py",
        "src/run_round11_paired_sync.py",
        "src/run_round10_training_intervention.py",
        "src/run_pilot.py",
    ]
    content = "\n".join(f"{sha256(ROOT / name)}  {name}" for name in names) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "PRETARGET_RELEASE_SHA256.txt").write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    (output / "PRETARGET_MANIFEST_DIGEST.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


def training_arrays(cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(cfg["training"]["samples_per_class"]) * len(cfg["classes"])
    rng = np.random.default_rng(stable_seed(seed, "round12_base_training"))
    labels = np.arange(count, dtype=int) % len(cfg["classes"])
    rng.shuffle(labels)
    baseline = cfg["round10"]["regimes"]["baseline"]
    theta = rng.uniform(float(baseline["low"]), float(baseline["high"]), size=(count, 3))
    features = np.empty((count, len(FEATURE_NAMES)), dtype=float)
    for index, label in enumerate(labels):
        iq = generate_iq(cfg["classes"][int(label)], theta[index], rng, cfg["signal"])
        features[index] = extract_features(iq)
    return features, labels, theta


def validation_arrays(cfg: dict, smoke: dict) -> tuple[np.ndarray, np.ndarray]:
    count = int(smoke["validation"]["samples_per_class"]) * len(cfg["classes"])
    rng = np.random.default_rng(stable_seed(smoke["master_seed"], "baseline_validation"))
    labels = np.arange(count, dtype=int) % len(cfg["classes"])
    rng.shuffle(labels)
    baseline = cfg["round10"]["regimes"]["baseline"]
    theta = rng.uniform(float(baseline["low"]), float(baseline["high"]), size=(count, 3))
    features = np.empty((count, len(FEATURE_NAMES)), dtype=float)
    for index, label in enumerate(labels):
        iq = generate_iq(cfg["classes"][int(label)], theta[index], rng, cfg["signal"])
        features[index] = extract_features(iq)
    return features, labels


def fit_model(features: np.ndarray, labels: np.ndarray, cfg: dict, seed: int) -> ExtraTreesClassifier:
    model_seed = stable_seed(seed, "round12_paired_model") % (2**32 - 1)
    model = ExtraTreesClassifier(
        n_estimators=int(cfg["training"]["n_estimators"]),
        min_samples_leaf=int(cfg["training"]["min_samples_leaf"]),
        max_features="sqrt",
        random_state=model_seed,
        n_jobs=int(cfg["training"].get("n_jobs", 1)),
    )
    model.fit(features, labels)
    return model


def first_persistent_nonpositive(values: np.ndarray, anchor: int, confirmation: int) -> int:
    for time_index in range(anchor + 1, len(values) - confirmation + 1):
        if bool(np.all(values[time_index : time_index + confirmation] <= 0)):
            return time_index
    return -1


def rank_unlabeled(
    probabilities: dict[str, np.ndarray],
    anchor: int,
    confirmation: int,
    method: str,
    seed: int,
) -> pd.DataFrame:
    """Rank unique path/sample trajectories using probabilities only."""
    if method not in {"random", "uncertainty", "blind_flux"}:
        raise ValueError(f"unknown blind selector: {method}")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for path, array in probabilities.items():
        for sample in range(array.shape[1]):
            trajectory = array[:, sample]
            anchor_class = int(np.argmax(trajectory[anchor]))
            anchor_probability = trajectory[:, anchor_class]
            other = trajectory.copy()
            other[:, anchor_class] = -np.inf
            proxy_margin = anchor_probability - np.max(other, axis=1)
            ordered = np.sort(trajectory, axis=1)
            uncertainty_gap = ordered[:, -1] - ordered[:, -2]
            proxy_first = first_persistent_nonpositive(proxy_margin, anchor, confirmation)
            if proxy_first >= 0:
                repair_time = proxy_first
                persistence = float(np.mean(proxy_margin[proxy_first:] <= 0))
                incident = 1.0
            else:
                repair_time = int(anchor + 1 + np.argmin(proxy_margin[anchor + 1 :]))
                persistence = 0.0
                incident = 0.0
            descent = float(max(0.0, proxy_margin[anchor] - np.min(proxy_margin[anchor + 1 :])))
            negative_area = float(np.mean(np.maximum(0.0, -proxy_margin[anchor + 1 :])))
            flux_score = 4.0 * incident + 2.0 * persistence + descent + negative_area
            uncertainty_time = int(anchor + 1 + np.argmin(uncertainty_gap[anchor + 1 :]))
            if method == "blind_flux":
                score = flux_score
                chosen_time = repair_time
            elif method == "uncertainty":
                score = -float(uncertainty_gap[uncertainty_time])
                chosen_time = uncertainty_time
            else:
                score = float(rng.random())
                chosen_time = int(rng.integers(anchor + 1, len(trajectory)))
            rows.append(
                {
                    "path": path,
                    "sample_index": sample,
                    "chosen_time": chosen_time,
                    "score": score,
                    "anchor_predicted_class": anchor_class,
                    "proxy_first_crossing_time": proxy_first,
                    "proxy_persistence": persistence,
                    "proxy_margin_descent": descent,
                    "proxy_negative_area": negative_area,
                    "minimum_uncertainty_gap": float(np.min(uncertainty_gap[anchor + 1 :])),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["score", "path", "sample_index"], ascending=[False, True, True]
    )
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def attach_hidden_diagnostics(
    ranking: pd.DataFrame,
    panels: dict[str, dict],
    baseline_probabilities: dict[str, np.ndarray],
    anchor: int,
    confirmation: int,
) -> pd.DataFrame:
    diagnostics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path, panel in panels.items():
        margin = np.vstack(
            [true_margin(item, panel["labels"]) for item in baseline_probabilities[path]]
        )
        first = confirmed_first_crossing(margin, anchor, confirmation)
        diagnostics[path] = (panel["labels"], first)
    result = ranking.copy()
    result["class_index"] = [
        int(diagnostics[row.path][0][int(row.sample_index)]) for row in result.itertuples()
    ]
    result["true_first_crossing_time"] = [
        int(diagnostics[row.path][1][int(row.sample_index)]) for row in result.itertuples()
    ]
    result["true_incident_crossing"] = result["true_first_crossing_time"] >= 0
    return result


def selected_arrays(
    ranking: pd.DataFrame, panels: dict[str, dict], budget: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    selected = ranking.head(budget).copy()
    features = np.vstack(
        [
            panels[row.path]["features"][int(row.chosen_time), int(row.sample_index)]
            for row in selected.itertuples()
        ]
    )
    labels = selected["class_index"].to_numpy(dtype=int)
    return features, labels, selected


def oracle_u_arrays(cfg: dict, smoke: dict, budget: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(stable_seed(smoke["master_seed"], "oracle_u_repair"))
    labels = np.arange(budget, dtype=int) % len(cfg["classes"])
    rng.shuffle(labels)
    paths = path_specs(cfg)
    assignments = rng.integers(0, len(paths), size=budget)
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    jitter = float(smoke["repair"]["oracle_tube_jitter_std"])
    theta = np.empty((budget, 3), dtype=float)
    path_names: list[str] = []
    for index, assignment in enumerate(assignments):
        path = paths[int(assignment)]
        direction = path["direction"] / max(np.linalg.norm(path["direction"]), 1e-12)
        scalar = rng.uniform(path["start_scalar"], path["end_scalar"])
        theta[index] = center + scalar * direction + rng.normal(0.0, jitter, size=3)
        path_names.append(path["name"])
    theta = np.clip(
        theta,
        float(smoke["repair"]["oracle_clip_low"]),
        float(smoke["repair"]["oracle_clip_high"]),
    )
    features = np.empty((budget, len(FEATURE_NAMES)), dtype=float)
    for index, label in enumerate(labels):
        iq = generate_iq(cfg["classes"][int(label)], theta[index], rng, cfg["signal"])
        features[index] = extract_features(iq)
    ledger = pd.DataFrame(
        {
            "path": path_names,
            "sample_index": np.arange(budget),
            "chosen_time": -1,
            "score": np.nan,
            "rank": np.arange(1, budget + 1),
            "class_index": labels,
            "true_incident_crossing": np.nan,
            "theta_noise": theta[:, 0],
            "theta_phase": theta[:, 1],
            "theta_nonlinearity": theta[:, 2],
        }
    )
    return features, labels, ledger


def replace_same_class(
    base_features: np.ndarray,
    base_labels: np.ndarray,
    repair_features: np.ndarray,
    repair_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = base_features.copy()
    labels = base_labels.copy()
    replacement: list[int] = []
    rng = np.random.default_rng(stable_seed(seed, "round12_replacement_indices"))
    available = {
        label: list(rng.permutation(np.flatnonzero(base_labels == label)))
        for label in np.unique(base_labels)
    }
    used = {label: 0 for label in available}
    for feature, label in zip(repair_features, repair_labels):
        label = int(label)
        index = int(available[label][used[label]])
        used[label] += 1
        features[index] = feature
        labels[index] = label
        replacement.append(index)
    return features, labels, np.asarray(replacement, dtype=int)


def light_path_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    anchor: int,
    confirmation: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    margin = np.vstack([true_margin(item, labels) for item in probabilities])
    risk = np.mean(margin <= 0, axis=1)
    first = confirmed_first_crossing(margin, anchor, confirmation)
    incident = first >= 0
    persistence = float(np.mean(margin[-1, incident] <= 0)) if np.any(incident) else 0.0
    return (
        {
            "anchor_risk": float(risk[anchor]),
            "end_risk": float(risk[-1]),
            "risk_area": float(np.mean(risk[anchor:])),
            "net_risk_change": float(risk[-1] - risk[anchor]),
            "incident_crossing_fraction_all": float(np.mean(incident)),
            "incident_persistence": persistence,
        },
        risk,
        margin,
    )


def confirmed_cliff_time(risk: np.ndarray, boundary: float, anchor: int, confirmation: int) -> int:
    for time_index in range(anchor, len(risk) - confirmation + 1):
        if bool(np.all(risk[time_index : time_index + confirmation] >= boundary)):
            return time_index
    return -1


def make_figure(path_frame: pd.DataFrame, diagnostics: pd.DataFrame, output: Path) -> None:
    colors = {
        "baseline": "#4c566a",
        "random": "#a3be8c",
        "uncertainty": "#ebcb8b",
        "blind_flux": "#5e81ac",
        "oracle_u": "#b48ead",
    }
    order = list(colors)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    mean_path = path_frame.groupby("arm", as_index=False)[
        ["end_risk", "incident_crossing_fraction_all"]
    ].mean().set_index("arm")
    axes[0].bar(order, [mean_path.loc[item, "end_risk"] for item in order], color=[colors[x] for x in order])
    axes[0].set_title("Fresh-stream terminal risk")
    axes[1].bar(
        order,
        [mean_path.loc[item, "incident_crossing_fraction_all"] for item in order],
        color=[colors[x] for x in order],
    )
    axes[1].set_title("Fresh incident crossing")
    selector = diagnostics.groupby("arm")["true_incident_crossing"].mean()
    selector_order = ["random", "uncertainty", "blind_flux"]
    axes[2].bar(selector_order, [selector[item] for item in selector_order], color=[colors[x] for x in selector_order])
    axes[2].set_title("Selected true-incident precision")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=28)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round12a_blind_flux_repair_smoke.json")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--expected-pretarget-sha")
    args = parser.parse_args()
    smoke = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / "results" / smoke["output_tag"]
    pretarget_digest = freeze(args.config, output)
    if args.freeze_only:
        print(pretarget_digest)
        return
    if args.expected_pretarget_sha is None or args.expected_pretarget_sha != pretarget_digest:
        raise SystemExit(
            f"pretarget mismatch or missing: observed={pretarget_digest} "
            f"expected={args.expected_pretarget_sha}"
        )

    cfg = copy.deepcopy(json.loads((ROOT / smoke["source_round10_config"]).read_text(encoding="utf-8")))
    cfg["training"].update(smoke["training_override"])
    seed = int(smoke["replicate_seed"])
    base_features, base_labels, base_theta = training_arrays(cfg, seed)
    validation_features, validation_labels = validation_arrays(cfg, smoke)
    base_model = fit_model(base_features, base_labels, cfg, seed)
    anchor = int(cfg["deployment"]["pre_plateau_windows"])
    confirmation = int(smoke["repair"]["confirmation_windows"])
    budget = int(round(len(base_labels) * float(smoke["repair"]["fraction"])))

    acquisition_spec = {
        "master_seed": int(smoke["acquisition"]["master_seed"]),
        "paired_deployment": {"samples_per_path": int(smoke["acquisition"]["samples_per_path"])},
    }
    acquisition_panels, acquisition_hash = paired_panel(cfg, acquisition_spec)
    acquisition_probabilities = {
        path: np.stack([base_model.predict_proba(item) for item in panel["features"]], axis=0)
        for path, panel in acquisition_panels.items()
    }

    repair_arrays: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    diagnostic_frames: list[pd.DataFrame] = []
    for method in ["random", "uncertainty", "blind_flux"]:
        ranking = rank_unlabeled(
            acquisition_probabilities,
            anchor,
            confirmation,
            method,
            stable_seed(smoke["master_seed"], "selector", method),
        )
        ranking = attach_hidden_diagnostics(
            ranking, acquisition_panels, acquisition_probabilities, anchor, confirmation
        )
        features, labels, selected = selected_arrays(ranking, acquisition_panels, budget)
        selected.insert(0, "arm", method)
        repair_arrays[method] = (features, labels, selected)
        diagnostic_frames.append(selected)
    oracle = oracle_u_arrays(cfg, smoke, budget)
    oracle_ledger = oracle[2].copy()
    oracle_ledger.insert(0, "arm", "oracle_u")
    repair_arrays["oracle_u"] = (oracle[0], oracle[1], oracle_ledger)
    diagnostic_frames.append(oracle_ledger)

    models = {"baseline": base_model}
    training_rows = [
        {
            "arm": "baseline",
            "training_count": len(base_labels),
            "repair_count": 0,
            "model_random_state": int(base_model.random_state),
            "validation_accuracy": float(accuracy_score(validation_labels, base_model.predict(validation_features))),
            "training_hash": hash_arrays([("features", base_features), ("labels", base_labels)]),
            "replacement_indices_hash": "none",
        }
    ]
    for arm in ["random", "uncertainty", "blind_flux", "oracle_u"]:
        repair_features, repair_labels, _ = repair_arrays[arm]
        features, labels, replacement = replace_same_class(
            base_features, base_labels, repair_features, repair_labels, seed
        )
        model = fit_model(features, labels, cfg, seed)
        models[arm] = model
        training_rows.append(
            {
                "arm": arm,
                "training_count": len(labels),
                "repair_count": len(repair_labels),
                "model_random_state": int(model.random_state),
                "validation_accuracy": float(accuracy_score(validation_labels, model.predict(validation_features))),
                "training_hash": hash_arrays([("features", features), ("labels", labels)]),
                "replacement_indices_hash": hashlib.sha256(replacement.tobytes()).hexdigest(),
            }
        )

    evaluation_spec = {
        "master_seed": int(smoke["evaluation"]["master_seed"]),
        "paired_deployment": {"samples_per_path": int(smoke["evaluation"]["samples_per_path"])},
    }
    evaluation_panels, evaluation_hash = paired_panel(cfg, evaluation_spec)
    path_rows: list[dict] = []
    risk_store: dict[tuple[str, str], np.ndarray] = {}
    for arm, model in models.items():
        for path, panel in evaluation_panels.items():
            probabilities = np.stack(
                [model.predict_proba(item) for item in panel["features"]], axis=0
            )
            metrics, risk, _ = light_path_metrics(
                probabilities, panel["labels"], anchor, confirmation
            )
            metrics.update({"arm": arm, "path": path})
            path_rows.append(metrics)
            risk_store[(arm, path)] = risk
    path_frame = pd.DataFrame(path_rows)
    for path in evaluation_panels:
        boundary = float(risk_store[("baseline", path)][anchor] + smoke["evaluation"]["relative_cliff_margin"])
        for arm in smoke["arms"]:
            mask = (path_frame["arm"] == arm) & (path_frame["path"] == path)
            cliff_time = confirmed_cliff_time(risk_store[(arm, path)], boundary, anchor, confirmation)
            path_frame.loc[mask, "common_relative_boundary"] = boundary
            path_frame.loc[mask, "common_cliff_time"] = cliff_time
            path_frame.loc[mask, "common_cliff_crossed"] = cliff_time >= 0

    mean_path = path_frame.groupby("arm")[[
        "anchor_risk", "end_risk", "risk_area", "incident_crossing_fraction_all"
    ]].mean()
    baseline = mean_path.loc["baseline"]
    effects = pd.DataFrame(
        [
            {
                "arm": arm,
                "mean_end_risk_reduction": float(baseline.end_risk - mean_path.loc[arm].end_risk),
                "mean_incident_crossing_reduction": float(
                    baseline.incident_crossing_fraction_all
                    - mean_path.loc[arm].incident_crossing_fraction_all
                ),
                "mean_risk_area_reduction": float(baseline.risk_area - mean_path.loc[arm].risk_area),
                "mean_anchor_risk_change": float(mean_path.loc[arm].anchor_risk - baseline.anchor_risk),
            }
            for arm in ["random", "uncertainty", "blind_flux", "oracle_u"]
        ]
    )
    effect_index = effects.set_index("arm")
    selections = pd.concat(diagnostic_frames, ignore_index=True, sort=False)
    selector_precision = selections[
        selections["arm"].isin(["random", "uncertainty", "blind_flux"])
    ].groupby("arm")["true_incident_crossing"].mean()
    training_frame = pd.DataFrame(training_rows)
    validation = training_frame.set_index("arm")["validation_accuracy"]
    baseline_cliff_fraction = float(
        path_frame[path_frame["arm"] == "baseline"]["common_cliff_crossed"].mean()
    )
    blind = effect_index.loc["blind_flux"]
    random_effect = effect_index.loc["random"]
    uncertainty_effect = effect_index.loc["uncertainty"]
    oracle_effect = effect_index.loc["oracle_u"]
    values = {
        "training_budget": budget,
        "all_training_counts_equal": training_frame["training_count"].nunique() == 1,
        "all_model_random_states_equal": training_frame["model_random_state"].nunique() == 1,
        "acquisition_evaluation_streams_disjoint": acquisition_hash != evaluation_hash,
        "baseline_common_cliff_fraction": baseline_cliff_fraction,
        "random_selector_incident_precision": float(selector_precision["random"]),
        "uncertainty_selector_incident_precision": float(selector_precision["uncertainty"]),
        "blind_flux_selector_incident_precision": float(selector_precision["blind_flux"]),
        "blind_flux_selector_precision_advantage_over_random": float(
            selector_precision["blind_flux"] - selector_precision["random"]
        ),
        "blind_flux_mean_end_risk_reduction": float(blind.mean_end_risk_reduction),
        "blind_flux_mean_incident_reduction": float(blind.mean_incident_crossing_reduction),
        "blind_flux_end_risk_advantage_over_random": float(
            blind.mean_end_risk_reduction - random_effect.mean_end_risk_reduction
        ),
        "blind_flux_end_risk_advantage_over_uncertainty": float(
            blind.mean_end_risk_reduction - uncertainty_effect.mean_end_risk_reduction
        ),
        "blind_flux_validation_accuracy_loss": float(validation["baseline"] - validation["blind_flux"]),
        "blind_flux_mean_anchor_risk_increase": float(blind.mean_anchor_risk_change),
        "oracle_u_mean_end_risk_reduction": float(oracle_effect.mean_end_risk_reduction),
    }
    gates = smoke["smoke_gates"]
    checks = {
        "equal_training_counts": bool(values["all_training_counts_equal"]),
        "paired_model_random_state": bool(values["all_model_random_states_equal"]),
        "disjoint_acquisition_evaluation_streams": bool(values["acquisition_evaluation_streams_disjoint"]),
        "baseline_cliff_present": values["baseline_common_cliff_fraction"]
        >= gates["minimum_baseline_common_cliff_fraction"],
        "blind_selector_has_true_incident_signal": values["blind_flux_selector_incident_precision"]
        >= gates["minimum_blind_flux_selector_incident_precision"],
        "blind_selector_beats_random": values["blind_flux_selector_precision_advantage_over_random"]
        >= gates["minimum_blind_flux_selector_precision_advantage_over_random"],
        "blind_reduces_end_risk": values["blind_flux_mean_end_risk_reduction"]
        >= gates["minimum_blind_flux_mean_end_risk_reduction"],
        "blind_reduces_incident_crossing": values["blind_flux_mean_incident_reduction"]
        >= gates["minimum_blind_flux_mean_incident_reduction"],
        "blind_beats_random_end_risk": values["blind_flux_end_risk_advantage_over_random"]
        >= gates["minimum_blind_flux_end_risk_advantage_over_random"],
        "blind_beats_uncertainty_end_risk": values["blind_flux_end_risk_advantage_over_uncertainty"]
        >= gates["minimum_blind_flux_end_risk_advantage_over_uncertainty"],
        "blind_preserves_baseline_validation": values["blind_flux_validation_accuracy_loss"]
        <= gates["maximum_blind_flux_validation_accuracy_loss"],
        "blind_does_not_raise_anchor_risk": values["blind_flux_mean_anchor_risk_increase"]
        <= gates["maximum_blind_flux_mean_anchor_risk_increase"],
        "oracle_u_upper_bound_is_active": values["oracle_u_mean_end_risk_reduction"]
        >= gates["minimum_oracle_u_mean_end_risk_reduction"],
    }
    passed = int(sum(checks.values()))
    decision = "ADVANCE_TO_FRESH_MULTI_SEED_PILOT" if passed == len(checks) else "SMOKE_STOP_DIAGNOSE"

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", smoke)
    training_frame.to_csv(output / "training_summary.csv", index=False)
    selections.to_csv(output / "selected_repair_examples.csv", index=False)
    path_frame.to_csv(output / "path_summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    summary = {
        "pretarget_manifest_digest": pretarget_digest,
        "runtime": {
            "kind": RUNTIME_KIND,
            "python": platform.python_version(),
            "torchsig_tag": TORCHSIG_TAG,
            "torchsig_source_commit": TORCHSIG_SOURCE_COMMIT,
            "standard_torchsig_package_runtime": False,
        },
        "base_training_sha256": hash_arrays(
            [("features", base_features), ("labels", base_labels), ("theta", base_theta)]
        ),
        "acquisition_panel_sha256": acquisition_hash,
        "evaluation_panel_sha256": evaluation_hash,
        "checks": {"values": values, "checks": checks, "passed": passed, "total": len(checks)},
        "decision": decision,
        "claim_status": "single-seed smoke only; no manuscript claim",
    }
    write_json(output / "summary.json", summary)
    make_figure(
        path_frame,
        selections[selections["arm"].isin(["random", "uncertainty", "blind_flux"])],
        ROOT / "figures" / smoke["output_tag"] / f"{smoke['output_tag']}.png",
    )
    print(json.dumps(serializable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
