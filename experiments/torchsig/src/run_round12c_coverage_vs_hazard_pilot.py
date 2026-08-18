#!/usr/bin/env python3
"""Round 12C: five-seed coverage-versus-hazard repair pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

from run_pilot import stable_seed  # noqa: E402
from run_round10_training_intervention import hash_arrays  # noqa: E402
from run_round11_paired_sync import paired_panel, true_margin  # noqa: E402
from run_round12a_common_query_repair_v4 import (  # noqa: E402
    confirmed_cliff_time,
    fit_model,
    light_path_metrics,
    rank_unlabeled,
    replace_same_class,
    training_arrays,
    validation_arrays,
)


REPAIR_ARMS = [
    "random_unstratified",
    "hazard_concentrated",
    "coverage_random",
    "coverage_hazard",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def freeze(config_relative: str, output: Path) -> str:
    cfg = json.loads((ROOT / config_relative).read_text(encoding="utf-8"))
    names = [
        config_relative,
        cfg["source_round10_config"],
        "ROUND12C_COVERAGE_VS_HAZARD_PILOT_PROTOCOL.md",
        "src/run_round12c_coverage_vs_hazard_pilot.py",
        "src/run_round12a_common_query_repair_v4.py",
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


def query_by_path_and_prediction(
    ranking: pd.DataFrame, budget: int, class_count: int, path_names: list[str]
) -> pd.DataFrame:
    strata = class_count * len(path_names)
    if budget % strata != 0:
        raise ValueError("query budget must divide path x predicted-class strata")
    quota = budget // strata
    groups = []
    for path in path_names:
        for predicted_class in range(class_count):
            group = ranking[
                (ranking["path"] == path)
                & (ranking["anchor_predicted_class"] == predicted_class)
            ].head(quota)
            if len(group) != quota:
                raise RuntimeError(
                    f"insufficient query candidates for path={path}, "
                    f"predicted_class={predicted_class}"
                )
            groups.append(group)
    result = pd.concat(groups, ignore_index=True)
    return result.sort_values(["path", "anchor_predicted_class", "score"], ascending=[True, True, False]).reset_index(drop=True)


def local_band_hazard(
    margin: np.ndarray, start: int, end: int, confirmation: int
) -> tuple[float, int, bool, float, float, float]:
    if not (1 <= start <= end < len(margin)):
        raise ValueError("invalid local time band")
    crossing = -1
    for time_index in range(start, end - confirmation + 2):
        if margin[time_index - 1] > 0 and bool(
            np.all(margin[time_index : time_index + confirmation] <= 0)
        ):
            crossing = time_index
            break
    if crossing >= 0:
        chosen_time = crossing
        persistence = float(np.mean(margin[crossing : end + 1] <= 0))
        incident = True
    else:
        chosen_time = int(start + np.argmin(margin[start : end + 1]))
        persistence = 0.0
        incident = False
    descent = float(max(0.0, margin[start - 1] - np.min(margin[start : end + 1])))
    negative_area = float(np.mean(np.maximum(0.0, -margin[start : end + 1])))
    score = 4.0 * float(incident) + 2.0 * persistence + descent + negative_area
    return score, chosen_time, incident, persistence, descent, negative_area


def candidate_bands(
    queried: pd.DataFrame,
    panels: dict[str, dict],
    probabilities: dict[str, np.ndarray],
    bands: dict[str, tuple[int, int]],
    confirmation: int,
    seed: int,
) -> pd.DataFrame:
    margins = {
        path: np.vstack(
            [true_margin(item, panels[path]["labels"]) for item in probabilities[path]]
        )
        for path in panels
    }
    rng = np.random.default_rng(seed)
    rows = []
    for query in queried.itertuples():
        sample = int(query.sample_index)
        label = int(panels[query.path]["labels"][sample])
        for phase, (start, end) in bands.items():
            score, chosen, incident, persistence, descent, negative_area = local_band_hazard(
                margins[query.path][:, sample], start, end, confirmation
            )
            rows.append(
                {
                    "path": query.path,
                    "sample_index": sample,
                    "anchor_predicted_class": int(query.anchor_predicted_class),
                    "class_index": label,
                    "phase": phase,
                    "band_start": start,
                    "band_end": end,
                    "chosen_time": chosen,
                    "local_hazard_score": score,
                    "local_incident": incident,
                    "local_persistence": persistence,
                    "local_margin_descent": descent,
                    "local_negative_area": negative_area,
                    "random_score": float(rng.random()),
                    "cell": f"class{label}|{query.path}|{phase}",
                }
            )
    return pd.DataFrame(rows)


def select_global(candidates: pd.DataFrame, budget: int, score_column: str) -> pd.DataFrame:
    ordered = candidates.sort_values(
        [score_column, "path", "sample_index", "phase"],
        ascending=[False, True, True, True],
    )
    selected = ordered.drop_duplicates(["path", "sample_index"], keep="first").head(budget)
    if len(selected) != budget:
        raise RuntimeError("insufficient unique trajectories for global selection")
    return selected.reset_index(drop=True)


def select_coverage(
    candidates: pd.DataFrame,
    class_count: int,
    path_names: list[str],
    phases: list[str],
    score_column: str,
) -> pd.DataFrame:
    selected = []
    used: set[tuple[str, int]] = set()
    for label in range(class_count):
        for path in path_names:
            for phase in phases:
                group = candidates[
                    (candidates["class_index"] == label)
                    & (candidates["path"] == path)
                    & (candidates["phase"] == phase)
                ].sort_values(
                    [score_column, "sample_index"], ascending=[False, True]
                )
                available = group[
                    ~group.apply(
                        lambda row: (str(row["path"]), int(row["sample_index"])) in used,
                        axis=1,
                    )
                ]
                if available.empty:
                    raise RuntimeError(f"unpopulated unique coverage cell class{label}|{path}|{phase}")
                row = available.iloc[0]
                selected.append(row)
                used.add((str(row["path"]), int(row["sample_index"])))
    return pd.DataFrame(selected).reset_index(drop=True)


def selection_diagnostics(selected: pd.DataFrame, total_cells: int) -> dict:
    counts = selected["cell"].value_counts()
    probabilities = counts.to_numpy(dtype=float) / max(float(len(selected)), 1.0)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / math.log(total_cells) if total_cells > 1 else 1.0
    return {
        "repair_count": int(len(selected)),
        "occupied_cells": int(len(counts)),
        "coverage_fraction": float(len(counts) / total_cells),
        "normalized_cell_entropy": normalized_entropy,
        "mean_local_hazard_score": float(selected["local_hazard_score"].mean()),
        "true_local_incident_precision": float(selected["local_incident"].mean()),
        "unique_source_trajectories": int(
            selected[["path", "sample_index"]].drop_duplicates().shape[0]
        ),
    }


def extract_selected_arrays(
    selected: pd.DataFrame, panels: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack(
        [
            panels[row.path]["features"][int(row.chosen_time), int(row.sample_index)]
            for row in selected.itertuples()
        ]
    )
    return features, selected["class_index"].to_numpy(dtype=int)


def cluster_bootstrap_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def compute_seed_contrasts(effects: pd.DataFrame) -> pd.DataFrame:
    pivot = effects.pivot(index="replicate_seed", columns="arm", values="mean_end_risk_reduction")
    frame = pd.DataFrame(index=pivot.index)
    frame["coverage_random_minus_hazard_concentrated"] = (
        pivot["coverage_random"] - pivot["hazard_concentrated"]
    )
    frame["coverage_gain_at_high_hazard"] = (
        pivot["coverage_hazard"] - pivot["hazard_concentrated"]
    )
    frame["hazard_gain_at_high_coverage"] = (
        pivot["coverage_hazard"] - pivot["coverage_random"]
    )
    frame["coverage_minus_hazard_contribution"] = (
        frame["coverage_gain_at_high_hazard"] - frame["hazard_gain_at_high_coverage"]
    )
    frame["coverage_hazard_minus_best_single"] = pivot["coverage_hazard"] - np.maximum(
        pivot["coverage_random"], pivot["hazard_concentrated"]
    )
    return frame.reset_index()


def write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    content = "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n"
    (output / "SHA256SUMS.txt").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round12c_coverage_vs_hazard_pilot.json")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--expected-pretarget-sha")
    args = parser.parse_args()

    pilot = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / "results" / pilot["output_tag"]
    pretarget_digest = freeze(args.config, output)
    if args.freeze_only:
        print(pretarget_digest)
        return
    if args.expected_pretarget_sha != pretarget_digest:
        raise SystemExit(
            f"pretarget mismatch or missing: observed={pretarget_digest} "
            f"expected={args.expected_pretarget_sha}"
        )

    cfg = copy.deepcopy(
        json.loads((ROOT / pilot["source_round10_config"]).read_text(encoding="utf-8"))
    )
    cfg["training"].update(pilot["training_override"])
    anchor = int(cfg["deployment"]["pre_plateau_windows"])
    confirmation = int(pilot["repair"]["confirmation_windows"])
    class_count = len(cfg["classes"])
    path_names = [item["name"] for item in cfg["deployment"]["paths"]]
    bands = {
        "early": tuple(int(value) for value in pilot["repair"]["early_band"]),
        "late": tuple(int(value) for value in pilot["repair"]["late_band"]),
    }
    total_cells = class_count * len(path_names) * len(bands)

    prepared = []
    query_frames = []
    candidate_frames = []
    selection_frames = []
    selection_summary_rows = []
    acquisition_hashes = []

    # Complete acquisition and cell-feasibility checks before generating any evaluation panel.
    for replicate_seed in pilot["replicate_seeds"]:
        replicate_seed = int(replicate_seed)
        base_features, base_labels, base_theta = training_arrays(cfg, replicate_seed)
        base_model = fit_model(base_features, base_labels, cfg, replicate_seed)
        budget = int(round(len(base_labels) * float(pilot["repair"]["fraction"])))
        query_budget = int(round(len(base_labels) * float(pilot["repair"]["query_fraction"])))
        if budget != total_cells:
            raise RuntimeError("repair budget must equal the frozen 16-cell design")

        acquisition_seed = stable_seed(replicate_seed, "round12c_acquisition")
        acquisition_spec = {
            "master_seed": acquisition_seed,
            "paired_deployment": {
                "samples_per_path": int(pilot["acquisition"]["samples_per_path"])
            },
        }
        panels, acquisition_hash = paired_panel(cfg, acquisition_spec)
        acquisition_hashes.append(acquisition_hash)
        probabilities = {
            path: np.stack(
                [base_model.predict_proba(item) for item in panel["features"]], axis=0
            )
            for path, panel in panels.items()
        }
        ranking = rank_unlabeled(
            probabilities,
            anchor,
            confirmation,
            "random",
            stable_seed(replicate_seed, "round12c_common_query_ranking"),
        )
        queried = query_by_path_and_prediction(
            ranking, query_budget, class_count, path_names
        )
        queried.insert(0, "replicate_seed", replicate_seed)
        candidates = candidate_bands(
            queried,
            panels,
            probabilities,
            bands,
            confirmation,
            stable_seed(replicate_seed, "round12c_candidate_random_scores"),
        )
        candidates.insert(0, "replicate_seed", replicate_seed)
        query_frames.append(queried)
        candidate_frames.append(candidates)

        try:
            selections = {
                "random_unstratified": select_global(candidates, budget, "random_score"),
                "hazard_concentrated": select_global(
                    candidates, budget, "local_hazard_score"
                ),
                "coverage_random": select_coverage(
                    candidates,
                    class_count,
                    path_names,
                    list(bands),
                    "random_score",
                ),
                "coverage_hazard": select_coverage(
                    candidates,
                    class_count,
                    path_names,
                    list(bands),
                    "local_hazard_score",
                ),
            }
        except RuntimeError as error:
            output.mkdir(parents=True, exist_ok=True)
            pd.concat(query_frames, ignore_index=True).to_csv(
                output / "queried_trajectories.csv", index=False
            )
            pd.concat(candidate_frames, ignore_index=True).to_csv(
                output / "candidate_bands.csv", index=False
            )
            abort = {
                "pretarget_manifest_digest": pretarget_digest,
                "decision": "PRETARGET_ACQUISITION_ABORT",
                "replicate_seed": replicate_seed,
                "reason": str(error),
                "evaluation_generated": False,
            }
            write_json(output / "ABORT.json", abort)
            write_checksums(output)
            print(json.dumps(abort, indent=2, sort_keys=True))
            return

        repair_arrays = {}
        for arm, selected in selections.items():
            selected = selected.copy()
            selected.insert(0, "arm", arm)
            selection_frames.append(selected)
            diagnostics = selection_diagnostics(selected, total_cells)
            diagnostics.update({"replicate_seed": replicate_seed, "arm": arm})
            selection_summary_rows.append(diagnostics)
            repair_arrays[arm] = extract_selected_arrays(selected, panels)
        prepared.append(
            {
                "replicate_seed": replicate_seed,
                "base_features": base_features,
                "base_labels": base_labels,
                "base_theta": base_theta,
                "base_model": base_model,
                "repair_arrays": repair_arrays,
                "acquisition_hash": acquisition_hash,
            }
        )

    training_rows = []
    path_rows = []
    path_effect_rows = []
    evaluation_hashes = []
    for item in prepared:
        replicate_seed = item["replicate_seed"]
        base_features = item["base_features"]
        base_labels = item["base_labels"]
        models = {"baseline": item["base_model"]}
        validation_spec = {
            "master_seed": stable_seed(replicate_seed, "round12c_validation"),
            "validation": pilot["validation"],
        }
        validation_features, validation_labels = validation_arrays(cfg, validation_spec)
        baseline_accuracy = float(
            accuracy_score(validation_labels, models["baseline"].predict(validation_features))
        )
        training_rows.append(
            {
                "replicate_seed": replicate_seed,
                "arm": "baseline",
                "training_count": len(base_labels),
                "repair_count": 0,
                "model_random_state": int(models["baseline"].random_state),
                "validation_accuracy": baseline_accuracy,
                "validation_accuracy_loss": 0.0,
                "training_hash": hash_arrays(
                    [("features", base_features), ("labels", base_labels)]
                ),
                "replacement_indices_hash": "none",
            }
        )
        for arm in REPAIR_ARMS:
            repair_features, repair_labels = item["repair_arrays"][arm]
            features, labels, replacement = replace_same_class(
                base_features,
                base_labels,
                repair_features,
                repair_labels,
                replicate_seed,
            )
            model = fit_model(features, labels, cfg, replicate_seed)
            models[arm] = model
            accuracy = float(accuracy_score(validation_labels, model.predict(validation_features)))
            training_rows.append(
                {
                    "replicate_seed": replicate_seed,
                    "arm": arm,
                    "training_count": len(labels),
                    "repair_count": len(repair_labels),
                    "model_random_state": int(model.random_state),
                    "validation_accuracy": accuracy,
                    "validation_accuracy_loss": baseline_accuracy - accuracy,
                    "training_hash": hash_arrays(
                        [("features", features), ("labels", labels)]
                    ),
                    "replacement_indices_hash": hashlib.sha256(
                        replacement.tobytes()
                    ).hexdigest(),
                }
            )

        evaluation_seed = stable_seed(replicate_seed, "round12c_evaluation")
        evaluation_spec = {
            "master_seed": evaluation_seed,
            "paired_deployment": {
                "samples_per_path": int(pilot["evaluation"]["samples_per_path"])
            },
        }
        panels, evaluation_hash = paired_panel(cfg, evaluation_spec)
        evaluation_hashes.append(evaluation_hash)
        risk_store = {}
        seed_rows = []
        for arm, model in models.items():
            for path, panel in panels.items():
                probabilities = np.stack(
                    [model.predict_proba(features) for features in panel["features"]], axis=0
                )
                metrics, risk, _ = light_path_metrics(
                    probabilities, panel["labels"], anchor, confirmation
                )
                metrics.update(
                    {"replicate_seed": replicate_seed, "arm": arm, "path": path}
                )
                seed_rows.append(metrics)
                risk_store[(arm, path)] = risk
        seed_frame = pd.DataFrame(seed_rows)
        for path in path_names:
            boundary = float(
                risk_store[("baseline", path)][anchor]
                + pilot["evaluation"]["relative_cliff_margin"]
            )
            for arm in pilot["arms"]:
                mask = (seed_frame["arm"] == arm) & (seed_frame["path"] == path)
                cliff_time = confirmed_cliff_time(
                    risk_store[(arm, path)], boundary, anchor, confirmation
                )
                seed_frame.loc[mask, "common_relative_boundary"] = boundary
                seed_frame.loc[mask, "common_cliff_time"] = cliff_time
                seed_frame.loc[mask, "common_cliff_crossed"] = cliff_time >= 0
        path_rows.append(seed_frame)

        baseline_by_path = seed_frame[seed_frame["arm"] == "baseline"].set_index("path")
        for arm in REPAIR_ARMS:
            arm_by_path = seed_frame[seed_frame["arm"] == arm].set_index("path")
            for path in path_names:
                path_effect_rows.append(
                    {
                        "replicate_seed": replicate_seed,
                        "arm": arm,
                        "path": path,
                        "end_risk_reduction": float(
                            baseline_by_path.loc[path, "end_risk"]
                            - arm_by_path.loc[path, "end_risk"]
                        ),
                        "incident_crossing_reduction": float(
                            baseline_by_path.loc[path, "incident_crossing_fraction_all"]
                            - arm_by_path.loc[path, "incident_crossing_fraction_all"]
                        ),
                        "risk_area_reduction": float(
                            baseline_by_path.loc[path, "risk_area"]
                            - arm_by_path.loc[path, "risk_area"]
                        ),
                        "anchor_risk_change": float(
                            arm_by_path.loc[path, "anchor_risk"]
                            - baseline_by_path.loc[path, "anchor_risk"]
                        ),
                    }
                )

    query_frame = pd.concat(query_frames, ignore_index=True)
    candidate_frame = pd.concat(candidate_frames, ignore_index=True)
    selection_frame = pd.concat(selection_frames, ignore_index=True)
    selection_summary = pd.DataFrame(selection_summary_rows)
    training_frame = pd.DataFrame(training_rows)
    path_frame = pd.concat(path_rows, ignore_index=True)
    path_effects = pd.DataFrame(path_effect_rows)
    effects = (
        path_effects.groupby(["replicate_seed", "arm"], as_index=False)
        .agg(
            mean_end_risk_reduction=("end_risk_reduction", "mean"),
            mean_incident_crossing_reduction=("incident_crossing_reduction", "mean"),
            mean_risk_area_reduction=("risk_area_reduction", "mean"),
            mean_anchor_risk_change=("anchor_risk_change", "mean"),
        )
    )
    contrasts = compute_seed_contrasts(effects)
    primary = contrasts["coverage_random_minus_hazard_concentrated"].to_numpy()
    ci_lower, ci_upper = cluster_bootstrap_ci(
        primary,
        int(pilot["bootstrap_draws"]),
        stable_seed(pilot["master_seed"], "round12c_seed_cluster_bootstrap"),
    )

    selection_means = selection_summary.groupby("arm").mean(numeric_only=True)
    baseline_cliff_fraction = float(
        path_frame[path_frame["arm"] == "baseline"]["common_cliff_crossed"].mean()
    )
    pathwise_coverage_medians = (
        path_effects[path_effects["arm"] == "coverage_random"]
        .groupby("path")["end_risk_reduction"]
        .median()
    )
    maximum_validation_loss = float(
        training_frame[training_frame["arm"] != "baseline"]["validation_accuracy_loss"].max()
    )
    values = {
        "replicate_count": len(pilot["replicate_seeds"]),
        "repair_budget": total_cells,
        "query_budget_per_seed": int(len(query_frame) / len(pilot["replicate_seeds"])),
        "all_training_counts_equal_within_seed": bool(
            training_frame.groupby("replicate_seed")["training_count"].nunique().eq(1).all()
        ),
        "all_model_random_states_equal_within_seed": bool(
            training_frame.groupby("replicate_seed")["model_random_state"].nunique().eq(1).all()
        ),
        "all_acquisition_evaluation_streams_disjoint": bool(
            all(a != e for a, e in zip(acquisition_hashes, evaluation_hashes))
        ),
        "coverage_random_mean_coverage_fraction": float(
            selection_means.loc["coverage_random", "coverage_fraction"]
        ),
        "hazard_concentrated_mean_coverage_fraction": float(
            selection_means.loc["hazard_concentrated", "coverage_fraction"]
        ),
        "coverage_advantage": float(
            selection_means.loc["coverage_random", "coverage_fraction"]
            - selection_means.loc["hazard_concentrated", "coverage_fraction"]
        ),
        "coverage_random_mean_hazard_score": float(
            selection_means.loc["coverage_random", "mean_local_hazard_score"]
        ),
        "hazard_concentrated_mean_hazard_score": float(
            selection_means.loc["hazard_concentrated", "mean_local_hazard_score"]
        ),
        "hazard_score_advantage": float(
            selection_means.loc["hazard_concentrated", "mean_local_hazard_score"]
            - selection_means.loc["coverage_random", "mean_local_hazard_score"]
        ),
        "baseline_common_cliff_fraction": baseline_cliff_fraction,
        "primary_mean_end_risk_advantage": float(np.mean(primary)),
        "primary_positive_seed_fraction": float(np.mean(primary > 0)),
        "primary_seed_cluster_ci95": [ci_lower, ci_upper],
        "coverage_gain_at_high_hazard": float(
            contrasts["coverage_gain_at_high_hazard"].mean()
        ),
        "hazard_gain_at_high_coverage": float(
            contrasts["hazard_gain_at_high_coverage"].mean()
        ),
        "coverage_minus_hazard_contribution": float(
            contrasts["coverage_minus_hazard_contribution"].mean()
        ),
        "minimum_pathwise_median_coverage_random_reduction": float(
            pathwise_coverage_medians.min()
        ),
        "maximum_validation_accuracy_loss": maximum_validation_loss,
    }
    gates = pilot["pilot_gates"]
    checks = {
        "paired_training_counts": values["all_training_counts_equal_within_seed"],
        "paired_model_random_states": values["all_model_random_states_equal_within_seed"],
        "disjoint_acquisition_evaluation_streams": values[
            "all_acquisition_evaluation_streams_disjoint"
        ],
        "hazard_manipulation_active": values["hazard_score_advantage"]
        > gates["minimum_hazard_score_advantage"],
        "coverage_random_fills_all_cells": values[
            "coverage_random_mean_coverage_fraction"
        ]
        == 1.0,
        "coverage_manipulation_active": values["coverage_advantage"]
        >= gates["minimum_coverage_advantage"],
        "baseline_cliff_present": values["baseline_common_cliff_fraction"]
        >= gates["minimum_baseline_common_cliff_fraction"],
        "coverage_random_beats_hazard_concentrated_mean": values[
            "primary_mean_end_risk_advantage"
        ]
        >= gates["minimum_primary_mean_end_risk_advantage"],
        "coverage_random_beats_hazard_in_four_of_five_seeds": values[
            "primary_positive_seed_fraction"
        ]
        >= gates["minimum_positive_seed_fraction"],
        "primary_cluster_interval_excludes_zero": ci_lower
        > gates["minimum_primary_ci_lower"],
        "coverage_addition_helps_at_high_hazard": values[
            "coverage_gain_at_high_hazard"
        ]
        > gates["minimum_coverage_at_high_hazard_gain"],
        "coverage_contribution_exceeds_hazard_contribution": values[
            "coverage_minus_hazard_contribution"
        ]
        > gates["minimum_coverage_minus_hazard_contribution"],
        "coverage_random_helps_both_paths": values[
            "minimum_pathwise_median_coverage_random_reduction"
        ]
        > gates["minimum_pathwise_median_coverage_random_reduction"],
        "baseline_validation_preserved": values["maximum_validation_accuracy_loss"]
        <= gates["maximum_validation_accuracy_loss"],
    }
    passed = int(sum(checks.values()))
    decision = (
        "PILOT_SUPPORTS_COVERAGE_DOMINANCE"
        if passed == len(checks)
        else "PILOT_DOES_NOT_SUPPORT_COVERAGE_DOMINANCE"
    )

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", pilot)
    query_frame.to_csv(output / "queried_trajectories.csv", index=False)
    candidate_frame.to_csv(output / "candidate_bands.csv", index=False)
    selection_frame.to_csv(output / "selected_repair_examples.csv", index=False)
    selection_summary.to_csv(output / "selection_summary.csv", index=False)
    training_frame.to_csv(output / "training_summary.csv", index=False)
    path_frame.to_csv(output / "path_summary.csv", index=False)
    path_effects.to_csv(output / "pathwise_paired_effects.csv", index=False)
    effects.to_csv(output / "seed_paired_effects.csv", index=False)
    contrasts.to_csv(output / "seed_contrasts.csv", index=False)
    write_json(
        output / "bootstrap_summary.json",
        {
            "estimand": "coverage_random minus hazard_concentrated mean end-risk reduction",
            "cluster": "training seed",
            "draws": int(pilot["bootstrap_draws"]),
            "mean": float(np.mean(primary)),
            "ci95": [ci_lower, ci_upper],
        },
    )
    summary = {
        "pretarget_manifest_digest": pretarget_digest,
        "runtime": {
            "kind": RUNTIME_KIND,
            "python": platform.python_version(),
            "torchsig_tag": TORCHSIG_TAG,
            "torchsig_source_commit": TORCHSIG_SOURCE_COMMIT,
            "standard_torchsig_package_runtime": False,
        },
        "base_training_sha256": {
            str(item["replicate_seed"]): hash_arrays(
                [
                    ("features", item["base_features"]),
                    ("labels", item["base_labels"]),
                    ("theta", item["base_theta"]),
                ]
            )
            for item in prepared
        },
        "acquisition_panel_sha256": dict(
            zip([str(x) for x in pilot["replicate_seeds"]], acquisition_hashes)
        ),
        "evaluation_panel_sha256": dict(
            zip([str(x) for x in pilot["replicate_seeds"]], evaluation_hashes)
        ),
        "checks": {
            "values": values,
            "checks": checks,
            "passed": passed,
            "total": len(checks),
        },
        "decision": decision,
        "claim_status": "fresh five-seed post-paper pilot; not Paper V5 evidence",
    }
    write_json(output / "summary.json", summary)
    write_checksums(output)
    print(json.dumps(serializable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
