#!/usr/bin/env python3
"""Matched-order knockout of the persistent Cliff precursor mechanism.

This probe reuses the sealed Round 7 current-risk telemetry.  Within every
event replicate it keeps the complete pre-crossing telemetry multiset and its
terminal state fixed, permutes only temporal order, and recomputes the frozen
six-window, five-step risk forecast.  No model training or target generation is
performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def crossing_map(outcomes: pd.DataFrame) -> dict[str, int]:
    events = outcomes[outcomes["actual_event"]].copy()
    return {
        str(row.trajectory_id): int(row.actual_crossing_time)
        for row in events.itertuples(index=False)
    }


def forecast_window(values: np.ndarray, times: np.ndarray, crossing: int,
                    history: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    lookup = dict(zip(times.tolist(), values.tolist()))
    slopes: list[float] = []
    scores: list[float] = []
    centered = np.arange(history, dtype=float)
    centered -= centered.mean()
    denominator = float(np.sum(centered**2))
    for time_index in range(crossing - horizon, crossing):
        risk_history = np.asarray(
            [lookup[item] for item in range(time_index - history + 1, time_index + 1)],
            dtype=float,
        )
        slope = float(np.sum(centered * risk_history) / denominator)
        slopes.append(slope)
        scores.append(float(risk_history[-1] + horizon * max(slope, 0.0)))
    return np.asarray(slopes), np.asarray(scores)


def bootstrap_mean_interval(values: np.ndarray, draws: int, confidence: float,
                            rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(output: Path) -> None:
    manifest = output / "SHA256SUMS.txt"
    files = sorted(path for path in output.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files),
        encoding="utf-8",
    )


def plot_summary(pairs: pd.DataFrame, direction: pd.DataFrame, threshold: float,
                 output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(
        pairs["shuffle_forecast_mean"], pairs["ordered_forecast_mean"],
        c=pairs["direction"].map({"noise": "#2f5597", "phase": "#c55a11",
                                  "mixed_gradient": "#548235"}),
        alpha=0.55, s=20,
    )
    limits = [
        float(min(pairs["shuffle_forecast_mean"].min(), pairs["ordered_forecast_mean"].min())),
        float(max(pairs["shuffle_forecast_mean"].max(), pairs["ordered_forecast_mean"].max())),
    ]
    ax.plot(limits, limits, "--", color="#666666", linewidth=1)
    ax.set_xlabel("Fixed-terminal shuffled forecast")
    ax.set_ylabel("Persistent ordered forecast")
    ax.set_title("Matched forecast score")

    ax = axes[0, 1]
    x = np.arange(len(direction))
    width = 0.36
    ax.bar(x - width / 2, direction["ordered_forecast_mean"], width,
           label="ordered", color="#2f5597")
    ax.bar(x + width / 2, direction["shuffle_forecast_mean"], width,
           label="shuffled", color="#a5a5a5")
    ax.axhline(threshold, color="#7030a0", linestyle=":", label="alarm threshold")
    ax.set_xticks(x, direction["direction"])
    ax.set_ylabel("Mean five-window forecast score")
    ax.set_title("Effect replicated across directions")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["ordered", "shuffled", "sudden-jump proxy"]
    values = [
        pairs["ordered_positive_slope_fraction"].mean(),
        pairs["shuffle_positive_slope_fraction"].mean(),
        pairs["stationary_positive_slope_fraction"].mean(),
    ]
    ax.bar(labels, values, color=["#2f5597", "#a5a5a5", "#c00000"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of positive local slopes")
    ax.set_title("Temporal persistence is knocked out")

    ax = axes[1, 1]
    alarm_values = [
        pairs["ordered_any_alarm"].mean(),
        pairs["shuffle_any_alarm"].mean(),
        pairs["stationary_any_alarm"].mean(),
    ]
    ax.bar(labels, alarm_values, color=["#2f5597", "#a5a5a5", "#c00000"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Alarm in five-window decision interval")
    ax.set_title("Warning consequence of order removal")

    fig.suptitle("Round 8 precursor mechanism knockout", fontsize=14)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/precursor_order_knockout_probe.json",
        help="Configuration path relative to the repository root",
    )
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    source = ROOT / "results" / cfg["source_run"]
    output = ROOT / "results" / cfg["output_tag"]
    figures = ROOT / "figures" / cfg["output_tag"]
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    all_time_path = source / "sealed_all_time_risk_coordinates.csv"
    predictions = pd.read_csv(
        all_time_path if all_time_path.exists() else source / "sealed_online_predictions.csv"
    )
    outcomes = pd.read_csv(source / "trajectory_outcomes.csv")
    pretarget = json.loads(
        (source / "pretarget_formal_early_warning_gate.json").read_text(encoding="utf-8")
    )
    boundary = float(pretarget["metrics"]["relative_warning_boundary"])
    buffer = float(pretarget["metrics"]["forecast_buffer_frozen_from_calibration"])
    threshold = boundary + buffer
    history = int(cfg["history_windows"])
    horizon = int(cfg["forecast_horizon"])
    draws = int(cfg["shuffle_draws_per_replicate"])
    analysis_seed = int(cfg["analysis_seed"])
    rng = np.random.default_rng(analysis_seed)
    bootstrap_rng = np.random.default_rng(analysis_seed + 1)

    current = predictions[predictions["method"] == cfg["method"]].copy()
    crossings = crossing_map(outcomes)
    pair_rows: list[dict] = []
    shuffle_rows: list[dict] = []
    multiset_invariant_violations = 0
    terminal_invariant_violations = 0

    for trajectory_id, crossing in crossings.items():
        if not trajectory_id.endswith(":event"):
            continue
        direction = trajectory_id.split(":", maxsplit=1)[0]
        stationary_id = trajectory_id.replace(":event", ":stationary_safe")
        event_frame = current[current["trajectory_id"] == trajectory_id]
        stationary_frame = current[current["trajectory_id"] == stationary_id]
        replicates = sorted(event_frame["replicate"].unique())
        for replicate in replicates:
            event = event_frame[event_frame["replicate"] == replicate].set_index("time_index")
            stationary = stationary_frame[
                stationary_frame["replicate"] == replicate
            ].set_index("time_index")
            start = max(
                int(cfg["matched_design"]["evaluation_start_time"]),
                crossing - horizon - history + 1,
            )
            times = np.arange(start, crossing)
            event_values = event.loc[times, "estimated_current_risk"].to_numpy(dtype=float)
            stationary_values = stationary.loc[times, "estimated_current_risk"].to_numpy(dtype=float)
            ordered_slopes, ordered_scores = forecast_window(
                event_values, times, crossing, history, horizon
            )
            stationary_slopes, stationary_scores = forecast_window(
                stationary_values, times, crossing, history, horizon
            )

            shuffled_slope_means: list[float] = []
            shuffled_score_means: list[float] = []
            shuffled_positive: list[float] = []
            shuffled_alarm: list[float] = []
            for draw in range(draws):
                shuffled = event_values.copy()
                shuffled[:-1] = shuffled[:-1][rng.permutation(len(shuffled) - 1)]
                if not np.array_equal(np.sort(shuffled), np.sort(event_values)):
                    multiset_invariant_violations += 1
                if shuffled[-1] != event_values[-1]:
                    terminal_invariant_violations += 1
                shuffled_slopes, shuffled_scores = forecast_window(
                    shuffled, times, crossing, history, horizon
                )
                shuffled_slope_means.append(float(shuffled_slopes.mean()))
                shuffled_score_means.append(float(shuffled_scores.mean()))
                shuffled_positive.append(float(np.mean(shuffled_slopes > 0)))
                shuffled_alarm.append(float(np.any(shuffled_scores >= threshold)))
                shuffle_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "direction": direction,
                        "replicate": int(replicate),
                        "shuffle_draw": draw,
                        "integrated_slope": float(shuffled_slopes.mean()),
                        "integrated_forecast": float(shuffled_scores.mean()),
                        "positive_slope_fraction": float(np.mean(shuffled_slopes > 0)),
                        "any_alarm": bool(np.any(shuffled_scores >= threshold)),
                    }
                )

            pair_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "direction": direction,
                    "replicate": int(replicate),
                    "crossing_time": crossing,
                    "ordered_slope_mean": float(ordered_slopes.mean()),
                    "shuffle_slope_mean": float(np.mean(shuffled_slope_means)),
                    "slope_difference": float(
                        ordered_slopes.mean() - np.mean(shuffled_slope_means)
                    ),
                    "ordered_forecast_mean": float(ordered_scores.mean()),
                    "shuffle_forecast_mean": float(np.mean(shuffled_score_means)),
                    "forecast_difference": float(
                        ordered_scores.mean() - np.mean(shuffled_score_means)
                    ),
                    "ordered_positive_slope_fraction": float(np.mean(ordered_slopes > 0)),
                    "shuffle_positive_slope_fraction": float(np.mean(shuffled_positive)),
                    "positive_slope_fraction_difference": float(
                        np.mean(ordered_slopes > 0) - np.mean(shuffled_positive)
                    ),
                    "ordered_any_alarm": bool(np.any(ordered_scores >= threshold)),
                    "shuffle_any_alarm": float(np.mean(shuffled_alarm)),
                    "any_alarm_difference": float(
                        np.any(ordered_scores >= threshold) - np.mean(shuffled_alarm)
                    ),
                    "stationary_slope_mean": float(stationary_slopes.mean()),
                    "stationary_forecast_mean": float(stationary_scores.mean()),
                    "stationary_positive_slope_fraction": float(
                        np.mean(stationary_slopes > 0)
                    ),
                    "stationary_any_alarm": bool(np.any(stationary_scores >= threshold)),
                }
            )

    pairs = pd.DataFrame(pair_rows)
    shuffles = pd.DataFrame(shuffle_rows)
    if len(pairs) == 0:
        raise RuntimeError("No matched event replicates were available")

    slope_ci = bootstrap_mean_interval(
        pairs["slope_difference"].to_numpy(), int(cfg["bootstrap_draws"]),
        float(cfg["bootstrap_confidence"]), bootstrap_rng,
    )
    forecast_ci = bootstrap_mean_interval(
        pairs["forecast_difference"].to_numpy(), int(cfg["bootstrap_draws"]),
        float(cfg["bootstrap_confidence"]), bootstrap_rng,
    )
    direction = pairs.groupby("direction", as_index=False).agg(
        n_pairs=("replicate", "size"),
        ordered_slope_mean=("ordered_slope_mean", "mean"),
        shuffle_slope_mean=("shuffle_slope_mean", "mean"),
        slope_difference=("slope_difference", "mean"),
        ordered_forecast_mean=("ordered_forecast_mean", "mean"),
        shuffle_forecast_mean=("shuffle_forecast_mean", "mean"),
        forecast_difference=("forecast_difference", "mean"),
        ordered_alarm_rate=("ordered_any_alarm", "mean"),
        shuffle_alarm_rate=("shuffle_any_alarm", "mean"),
        stationary_alarm_rate=("stationary_any_alarm", "mean"),
    )

    metrics = {
        "matched_pairs": int(len(pairs)),
        "shuffle_replays": int(len(shuffles)),
        "mean_integrated_slope_difference": float(pairs["slope_difference"].mean()),
        "slope_difference_bootstrap_interval": list(slope_ci),
        "mean_integrated_forecast_difference": float(pairs["forecast_difference"].mean()),
        "forecast_difference_bootstrap_interval": list(forecast_ci),
        "pair_superiority_rate": float(np.mean(pairs["forecast_difference"] > 0)),
        "positive_slope_fraction_difference": float(
            pairs["positive_slope_fraction_difference"].mean()
        ),
        "ordered_any_window_alarm_rate": float(pairs["ordered_any_alarm"].mean()),
        "shuffled_any_window_alarm_rate": float(pairs["shuffle_any_alarm"].mean()),
        "any_window_alarm_rate_difference": float(pairs["any_alarm_difference"].mean()),
        "sudden_jump_proxy_pre_warning_rate": float(pairs["stationary_any_alarm"].mean()),
        "minimum_direction_forecast_difference": float(direction["forecast_difference"].min()),
        "alarm_threshold": threshold,
        "multiset_invariant_violations": multiset_invariant_violations,
        "terminal_state_invariant_violations": terminal_invariant_violations,
    }
    gate_spec = cfg["probe_gates"]
    gates = {
        "slope_effect": metrics["mean_integrated_slope_difference"]
        >= gate_spec["minimum_mean_integrated_slope_difference"],
        "forecast_effect": metrics["mean_integrated_forecast_difference"]
        >= gate_spec["minimum_mean_integrated_forecast_difference"],
        "pair_superiority": metrics["pair_superiority_rate"]
        >= gate_spec["minimum_pair_superiority_rate"],
        "positive_slope_fraction_effect": metrics["positive_slope_fraction_difference"]
        >= gate_spec["minimum_positive_slope_fraction_difference"],
        "alarm_consequence": metrics["any_window_alarm_rate_difference"]
        >= gate_spec["minimum_any_window_alarm_rate_difference"],
        "sudden_jump_limit": metrics["sudden_jump_proxy_pre_warning_rate"]
        <= gate_spec["maximum_sudden_jump_proxy_pre_warning_rate"],
        "direction_replication": metrics["minimum_direction_forecast_difference"] > 0,
        "bootstrap_slope_positive": slope_ci[0] > 0,
        "bootstrap_forecast_positive": forecast_ci[0] > 0,
        "state_multiset_exact": multiset_invariant_violations == 0,
        "terminal_state_exact": terminal_invariant_violations == 0,
    }
    checks = {
        "pretarget_all_passed": bool(pretarget["all_passed"]),
        "all_passed": bool(all(gates.values())),
        "gates": gates,
        "metrics": metrics,
        "semantics": cfg["semantics"],
        "decision": (
            cfg.get(
                "decision_if_passed",
                "matched temporal-order knockout supports persistent accumulation as the "
                "source of the prospective telemetry signal; new-seed confirmation remains required",
            )
            if all(gates.values())
            else cfg.get(
                "decision_if_failed",
                "matched temporal-order knockout does not pass the frozen probe gates",
            )
        ),
    }

    pairs.to_csv(output / "matched_pair_metrics.csv", index=False)
    shuffles.to_csv(output / "shuffle_replay_ledger.csv", index=False)
    direction.to_csv(output / "direction_summary.csv", index=False)
    (output / "checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "source_run": cfg["source_run"],
                "source_risk_coordinate_file": str(
                    all_time_path.name
                    if all_time_path.exists()
                    else "sealed_online_predictions.csv"
                ),
                "source_risk_coordinate_sha256": sha256(
                    all_time_path
                    if all_time_path.exists()
                    else source / "sealed_online_predictions.csv"
                ),
                "source_predictions_sha256": sha256(source / "sealed_online_predictions.csv"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_summary(
        pairs, direction, threshold,
        figures / "precursor_order_knockout_probe.png",
    )
    write_manifest(output)
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
