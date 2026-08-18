#!/usr/bin/env python3
"""Posttarget diagnosis of the failed Round 11A velocity-matching gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "round11_paired_sync_smoke"
ANCHOR = 6
CONFIRMATION = 2
DRAW_COUNT = 4096
SEED = 2026110199


def first_crossing(margin: np.ndarray) -> np.ndarray:
    output = np.full(margin.shape[1], -1, dtype=int)
    for sample in np.flatnonzero(margin[ANCHOR] > 0):
        for time in range(ANCHOR + 1, margin.shape[0] - CONFIRMATION + 1):
            if np.all(margin[time : time + CONFIRMATION, sample] <= 0):
                output[sample] = time
                break
    return output


def classwise_permute(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = values.copy()
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        result[index] = values[rng.permutation(index)]
    return result


def main() -> None:
    archive = np.load(RESULT / "paired_margin_trajectories.npz", allow_pickle=False)
    path_rows = []
    transition_rows = []
    histogram_rows = []
    bin_rows = []
    rng = np.random.default_rng(SEED)
    for path in ("noise", "mixed_gradient"):
        prefix = f"seed20261111_baseline_{path}"
        margin = archive[f"{prefix}_margins"].astype(float)
        labels = archive[f"{prefix}_labels"].astype(int)
        risk = np.mean(margin <= 0, axis=1)
        crossing = first_crossing(margin)
        incident_mask = crossing >= 0
        incident = crossing[incident_mask]
        counts = np.asarray([np.sum(crossing == t) for t in range(ANCHOR + 1, len(risk))])
        probabilities = counts / len(labels)
        three_window_peak = max(
            (probabilities[i : i + 3].sum() for i in range(max(len(probabilities) - 2, 1))),
            default=0.0,
        )
        positive = probabilities[probabilities > 0]
        normalized_entropy = (
            float(-np.sum((positive / positive.sum()) * np.log(positive / positive.sum())) / np.log(len(probabilities)))
            if len(positive) > 1
            else 0.0
        )
        persistence = float(np.mean(margin[-1, incident_mask] <= 0)) if len(incident) else 0.0
        sign_changes = np.sum(np.diff(margin[ANCHOR:] <= 0, axis=0) != 0, axis=0)
        path_rows.append(
            {
                "path": path,
                "incident_count": int(len(incident)),
                "incident_fraction": float(len(incident) / len(labels)),
                "single_window_peak": float(probabilities.max()),
                "three_window_peak": float(three_window_peak),
                "normalized_first_crossing_entropy": normalized_entropy,
                "first_crossing_iqr_windows": float(np.subtract(*np.percentile(incident, [75, 25])))
                if len(incident)
                else 0.0,
                "end_wrong_fraction_among_incident": persistence,
                "median_sign_changes_after_anchor": float(np.median(sign_changes)),
                "risk_forward_flux_spearman": float(
                    spearmanr(
                        np.diff(risk[ANCHOR:]),
                        [
                            np.mean((margin[t] > 0) & (margin[t + 1] <= 0))
                            for t in range(ANCHOR, len(risk) - 1)
                        ],
                    ).statistic
                ),
            }
        )
        for time, count in zip(range(ANCHOR + 1, len(risk)), counts):
            histogram_rows.append({"path": path, "time_index": time, "confirmed_first_crossings": int(count)})
        for time in range(ANCHOR, len(risk) - 1):
            current = margin[time]
            velocity = margin[time + 1] - current
            observed = int(np.sum((current > 0) & (current + velocity <= 0)))
            null = np.empty(DRAW_COUNT, dtype=int)
            for draw in range(DRAW_COUNT):
                shuffled = classwise_permute(velocity, labels, rng)
                null[draw] = np.sum((current > 0) & (current + shuffled <= 0))
            transition_rows.append(
                {
                    "path": path,
                    "time_index": time,
                    "observed_forward_crossings": observed,
                    "shuffle_median": float(np.median(null)),
                    "shuffle_q95": float(np.quantile(null, 0.95)),
                    "shuffle_percentile": float(np.mean(null < observed)),
                    "margin_velocity_spearman": float(spearmanr(current, velocity).statistic),
                }
            )
            bins = [(-np.inf, 0.0, "wrong"), (0.0, 0.05, "near_0_005"),
                    (0.05, 0.15, "near_005_015"), (0.15, 0.30, "mid_015_030"),
                    (0.30, np.inf, "far_gt_030")]
            for low, high, name in bins:
                selected = (current > low) & (current <= high)
                if np.any(selected):
                    bin_rows.append(
                        {
                            "path": path,
                            "time_index": time,
                            "margin_bin": name,
                            "count": int(selected.sum()),
                            "mean_velocity": float(velocity[selected].mean()),
                            "median_velocity": float(np.median(velocity[selected])),
                            "fraction_harmful_velocity": float(np.mean(velocity[selected] < 0)),
                        }
                    )
    path_frame = pd.DataFrame(path_rows)
    transition_frame = pd.DataFrame(transition_rows)
    histogram_frame = pd.DataFrame(histogram_rows)
    bin_frame = pd.DataFrame(bin_rows)
    transition_frame.to_csv(RESULT / "posttarget_velocity_matching_diagnostic.csv", index=False)
    histogram_frame.to_csv(RESULT / "posttarget_first_crossing_histogram.csv", index=False)
    bin_frame.to_csv(RESULT / "posttarget_local_velocity_bins.csv", index=False)
    active = transition_frame[transition_frame["shuffle_q95"] > 0].copy()
    exceedances = int(np.sum(active["observed_forward_crossings"] > active["shuffle_q95"]))
    exceedance_p = float(binomtest(exceedances, len(active), 0.05, alternative="greater").pvalue)
    summary = {
        "status": "POSTTARGET_DIAGNOSTIC_ONLY",
        "draws_per_transition": DRAW_COUNT,
        "paths": path_frame.to_dict(orient="records"),
        "active_transitions_above_shuffle_q95": exceedances,
        "active_transitions_total": int(len(active)),
        "binomial_exceedance_p_value": exceedance_p,
        "median_observed_shuffle_percentile": float(
            transition_frame["shuffle_percentile"].median()
        ),
        "median_margin_velocity_spearman": float(
            transition_frame["margin_velocity_spearman"].median()
        ),
        "diagnosis": (
            "DISTRIBUTED_BOUNDARY_FLUX_WITHOUT_REPLICATED_EXTRA_POSITION_VELOCITY_COUPLING"
            if exceedance_p >= 0.05
            else "POSITION_VELOCITY_COUPLING_REQUIRES_FRESH_CONFIRMATION"
        ),
    }
    (RESULT / "posttarget_failure_diagnostic.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
