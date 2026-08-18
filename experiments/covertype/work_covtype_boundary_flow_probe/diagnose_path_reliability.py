#!/usr/bin/env python3
"""Posttarget diagnostic: split-half reliability of path velocities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import run_boundary_flow as bf


ROOT = Path(__file__).resolve().parent


def corr(a: list[float], b: list[float]) -> float:
    return float(spearmanr(a, b).statistic) if len(a) >= 3 else float("nan")


def main() -> None:
    cfg = bf.config()
    x, y, _ = bf.load_source(cfg)
    panels = bf.load_panels()
    n_windows = cfg["windows"]["count"]
    rows = []
    rng = np.random.default_rng(2026081599)
    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(n_windows))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx = bf.ordered_indices(panels, "fit", ordered, list(range(origin)))
            scaler, model, train_hidden, _ = bf.fit_model(x, y, train_idx, cfg)
            train_labels = y[train_idx]
            train_hidden_z = bf.standardize_space(train_hidden, train_hidden)[0]
            positions = {"a": [], "b": []}
            distances = {"a": [], "b": []}
            for position in range(max(0, origin - 2), min(n_windows, origin + 6)):
                w = ordered[position]
                halves = {"a": [], "b": []}
                for cls in (0, 1):
                    idx = panels[f"flow_w{w:02d}_c{cls}"].copy()
                    rng.shuffle(idx)
                    halves["a"].append(idx[:idx.size // 2])
                    halves["b"].append(idx[idx.size // 2:])
                for half in ("a", "b"):
                    idx = np.concatenate(halves[half])
                    z = scaler.transform(x[idx])
                    hidden = bf.forward_hidden_logits(model, z)[0]
                    hidden_z = bf.standardize_space(train_hidden, hidden)[1]
                    pos = bf.position_vector(hidden_z, y[idx], train_hidden_z, train_labels)
                    positions[half].append(pos)
                    distances[half].append(float(np.linalg.norm(pos) / np.sqrt(pos.size)))
            for j in range(1, len(positions["a"])):
                va = positions["a"][j] - positions["a"][j - 1]
                vb = positions["b"][j] - positions["b"][j - 1]
                rows.append({"direction": direction, "origin": origin, "step": j,
                             "velocity_split_cosine": bf.cosine(va, vb),
                             "distance_a": distances["a"][j], "distance_b": distances["b"][j]})
    cosines = [r["velocity_split_cosine"] for r in rows]
    summary = {"n_velocity_steps": len(cosines),
               "median_velocity_split_cosine": float(np.nanmedian(cosines)),
               "fraction_velocity_split_cosine_positive": float(np.mean(np.asarray(cosines) > 0)),
               "distance_split_spearman": corr([r["distance_a"] for r in rows],
                                                 [r["distance_b"] for r in rows]),
               "interpretation": "velocity_unreliable" if np.nanmedian(cosines) < 0.5 else "velocity_measurable"}
    (ROOT / "results" / "path_reliability_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
