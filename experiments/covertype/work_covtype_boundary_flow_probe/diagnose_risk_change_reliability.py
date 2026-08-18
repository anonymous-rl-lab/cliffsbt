#!/usr/bin/env python3
"""Posttarget split-half reliability of Covertype risk levels and changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

import run_boundary_flow as bf
import run_boundary_accounting as ba


ROOT = Path(__file__).resolve().parent


def rho(a, b):
    return float(spearmanr(a, b).statistic) if len(a) >= 3 else float("nan")


def main() -> None:
    cfg = bf.config()
    x, y, _ = bf.load_source(cfg)
    p = ba.panels()
    rng = np.random.default_rng(2026081596)
    rows = []
    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(cfg["windows"]["count"]))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx, scaler, model, _, _ = ba.fit_case(x, y, p, ordered, origin, cfg)
            curves = {"a": {"error": [], "brier": []}, "b": {"error": [], "brier": []}}
            for position in range(max(0, origin - 2), min(len(ordered), origin + 6)):
                w = ordered[position]
                split = {"a": [], "b": []}
                for cls in (0, 1):
                    idx = p[f"flow_w{w:02d}_c{cls}"].copy()
                    rng.shuffle(idx)
                    split["a"].append(idx[:idx.size // 2])
                    split["b"].append(idx[idx.size // 2:])
                for half in ("a", "b"):
                    idx = np.concatenate(split[half])
                    logits = bf.forward_hidden_logits(model, scaler.transform(x[idx]))[1]
                    prob = expit(logits)
                    curves[half]["error"].append(float(np.mean((prob >= 0.5) != y[idx])))
                    curves[half]["brier"].append(float(np.mean((prob - y[idx]) ** 2)))
            for metric in ("error", "brier"):
                a = np.asarray(curves["a"][metric])
                b = np.asarray(curves["b"][metric])
                da, db = np.diff(a), np.diff(b)
                for j in range(len(da)):
                    rows.append({"direction": direction, "origin": origin, "metric": metric,
                                 "step": j, "level_a": a[j + 1], "level_b": b[j + 1],
                                 "delta_a": da[j], "delta_b": db[j],
                                 "target_step": j >= 1})
    summary = {}
    for metric in ("error", "brier"):
        part = [r for r in rows if r["metric"] == metric]
        target = [r for r in part if r["target_step"]]
        nz = [r for r in target if abs(r["delta_a"]) > 1e-12 and abs(r["delta_b"]) > 1e-12]
        summary[metric] = {
            "n_levels": len(part),
            "level_split_spearman": rho([r["level_a"] for r in part], [r["level_b"] for r in part]),
            "delta_split_spearman": rho([r["delta_a"] for r in target], [r["delta_b"] for r in target]),
            "delta_sign_agreement_nonzero": float(np.mean([
                np.sign(r["delta_a"]) == np.sign(r["delta_b"]) for r in nz])) if nz else float("nan"),
            "n_nonzero_pairs": len(nz)
        }
    result = {"posttarget_diagnostic": True, "summary": summary}
    (ROOT / "results" / "risk_change_reliability.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
