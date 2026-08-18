#!/usr/bin/env python3
"""Posttarget diagnostic of the minimum reproducible path/risk change scale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

import run_boundary_flow as bf
import run_boundary_accounting as ba


ROOT = Path(__file__).resolve().parent
LAGS = (1, 2, 3, 4)


def rho(a, b):
    return float(spearmanr(a, b).statistic) if len(a) >= 3 else float("nan")


def main() -> None:
    cfg = bf.config()
    x, y, _ = bf.load_source(cfg)
    p = ba.panels()
    rng = np.random.default_rng(2026081595)
    collected = {lag: {"error_a": [], "error_b": [], "brier_a": [], "brier_b": [],
                       "hidden_cos": [], "boundary_cos": []} for lag in LAGS}
    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(cfg["windows"]["count"]))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx, scaler, model, train_hidden, _ = ba.fit_case(x, y, p, ordered, origin, cfg)
            mu, sd = ba.hidden_standardizer(train_hidden)
            train_hz = (train_hidden - mu) / sd
            w = model.coefs_[-1].reshape(-1)
            b0 = float(model.intercepts_[-1].reshape(-1)[0])
            train_boundary = np.where(y[train_idx] == 1, train_hidden @ w + b0,
                                      -(train_hidden @ w + b0)).reshape(-1, 1)
            curves = {half: {"error": [], "brier": [], "hidden": [], "boundary": []}
                      for half in ("a", "b")}
            for position in range(origin, min(len(ordered), origin + 6)):
                window = ordered[position]
                split = {"a": [], "b": []}
                for cls in (0, 1):
                    idx = p[f"flow_w{window:02d}_c{cls}"].copy()
                    rng.shuffle(idx)
                    split["a"].append(idx[:idx.size // 2])
                    split["b"].append(idx[idx.size // 2:])
                for half in ("a", "b"):
                    idx = np.concatenate(split[half])
                    hidden, logits = bf.forward_hidden_logits(model, scaler.transform(x[idx]))
                    hz = (hidden - mu) / sd
                    prob = expit(logits)
                    margin = np.where(y[idx] == 1, logits, -logits).reshape(-1, 1)
                    curves[half]["error"].append(float(np.mean((prob >= 0.5) != y[idx])))
                    curves[half]["brier"].append(float(np.mean((prob - y[idx]) ** 2)))
                    curves[half]["hidden"].append(
                        bf.position_vector(hz, y[idx], train_hz, y[train_idx]))
                    curves[half]["boundary"].append(
                        bf.position_vector(margin, y[idx], train_boundary, y[train_idx]))
            for lag in LAGS:
                for start in range(0, len(curves["a"]["error"]) - lag):
                    for metric in ("error", "brier"):
                        da = curves["a"][metric][start + lag] - curves["a"][metric][start]
                        db = curves["b"][metric][start + lag] - curves["b"][metric][start]
                        collected[lag][f"{metric}_a"].append(da)
                        collected[lag][f"{metric}_b"].append(db)
                    for chart in ("hidden", "boundary"):
                        va = curves["a"][chart][start + lag] - curves["a"][chart][start]
                        vb = curves["b"][chart][start + lag] - curves["b"][chart][start]
                        collected[lag][f"{chart}_cos"].append(bf.cosine(va, vb))
    summary = {}
    for lag, d in collected.items():
        row = {"meters": lag * cfg["windows"]["width_m"]}
        for metric in ("error", "brier"):
            aa, bb = np.asarray(d[f"{metric}_a"]), np.asarray(d[f"{metric}_b"])
            nz = (np.abs(aa) > 1e-12) & (np.abs(bb) > 1e-12)
            row[f"{metric}_delta_spearman"] = rho(aa.tolist(), bb.tolist())
            row[f"{metric}_sign_agreement"] = float(np.mean(np.sign(aa[nz]) == np.sign(bb[nz])))
            row[f"{metric}_n"] = int(aa.size)
        for chart in ("hidden", "boundary"):
            vals = np.asarray(d[f"{chart}_cos"], dtype=float)
            row[f"{chart}_median_split_cosine"] = float(np.nanmedian(vals))
            row[f"{chart}_fraction_positive"] = float(np.mean(vals > 0))
        summary[str(lag)] = row
    eligible = [int(lag) for lag, row in summary.items()
                if row["brier_delta_spearman"] >= 0.5
                and row["hidden_median_split_cosine"] >= 0.5]
    result = {"posttarget_diagnostic": True, "lags": summary,
              "minimum_jointly_reproducible_lag": min(eligible) if eligible else None,
              "minimum_jointly_reproducible_meters": min(eligible) * 15 if eligible else None}
    (ROOT / "results" / "coherence_scale_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
