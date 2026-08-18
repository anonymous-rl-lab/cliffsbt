#!/usr/bin/env python3
"""Risk-blind diagnostic of split-half velocity reliability by coordinate chart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

import run_boundary_flow as bf
import run_boundary_accounting as ba


ROOT = Path(__file__).resolve().parent
CANDIDATES = (2, 4, 8, 16, 32)


def pos(rep, labels, train_rep, train_labels):
    return bf.position_vector(rep, labels, train_rep, train_labels)


def main() -> None:
    cfg = bf.config()
    x, y, _ = bf.load_source(cfg)
    p = ba.panels()
    rng = np.random.default_rng(2026081597)
    values = {"hidden_full": []}
    for k in CANDIDATES:
        values[f"hidden_pca{k}"] = []
        values[f"hidden_pca{k}_plus_boundary"] = []
        values[f"raw_pca{k}"] = []
    values["boundary_only"] = []

    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(cfg["windows"]["count"]))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx, scaler, model, train_hidden, _ = ba.fit_case(x, y, p, ordered, origin, cfg)
            train_mu, train_sd = ba.hidden_standardizer(train_hidden)
            train_hz = (train_hidden - train_mu) / train_sd
            train_raw = np.delete(scaler.transform(x[train_idx]), 0, axis=1)
            raw_mu, raw_sd = train_raw.mean(axis=0), train_raw.std(axis=0)
            raw_sd[raw_sd < 1e-6] = 1.0
            train_rz = (train_raw - raw_mu) / raw_sd
            hpca = {k: PCA(n_components=k, random_state=0).fit(train_hz) for k in CANDIDATES}
            rpca = {k: PCA(n_components=k, random_state=0).fit(train_rz) for k in CANDIDATES}
            train_hp = {k: hpca[k].transform(train_hz) for k in CANDIDATES}
            train_rp = {k: rpca[k].transform(train_rz) for k in CANDIDATES}
            w = model.coefs_[-1].reshape(-1)
            b = float(model.intercepts_[-1].reshape(-1)[0])
            train_boundary = np.where(y[train_idx] == 1, train_hidden @ w + b,
                                      -(train_hidden @ w + b)).reshape(-1, 1)
            charts = {name: {"a": [], "b": []} for name in values}
            for position in range(max(0, origin - 2), min(len(ordered), origin + 6)):
                window = ordered[position]
                split = {"a": [], "b": []}
                for cls in (0, 1):
                    idx = p[f"flow_w{window:02d}_c{cls}"].copy()
                    rng.shuffle(idx)
                    split["a"].append(idx[:idx.size // 2])
                    split["b"].append(idx[idx.size // 2:])
                for half in ("a", "b"):
                    idx = np.concatenate(split[half])
                    z = scaler.transform(x[idx])
                    h, logits = bf.forward_hidden_logits(model, z)
                    hz = (h - train_mu) / train_sd
                    raw = np.delete(z, 0, axis=1)
                    rz = (raw - raw_mu) / raw_sd
                    boundary = np.where(y[idx] == 1, logits, -logits).reshape(-1, 1)
                    charts["hidden_full"][half].append(pos(hz, y[idx], train_hz, y[train_idx]))
                    charts["boundary_only"][half].append(pos(boundary, y[idx], train_boundary, y[train_idx]))
                    for k in CANDIDATES:
                        hpk = hpca[k].transform(hz)
                        rpk = rpca[k].transform(rz)
                        charts[f"hidden_pca{k}"][half].append(pos(hpk, y[idx], train_hp[k], y[train_idx]))
                        charts[f"raw_pca{k}"][half].append(pos(rpk, y[idx], train_rp[k], y[train_idx]))
                        aug = np.column_stack([hpk, boundary])
                        train_aug = np.column_stack([train_hp[k], train_boundary])
                        charts[f"hidden_pca{k}_plus_boundary"][half].append(
                            pos(aug, y[idx], train_aug, y[train_idx]))
            for name in charts:
                for j in range(1, len(charts[name]["a"])):
                    va = charts[name]["a"][j] - charts[name]["a"][j - 1]
                    vb = charts[name]["b"][j] - charts[name]["b"][j - 1]
                    values[name].append(bf.cosine(va, vb))

    summary = {}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        summary[name] = {"n": int(arr.size), "median_split_cosine": float(np.nanmedian(arr)),
                         "fraction_positive": float(np.mean(arr > 0)),
                         "fraction_ge_0_5": float(np.mean(arr >= 0.5))}
    ranked = sorted(summary, key=lambda k: summary[k]["median_split_cosine"], reverse=True)
    result = {"risk_outcomes_read": 0, "ranking": ranked, "charts": summary}
    (ROOT / "results" / "reduced_velocity_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
