#!/usr/bin/env python3
"""Posttarget 45 m signed-boundary accounting diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

import run_boundary_flow as bf
import run_boundary_accounting as ba


ROOT = Path(__file__).resolve().parent
LAG = 3
PLACEBOS = 128
PLACEBO_SEED = 2026081594


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze() -> str:
    files = [ROOT / "config.json", ROOT / "accounting_config.json",
             ROOT / "SCALED_ACCOUNTING_PROTOCOL.md", ROOT / "run_boundary_flow.py",
             ROOT / "run_boundary_accounting.py", ROOT / "run_scaled_accounting.py",
             ROOT / "data" / "accounting_panels.npz",
             ROOT / "results" / "coherence_scale_diagnostic.json"]
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (ROOT / "SCALED_ACCOUNTING_PRETARGET_MANIFEST.json").write_text(
        json.dumps({"files": manifest, "sha256": digest}, indent=2))
    return digest


def rho(a, b):
    return float(spearmanr(a, b).statistic) if len(a) >= 3 else float("nan")


def run(expected_hash: str) -> None:
    if freeze() != expected_hash:
        raise RuntimeError("scaled accounting manifest mismatch")
    cfg, ac = bf.config(), ba.acfg()
    x, y, _ = bf.load_source(cfg)
    p = ba.panels()
    rng = np.random.default_rng(PLACEBO_SEED)
    rows = []
    placebo_names = [f"placebo_{i:03d}" for i in range(PLACEBOS)]
    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(cfg["windows"]["count"]))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx, scaler, model, train_hidden, train_margin = ba.fit_case(
                x, y, p, ordered, origin, cfg)
            mu, sd = ba.hidden_standardizer(train_hidden)
            train_hz = (train_hidden - mu) / sd
            epsilon = float(np.quantile(np.abs(train_margin),
                ac["boundary"]["epsilon_quantile_of_absolute_training_margin"]))
            w = model.coefs_[-1].reshape(-1).astype(float)
            b0 = float(model.intercepts_[-1].reshape(-1)[0])
            perms = [rng.permutation(w) for _ in range(PLACEBOS)]
            perm_eps = []
            for wp in perms:
                pm = np.where(y[train_idx] == 1, train_hidden @ wp + b0,
                              -(train_hidden @ wp + b0))
                perm_eps.append(float(np.quantile(np.abs(pm),
                    ac["boundary"]["epsilon_quantile_of_absolute_training_margin"])))
            seq = []
            for position in range(origin, min(len(ordered), origin + 6)):
                q = ba.load_window(x, y, p, scaler, model, mu, sd, ordered[position])
                q["position_vec"] = bf.position_vector(q["hidden_z"], q["y"], train_hz, y[train_idx])
                prob = expit(np.where(q["y"] == 1, q["margin"], -q["margin"]))
                q["brier"] = float(np.mean((prob - q["y"]) ** 2))
                seq.append((position, q))
            for start in range(len(seq) - LAG):
                position, cur = seq[start]
                _, nxt = seq[start + LAG]
                true_terms, speed_terms = [], []
                for cls in (0, 1):
                    s = 1.0 if cls == 1 else -1.0
                    hc = cur["hidden"][cur["y"] == cls]
                    hn = nxt["hidden"][nxt["y"] == cls]
                    inward = -s * float((hn.mean(axis=0) - hc.mean(axis=0)) @ w)
                    crowd = float(np.mean(np.abs(cur["margin"][cur["y"] == cls]) <= epsilon))
                    true_terms.append(crowd * inward / max(epsilon, 1e-8))
                    speed_terms.append(inward)
                dc = float(np.linalg.norm(cur["position_vec"]) / np.sqrt(cur["position_vec"].size))
                dn = float(np.linalg.norm(nxt["position_vec"]) / np.sqrt(nxt["position_vec"].size))
                row = {"direction": direction, "origin": origin, "position": position,
                       "actual_error_delta": nxt["error"] - cur["error"],
                       "actual_brier_delta": nxt["brier"] - cur["brier"],
                       "hazard": float(np.mean(true_terms)),
                       "normal_speed": float(np.mean(speed_terms)),
                       "unsigned_distance_delta": dn - dc}
                for name, wp, ep in zip(placebo_names, perms, perm_eps):
                    terms = []
                    for cls in (0, 1):
                        s = 1.0 if cls == 1 else -1.0
                        hc = cur["hidden"][cur["y"] == cls]
                        hn = nxt["hidden"][nxt["y"] == cls]
                        inward = -s * float((hn.mean(axis=0) - hc.mean(axis=0)) @ wp)
                        mcur = s * (hc @ wp + b0)
                        terms.append(float(np.mean(np.abs(mcur) <= ep)) * inward / max(ep, 1e-8))
                    row[name] = float(np.mean(terms))
                rows.append(row)
    df = pd.DataFrame(rows)
    actual = df["actual_error_delta"].to_numpy()
    hazard = df["hazard"].to_numpy()
    nz = np.abs(actual) > 1e-12
    r_h = rho(hazard.tolist(), actual.tolist())
    r_s = rho(df["normal_speed"].tolist(), actual.tolist())
    r_d = rho(df["unsigned_distance_delta"].tolist(), actual.tolist())
    placebo_rhos = [rho(df[c].tolist(), actual.tolist()) for c in placebo_names]
    nonincrease = actual <= 0
    sign_acc = float(np.mean(np.sign(hazard[nz]) == np.sign(actual[nz])))
    explained = float(np.mean(hazard[nonincrease] <= 0)) if nonincrease.any() else np.nan
    q95 = float(np.nanquantile(placebo_rhos, 0.95))
    checks = {"sign_accuracy": sign_acc >= 0.80,
              "hazard_spearman": r_h >= 0.50,
              "advantage_over_distance": r_h - r_d >= 0.20,
              "nonincrease_explained": explained >= 0.75,
              "true_boundary_over_placebo_q95": r_h > q95}
    summary = {"pretarget_sha256": expected_hash, "lag_windows": LAG,
               "meters": LAG * cfg["windows"]["width_m"], "n_transitions": int(len(df)),
               "sign_accuracy": sign_acc, "hazard_error_spearman": r_h,
               "normal_speed_error_spearman": r_s,
               "unsigned_distance_error_spearman": r_d,
               "hazard_advantage": r_h - r_d,
               "hazard_brier_spearman": rho(df["hazard"].tolist(), df["actual_brier_delta"].tolist()),
               "nonincrease_n": int(nonincrease.sum()),
               "nonincrease_explained_fraction": explained,
               "placebo_median_spearman": float(np.nanmedian(placebo_rhos)),
               "placebo_q95_spearman": q95,
               "checks": {k: bool(v) for k, v in checks.items()},
               "decision": "LICENSE_MULTI_SEED_PILOT" if all(checks.values()) else "STOP_MECHANISM_ROUTE"}
    df.to_csv(ROOT / "results" / "scaled_accounting_transitions.csv", index=False)
    (ROOT / "results" / "scaled_accounting_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    parser.add_argument("--expected-hash")
    args = parser.parse_args()
    if args.mode == "freeze":
        print(freeze())
    else:
        run(args.expected_hash)
