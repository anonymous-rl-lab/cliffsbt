#!/usr/bin/env python3
"""Redesigned contemporaneous distribution-to-boundary accounting probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import run_boundary_flow as bf


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def acfg() -> dict:
    return json.loads((ROOT / "accounting_config.json").read_text())


def prepare() -> None:
    cfg = bf.config()
    a = acfg()
    x, y, _ = bf.load_source(cfg)
    with np.load(ROOT / "data" / "prior_fixed_panels.npz") as z:
        prior = np.concatenate([z[k] for k in sorted(z.files)]).astype(np.int64)
    excluded = np.zeros(y.size, dtype=bool)
    excluded[prior] = True
    rng = np.random.default_rng(a["panel"]["selection_seed"])
    payload = {}
    audit = []
    wc = cfg["windows"]
    fit_n = a["panel"]["fit_examples_per_class"]
    flow_n = a["panel"]["flow_examples_per_class"]
    for window, lower in enumerate(range(wc["lower_inclusive_m"],
                                         wc["upper_exclusive_m"], wc["width_m"])):
        upper = lower + wc["width_m"]
        for cls in (0, 1):
            pool = np.flatnonzero((~excluded) & (x[:, 0] >= lower) & (x[:, 0] < upper) & (y == cls))
            if pool.size < fit_n + flow_n:
                raise RuntimeError(f"window {window} class {cls}: {pool.size} < {fit_n + flow_n}")
            pool = pool.copy()
            rng.shuffle(pool)
            fit = np.sort(pool[:fit_n])
            flow = np.sort(pool[fit_n:fit_n + flow_n])
            payload[f"fit_w{window:02d}_c{cls}"] = fit
            payload[f"flow_w{window:02d}_c{cls}"] = flow
            audit.append({"window": window, "lower_m": lower, "upper_m": upper,
                          "class": cls, "available": int(pool.size),
                          "fit": fit_n, "flow": flow_n})
    np.savez_compressed(ROOT / "data" / "accounting_panels.npz", **payload)
    (ROOT / "data" / "accounting_panel_audit.json").write_text(json.dumps(audit, indent=2))


def frozen_hash(stage: str) -> str:
    files = [ROOT / "config.json", ROOT / "accounting_config.json",
             ROOT / "ACCOUNTING_PROTOCOL.md", ROOT / "run_boundary_flow.py",
             ROOT / "run_boundary_accounting.py", ROOT / "FAILED_RUNS.md",
             ROOT / "data" / "prior_fixed_panels.npz",
             ROOT / "data" / "accounting_panels.npz",
             ROOT / "data" / "accounting_panel_audit.json"]
    if stage == "account" and (ROOT / "results" / "accounting_reliability.json").exists():
        files.append(ROOT / "results" / "accounting_reliability.json")
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    payload = json.dumps({"stage": stage, "files": manifest}, sort_keys=True,
                         separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (ROOT / f"{stage.upper()}_PRETARGET_MANIFEST.json").write_text(
        json.dumps({"stage": stage, "files": manifest, "sha256": digest}, indent=2))
    return digest


def panels() -> dict[str, np.ndarray]:
    with np.load(ROOT / "data" / "accounting_panels.npz") as z:
        return {k: z[k] for k in z.files}


def fit_case(x, y, p, ordered, origin, cfg):
    train_idx = bf.ordered_indices(p, "fit", ordered, list(range(origin)))
    scaler, model, train_hidden, train_margin = bf.fit_model(x, y, train_idx, cfg)
    return train_idx, scaler, model, train_hidden, train_margin


def hidden_standardizer(train_hidden: np.ndarray):
    mu = train_hidden.mean(axis=0)
    sd = train_hidden.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def class_position(hidden: np.ndarray, labels: np.ndarray,
                   train_hidden_z: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    return bf.position_vector(hidden, labels, train_hidden_z, train_labels)


def load_window(x, y, p, scaler, model, train_mu, train_sd, w):
    idx = np.concatenate([p[f"flow_w{w:02d}_c0"], p[f"flow_w{w:02d}_c1"]])
    hidden, logits = bf.forward_hidden_logits(model, scaler.transform(x[idx]))
    hidden_z = (hidden - train_mu) / train_sd
    margin = np.where(y[idx] == 1, logits, -logits)
    return {"idx": idx, "y": y[idx], "hidden": hidden, "hidden_z": hidden_z,
            "margin": margin, "error": float(np.mean(margin <= 0))}


def reliability(expected_hash: str) -> None:
    if frozen_hash("reliability") != expected_hash:
        raise RuntimeError("reliability manifest mismatch")
    cfg, a = bf.config(), acfg()
    x, y, _ = bf.load_source(cfg)
    p = panels()
    rng = np.random.default_rng(2026081598)
    cosines = []
    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(cfg["windows"]["count"]))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx, scaler, model, train_hidden, _ = fit_case(x, y, p, ordered, origin, cfg)
            train_mu, train_sd = hidden_standardizer(train_hidden)
            train_hidden_z = (train_hidden - train_mu) / train_sd
            pos = {"a": [], "b": []}
            for position in range(max(0, origin - 2), min(len(ordered), origin + 6)):
                w = ordered[position]
                split = {"a": [], "b": []}
                for cls in (0, 1):
                    idx = p[f"flow_w{w:02d}_c{cls}"].copy()
                    rng.shuffle(idx)
                    split["a"].append(idx[:idx.size // 2])
                    split["b"].append(idx[idx.size // 2:])
                for half in ("a", "b"):
                    chosen = split[half]
                    idx = np.concatenate(chosen)
                    hidden = bf.forward_hidden_logits(model, scaler.transform(x[idx]))[0]
                    hidden_z = (hidden - train_mu) / train_sd
                    pos[half].append(class_position(hidden_z, y[idx], train_hidden_z, y[train_idx]))
            for j in range(1, len(pos["a"])):
                cosines.append(bf.cosine(pos["a"][j] - pos["a"][j - 1],
                                         pos["b"][j] - pos["b"][j - 1]))
    summary = {"pretarget_sha256": expected_hash, "n_velocity_steps": len(cosines),
               "median_velocity_split_cosine": float(np.nanmedian(cosines)),
               "fraction_positive": float(np.mean(np.asarray(cosines) > 0)),
               "gate": float(np.nanmedian(cosines)) >= a["gate"]["minimum_velocity_split_cosine"]}
    (ROOT / "results" / "accounting_reliability.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def safe_rho(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[keep], y[keep]).statistic) if keep.sum() >= 3 else np.nan


def account(expected_hash: str) -> None:
    if frozen_hash("account") != expected_hash:
        raise RuntimeError("account manifest mismatch")
    rel = json.loads((ROOT / "results" / "accounting_reliability.json").read_text())
    if not rel["gate"]:
        raise RuntimeError("velocity reliability gate failed; target accounting prohibited")
    cfg, a = bf.config(), acfg()
    x, y, _ = bf.load_source(cfg)
    p = panels()
    rng = np.random.default_rng(a["boundary"]["placebo_seed"])
    rows = []
    placebo_columns = []
    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(cfg["windows"]["count"]))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx, scaler, model, train_hidden, train_margin = fit_case(x, y, p, ordered, origin, cfg)
            train_mu, train_sd = hidden_standardizer(train_hidden)
            train_hidden_z = (train_hidden - train_mu) / train_sd
            epsilon = float(np.quantile(np.abs(train_margin),
                                        a["boundary"]["epsilon_quantile_of_absolute_training_margin"]))
            w_true = model.coefs_[-1].reshape(-1).astype(float)
            b_true = float(model.intercepts_[-1].reshape(-1)[0])
            perms = [rng.permutation(w_true) for _ in range(a["boundary"]["placebo_permutations"])]
            placebo_eps = []
            for wp in perms:
                train_pm = np.where(y[train_idx] == 1, train_hidden @ wp + b_true,
                                    -(train_hidden @ wp + b_true))
                placebo_eps.append(float(np.quantile(
                    np.abs(train_pm), a["boundary"]["epsilon_quantile_of_absolute_training_margin"])))
            seq = []
            for position in range(max(0, origin - 1), min(len(ordered), origin + 6)):
                q = load_window(x, y, p, scaler, model, train_mu, train_sd, ordered[position])
                q["position_vec"] = class_position(q["hidden_z"], q["y"], train_hidden_z, y[train_idx])
                seq.append((position, q))
            for j in range(len(seq) - 1):
                position, cur = seq[j]
                _, nxt = seq[j + 1]
                if position < origin:
                    continue
                cls_terms = []
                for cls in (0, 1):
                    s = 1.0 if cls == 1 else -1.0
                    hc = cur["hidden"][cur["y"] == cls]
                    hn = nxt["hidden"][nxt["y"] == cls]
                    dm = s * float((hn.mean(axis=0) - hc.mean(axis=0)) @ w_true)
                    crowd = float(np.mean(np.abs(cur["margin"][cur["y"] == cls]) <= epsilon))
                    cls_terms.append(crowd * (-dm / max(epsilon, 1e-8)))
                hazard = float(np.mean(cls_terms))
                distance_cur = float(np.linalg.norm(cur["position_vec"]) / np.sqrt(cur["position_vec"].size))
                distance_nxt = float(np.linalg.norm(nxt["position_vec"]) / np.sqrt(nxt["position_vec"].size))
                row = {"direction": direction, "origin": origin, "position": position,
                       "late": position >= origin + 2,
                       "actual_delta": nxt["error"] - cur["error"],
                       "current_error": cur["error"], "next_error": nxt["error"],
                       "hazard": hazard, "unsigned_distance_delta": distance_nxt - distance_cur,
                       "epsilon": epsilon}
                for k, (wp, eps_p) in enumerate(zip(perms, placebo_eps)):
                    terms = []
                    for cls in (0, 1):
                        s = 1.0 if cls == 1 else -1.0
                        hc = cur["hidden"][cur["y"] == cls]
                        hn = nxt["hidden"][nxt["y"] == cls]
                        mcur = s * (hc @ wp + b_true)
                        dm = s * float((hn.mean(axis=0) - hc.mean(axis=0)) @ wp)
                        terms.append(float(np.mean(np.abs(mcur) <= eps_p)) * (-dm / max(eps_p, 1e-8)))
                    col = f"placebo_{k:03d}"
                    row[col] = float(np.mean(terms))
                    placebo_columns.append(col)
                rows.append(row)

    df = pd.DataFrame(rows)
    rho = safe_rho(df["hazard"], df["actual_delta"])
    rho_distance = safe_rho(df["unsigned_distance_delta"], df["actual_delta"])
    nz = df[np.abs(df["actual_delta"]) > 1e-12]
    sign_acc = float(np.mean(np.sign(nz["hazard"]) == np.sign(nz["actual_delta"])))
    late_non = df[df["late"] & (df["actual_delta"] <= 0)]
    explained = float(np.mean(late_non["hazard"] <= 0)) if len(late_non) else np.nan
    unique_placebos = sorted(set(placebo_columns))
    placebo_rhos = [safe_rho(df[c], df["actual_delta"]) for c in unique_placebos]
    placebo_q95 = float(np.nanquantile(placebo_rhos, 0.95))
    g = a["gate"]
    checks = {"sign_accuracy": sign_acc >= g["minimum_sign_accuracy"],
              "hazard_spearman": rho >= g["minimum_hazard_spearman"],
              "advantage_over_unsigned_distance": rho - rho_distance >= g["minimum_advantage_over_unsigned_distance"],
              "late_nonincrease": explained >= g["minimum_late_nonincrease_explained_fraction"],
              "true_boundary_over_placebo_q95": rho > placebo_q95}
    summary = {"pretarget_sha256": expected_hash, "n_transitions": int(len(df)),
               "n_nonzero": int(len(nz)), "sign_accuracy": sign_acc,
               "hazard_spearman": rho, "unsigned_distance_spearman": rho_distance,
               "hazard_advantage": rho - rho_distance,
               "late_nonincrease_n": int(len(late_non)),
               "late_nonincrease_explained_fraction": explained,
               "placebo_q95_spearman": placebo_q95,
               "placebo_median_spearman": float(np.nanmedian(placebo_rhos)),
               "checks": {k: bool(v) for k, v in checks.items()},
               "decision": "ADVANCE_TO_FIVE_SEED_CONFIRMATION" if all(checks.values()) else "STOP_OR_REDESIGN"}
    df.to_csv(ROOT / "results" / "accounting_transitions.csv", index=False)
    (ROOT / "results" / "accounting_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "freeze-reliability", "reliability",
                                         "freeze-account", "account"])
    parser.add_argument("--expected-hash")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "freeze-reliability":
        print(frozen_hash("reliability"))
    elif args.mode == "freeze-account":
        print(frozen_hash("account"))
    elif args.mode == "reliability":
        reliability(args.expected_hash)
    else:
        account(args.expected_hash)


if __name__ == "__main__":
    main()
