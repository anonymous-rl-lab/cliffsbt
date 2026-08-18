#!/usr/bin/env python3
"""Frozen Covertype distribution-to-boundary flow smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config() -> dict:
    return json.loads((ROOT / "config.json").read_text())


def source_paths(cfg: dict) -> tuple[Path, Path]:
    base = (ROOT / cfg["source"]["data_home"] / "covertype").resolve()
    return base / "samples_py3", base / "targets_py3"


def load_source(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp, tp = source_paths(cfg)
    if sha256(sp) != cfg["source"]["samples_sha256"]:
        raise RuntimeError("samples cache checksum mismatch")
    if sha256(tp) != cfg["source"]["targets_sha256"]:
        raise RuntimeError("targets cache checksum mismatch")
    data = fetch_covtype(data_home=str((ROOT / cfg["source"]["data_home"]).resolve()),
                         download_if_missing=False)
    mask = np.isin(data.target, cfg["task"]["labels"])
    original = np.flatnonzero(mask)
    x = data.data[mask].astype(np.float32, copy=False)
    y = (data.target[mask] == cfg["task"]["positive_label"]).astype(np.int64)
    return x, y, original


def prepare_panels(cfg: dict) -> None:
    x, y, _ = load_source(cfg)
    prior_file = ROOT / "data" / "prior_fixed_panels.npz"
    with np.load(prior_file) as z:
        # The original source-smoke runner selected panels after filtering to labels
        # 1/2, so these are already indices into the binary task arrays.
        prior_binary = np.concatenate([z[k] for k in sorted(z.files)]).astype(np.int64)
    excluded = np.zeros(y.size, dtype=bool)
    excluded[prior_binary] = True
    wc = cfg["windows"]
    rng = np.random.default_rng(wc["selection_seed"])
    payload: dict[str, np.ndarray] = {}
    audit = []
    for window, lower in enumerate(range(wc["lower_inclusive_m"],
                                         wc["upper_exclusive_m"], wc["width_m"])):
        upper = lower + wc["width_m"]
        for cls in (0, 1):
            pool = np.flatnonzero((~excluded) & (x[:, 0] >= lower) & (x[:, 0] < upper) & (y == cls))
            need = wc["fit_examples_per_class"] + wc["flow_examples_per_class"]
            if pool.size < need:
                raise RuntimeError(f"window {window} class {cls}: {pool.size} < {need}")
            pool = pool.copy()
            rng.shuffle(pool)
            fit = np.sort(pool[:wc["fit_examples_per_class"]])
            flow = np.sort(pool[wc["fit_examples_per_class"]:need])
            payload[f"fit_w{window:02d}_c{cls}"] = fit
            payload[f"flow_w{window:02d}_c{cls}"] = flow
            audit.append({"window": window, "lower_m": lower, "upper_m": upper,
                          "class": cls, "available_after_prior_exclusion": int(pool.size),
                          "fit": int(fit.size), "flow": int(flow.size)})
    np.savez_compressed(ROOT / "data" / "flow_panels.npz", **payload)
    (ROOT / "data" / "panel_audit.json").write_text(json.dumps(audit, indent=2))


def freeze() -> str:
    files = [ROOT / "config.json", ROOT / "PROTOCOL.md", ROOT / "run_boundary_flow.py",
             ROOT / "FAILED_RUNS.md",
             ROOT / "data" / "prior_fixed_panels.npz", ROOT / "data" / "flow_panels.npz",
             ROOT / "data" / "panel_audit.json"]
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (ROOT / "PRETARGET_MANIFEST.json").write_text(
        json.dumps({"files": manifest, "sha256": digest}, indent=2))
    return digest


def load_panels() -> dict[str, np.ndarray]:
    with np.load(ROOT / "data" / "flow_panels.npz") as z:
        return {k: z[k] for k in z.files}


def ordered_indices(panels: dict[str, np.ndarray], kind: str, ordered: list[int],
                    positions: list[int]) -> np.ndarray:
    parts = []
    for pos in positions:
        w = ordered[pos]
        parts.extend([panels[f"{kind}_w{w:02d}_c0"], panels[f"{kind}_w{w:02d}_c1"]])
    return np.concatenate(parts)


def forward_hidden_logits(model: MLPClassifier, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = z
    for coef, intercept in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = np.maximum(h @ coef + intercept, 0.0)
    logits = (h @ model.coefs_[-1] + model.intercepts_[-1]).reshape(-1)
    return h.astype(np.float64), logits.astype(np.float64)


def fit_model(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, cfg: dict):
    scaler = StandardScaler().fit(x[train_idx])
    z = scaler.transform(x[train_idx])
    mc = cfg["model"]
    model = MLPClassifier(hidden_layer_sizes=tuple(mc["hidden_layer_sizes"]),
                          activation="relu", solver="adam", alpha=mc["alpha"],
                          batch_size=mc["batch_size"], learning_rate_init=mc["learning_rate_init"],
                          max_iter=mc["max_iter"], shuffle=True, random_state=mc["seed"],
                          early_stopping=False, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(z, y[train_idx])
    train_hidden, train_logits = forward_hidden_logits(model, z)
    train_margin = np.where(y[train_idx] == 1, train_logits, -train_logits)
    return scaler, model, train_hidden, train_margin


def standardize_space(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (train - mu) / sd, (query - mu) / sd, sd


def position_vector(rep: np.ndarray, labels: np.ndarray,
                    train_rep: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    chunks = []
    for cls in (0, 1):
        chunks.append(rep[labels == cls].mean(axis=0) - train_rep[train_labels == cls].mean(axis=0))
    return np.concatenate(chunks)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else np.nan


def couple(prev_rep: np.ndarray, curr_rep: np.ndarray, prev_y: np.ndarray, curr_y: np.ndarray,
           prev_margin: np.ndarray, curr_margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curr_order = []
    delta_margin = []
    for cls in (0, 1):
        ip = np.flatnonzero(prev_y == cls)
        ic = np.flatnonzero(curr_y == cls)
        cost = ((prev_rep[ip, None, :] - curr_rep[None, ic, :]) ** 2).sum(axis=2)
        rp, rc = linear_sum_assignment(cost)
        curr_order.append(ic[rc])
        delta_margin.append(curr_margin[ic[rc]] - prev_margin[ip[rp]])
    return np.concatenate(curr_order), np.concatenate(delta_margin)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    keep = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    if keep.sum() < 3:
        return np.nan
    return float(spearmanr(a.to_numpy()[keep], b.to_numpy()[keep]).statistic)


def run(expected_hash: str) -> None:
    if freeze() != expected_hash:
        raise RuntimeError("pretarget manifest mismatch")
    cfg = config()
    x, y, _ = load_source(cfg)
    panels = load_panels()
    n_windows = cfg["windows"]["count"]
    case_rows = []
    window_rows = []
    forecast_rows = []

    for direction in cfg["deployment"]["directions"]:
        ordered = list(range(n_windows))
        if direction == "descending":
            ordered.reverse()
        for origin in cfg["deployment"]["origin_positions"]:
            train_idx = ordered_indices(panels, "fit", ordered, list(range(origin)))
            scaler, model, train_hidden, train_margin = fit_model(x, y, train_idx, cfg)
            train_z = scaler.transform(x[train_idx]).astype(np.float64)
            train_labels = y[train_idx]
            train_hidden_z, _, _ = standardize_space(train_hidden, train_hidden)
            raw_train = np.delete(train_z, cfg["task"]["elevation_feature_index"], axis=1)
            raw_train_z, _, _ = standardize_space(raw_train, raw_train)
            positive = train_margin[train_margin > 0]
            epsilon = float(np.quantile(positive, cfg["transport"]["epsilon_quantile_of_positive_training_margin"]))

            start = max(0, origin - 2)
            stop = min(n_windows, origin + cfg["deployment"]["target_horizon_windows"])
            per_window = []
            for position in range(start, stop):
                w = ordered[position]
                idx = np.concatenate([panels[f"flow_w{w:02d}_c0"], panels[f"flow_w{w:02d}_c1"]])
                z = scaler.transform(x[idx]).astype(np.float64)
                hidden, logits = forward_hidden_logits(model, z)
                margin = np.where(y[idx] == 1, logits, -logits)
                hidden_z = standardize_space(train_hidden, hidden)[1]
                raw = np.delete(z, cfg["task"]["elevation_feature_index"], axis=1)
                raw_z = standardize_space(raw_train, raw)[1]
                hidden_pos = position_vector(hidden_z, y[idx], train_hidden_z, train_labels)
                raw_pos = position_vector(raw_z, y[idx], raw_train_z, train_labels)
                row = {"direction": direction, "origin": origin, "position": position,
                       "physical_window": w, "phase": "target" if position >= origin else "history",
                       "error": float(np.mean(margin <= 0)), "mean_margin": float(np.mean(margin)),
                       "boundary_crowding": float(np.mean((margin > 0) & (margin <= epsilon))),
                       "epsilon": epsilon, "hidden_distance": float(np.linalg.norm(hidden_pos) / np.sqrt(hidden_pos.size)),
                       "raw_no_elevation_distance": float(np.linalg.norm(raw_pos) / np.sqrt(raw_pos.size))}
                window_rows.append(row)
                per_window.append({"row": row, "idx": idx, "y": y[idx], "margin": margin,
                                   "hidden": hidden_z, "raw_no_elevation": raw_z,
                                   "hidden_pos": hidden_pos, "raw_no_elevation_pos": raw_pos})

            for j in range(1, len(per_window)):
                prev, curr = per_window[j - 1], per_window[j]
                for space in cfg["transport"]["spaces"]:
                    order_curr, dm = couple(prev[space], curr[space], prev["y"], curr["y"],
                                            prev["margin"], curr["margin"])
                    curr_margin = curr["margin"][order_curr]
                    near = (curr_margin > 0) & (curr_margin <= epsilon)
                    inward = float(np.mean(-dm[near])) if near.any() else 0.0
                    hazard = float(curr["row"]["boundary_crowding"] * inward)
                    pred_next_margin = curr_margin + dm
                    pred_next_error = float(np.mean(pred_next_margin <= 0))
                    curr_error = curr["row"]["error"]
                    curr["row"][f"{space}_inward_velocity"] = inward
                    curr["row"][f"{space}_hazard"] = hazard
                    curr["row"][f"{space}_pred_next_error"] = pred_next_error
                    curr["row"][f"{space}_pred_delta"] = pred_next_error - curr_error
                    curr["row"][f"{space}_speed"] = float(np.linalg.norm(curr[f"{space}_pos"] - prev[f"{space}_pos"]) /
                                                               np.sqrt(curr[f"{space}_pos"].size))
                    curr["row"][f"{space}_distance_delta"] = curr["row"][f"{space}_distance"] - prev["row"][f"{space}_distance"]
                    if j >= 2:
                        older = per_window[j - 2]
                        v0 = prev[f"{space}_pos"] - older[f"{space}_pos"]
                        v1 = curr[f"{space}_pos"] - prev[f"{space}_pos"]
                        curr["row"][f"{space}_turn_cosine"] = cosine(v0, v1)

            for j in range(1, len(per_window) - 1):
                curr, nxt = per_window[j], per_window[j + 1]
                if curr["row"]["position"] < origin:
                    continue
                for space in cfg["transport"]["spaces"]:
                    forecast_rows.append({"direction": direction, "origin": origin,
                                          "position": curr["row"]["position"], "space": space,
                                          "actual_delta": nxt["row"]["error"] - curr["row"]["error"],
                                          "pred_delta": curr["row"].get(f"{space}_pred_delta", np.nan),
                                          "hazard": curr["row"].get(f"{space}_hazard", np.nan),
                                          "distance_delta": curr["row"].get(f"{space}_distance_delta", np.nan),
                                          "turn_cosine": curr["row"].get(f"{space}_turn_cosine", np.nan),
                                          "late": curr["row"]["position"] >= origin + 2})

            target = [v["row"] for v in per_window if v["row"]["position"] >= origin]
            case_rows.append({"direction": direction, "origin": origin,
                              "target_start_error": target[0]["error"],
                              "target_end_error": target[-1]["error"],
                              "target_delta": target[-1]["error"] - target[0]["error"],
                              "epsilon": epsilon, "iterations": int(model.n_iter_),
                              "train_accuracy": float(model.score(scaler.transform(x[train_idx]), y[train_idx]))})
            print(f"{direction:10s} origin={origin:02d} train={train_idx.size} "
                  f"risk={target[0]['error']:.3f}->{target[-1]['error']:.3f}", flush=True)

    windows_df = pd.DataFrame(window_rows)
    forecasts = pd.DataFrame(forecast_rows)
    cases = pd.DataFrame(case_rows)
    metrics = {}
    for space in cfg["transport"]["spaces"]:
        part = forecasts[forecasts["space"] == space].copy()
        nonzero = part[np.abs(part["actual_delta"]) > 1e-12]
        sign_acc = float(np.mean(np.sign(nonzero["pred_delta"]) == np.sign(nonzero["actual_delta"]))) if len(nonzero) else np.nan
        rho_flux = safe_spearman(part["pred_delta"], part["actual_delta"])
        rho_dist = safe_spearman(part["distance_delta"], part["actual_delta"])
        late_nonincrease = part[part["late"] & (part["actual_delta"] <= 0)]
        explained = float(np.mean(late_nonincrease["pred_delta"] <= 0)) if len(late_nonincrease) else np.nan
        metrics[space] = {"n_forecasts": int(len(part)), "n_nonzero": int(len(nonzero)),
                          "sign_accuracy_nonzero": sign_acc, "next_step_spearman_flux": rho_flux,
                          "next_step_spearman_unsigned_distance_delta": rho_dist,
                          "flux_advantage": rho_flux - rho_dist,
                          "late_nonincrease_n": int(len(late_nonincrease)),
                          "late_nonincrease_explained_fraction": explained}

    gate = cfg["smoke_gate"]
    checks = {}
    for space, m in metrics.items():
        checks[f"{space}_sign"] = m["sign_accuracy_nonzero"] >= gate["minimum_transition_sign_accuracy"]
        checks[f"{space}_rho"] = m["next_step_spearman_flux"] >= gate["minimum_next_step_spearman"]
        checks[f"{space}_advantage"] = m["flux_advantage"] >= gate["minimum_flux_advantage_over_unsigned_distance_spearman"]
        checks[f"{space}_late"] = m["late_nonincrease_explained_fraction"] >= gate["minimum_late_nonincrease_explained_fraction"]
    checks["space_sign_agreement"] = np.sign(metrics["hidden"]["next_step_spearman_flux"]) == np.sign(metrics["raw_no_elevation"]["next_step_spearman_flux"])
    decision = "ADVANCE_TO_FIVE_SEED_CONFIRMATION" if all(checks.values()) else "STOP_OR_REDESIGN"
    summary = {"pretarget_sha256": expected_hash, "cases": int(len(cases)),
               "forecasts": int(len(forecasts)), "metrics": metrics,
               "checks": {k: bool(v) for k, v in checks.items()}, "decision": decision}
    out = ROOT / "results"
    cases.to_csv(out / "cases.csv", index=False)
    windows_df.to_csv(out / "windows.csv", index=False)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "freeze", "run"])
    parser.add_argument("--expected-hash")
    args = parser.parse_args()
    cfg = config()
    if args.mode == "prepare":
        prepare_panels(cfg)
    elif args.mode == "freeze":
        print(freeze())
    else:
        if not args.expected_hash:
            parser.error("run requires --expected-hash")
        run(args.expected_hash)


if __name__ == "__main__":
    main()
