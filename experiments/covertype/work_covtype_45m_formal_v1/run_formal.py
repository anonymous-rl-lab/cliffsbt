#!/usr/bin/env python3
"""Frozen 45 m Covertype signed-boundary formal confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
FLOW_ROOT = ROOT.parent / "work_covtype_boundary_flow_probe"
PILOT_ROOT = ROOT.parent / "work_covtype_45m_multiseed_pilot"
sys.path.insert(0, str(FLOW_ROOT))
import run_boundary_flow as bf  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg() -> dict:
    return json.loads((ROOT / "config.json").read_text())


def base_cfg(c: dict, seed: int | None = None) -> dict:
    b = {
        "source": c["source"], "task": c["task"],
        "model": {"hidden_layer_sizes": c["model"]["hidden_layer_sizes"],
                  "alpha": c["model"]["alpha"],
                  "learning_rate_init": c["model"]["learning_rate_init"],
                  "batch_size": c["model"]["batch_size"],
                  "max_iter": c["model"]["max_iter"],
                  "seed": c["model"]["seeds"][0] if seed is None else seed}}
    return b


def load_source(c: dict):
    return bf.load_source(base_cfg(c))


def prior_panel_files() -> list[Path]:
    return [FLOW_ROOT / "data" / "prior_fixed_panels.npz",
            FLOW_ROOT / "data" / "flow_panels.npz",
            FLOW_ROOT / "data" / "accounting_panels.npz",
            PILOT_ROOT / "data" / "block_panels.npz"]


def prepare() -> None:
    c = cfg()
    x, y, _ = load_source(c)
    prior_arrays = []
    for path in prior_panel_files():
        with np.load(path) as z:
            prior_arrays.extend([z[k] for k in z.files])
    known_used = np.unique(np.concatenate(prior_arrays)).astype(np.int64)
    np.save(ROOT / "data" / "known_used_indices.npy", known_used)
    excluded = np.zeros(y.size, dtype=bool)
    excluded[known_used] = True
    bc = c["blocks"]
    rng = np.random.default_rng(bc["selection_seed"])
    payload = {}
    audit = []
    for block, lower in enumerate(range(bc["lower_inclusive_m"],
                                        bc["upper_exclusive_m"], bc["width_m"])):
        upper = lower + bc["width_m"]
        for cls in (0, 1):
            pool = np.flatnonzero((~excluded) & (x[:, 0] >= lower) & (x[:, 0] < upper) & (y == cls))
            eval_n = bc["evaluation_examples_per_class"] if block >= bc["evaluation_from_block"] else 0
            need = bc["fit_examples_per_class"] + eval_n
            if pool.size < need:
                raise RuntimeError(f"block {block} class {cls}: {pool.size} < {need}")
            pool = pool.copy()
            rng.shuffle(pool)
            fit = np.sort(pool[:bc["fit_examples_per_class"]])
            payload[f"fit_b{block:02d}_c{cls}"] = fit
            if eval_n:
                evaluation = np.sort(pool[bc["fit_examples_per_class"]:need])
                payload[f"eval_b{block:02d}_c{cls}"] = evaluation
            audit.append({"block": block, "lower_m": lower, "upper_m": upper,
                          "class": cls, "available_after_exclusion": int(pool.size),
                          "fit": int(bc["fit_examples_per_class"]), "evaluation": int(eval_n)})
    np.savez_compressed(ROOT / "data" / "block_panels.npz", **payload)
    (ROOT / "data" / "panel_audit.json").write_text(json.dumps(audit, indent=2))


def freeze() -> str:
    files = [ROOT / "config.json", ROOT / "PROTOCOL.md", ROOT / "run_formal.py",
             ROOT / "data" / "known_used_indices.npy", ROOT / "data" / "block_panels.npz",
             ROOT / "data" / "panel_audit.json", FLOW_ROOT / "run_boundary_flow.py",
             PILOT_ROOT / "data" / "block_panels.npz"]
    manifest = {str(p.relative_to(ROOT.parent)): sha256(p) for p in files}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (ROOT / "PRETARGET_MANIFEST.json").write_text(
        json.dumps({"files": manifest, "sha256": digest}, indent=2))
    return digest


def panels() -> dict[str, np.ndarray]:
    with np.load(ROOT / "data" / "block_panels.npz") as z:
        return {k: z[k] for k in z.files}


def train_indices(p: dict[str, np.ndarray], ordered: list[int], origin: int) -> np.ndarray:
    return np.concatenate([p[f"fit_b{ordered[pos]:02d}_c{cls}"]
                           for pos in range(origin) for cls in (0, 1)])


def load_eval(x, y, p, scaler, model, block):
    idx = np.concatenate([p[f"eval_b{block:02d}_c0"], p[f"eval_b{block:02d}_c1"]])
    hidden, logits = bf.forward_hidden_logits(model, scaler.transform(x[idx]))
    margin = np.where(y[idx] == 1, logits, -logits)
    prob = expit(logits)
    return {"idx": idx, "y": y[idx], "hidden": hidden, "margin": margin,
            "error": float(np.mean(margin <= 0)),
            "brier": float(np.mean((prob - y[idx]) ** 2))}


def safe_rho(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return np.nan
    return float(spearmanr(a[keep], b[keep]).statistic)


def resample_grid(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    seeds = df["seed"].unique()
    sources = df["source"].unique()
    sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
    sampled_sources = rng.choice(sources, len(sources), replace=True)
    pieces = []
    for i, seed in enumerate(sampled_seeds):
        for j, source in enumerate(sampled_sources):
            q = df[(df["seed"] == seed) & (df["source"] == source)].copy()
            q["boot_seed"] = i
            q["boot_source"] = j
            pieces.append(q)
    return pd.concat(pieces, ignore_index=True)


def run(expected_hash: str) -> None:
    if freeze() != expected_hash:
        raise RuntimeError("pretarget manifest mismatch")
    c = cfg()
    x, y, _ = load_source(c)
    p = panels()
    rng_placebo = np.random.default_rng(c["boundary"]["placebo_seed"])
    rows = []
    placebo_names = [f"placebo_{k:03d}" for k in range(c["boundary"]["placebo_permutations"])]
    for seed in c["model"]["seeds"]:
        mc = base_cfg(c, seed)
        for direction in c["deployment"]["directions"]:
            ordered = list(range(c["blocks"]["count"]))
            if direction == "descending":
                ordered.reverse()
            for origin in c["deployment"]["origin_blocks"]:
                train_idx = train_indices(p, ordered, origin)
                scaler, model, train_hidden, train_margin = bf.fit_model(x, y, train_idx, mc)
                train_labels = y[train_idx]
                hmu, hsd = train_hidden.mean(axis=0), train_hidden.std(axis=0)
                hsd[hsd < 1e-6] = 1.0
                train_hz = (train_hidden - hmu) / hsd
                train_pos = {cls: train_hz[train_labels == cls].mean(axis=0) for cls in (0, 1)}
                epsilon = float(np.quantile(np.abs(train_margin),
                    c["boundary"]["epsilon_quantile_of_absolute_training_margin"]))
                w = model.coefs_[-1].reshape(-1).astype(float)
                b0 = float(model.intercepts_[-1].reshape(-1)[0])
                Wp = np.column_stack([rng_placebo.permutation(w) for _ in placebo_names])
                train_lp = train_hidden @ Wp + b0
                train_mp = np.where(train_labels[:, None] == 1, train_lp, -train_lp)
                eps_p = np.quantile(np.abs(train_mp),
                    c["boundary"]["epsilon_quantile_of_absolute_training_margin"], axis=0)
                horizon = c["deployment"]["target_horizon_blocks"]
                seq = []
                for position in range(origin, origin + horizon):
                    block = ordered[position]
                    q = load_eval(x, y, p, scaler, model, block)
                    hz = (q["hidden"] - hmu) / hsd
                    q["position_vec"] = np.concatenate([
                        hz[q["y"] == cls].mean(axis=0) - train_pos[cls] for cls in (0, 1)])
                    q["block"] = block
                    seq.append(q)
                source = f"{direction}_{origin}"
                for step in range(horizon - 1):
                    cur, nxt = seq[step], seq[step + 1]
                    true_terms, speed_terms = [], []
                    placebo_terms = []
                    for cls in (0, 1):
                        s = 1.0 if cls == 1 else -1.0
                        hc = cur["hidden"][cur["y"] == cls]
                        hn = nxt["hidden"][nxt["y"] == cls]
                        delta_h = hn.mean(axis=0) - hc.mean(axis=0)
                        inward = -s * float(delta_h @ w)
                        crowd = float(np.mean(np.abs(cur["margin"][cur["y"] == cls]) <= epsilon))
                        true_terms.append(crowd * inward / max(epsilon, 1e-8))
                        speed_terms.append(inward)
                        inward_p = -s * (delta_h @ Wp)
                        mcur_p = s * (hc @ Wp + b0)
                        crowd_p = np.mean(np.abs(mcur_p) <= eps_p, axis=0)
                        placebo_terms.append(crowd_p * inward_p / np.maximum(eps_p, 1e-8))
                    dc = float(np.linalg.norm(cur["position_vec"]) / np.sqrt(cur["position_vec"].size))
                    dn = float(np.linalg.norm(nxt["position_vec"]) / np.sqrt(nxt["position_vec"].size))
                    row = {"seed": seed, "direction": direction, "origin": origin,
                           "source": source, "step": step, "current_block": cur["block"],
                           "next_block": nxt["block"],
                           "actual_error_delta": nxt["error"] - cur["error"],
                           "actual_brier_delta": nxt["brier"] - cur["brier"],
                           "hazard": float(np.mean(true_terms)),
                           "normal_speed": float(np.mean(speed_terms)),
                           "unsigned_distance_delta": dn - dc,
                           "current_error": cur["error"], "next_error": nxt["error"]}
                    pv = np.mean(np.vstack(placebo_terms), axis=0)
                    for name, value in zip(placebo_names, pv):
                        row[name] = float(value)
                    rows.append(row)
                print(f"seed={seed} {direction:10s} origin={origin:02d} "
                      f"risk={seq[0]['error']:.3f}->{seq[-1]['error']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    actual = df["actual_error_delta"].to_numpy()
    hazard = df["hazard"].to_numpy()
    nz = np.abs(actual) > 1e-12
    rho_h = safe_rho(hazard, actual)
    rho_b = safe_rho(df["hazard"], df["actual_brier_delta"])
    rho_d = safe_rho(df["unsigned_distance_delta"], actual)
    sign_acc = float(np.mean(np.sign(hazard[nz]) == np.sign(actual[nz])))
    source_sign = {}
    for source, q in df.groupby("source"):
        keep = np.abs(q["actual_error_delta"].to_numpy()) > 1e-12
        source_sign[source] = float(np.mean(np.sign(q["hazard"].to_numpy()[keep]) ==
                                            np.sign(q["actual_error_delta"].to_numpy()[keep])))
    direction_rho = {direction: safe_rho(q["hazard"], q["actual_error_delta"])
                     for direction, q in df.groupby("direction")}
    placebo_rhos = [safe_rho(df[name], actual) for name in placebo_names]
    placebo_q95 = float(np.nanquantile(placebo_rhos, 0.95))
    rng_boot = np.random.default_rng(c["bootstrap"]["seed"])
    boot_h, boot_adv = [], []
    for _ in range(c["bootstrap"]["replicates"]):
        q = resample_grid(df, rng_boot)
        rh = safe_rho(q["hazard"], q["actual_error_delta"])
        rd = safe_rho(q["unsigned_distance_delta"], q["actual_error_delta"])
        boot_h.append(rh)
        boot_adv.append(rh - rd)
    ci_h = np.nanquantile(boot_h, [0.025, 0.5, 0.975]).tolist()
    ci_adv = np.nanquantile(boot_adv, [0.025, 0.5, 0.975]).tolist()
    g = c["gate"]
    checks = {"error_spearman": rho_h >= g["minimum_error_spearman"],
              "brier_spearman": rho_b >= g["minimum_brier_spearman"],
              "sign_accuracy": sign_acc >= g["minimum_sign_accuracy"],
              "advantage": rho_h - rho_d >= g["minimum_advantage_over_unsigned_distance"],
              "source_replication": sum(v >= 0.6 for v in source_sign.values()) >= g["minimum_sources_with_sign_accuracy_at_least_0_6"],
              "direction_replication": all(v > 0 for v in direction_rho.values()),
              "placebo": rho_h > placebo_q95,
              "bootstrap_hazard": ci_h[0] > 0,
              "bootstrap_advantage": ci_adv[0] > 0}
    summary = {"pretarget_sha256": expected_hash, "models": 30,
               "transitions": int(len(df)), "error_spearman": rho_h,
               "brier_spearman": rho_b, "sign_accuracy": sign_acc,
               "unsigned_distance_spearman": rho_d,
               "hazard_advantage": rho_h - rho_d,
               "normal_speed_spearman": safe_rho(df["normal_speed"], actual),
               "source_sign_accuracy": source_sign, "direction_spearman": direction_rho,
               "placebo_median_spearman": float(np.nanmedian(placebo_rhos)),
               "placebo_q95_spearman": placebo_q95,
               "two_way_bootstrap_hazard_ci": ci_h,
               "two_way_bootstrap_advantage_ci": ci_adv,
               "checks": {k: bool(v) for k, v in checks.items()},
               "decision": "FORMAL_PASS" if all(checks.values()) else "FORMAL_STOP"}
    df.to_csv(ROOT / "results" / "transitions.csv", index=False)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "freeze", "run"])
    parser.add_argument("--expected-hash")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "freeze":
        print(freeze())
    else:
        if not args.expected_hash:
            parser.error("run requires --expected-hash")
        run(args.expected_hash)


if __name__ == "__main__":
    main()
