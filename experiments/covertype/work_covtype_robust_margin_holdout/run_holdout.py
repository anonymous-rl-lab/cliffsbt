#!/usr/bin/env python3
"""Frozen fresh-holdout test of robust margin transport."""

from __future__ import annotations

import argparse
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
FORMAL_ROOT = ROOT.parent / "work_covtype_45m_formal_v1"
sys.path.insert(0, str(FLOW_ROOT))
import run_boundary_flow as bf  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg():
    return json.loads((ROOT / "config.json").read_text())


def formal_cfg():
    return json.loads((FORMAL_ROOT / "config.json").read_text())


def model_cfg(seed):
    c = formal_cfg()
    return {"source": c["source"], "task": c["task"],
            "model": {"hidden_layer_sizes": c["model"]["hidden_layer_sizes"],
                      "alpha": c["model"]["alpha"],
                      "learning_rate_init": c["model"]["learning_rate_init"],
                      "batch_size": c["model"]["batch_size"],
                      "max_iter": c["model"]["max_iter"], "seed": seed}}


def exclusion_files():
    return [FLOW_ROOT / "data" / "prior_fixed_panels.npz",
            FLOW_ROOT / "data" / "flow_panels.npz",
            FLOW_ROOT / "data" / "accounting_panels.npz",
            PILOT_ROOT / "data" / "block_panels.npz",
            FORMAL_ROOT / "data" / "block_panels.npz"]


def prepare():
    c, fc = cfg(), formal_cfg()
    x, y, _ = bf.load_source(model_cfg(c["model_seeds"][0]))
    arrays = []
    for path in exclusion_files():
        with np.load(path) as z:
            arrays.extend([z[k] for k in z.files])
    used = np.unique(np.concatenate(arrays)).astype(np.int64)
    np.save(ROOT / "data" / "known_used_indices.npy", used)
    excluded = np.zeros(y.size, dtype=bool)
    excluded[used] = True
    ec = c["evaluation"]
    rng = np.random.default_rng(ec["selection_seed"])
    payload, audit = {}, []
    for block, lower in enumerate(range(ec["lower_inclusive_m"],
                                        ec["upper_exclusive_m"], ec["block_width_m"])):
        upper = lower + ec["block_width_m"]
        for cls in (0, 1):
            pool = np.flatnonzero((~excluded) & (x[:, 0] >= lower) &
                                  (x[:, 0] < upper) & (y == cls))
            need = ec["examples_per_class_per_block"]
            if pool.size < need:
                raise RuntimeError(f"block {block} class {cls}: {pool.size} < {need}")
            pool = pool.copy(); rng.shuffle(pool)
            selected = np.sort(pool[:need])
            payload[f"eval_b{block:02d}_c{cls}"] = selected
            audit.append({"block": block, "lower_m": lower, "upper_m": upper,
                          "class": cls, "available_after_exclusion": int(pool.size),
                          "selected": int(need)})
    np.savez_compressed(ROOT / "data" / "holdout_panels.npz", **payload)
    (ROOT / "data" / "panel_audit.json").write_text(json.dumps(audit, indent=2))


def freeze():
    files = [ROOT / "config.json", ROOT / "PROTOCOL.md", ROOT / "run_holdout.py",
             ROOT / "data" / "known_used_indices.npy", ROOT / "data" / "holdout_panels.npz",
             ROOT / "data" / "panel_audit.json", FORMAL_ROOT / "config.json",
             FORMAL_ROOT / "data" / "block_panels.npz", FLOW_ROOT / "run_boundary_flow.py"]
    manifest = {str(p.relative_to(ROOT.parent)): sha256(p) for p in files}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (ROOT / "PRETARGET_MANIFEST.json").write_text(
        json.dumps({"files": manifest, "sha256": digest}, indent=2))
    return digest


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def rho(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)


def transport(a, b, epsilon, trim):
    a, b = np.sort(a), np.sort(b)
    v = b - a
    density = float(np.mean(np.abs(a) <= epsilon) / max(2 * epsilon, 1e-8))
    lo, hi = np.quantile(v, [trim, 1 - trim])
    vt = v[(v >= lo) & (v <= hi)]
    return {"mean": -density * float(np.mean(v)),
            "median": -density * float(np.median(v)),
            "trimmed": -density * float(np.mean(vt))}


def bootstrap_grid(df, rng):
    seeds, sources = df.seed.unique(), df.source.unique()
    ss = rng.choice(seeds, len(seeds), replace=True)
    uu = rng.choice(sources, len(sources), replace=True)
    return pd.concat([df[(df.seed == s) & (df.source == u)] for s in ss for u in uu],
                     ignore_index=True)


def run(expected_hash):
    if freeze() != expected_hash:
        raise RuntimeError("pretarget manifest mismatch")
    c, fc = cfg(), formal_cfg()
    fit = load_npz(FORMAL_ROOT / "data" / "block_panels.npz")
    ev = load_npz(ROOT / "data" / "holdout_panels.npz")
    rng_p = np.random.default_rng(c["controls"]["seed"])
    placebo_names = [f"placebo_{k:03d}" for k in range(c["controls"]["permuted_boundaries"])]
    rows = []
    for seed in c["model_seeds"]:
        mc = model_cfg(seed)
        x, y, _ = bf.load_source(mc)
        for direction in c["directions"]:
            ordered = list(range(c["evaluation"]["block_count"]))
            if direction == "descending": ordered.reverse()
            for origin in c["origin_blocks"]:
                train_idx = np.concatenate([fit[f"fit_b{ordered[pos]:02d}_c{cls}"]
                                            for pos in range(origin) for cls in (0, 1)])
                scaler, model, train_hidden, train_margin = bf.fit_model(x, y, train_idx, mc)
                epsilon = float(np.quantile(np.abs(train_margin),
                    c["margin_transport"]["boundary_band_training_abs_margin_quantile"]))
                w = model.coefs_[-1].reshape(-1).astype(float)
                b0 = float(model.intercepts_[-1].reshape(-1)[0])
                Wp = np.column_stack([rng_p.permutation(w) for _ in placebo_names])
                train_labels = y[train_idx]
                train_lp = train_hidden @ Wp + b0
                train_mp = np.where(train_labels[:, None] == 1, train_lp, -train_lp)
                eps_p = np.quantile(np.abs(train_mp),
                    c["margin_transport"]["boundary_band_training_abs_margin_quantile"], axis=0)
                hmu, hsd = train_hidden.mean(axis=0), train_hidden.std(axis=0)
                hsd[hsd < 1e-6] = 1.0
                thz = (train_hidden - hmu) / hsd
                train_pos = {cls: thz[train_labels == cls].mean(axis=0) for cls in (0, 1)}
                seq = []
                for position in range(origin, origin + c["target_horizon_blocks"]):
                    block = ordered[position]
                    idx = np.concatenate([ev[f"eval_b{block:02d}_c0"], ev[f"eval_b{block:02d}_c1"]])
                    hidden, logits = bf.forward_hidden_logits(model, scaler.transform(x[idx]))
                    margin = np.where(y[idx] == 1, logits, -logits)
                    prob = expit(logits)
                    hz = (hidden - hmu) / hsd
                    pos = np.concatenate([hz[y[idx] == cls].mean(axis=0) - train_pos[cls]
                                          for cls in (0, 1)])
                    seq.append({"y": y[idx], "hidden": hidden, "margin": margin,
                                "error": float(np.mean(margin <= 0)),
                                "brier": float(np.mean((prob - y[idx]) ** 2)), "position": pos})
                for step in range(2):
                    cur, nxt = seq[step], seq[step + 1]
                    true = {k: [] for k in ("mean", "median", "trimmed")}
                    placebo = np.zeros(len(placebo_names), dtype=float)
                    for cls in (0, 1):
                        s = 1.0 if cls == 1 else -1.0
                        q0, q1 = cur["y"] == cls, nxt["y"] == cls
                        values = transport(cur["margin"][q0], nxt["margin"][q1], epsilon,
                            c["margin_transport"]["trim_fraction_each_tail"])
                        for name in true: true[name].append(values[name])
                        lp0 = s * (cur["hidden"][q0] @ Wp + b0)
                        lp1 = s * (nxt["hidden"][q1] @ Wp + b0)
                        v = np.sort(lp1, axis=0) - np.sort(lp0, axis=0)
                        dens = np.mean(np.abs(lp0) <= eps_p, axis=0) / np.maximum(2 * eps_p, 1e-8)
                        placebo += -dens * np.median(v, axis=0) / 2.0
                    dc = np.linalg.norm(cur["position"]) / np.sqrt(cur["position"].size)
                    dn = np.linalg.norm(nxt["position"]) / np.sqrt(nxt["position"].size)
                    row = {"seed": seed, "direction": direction, "origin": origin,
                           "source": f"{direction}_{origin}", "step": step,
                           "actual_error_delta": nxt["error"] - cur["error"],
                           "actual_brier_delta": nxt["brier"] - cur["brier"],
                           "mean_hazard": float(np.mean(true["mean"])),
                           "median_hazard": float(np.mean(true["median"])),
                           "trimmed_hazard": float(np.mean(true["trimmed"])),
                           "unsigned_distance_delta": float(dn - dc)}
                    for name, value in zip(placebo_names, placebo): row[name] = float(value)
                    rows.append(row)
    df = pd.DataFrame(rows)
    actual = df.actual_error_delta
    rmed, rmean = rho(df.median_hazard, actual), rho(df.mean_hazard, actual)
    rdist = rho(df.unsigned_distance_delta, actual)
    sign_acc = float(np.mean(np.sign(df.median_hazard) == np.sign(actual)))
    dirs = {d: rho(q.median_hazard, q.actual_error_delta) for d, q in df.groupby("direction")}
    source_sign = {s: float(np.mean(np.sign(q.median_hazard) == np.sign(q.actual_error_delta)))
                   for s, q in df.groupby("source")}
    placebo_rhos = [rho(df[name], actual) for name in placebo_names]
    rng_b = np.random.default_rng(c["bootstrap"]["seed"])
    bm, ba, bd = [], [], []
    for _ in range(c["bootstrap"]["replicates"]):
        q = bootstrap_grid(df, rng_b)
        rm = rho(q.median_hazard, q.actual_error_delta)
        bm.append(rm); ba.append(rm - rho(q.mean_hazard, q.actual_error_delta))
        bd.append(rm - rho(q.unsigned_distance_delta, q.actual_error_delta))
    ci_m = np.nanquantile(bm, [0.025, 0.5, 0.975]).tolist()
    ci_a = np.nanquantile(ba, [0.025, 0.5, 0.975]).tolist()
    ci_d = np.nanquantile(bd, [0.025, 0.5, 0.975]).tolist()
    g = c["gate"]
    checks = {"median_error_spearman": rmed >= g["minimum_median_error_spearman"],
              "sign_accuracy": sign_acc >= g["minimum_sign_accuracy"],
              "direction_replication": all(v >= g["minimum_each_direction_spearman"] for v in dirs.values()),
              "advantage_over_mean": rmed - rmean >= g["minimum_advantage_over_mean"],
              "advantage_over_distance": rmed - rdist >= g["minimum_advantage_over_unsigned_distance"],
              "source_replication": sum(v >= 0.6 for v in source_sign.values()) >= g["minimum_sources_with_sign_accuracy_at_least_0_6"],
              "placebo": rmed > float(np.nanquantile(placebo_rhos, 0.95)),
              "bootstrap_median": ci_m[0] >= g["minimum_two_way_bootstrap_median_lower"],
              "bootstrap_mean_advantage": ci_a[0] > 0,
              "bootstrap_distance_advantage": ci_d[0] > 0}
    summary = {"pretarget_sha256": expected_hash, "models": 30, "transitions": int(len(df)),
               "median_error_spearman": rmed, "mean_error_spearman": rmean,
               "trimmed_error_spearman": rho(df.trimmed_hazard, actual),
               "unsigned_distance_spearman": rdist, "sign_accuracy": sign_acc,
               "direction_spearman": dirs, "source_sign_accuracy": source_sign,
               "median_brier_spearman": rho(df.median_hazard, df.actual_brier_delta),
               "placebo_median": float(np.nanmedian(placebo_rhos)),
               "placebo_q95": float(np.nanquantile(placebo_rhos, 0.95)),
               "two_way_bootstrap_median_ci": ci_m,
               "two_way_bootstrap_mean_advantage_ci": ci_a,
               "two_way_bootstrap_distance_advantage_ci": ci_d,
               "checks": {k: bool(v) for k, v in checks.items()},
               "decision": "FRESH_HOLDOUT_PASS" if all(checks.values()) else "FRESH_HOLDOUT_STOP"}
    df.to_csv(ROOT / "results" / "transitions.csv", index=False)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "freeze", "run"])
    parser.add_argument("--expected-hash")
    args = parser.parse_args()
    if args.mode == "prepare": prepare()
    elif args.mode == "freeze": print(freeze())
    else: run(args.expected_hash)


if __name__ == "__main__": main()
