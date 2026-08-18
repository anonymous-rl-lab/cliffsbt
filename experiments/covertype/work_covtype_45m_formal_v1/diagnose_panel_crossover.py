#!/usr/bin/env python3
"""Posttarget one-seed crossover of pilot/formal fit and evaluation panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
FLOW_ROOT = ROOT.parent / "work_covtype_boundary_flow_probe"
PILOT_ROOT = ROOT.parent / "work_covtype_45m_multiseed_pilot"
sys.path.insert(0, str(FLOW_ROOT))
import run_boundary_flow as bf  # noqa: E402


SEED = 2026081541


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def main() -> None:
    c = json.loads((ROOT / "config.json").read_text())
    base = {"source": c["source"], "task": c["task"],
            "model": {"hidden_layer_sizes": c["model"]["hidden_layer_sizes"],
                      "alpha": c["model"]["alpha"],
                      "learning_rate_init": c["model"]["learning_rate_init"],
                      "batch_size": c["model"]["batch_size"],
                      "max_iter": c["model"]["max_iter"], "seed": SEED}}
    x, y, _ = bf.load_source(base)
    panel_sets = {"pilot": load_npz(PILOT_ROOT / "data" / "block_panels.npz"),
                  "formal": load_npz(ROOT / "data" / "block_panels.npz")}
    rows = []
    for fit_name, fit_panels in panel_sets.items():
        for direction in c["deployment"]["directions"]:
            ordered = list(range(c["blocks"]["count"]))
            if direction == "descending":
                ordered.reverse()
            for origin in c["deployment"]["origin_blocks"]:
                train_idx = np.concatenate([fit_panels[f"fit_b{ordered[pos]:02d}_c{cls}"]
                                            for pos in range(origin) for cls in (0, 1)])
                scaler, model, train_hidden, train_margin = bf.fit_model(x, y, train_idx, base)
                hmu, hsd = train_hidden.mean(axis=0), train_hidden.std(axis=0)
                hsd[hsd < 1e-6] = 1.0
                train_hz = (train_hidden - hmu) / hsd
                train_labels = y[train_idx]
                train_pos = {cls: train_hz[train_labels == cls].mean(axis=0) for cls in (0, 1)}
                epsilon = float(np.quantile(np.abs(train_margin), 0.2))
                w = model.coefs_[-1].reshape(-1)
                for eval_name, eval_panels in panel_sets.items():
                    seq = []
                    for position in range(origin, origin + 3):
                        block = ordered[position]
                        idx = np.concatenate([eval_panels[f"eval_b{block:02d}_c0"],
                                              eval_panels[f"eval_b{block:02d}_c1"]])
                        hidden, logits = bf.forward_hidden_logits(model, scaler.transform(x[idx]))
                        margin = np.where(y[idx] == 1, logits, -logits)
                        hz = (hidden - hmu) / hsd
                        position_vec = np.concatenate([
                            hz[y[idx] == cls].mean(axis=0) - train_pos[cls] for cls in (0, 1)])
                        seq.append({"y": y[idx], "hidden": hidden, "margin": margin,
                                    "error": float(np.mean(margin <= 0)), "position": position_vec})
                    for step in range(2):
                        cur, nxt = seq[step], seq[step + 1]
                        terms = []
                        for cls in (0, 1):
                            s = 1.0 if cls == 1 else -1.0
                            hc = cur["hidden"][cur["y"] == cls]
                            hn = nxt["hidden"][nxt["y"] == cls]
                            inward = -s * float((hn.mean(axis=0) - hc.mean(axis=0)) @ w)
                            crowd = float(np.mean(np.abs(cur["margin"][cur["y"] == cls]) <= epsilon))
                            terms.append(crowd * inward / max(epsilon, 1e-8))
                        dc = np.linalg.norm(cur["position"]) / np.sqrt(cur["position"].size)
                        dn = np.linalg.norm(nxt["position"]) / np.sqrt(nxt["position"].size)
                        rows.append({"fit_panel": fit_name, "eval_panel": eval_name,
                                     "direction": direction, "origin": origin, "step": step,
                                     "actual_delta": nxt["error"] - cur["error"],
                                     "hazard": float(np.mean(terms)),
                                     "distance_delta": float(dn - dc)})
    df = pd.DataFrame(rows)
    summary = {}
    for (fit_name, eval_name), q in df.groupby(["fit_panel", "eval_panel"]):
        nz = np.abs(q["actual_delta"].to_numpy()) > 1e-12
        rh = rho(q["hazard"], q["actual_delta"])
        rd = rho(q["distance_delta"], q["actual_delta"])
        summary[f"fit_{fit_name}__eval_{eval_name}"] = {
            "n": int(len(q)), "hazard_spearman": rh, "distance_spearman": rd,
            "advantage": rh - rd,
            "sign_accuracy": float(np.mean(np.sign(q["hazard"].to_numpy()[nz]) ==
                                            np.sign(q["actual_delta"].to_numpy()[nz]))),
            "ascending_spearman": rho(q[q["direction"] == "ascending"]["hazard"],
                                       q[q["direction"] == "ascending"]["actual_delta"]),
            "descending_spearman": rho(q[q["direction"] == "descending"]["hazard"],
                                        q[q["direction"] == "descending"]["actual_delta"])}
    result = {"posttarget_diagnostic": True, "seed": SEED, "summary": summary}
    df.to_csv(ROOT / "results" / "panel_crossover_transitions.csv", index=False)
    (ROOT / "results" / "panel_crossover_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
