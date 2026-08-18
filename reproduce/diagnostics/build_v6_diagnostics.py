#!/usr/bin/env python3
"""Post hoc committed-output diagnostics for Cliff NMI manuscript v6.

These analyses do not alter any preregistered CURE-OR H1--H3 decision.
They reuse frozen model outputs, telemetry and fixed calibration/confirmation
identity splits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--repo-root",
    type=Path,
    default=Path(os.environ.get("CLIFF_REPO_ROOT", "Cliff_boundary_transport_code_v6")),
)
PARSER.add_argument(
    "--out-dir",
    type=Path,
    default=Path(os.environ.get("CLIFF_V5_DIAG_OUT", "v6_diagnostics")),
)
ARGS = PARSER.parse_args()
ROOT = ARGS.repo_root.resolve()
OUT = ARGS.out_dir.resolve()
CURE = ROOT / "cure_or"
CODE = CURE / "code"
if not CODE.exists():
    raise FileNotFoundError(f"Expected CURE-OR code at {CODE}")
sys.path.insert(0, str(CODE))

from common import (  # noqa: E402
    MODEL_SEEDS,
    SCHEDULE_IDS,
    TARGET_FAMILIES,
    WINDOWS,
    fit_ridge,
    hybrid_temporal,
    persistent_cliff,
    score_hybrid,
)
from run_phase1 import (  # noqa: E402
    build_trajectory,
    grouped_streams,
    model_outputs,
    typed_rows,
)

OUT.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260818
FALSE_BUDGET = 0.075
HORIZON = 6


def first_alarm(scores: np.ndarray, threshold: float) -> int | None:
    for t in range(1, len(scores)):
        if float(scores[t]) >= threshold:
            return t
    return None


def eval_paths(paths: List[dict], scores_by_id: Dict[str, np.ndarray], threshold: float) -> dict:
    cliffs = controls = timely = false = 0
    leads: list[int] = []
    rows: list[dict] = []
    for p in paths:
        scores = np.asarray(scores_by_id[p["id"]], dtype=float)
        alarm = first_alarm(scores, threshold)
        event = p["event"]
        is_timely = event is not None and alarm is not None and alarm < event
        is_false = event is None and alarm is not None
        cliffs += int(event is not None)
        controls += int(event is None)
        timely += int(is_timely)
        false += int(is_false)
        if is_timely:
            leads.append(int(event - alarm))
        rows.append(
            {
                "id": p["id"],
                "seed": p["seed"],
                "schedule_id": p["schedule_id"],
                "family": p["family"],
                "event": event,
                "alarm": alarm,
                "timely": is_timely,
                "false_alarm": is_false,
                "lead": (event - alarm) if is_timely else np.nan,
            }
        )
    return {
        "cliffs": cliffs,
        "controls": controls,
        "timely": timely,
        "false": false,
        "timely_rate": timely / cliffs if cliffs else np.nan,
        "false_rate": false / controls if controls else np.nan,
        "median_lead": float(np.median(leads)) if leads else np.nan,
        "rows": rows,
    }


def choose_threshold(paths: List[dict], scores: Dict[str, np.ndarray]) -> Tuple[float, dict]:
    vals = np.concatenate([np.asarray(v, dtype=float)[1:] for v in scores.values()])
    grid = np.unique(np.quantile(vals, np.linspace(0, 1, 1001)))
    grid = np.concatenate([grid, [np.nextafter(np.max(vals), np.inf)]])
    best = None
    for thr in grid:
        m = eval_paths(paths, scores, float(thr))
        if m["false_rate"] <= FALSE_BUDGET + 1e-12:
            key = (
                m["timely_rate"],
                m["median_lead"] if not np.isnan(m["median_lead"]) else -1,
                -m["false_rate"],
                -float(thr),
            )
            if best is None or key > best[0]:
                best = (key, float(thr), m)
    if best is None:
        raise RuntimeError("No threshold satisfied the calibration false-alarm budget")
    return best[1], best[2]


def temporal_slope(values: np.ndarray, t: int, width: int = 2) -> float:
    start = max(0, t - width)
    if start == t:
        return 0.0
    return float((values[t] - values[start]) / (t - start))


def reconstruct_paths() -> tuple[list[dict], list[dict]]:
    cache = np.load(CURE / "raw_outputs/features.npz")
    train_ids, test_ids = cache["train_ids"], cache["test_ids"]
    train_lookup = {int(v): i for i, v in enumerate(train_ids)}
    clean = typed_rows(CURE / "data/TRAINING_BASELINE_FROZEN.csv")
    clean_idx = [train_lookup[x["image_id"]] for x in clean]
    clean_y = np.asarray([x["class_index"] for x in clean])
    warning_model = json.loads((CURE / "config/hybrid25_warning_model_frozen.json").read_text())
    streams = {role: grouped_streams(role) for role in ("calibration", "confirmation")}
    output = {"calibration": [], "confirmation": []}
    for seed in MODEL_SEEDS:
        head = fit_ridge(cache[f"train_features_seed{seed}"][clean_idx], clean_y, 1.0)
        probs, norms = model_outputs(head, test_ids, cache["test_features"])
        for role in output:
            for schedule in SCHEDULE_IDS:
                for family in TARGET_FAMILIES:
                    item = build_trajectory(streams[role][family], probs, norms, schedule, True)
                    risk = np.asarray(item["risk"], dtype=float)
                    event = persistent_cliff(risk.tolist())
                    identifier = f"{role}|{seed}|{schedule}|{family}"
                    path = {
                        "id": identifier,
                        "role": role,
                        "seed": int(seed),
                        "schedule_id": int(schedule),
                        "family": int(family),
                        "risk": risk,
                        "event": event,
                        "endpoint_risk": float(risk[-1]),
                        "baseline_risk": float(risk[0]),
                        "hybrid25": np.asarray(item["hybrid25"], dtype=float),
                        "flux25": np.asarray(item["flux25"], dtype=float),
                        "moments25": np.asarray(item["moments25"], dtype=float),
                        "registered_scores": np.asarray(score_hybrid(item["hybrid25"], warning_model), dtype=float),
                    }
                    path["initial_headroom"] = 0.5 - path["baseline_risk"]
                    path["endpoint_overshoot"] = path["endpoint_risk"] - 0.5
                    path["endpoint_delta"] = path["endpoint_risk"] - path["baseline_risk"]
                    if event is not None:
                        path["last_pre_event_headroom"] = 0.5 - risk[event - 1] if event > 0 else np.nan
                        path["pre_event_slope"] = temporal_slope(risk, event - 1, 2) if event > 0 else np.nan
                    else:
                        path["last_pre_event_headroom"] = np.nan
                        path["pre_event_slope"] = np.nan
                    output[role].append(path)

    # Exact reconstruction check against the frozen confirmation path table.
    frozen = pd.read_csv(CURE / "raw_outputs/path_level_results.csv")
    lookup = {
        (int(r.seed), int(r.schedule_id), int(r.family)): np.asarray(json.loads(r.risk), dtype=float)
        for _, r in frozen.iterrows()
    }
    max_error = 0.0
    for p in output["confirmation"]:
        expected = lookup[(p["seed"], p["schedule_id"], p["family"])]
        max_error = max(max_error, float(np.max(np.abs(expected - p["risk"]))))
    if max_error > 1e-12:
        raise RuntimeError(f"Confirmation reconstruction mismatch: {max_error}")
    print(f"confirmation reconstruction max error: {max_error:.3g}")
    return output["calibration"], output["confirmation"]


def path_feature(p: dict, t: int, kind: str) -> np.ndarray:
    if kind == "time":
        return np.asarray([t / 12.0])
    if kind == "current_state":
        return p["hybrid25"][t, :11]
    if kind == "static_state":
        return p["hybrid25"][t, [0, 1, 2, 3, 4, 5, 6, 9, 10]]
    if kind == "static_plus_net":
        return p["hybrid25"][t, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]
    if kind == "static_plus_persistence":
        return p["hybrid25"][t, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]]
    if kind == "entropy_margin":
        m = p["moments25"]
        vals = m[:, [20, 22]]
        return np.asarray(
            [
                vals[t, 0],
                vals[t, 1],
                vals[t, 0] - vals[t - 1, 0],
                vals[t, 1] - vals[t - 1, 1],
                temporal_slope(vals[:, 0], t, 2),
                temporal_slope(vals[:, 1], t, 2),
            ]
        )
    if kind == "unsigned_shift":
        m, f = p["moments25"], p["flux25"]
        vals = np.asarray(
            [
                [
                    np.linalg.norm(m[q, :10] - m[0, :10]),
                    np.linalg.norm(m[q, 10:20] - m[0, 10:20]),
                    abs(m[q, 24] - m[0, 24]),
                    np.sum(f[q, :10]),
                ]
                for q in range(WINDOWS)
            ]
        )
        return np.concatenate([vals[t], vals[t] - vals[t - 1]])
    if kind == "full34":
        return hybrid_temporal(p["hybrid25"], t)
    raise KeyError(kind)


def samples_from_paths(paths: list[dict], kind: str):
    X, y, groups, index = [], [], [], []
    for p in paths:
        event = p["event"]
        for t in range(1, WINDOWS):
            if event is not None and t >= event:
                continue
            X.append(path_feature(p, t, kind))
            y.append(int(event is not None and 0 < event - t <= HORIZON))
            groups.append(p["family"])
            index.append((p["id"], t))
    return np.asarray(X, float), np.asarray(y, int), np.asarray(groups, int), index


def scores_from_index(paths: list[dict], index: list[tuple[str, int]], values: np.ndarray):
    scores = {p["id"]: np.zeros(WINDOWS, dtype=float) for p in paths}
    for (pid, t), value in zip(index, values):
        scores[pid][t] = value
    return scores


def fit_refit_logistic(cal_paths: list[dict], kind: str, Cs=(0.03, 0.1, 0.3, 1.0, 3.0)):
    X, y, groups, index = samples_from_paths(cal_paths, kind)
    families = sorted(set(groups.tolist()))
    selection_rows = []
    best = None
    for C in Cs:
        prediction = np.zeros(len(y), dtype=float)
        for family in families:
            train = groups != family
            test = groups == family
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=RNG_SEED,
                ),
            )
            model.fit(X[train], y[train])
            prediction[test] = model.predict_proba(X[test])[:, 1]
        oof_scores = scores_from_index(cal_paths, index, prediction)
        threshold, metric = choose_threshold(cal_paths, oof_scores)
        key = (
            metric["timely_rate"],
            metric["median_lead"] if not np.isnan(metric["median_lead"]) else -1,
            -metric["false_rate"],
        )
        selection_rows.append(
            {
                "C": C,
                "threshold": threshold,
                **{k: metric[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
            }
        )
        if best is None or key > best[0]:
            best = (key, C)

    chosen_C = best[1]
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=chosen_C,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=RNG_SEED,
        ),
    )
    final.fit(X, y)
    # Threshold is calibrated on the fully fitted calibration model, exactly as reported.
    full_values = final.predict_proba(X)[:, 1]
    full_scores = scores_from_index(cal_paths, index, full_values)
    threshold, cal_metric = choose_threshold(cal_paths, full_scores)
    return final, threshold, cal_metric, pd.DataFrame(selection_rows)


def score_logistic(paths: list[dict], kind: str, model) -> dict[str, np.ndarray]:
    scores = {}
    for p in paths:
        matrix = np.asarray([path_feature(p, t, kind) for t in range(1, WINDOWS)], dtype=float)
        values = model.predict_proba(matrix)[:, 1]
        arr = np.zeros(WINDOWS, dtype=float)
        arr[1:] = values
        scores[p["id"]] = arr
    return scores


def fit_current_risk_proxy(cal_paths: list[dict], conf_paths: list[dict]):
    X, y, groups = [], [], []
    for p in cal_paths:
        for t in range(WINDOWS):
            X.append(p["hybrid25"][t, :11])
            y.append(p["risk"][t])
            groups.append(p["family"])
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups, int)
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        prediction = np.zeros(len(y), dtype=float)
        for family in sorted(set(groups.tolist())):
            train = groups != family
            test = groups == family
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(X[train], y[train])
            prediction[test] = model.predict(X[test])
        mse = float(np.mean((prediction - y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, alpha)
    alpha = best[1]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X, y)
    output = {}
    for paths in (cal_paths, conf_paths):
        for p in paths:
            output[p["id"]] = model.predict(p["hybrid25"][:, :11])
    return alpha, output


def derived_risk_scores(paths: list[dict], risk_series: dict[str, np.ndarray], kind: str):
    out = {}
    for p in paths:
        r = np.asarray(risk_series[p["id"]], dtype=float)
        s = np.zeros(WINDOWS, dtype=float)
        if kind == "risk_only":
            s = r.copy()
        elif kind == "risk_slope":
            for t in range(1, WINDOWS):
                s[t] = r[t] + HORIZON * max(temporal_slope(r, t, 2), 0.0)
        elif kind == "risk_cusum":
            value = 0.0
            for t in range(1, WINDOWS):
                value = max(0.0, value + r[t] - r[t - 1])
                s[t] = value
        else:
            raise KeyError(kind)
        out[p["id"]] = s
    return out


def registered_scores(paths: list[dict]):
    return {p["id"]: p["registered_scores"] for p in paths}


def cluster_false_alarm_range(rows: list[dict], replicates=20000):
    frame = pd.DataFrame(rows)
    per_seed = (
        frame.groupby("seed")
        .agg(false=("false_alarm", "sum"), controls=("event", lambda x: x.isna().sum()))
        .reset_index()
    )
    array = per_seed[["false", "controls"]].to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(array), size=len(array))
        total = array[index].sum(axis=0)
        values.append(total[0] / total[1])
    return per_seed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def cliff_difficulty(cal_paths: list[dict], conf_paths: list[dict]):
    cal_eval = eval_paths(cal_paths, registered_scores(cal_paths), 0.87)
    conf_eval = eval_paths(conf_paths, registered_scores(conf_paths), 0.87)

    def frame(paths, eval_rows, role):
        by_id = {row["id"]: row for row in eval_rows}
        rows = []
        for p in paths:
            if p["event"] is None:
                continue
            result = by_id[p["id"]]
            rows.append(
                {
                    "role": role,
                    "id": p["id"],
                    "seed": p["seed"],
                    "schedule_id": p["schedule_id"],
                    "family": p["family"],
                    "event": p["event"],
                    "alarm": result["alarm"],
                    "timely": result["timely"],
                    "lead": result["lead"],
                    "baseline_risk": p["baseline_risk"],
                    "initial_headroom": p["initial_headroom"],
                    "endpoint_risk": p["endpoint_risk"],
                    "endpoint_overshoot": p["endpoint_overshoot"],
                    "endpoint_delta": p["endpoint_delta"],
                    "last_pre_event_headroom": p["last_pre_event_headroom"],
                    "pre_event_slope": p["pre_event_slope"],
                }
            )
        return pd.DataFrame(rows)

    cal = frame(cal_paths, cal_eval["rows"], "calibration")
    conf = frame(conf_paths, conf_eval["rows"], "confirmation")
    metrics = [
        "endpoint_overshoot",
        "endpoint_delta",
        "initial_headroom",
        "event",
        "last_pre_event_headroom",
        "pre_event_slope",
    ]
    cuts = {metric: cal[metric].quantile([1 / 3, 2 / 3]).tolist() for metric in metrics}

    def add_bins(data):
        data = data.copy()
        for metric, (lower, upper) in cuts.items():
            lower, upper = min(lower, upper), max(lower, upper)
            if np.isclose(lower, upper):
                bins = [-np.inf, lower, np.inf]
                labels = ["low", "high"]
            else:
                bins = [-np.inf, lower, upper, np.inf]
                labels = ["low", "mid", "high"]
            data[metric + "_bin"] = pd.cut(data[metric], bins, labels=labels, include_lowest=True)
        return data

    cal, conf = add_bins(cal), add_bins(conf)
    summaries = []
    for role, data in (("calibration", cal), ("confirmation", conf)):
        for metric in metrics:
            for level, group in data.groupby(metric + "_bin", observed=False):
                if len(group) == 0:
                    continue
                summaries.append(
                    {
                        "role": role,
                        "stratifier": metric,
                        "stratum": str(level),
                        "n": len(group),
                        "timely": int(group.timely.sum()),
                        "timely_rate": float(group.timely.mean()),
                        "median_lead": float(group.loc[group.timely, "lead"].median()) if group.timely.any() else np.nan,
                        "median_value": float(group[metric].median()),
                    }
                )
    return pd.concat([cal, conf], ignore_index=True), pd.DataFrame(summaries), cuts


def trained_peer_boundaries():
    rows = []
    # CURE-OR: shared frozen representation, independently fitted ridge heads.
    arrays = np.load(CURE / "raw_outputs/blind_predictions.npz")
    predictions = arrays["baseline_predicted"]
    seeds = arrays["seeds"]
    schedules = arrays["schedule_ids"]
    families = arrays["families"]
    streams = grouped_streams("confirmation")
    truths = {int(fam): np.asarray([int(x["class_index"]) for x in streams[int(fam)]]) for fam in families}
    for si, seed in enumerate(seeds):
        for qi, schedule in enumerate(schedules):
            for fi, family in enumerate(families):
                truth = truths[int(family)]
                error = predictions[si, qi, fi] != truth[None, :]
                risk = error.mean(axis=1)
                delta = np.diff(risk)
                candidates = [
                    (abs((predictions[pj, qi, fi, 0] != truth).mean() - risk[0]), pj)
                    for pj in range(len(seeds))
                    if pj != si
                ]
                _, peer_index = min(candidates)
                peer_error = predictions[peer_index, qi, fi] != truth[None, :]
                peer_net = ((~peer_error[:-1]) & peer_error[1:]).mean(axis=1) - (
                    peer_error[:-1] & (~peer_error[1:])
                ).mean(axis=1)
                rows.append(
                    {
                        "domain": "CURE-OR",
                        "focal_seed": int(seed),
                        "peer_seed": int(seeds[peer_index]),
                        "schedule": int(schedule),
                        "family": str(int(family)),
                        "focal_baseline_risk": risk[0],
                        "peer_baseline_risk": peer_error[0].mean(),
                        "active": bool(np.any(np.abs(delta) > 1e-12)),
                        "cliff": persistent_cliff(risk.tolist()) is not None,
                        "self_rmse": 0.0,
                        "peer_rmse": float(np.sqrt(np.mean((delta - peer_net) ** 2))),
                    }
                )

    # CIFAR-10-C: independently trained small CNN seeds.
    arrays = np.load(ROOT / "round13_second_domain/results/cifar10c_official_v1/paired_outputs.npz", allow_pickle=True)
    predictions = arrays["predictions"]
    labels = arrays["labels"]
    seeds = arrays["seeds"]
    corruptions = arrays["corruptions"]
    clean_risks = [(predictions[i, 0, 0] != labels).mean() for i in range(len(seeds))]
    for si, seed in enumerate(seeds):
        for ci, corruption in enumerate(corruptions):
            error = predictions[si, ci] != labels[None, :]
            risk = error.mean(axis=1)
            delta = np.diff(risk)
            _, peer_index = min(
                (abs(clean_risks[pj] - clean_risks[si]), pj)
                for pj in range(len(seeds))
                if pj != si
            )
            peer_error = predictions[peer_index, ci] != labels[None, :]
            peer_net = ((~peer_error[:-1]) & peer_error[1:]).mean(axis=1) - (
                peer_error[:-1] & (~peer_error[1:])
            ).mean(axis=1)
            self_net = ((~error[:-1]) & error[1:]).mean(axis=1) - (
                error[:-1] & (~error[1:])
            ).mean(axis=1)
            rows.append(
                {
                    "domain": "CIFAR-10-C",
                    "focal_seed": int(seed),
                    "peer_seed": int(seeds[peer_index]),
                    "schedule": "severity",
                    "family": str(corruption),
                    "focal_baseline_risk": risk[0],
                    "peer_baseline_risk": peer_error[0].mean(),
                    "active": bool(np.any(np.abs(delta) > 1e-12)),
                    "cliff": bool(np.any(risk >= clean_risks[si] + 0.15)),
                    "self_rmse": float(np.sqrt(np.mean((delta - self_net) ** 2))),
                    "peer_rmse": float(np.sqrt(np.mean((delta - peer_net) ** 2))),
                }
            )

    frame = pd.DataFrame(rows)
    summaries = []
    for domain, domain_frame in frame.groupby("domain"):
        for subset, mask in (
            ("all", np.ones(len(domain_frame), dtype=bool)),
            ("active", domain_frame.active.to_numpy(bool)),
            ("cliff", domain_frame.cliff.to_numpy(bool)),
        ):
            group = domain_frame.loc[mask]
            if len(group) == 0:
                continue
            summaries.append(
                {
                    "domain": domain,
                    "subset": subset,
                    "n": len(group),
                    "self_rmse_max": group.self_rmse.max(),
                    "peer_rmse_mean": group.peer_rmse.mean(),
                    "peer_rmse_median": group.peer_rmse.median(),
                    "peer_rmse_min": group.peer_rmse.min(),
                    "peer_rmse_max": group.peer_rmse.max(),
                    "baseline_risk_gap_median": np.median(
                        np.abs(group.focal_baseline_risk - group.peer_baseline_risk)
                    ),
                }
            )
    return frame, pd.DataFrame(summaries)


def main():
    cal_paths, conf_paths = reconstruct_paths()
    results = []

    registered_cal = eval_paths(cal_paths, registered_scores(cal_paths), 0.87)
    registered_conf = eval_paths(conf_paths, registered_scores(conf_paths), 0.87)
    results.append(
        {
            "method": "Registered Hybrid25",
            "type": "registered",
            "threshold": 0.87,
            **{f"cal_{k}": registered_cal[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
            **{f"conf_{k}": registered_conf[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
        }
    )

    methods = [
        ("time", "Time only (refit)"),
        ("static_state", "Static current telemetry (refit)"),
        ("static_plus_net", "Static current + net departure-recovery (refit)"),
        ("static_plus_persistence", "Static current + persistent departure (refit)"),
        ("current_state", "Current active-state telemetry (refit)"),
        ("entropy_margin", "Entropy-margin trend (refit)"),
        ("unsigned_shift", "Unsigned shift (refit)"),
        ("full34", "Hybrid25 full temporal (refit)"),
    ]
    for kind, label in methods:
        model, threshold, cal_metric, selection = fit_refit_logistic(cal_paths, kind)
        selection.to_csv(OUT / f"{kind}_selection.csv", index=False)
        conf_scores = score_logistic(conf_paths, kind, model)
        conf_metric = eval_paths(conf_paths, conf_scores, threshold)
        results.append(
            {
                "method": label,
                "type": "refit",
                "threshold": threshold,
                **{f"cal_{k}": cal_metric[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
                **{f"conf_{k}": conf_metric[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
            }
        )

    alpha, risk_series = fit_current_risk_proxy(cal_paths, conf_paths)
    for kind, label in (
        ("risk_only", "Estimated current risk"),
        ("risk_slope", "Estimated risk + slope"),
        ("risk_cusum", "Risk-proxy CUSUM"),
    ):
        cal_scores = derived_risk_scores(cal_paths, risk_series, kind)
        threshold, cal_metric = choose_threshold(cal_paths, cal_scores)
        conf_scores = derived_risk_scores(conf_paths, risk_series, kind)
        conf_metric = eval_paths(conf_paths, conf_scores, threshold)
        results.append(
            {
                "method": label,
                "type": "refit",
                "threshold": threshold,
                "risk_ridge_alpha": alpha,
                **{f"cal_{k}": cal_metric[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
                **{f"conf_{k}": conf_metric[k] for k in ("timely_rate", "false_rate", "median_lead", "timely", "false", "cliffs", "controls")},
            }
        )

    baselines = pd.DataFrame(results)
    baselines.to_csv(OUT / "warning_fair_baselines.csv", index=False)

    per_seed, lower, upper = cluster_false_alarm_range(registered_conf["rows"])
    per_seed.to_csv(OUT / "registered_false_alarm_by_seed.csv", index=False)
    (OUT / "registered_false_alarm_cluster_range.json").write_text(
        json.dumps(
            {
                "pooled_false_rate": registered_conf["false_rate"],
                "seed_cluster_percentile_range": [lower, upper],
            },
            indent=2,
        )
        + "\n"
    )

    difficulty_rows, difficulty_summary, cuts = cliff_difficulty(cal_paths, conf_paths)
    difficulty_rows.to_csv(OUT / "cure_or_cliff_difficulty_rows.csv", index=False)
    difficulty_summary.to_csv(OUT / "cure_or_cliff_difficulty_summary.csv", index=False)
    (OUT / "cure_or_cliff_difficulty_cutpoints.json").write_text(json.dumps(cuts, indent=2) + "\n")

    peer_rows, peer_summary = trained_peer_boundaries()
    peer_rows.to_csv(OUT / "trained_peer_boundary_placebo_rows.csv", index=False)
    peer_summary.to_csv(OUT / "trained_peer_boundary_placebo_summary.csv", index=False)

    def clean_markdown(frame: pd.DataFrame, floatfmt: str | None = None) -> str:
        kwargs = {"index": False}
        if floatfmt is not None:
            kwargs["floatfmt"] = floatfmt
        rendered = frame.to_markdown(**kwargs)
        return rendered.replace(" nan ", " — ").replace(" nan|", " —|")

    report = [
        "# Cliff NMI v6 committed-output diagnostic report",
        "",
        "All analyses reuse frozen outputs and fixed identity splits. No classifier, challenge path or preregistered H1–H3 decision is altered. The comparison specification was written after confirmation reveal and is therefore post hoc, although model fitting and threshold selection are restricted to calibration data.",
        "",
        "## Fair refitted warning baselines",
        "",
        clean_markdown(baselines, ".4f"),
        "",
        "## Registered false alarms by classifier-head seed",
        "",
        clean_markdown(per_seed),
        "",
        f"Descriptive complete-seed bootstrap range: [{lower:.4f}, {upper:.4f}].",
        "",
        "## Cliff-difficulty strata",
        "",
        f"Calibration cutpoints: `{json.dumps(cuts)}`",
        "",
        clean_markdown(difficulty_summary, ".4f"),
        "",
        "## Trained peer-boundary placebo",
        "",
        clean_markdown(peer_summary, ".6f"),
        "",
    ]
    (OUT / "V6_DIAGNOSTIC_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(baselines.to_string(index=False))
    print(peer_summary.to_string(index=False))


if __name__ == "__main__":
    main()
