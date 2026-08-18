#!/usr/bin/env python3
"""Round 13E equal-budget coverage versus hazard-hit training repair."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10


BASE_PATH = Path(__file__).with_name("run_cifar10_paired_smoke.py")
SPEC = importlib.util.spec_from_file_location("round13_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
SMOKE_CORRUPTIONS = (
    "gaussian_noise", "glass_blur", "snow", "brightness", "contrast",
    "jpeg_compression",
)
ARMS = ("baseline", "random", "hazard", "coverage")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def stratified_split(labels: np.ndarray, calibration_size: int, holdout_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calibration = []
    holdout = []
    per_cal = calibration_size // 10
    per_hold = holdout_size // 10
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        calibration.extend(indices[:per_cal])
        holdout.extend(indices[per_cal : per_cal + per_hold])
    return np.asarray(sorted(calibration), dtype=int), np.asarray(sorted(holdout), dtype=int)


def build_candidates(
    predictions: np.ndarray,
    margins: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
) -> list[dict]:
    candidates = []
    mean_margin = margins.mean(axis=0)
    wrong_votes = (predictions != labels[None, None, None, :]).sum(axis=0)
    majority_wrong = wrong_votes >= 2
    for corruption_index, corruption in enumerate(CORRUPTIONS):
        for identity in calibration:
            if majority_wrong[corruption_index, 0, identity]:
                continue
            first = None
            for severity in range(1, 6):
                if (not majority_wrong[corruption_index, severity - 1, identity]) and majority_wrong[corruption_index, severity, identity]:
                    first = severity
                    break
            if first is None:
                continue
            candidates.append(
                {
                    "identity": int(identity),
                    "corruption": corruption,
                    "corruption_index": corruption_index,
                    "severity": first,
                    "label": int(labels[identity]),
                    "hazard_score": float(-mean_margin[corruption_index, first, identity]),
                }
            )
    return candidates


def select_sets(candidates: list[dict], calibration: np.ndarray, labels: np.ndarray, budget: int, seed: int) -> dict[str, list[dict]]:
    if len(candidates) < budget:
        raise RuntimeError(f"Only {len(candidates)} first-crossing candidates for budget {budget}")
    hazard = sorted(candidates, key=lambda row: (-row["hazard_score"], row["corruption_index"], row["identity"]))[:budget]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        key = (row["corruption"], row["label"], row["severity"])
        groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (-row["hazard_score"], row["identity"]))
    coverage = []
    offsets = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(coverage) < budget:
        advanced = False
        for key in keys:
            offset = offsets[key]
            if offset < len(groups[key]):
                coverage.append(groups[key][offset])
                offsets[key] += 1
                advanced = True
                if len(coverage) == budget:
                    break
        if not advanced:
            break

    rng = np.random.default_rng(seed)
    total_cells = len(calibration) * len(CORRUPTIONS) * 5
    flat = rng.choice(total_cells, size=budget, replace=False)
    random_rows = []
    for value in flat:
        severity_index = int(value % 5)
        value //= 5
        corruption_index = int(value % len(CORRUPTIONS))
        identity = int(calibration[int(value // len(CORRUPTIONS))])
        random_rows.append(
            {
                "identity": identity,
                "corruption": CORRUPTIONS[corruption_index],
                "corruption_index": corruption_index,
                "severity": severity_index + 1,
                "label": int(labels[identity]),
                "hazard_score": None,
            }
        )
    return {"baseline": [], "random": random_rows, "hazard": hazard, "coverage": coverage}


def selection_stats(rows: list[dict], candidate_keys: set[tuple]) -> dict:
    fragments = {(r["corruption"], r["label"], r["severity"]) for r in rows}
    hit = sum((r["identity"], r["corruption"], r["severity"]) in candidate_keys for r in rows)
    return {
        "budget": len(rows),
        "hazard_hit_rate": hit / len(rows) if rows else None,
        "unique_fragments": len(fragments),
        "families": len({r["corruption"] for r in rows}),
        "classes": len({r["label"] for r in rows}),
        "severities": len({r["severity"] for r in rows}),
        "fragment_counts": {"|".join(map(str, key)): sum((r["corruption"], r["label"], r["severity"]) == key for r in rows) for key in sorted(fragments)},
    }


def load_repair_images(rows: list[dict], arrays: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        return torch.empty((0, 3, 32, 32)), torch.empty((0,), dtype=torch.long)
    images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        source = (row["severity"] - 1) * 10000 + row["identity"]
        images[index] = arrays[row["corruption"]][source]
        labels[index] = row["label"]
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    return x, torch.from_numpy(labels)


def train_model(x: torch.Tensor, y: torch.Tensor, seed: int, epochs: int, batch_size: int) -> tuple[nn.Module, list[dict]]:
    BASE.set_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    model = BASE.SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        schedule.step()
        history.append({"epoch": epoch + 1, "loss": loss_sum / total, "accuracy": correct / total})
    return model, history


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(array, copy=True)).permute(0, 3, 1, 2).float().div_(255.0)


def evaluate(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    clean_data: CIFAR10,
    labels: np.ndarray,
    holdout: np.ndarray,
    corruptions: tuple[str, ...],
    beta: float | None,
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_np = labels[holdout]
    y = torch.from_numpy(y_np)
    clean_x = to_tensor(clean_data.data[holdout])
    clean_logits = BASE.predict(model, clean_x, batch_size)
    clean_pred = clean_logits.argmax(1).numpy()
    clean_margin = BASE.margin(clean_logits, y)
    clean_error = float((clean_pred != y_np).mean())
    if beta is None:
        beta = clean_error + 0.15
    predictions = np.empty((len(corruptions), 6, len(holdout)), dtype=np.int16)
    margins = np.empty_like(predictions, dtype=np.float32)
    cells = []
    for c_index, corruption in enumerate(corruptions):
        predictions[c_index, 0] = clean_pred
        margins[c_index, 0] = clean_margin
        for severity in range(1, 6):
            indices = (severity - 1) * 10000 + holdout
            x = to_tensor(arrays[corruption][indices])
            logits = BASE.predict(model, x, batch_size)
            predictions[c_index, severity] = logits.argmax(1).numpy()
            margins[c_index, severity] = BASE.margin(logits, y)
        report = BASE.analyze_path(predictions[c_index], margins[c_index], y_np)
        crossing = [level for level, error in enumerate(report["errors"]) if error >= beta]
        report["cliff_level"] = crossing[0] if crossing else None
        report["corruption"] = corruption
        cells.append(report)
    error_matrix = np.asarray([cell["errors"] for cell in cells])
    result = {
        "clean_error": clean_error,
        "risk_threshold": beta,
        "endpoint_error_mean": float(error_matrix[:, -1].mean()),
        "risk_area_mean": float(error_matrix[:, 1:].mean()),
        "endpoint_net_flux_mean": float(np.mean([cell["cumulative_net_flux"][-1] for cell in cells])),
        "crossing_fraction": float(np.mean([cell["cliff_level"] is not None for cell in cells])),
        "max_accounting_error": float(max(cell["max_abs_accounting_error"] for cell in cells)),
        "cells": cells,
    }
    return result, predictions, margins


def paired_bootstrap(differences: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(draws, len(differences)))
    distribution = differences[sampled].mean(axis=1)
    return {
        "estimate": float(differences.mean()),
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "seed_differences": differences.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--round13d-outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "smoke":
        seeds, train_size, cal_size, hold_size, budget, epochs = (67,), 5000, 500, 1000, 250, 3
        eval_corruptions = SMOKE_CORRUPTIONS
        bootstrap_draws = 1000
    else:
        seeds, train_size, cal_size, hold_size, budget, epochs = (71, 83, 97), 20000, 2000, 8000, 1000, 8
        eval_corruptions = CORRUPTIONS
        bootstrap_draws = 20000

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    train_data = CIFAR10(root=args.data_root, train=True, download=False)
    test_data = CIFAR10(root=args.data_root, train=False, download=False)
    labels = np.asarray(test_data.targets, dtype=np.int64)
    calibration, holdout = stratified_split(labels, cal_size, hold_size, 1305)
    reference = np.load(args.round13d_outputs)
    predictions_ref = reference["predictions"]
    margins_ref = reference["margins"]
    candidates = build_candidates(predictions_ref, margins_ref, labels, calibration)
    selections = select_sets(candidates, calibration, labels, budget, 1306)
    candidate_keys = {(r["identity"], r["corruption"], r["severity"]) for r in candidates}
    stats = {arm: selection_stats(rows, candidate_keys) for arm, rows in selections.items()}
    atomic_json(args.out / "selections.json", {"mode": args.mode, "calibration": calibration.tolist(), "holdout": holdout.tolist(), "stats": stats, "selections": selections})

    arrays = {corruption: np.load(args.data_root / "CIFAR-10-C" / f"{corruption}.npy", mmap_mode="r") for corruption in CORRUPTIONS}
    train_indices = BASE.stratified_indices(np.asarray(train_data.targets), train_size, 2026)
    x_clean, y_clean = BASE.as_tensors(train_data, train_indices)
    repair_tensors = {arm: load_repair_images(rows, arrays) for arm, rows in selections.items()}

    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": [], "started_at": time.time()}
    results: dict[str, dict[str, dict]] = {str(seed): {} for seed in seeds}
    baseline_beta: dict[int, float] = {}

    for seed in seeds:
        arm_order = ARMS
        for arm in arm_order:
            key = f"seed{seed}_{arm}"
            fit_dir = args.out / "fits" / key
            fit_dir.mkdir(parents=True, exist_ok=True)
            summary_path = fit_dir / "summary.json"
            if key in state["completed"] and summary_path.exists():
                result = json.loads(summary_path.read_text())
                results[str(seed)][arm] = result
                if arm == "baseline":
                    baseline_beta[seed] = result["metrics"]["clean_error"] + 0.15
                continue

            x_repair, y_repair = repair_tensors[arm]
            x_train = torch.cat([x_clean, x_repair])
            y_train = torch.cat([y_clean, y_repair])
            fit_started = time.time()
            model, history = train_model(x_train, y_train, seed, epochs, args.batch_size)
            checkpoint = fit_dir / "model.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed, "arm": arm, "train_indices": train_indices}, checkpoint)

            if arm == "baseline":
                metrics, pred, mar = evaluate(model, arrays, test_data, labels, holdout, eval_corruptions, None, args.batch_size)
                baseline_beta[seed] = metrics["risk_threshold"]
            else:
                metrics, pred, mar = evaluate(model, arrays, test_data, labels, holdout, eval_corruptions, baseline_beta[seed], args.batch_size)
            np.savez_compressed(fit_dir / "paired_outputs.npz", predictions=pred, margins=mar, labels=labels[holdout], holdout=holdout, corruptions=np.asarray(eval_corruptions), levels=np.arange(6))
            result = {
                "seed": seed, "arm": arm, "mode": args.mode,
                "budget": len(selections[arm]), "beta": baseline_beta[seed],
                "runtime_seconds": time.time() - fit_started,
                "training_history": history, "metrics": metrics,
            }
            atomic_json(summary_path, result)
            results[str(seed)][arm] = result
            state["completed"].append(key)
            state["last_completed"] = key
            state["updated_at"] = time.time()
            atomic_json(state_path, state)
            print(json.dumps({"completed": key, "runtime_seconds": result["runtime_seconds"], "metrics": {k: metrics[k] for k in ("clean_error", "endpoint_error_mean", "risk_area_mean", "crossing_fraction")}}), flush=True)

    all_metrics = ("endpoint_error_mean", "risk_area_mean", "crossing_fraction", "clean_error")
    matrices = {metric: {arm: np.asarray([results[str(seed)][arm]["metrics"][metric] for seed in seeds]) for arm in ARMS} for metric in all_metrics}
    comparisons = {}
    if len(seeds) > 1:
        for metric in all_metrics:
            for other in ("hazard", "baseline", "random"):
                name = f"coverage_minus_{other}_{metric}"
                comparisons[name] = paired_bootstrap(matrices[metric]["coverage"] - matrices[metric][other], bootstrap_draws, 1400 + len(comparisons))

    competence_threshold = 0.30 if args.mode == "smoke" else 0.45
    competence = all(1.0 - results[str(seed)][arm]["metrics"]["clean_error"] >= competence_threshold for seed in seeds for arm in ARMS)
    accounting = max(results[str(seed)][arm]["metrics"]["max_accounting_error"] for seed in seeds for arm in ARMS) <= 1e-12
    if args.mode == "smoke":
        gates = {
            "exact_budgets": all(stats[arm]["budget"] == budget for arm in ("random", "hazard", "coverage")),
            "matched_hazard_hit": stats["hazard"]["hazard_hit_rate"] == stats["coverage"]["hazard_hit_rate"] == 1.0,
            "coverage_separation": stats["coverage"]["unique_fragments"] > stats["hazard"]["unique_fragments"],
            "all_fits_complete": len(state["completed"]) == len(seeds) * len(ARMS),
            "finite_metrics": all(np.isfinite(results[str(seed)][arm]["metrics"][metric]) for seed in seeds for arm in ARMS for metric in all_metrics),
            "model_competence": competence,
            "exact_accounting": accounting,
        }
        decision = "ADVANCE_TO_FORMAL" if all(gates.values()) else "STOP_AND_DIAGNOSE"
    else:
        gates = {
            "equal_dangerous_hit_rate": stats["hazard"]["hazard_hit_rate"] == stats["coverage"]["hazard_hit_rate"] == 1.0 and stats["hazard"]["budget"] == stats["coverage"]["budget"] == budget,
            "coverage_separation": stats["coverage"]["unique_fragments"] >= 2 * stats["hazard"]["unique_fragments"] and stats["coverage"]["families"] >= 14,
            "model_competence": competence,
            "exact_paired_accounting": accounting,
            "coverage_beats_hazard_endpoint": comparisons["coverage_minus_hazard_endpoint_error_mean"]["ci95"][1] < 0,
            "coverage_beats_hazard_area": comparisons["coverage_minus_hazard_risk_area_mean"]["ci95"][1] < 0,
            "coverage_reduces_crossing": comparisons["coverage_minus_hazard_crossing_fraction"]["estimate"] < 0,
            "coverage_beats_baseline_endpoint": comparisons["coverage_minus_baseline_endpoint_error_mean"]["ci95"][1] < 0,
            "clean_risk_guard": comparisons["coverage_minus_hazard_clean_error"]["ci95"][1] <= 0.02,
        }
        decision = "COVERAGE_MECHANISM_CONFIRMED" if all(gates.values()) else "PARTIAL_OR_STOP"

    summary = {
        "experiment": "Round 13E coverage versus hazard-hit repair",
        "mode": args.mode,
        "config": {"seeds": seeds, "train_size": train_size, "calibration_size": cal_size, "holdout_size": hold_size, "budget": budget, "epochs": epochs, "arms": ARMS, "evaluation_corruptions": eval_corruptions, "threshold_rule": "baseline_clean_error_plus_0.15"},
        "runtime_seconds": time.time() - started,
        "selection_stats": stats,
        "results": results,
        "comparisons": comparisons,
        "gates": gates,
        "gate_count": int(sum(gates.values())),
        "gate_total": len(gates),
        "decision": decision,
    }
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({"decision": decision, "gates": gates, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
