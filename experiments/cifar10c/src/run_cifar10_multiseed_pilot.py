#!/usr/bin/env python3
"""Run the frozen Round 13C multi-seed paired transport pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import gaussian_blur


BASE_PATH = Path(__file__).with_name("run_cifar10_paired_smoke.py")
SPEC = importlib.util.spec_from_file_location("round13_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

PATHS = (
    "gaussian_noise",
    "gaussian_blur",
    "contrast_loss",
    "darkening",
    "pixelation",
    "central_occlusion",
)
LEVELS = tuple(range(6))
SEEDS = (31, 47, 61)


def corrupt(x: torch.Tensor, path: str, level: int, noise: torch.Tensor) -> torch.Tensor:
    if path in {"gaussian_noise", "gaussian_blur", "contrast_loss"}:
        return BASE.corrupt(x, path, level, noise)
    if level == 0:
        return x
    if path == "darkening":
        scale = (0.88, 0.72, 0.56, 0.40, 0.24)[level - 1]
        return x * scale
    if path == "pixelation":
        size = (28, 24, 20, 16, 10)[level - 1]
        small = nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return nn.functional.interpolate(small, size=(32, 32), mode="nearest")
    if path == "central_occlusion":
        width = (3, 5, 7, 10, 14)[level - 1]
        out = x.clone()
        start = (32 - width) // 2
        out[:, :, start : start + width, start : start + width] = 0.5
        return out
    raise ValueError(path)


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict]]:
    BASE.set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = BASE.SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(yb)
            total_correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        schedule.step()
        history.append(
            {"epoch": epoch + 1, "loss": total_loss / total, "accuracy": total_correct / total}
        )
    return model, history


def cluster_bootstrap(values: np.ndarray, draws: int, seed: int) -> dict:
    """Bootstrap training-seed clusters; paths remain nested within seed."""
    seed_means = np.nanmean(values, axis=1)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(seed_means), size=(draws, len(seed_means)))
    dist = seed_means[sampled].mean(axis=1)
    return {
        "estimate": float(seed_means.mean()),
        "ci95": [float(np.quantile(dist, 0.025)), float(np.quantile(dist, 0.975))],
        "seed_cluster_means": seed_means.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    train_data = CIFAR10(root=args.data_root, train=True, download=False)
    test_data = CIFAR10(root=args.data_root, train=False, download=False)
    train_indices = BASE.stratified_indices(np.asarray(train_data.targets), args.train_size, 2026)
    test_indices = np.arange(len(test_data), dtype=int)
    x_train, y_train = BASE.as_tensors(train_data, train_indices)
    x_test, y_test = BASE.as_tensors(test_data, test_indices)
    y_np = y_test.numpy()

    noise = torch.randn(x_test.shape, generator=torch.Generator().manual_seed(1313))
    rng = np.random.default_rng(1313)
    offsets = rng.integers(1, 10, size=len(y_np))
    pseudo_y = (y_np + offsets) % 10

    predictions = np.empty((len(SEEDS), len(PATHS), len(LEVELS), len(y_np)), dtype=np.int16)
    margins = np.empty_like(predictions, dtype=np.float32)
    cells = []
    training = {}

    for s_idx, seed in enumerate(SEEDS):
        model, history = train_model(x_train, y_train, seed, args.epochs, args.batch_size)
        training[str(seed)] = history
        torch.save({"state_dict": model.state_dict(), "train_indices": train_indices}, args.out / f"model_seed{seed}.pt")
        print(json.dumps({"seed": seed, "final_train": history[-1]}), flush=True)
        for p_idx, path in enumerate(PATHS):
            for level in LEVELS:
                logits = BASE.predict(model, corrupt(x_test, path, level, noise), args.batch_size)
                predictions[s_idx, p_idx, level] = logits.argmax(1).numpy()
                margins[s_idx, p_idx, level] = BASE.margin(logits, y_test)
            report = BASE.analyze_path(predictions[s_idx, p_idx], margins[s_idx, p_idx], y_np)
            beta = report["errors"][0] + 0.15
            crossing = [level for level, error in enumerate(report["errors"]) if error >= beta]
            report["risk_threshold"] = beta
            report["cliff_level"] = crossing[0] if crossing else None

            pseudo_wrong = predictions[s_idx, p_idx] != pseudo_y[None, :]
            pseudo_incident = np.logical_and(~pseudo_wrong[:-1], pseudo_wrong[1:]).mean(axis=1)
            pseudo_recovery = np.logical_and(pseudo_wrong[:-1], ~pseudo_wrong[1:]).mean(axis=1)
            placebo_net = pseudo_incident - pseudo_recovery
            true_delta = np.diff(np.asarray(report["errors"]))
            report["true_accounting_rmse"] = float(
                np.sqrt(np.mean((true_delta - np.asarray(report["net_flux"])) ** 2))
            )
            report["placebo_accounting_rmse"] = float(np.sqrt(np.mean((true_delta - placebo_net) ** 2)))
            report["placebo_minus_true_rmse"] = report["placebo_accounting_rmse"] - report["true_accounting_rmse"]
            report["seed"] = seed
            report["path"] = path
            cells.append(report)

    shape = (len(SEEDS), len(PATHS))
    def matrix(key: str) -> np.ndarray:
        return np.asarray([cell[key] for cell in cells], dtype=float).reshape(shape)

    summaries = {
        "endpoint_risk_increase": cluster_bootstrap(matrix("endpoint_risk_increase"), args.bootstrap_draws, 1313),
        "endpoint_persistence": cluster_bootstrap(matrix("endpoint_persistence"), args.bootstrap_draws, 1314),
        "first_crossing_entropy": cluster_bootstrap(matrix("first_crossing_entropy"), args.bootstrap_draws, 1315),
        "median_prior_margin_gap": cluster_bootstrap(matrix("median_prior_margin_gap"), args.bootstrap_draws, 1316),
        "placebo_minus_true_rmse": cluster_bootstrap(matrix("placebo_minus_true_rmse"), args.bootstrap_draws, 1317),
    }
    clean_accuracy = np.asarray(
        [1.0 - cells[s_idx * len(PATHS)]["errors"][0] for s_idx in range(len(SEEDS))]
    )
    material_fraction = float((matrix("endpoint_risk_increase") >= 0.10).mean())
    cliff_fraction = float(np.mean([cell["cliff_level"] is not None for cell in cells]))
    consistent_paths = int(sum(np.all(matrix("endpoint_risk_increase")[:, p_idx] > 0) for p_idx in range(len(PATHS))))

    gates = {
        "model_competence": bool(np.all(clean_accuracy >= 0.45)),
        "exact_true_boundary_accounting": bool(max(cell["max_abs_accounting_error"] for cell in cells) <= 1e-12),
        "material_risk_growth": material_fraction >= 0.80,
        "headroom_exhaustion": cliff_fraction >= 0.80,
        "persistent_crossings": summaries["endpoint_persistence"]["ci95"][0] > 0.70,
        "distributed_crossings": summaries["first_crossing_entropy"]["ci95"][0] > 0.50,
        "boundary_local_ordering": summaries["median_prior_margin_gap"]["ci95"][0] > 0.0,
        "boundary_specificity": summaries["placebo_minus_true_rmse"]["ci95"][0] > 0.01,
        "seed_direction_consistency": consistent_paths >= 5,
    }
    config = {
        "train_size": args.train_size,
        "test_size": len(test_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "bootstrap_draws": args.bootstrap_draws,
        "seeds": SEEDS,
        "paths": PATHS,
        "levels": LEVELS,
        "common_train_identity_seed": 2026,
        "common_corruption_seed": 1313,
        "risk_threshold_rule": "clean_error_plus_0.15",
    }
    summary = {
        "experiment": "Round 13C controlled CIFAR-10 multi-seed pilot",
        "official_cifar10_c_result": False,
        "config": config,
        "runtime_seconds": time.time() - started,
        "training": training,
        "clean_accuracy_by_seed": clean_accuracy.tolist(),
        "material_cell_fraction": material_fraction,
        "cliff_cell_fraction": cliff_fraction,
        "positive_all_seed_paths": consistent_paths,
        "cluster_summaries": summaries,
        "cells": cells,
        "gates": gates,
        "gate_count": int(sum(gates.values())),
        "gate_total": len(gates),
        "decision": "ADVANCE_TO_OFFICIAL_BENCHMARK" if all(gates.values()) else "STOP_AND_DIAGNOSE",
    }
    np.savez_compressed(
        args.out / "paired_outputs.npz",
        predictions=predictions,
        margins=margins,
        labels=y_np,
        pseudo_labels=pseudo_y,
        test_indices=test_indices,
        train_indices=train_indices,
        seeds=np.asarray(SEEDS),
        paths=np.asarray(PATHS),
        levels=np.asarray(LEVELS),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "gates": gates, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
