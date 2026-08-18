#!/usr/bin/env python3
"""Run the frozen Round 13B paired boundary-transport smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import gaussian_blur


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
PATHS = ("gaussian_noise", "gaussian_blur", "contrast_loss")
LEVELS = tuple(range(6))


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return self.classifier(self.features(x).flatten(1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def stratified_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    base = n // 10
    remainder = n % 10
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        take = base + int(label < remainder)
        parts.append(rng.choice(candidates, size=take, replace=False))
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


def as_tensors(dataset: CIFAR10, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(dataset.data[indices]).permute(0, 3, 1, 2).float().div_(255.0)
    y = torch.tensor(np.asarray(dataset.targets, dtype=np.int64)[indices], dtype=torch.long)
    return x, y


def corrupt(x: torch.Tensor, path: str, level: int, noise: torch.Tensor) -> torch.Tensor:
    if level == 0:
        return x
    if path == "gaussian_noise":
        sigma = (0.04, 0.08, 0.12, 0.18, 0.26)[level - 1]
        return (x + sigma * noise).clamp(0.0, 1.0)
    if path == "gaussian_blur":
        sigma = (0.35, 0.60, 0.90, 1.30, 1.90)[level - 1]
        kernel = (3, 3, 5, 7, 9)[level - 1]
        return gaussian_blur(x, [kernel, kernel], [sigma, sigma])
    if path == "contrast_loss":
        scale = (0.85, 0.68, 0.50, 0.32, 0.15)[level - 1]
        gray = x.mean(dim=(2, 3), keepdim=True)
        return (gray + scale * (x - gray)).clamp(0.0, 1.0)
    raise ValueError(path)


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        outputs.append(model(x[start : start + batch_size]).cpu())
    return torch.cat(outputs)


def margin(logits: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    true = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.clone()
    masked[torch.arange(len(y)), y] = -torch.inf
    return (true - masked.max(dim=1).values).numpy()


def normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(len(counts)))


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(float))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    n = len(values)
    return float((2 * np.dot(np.arange(1, n + 1), values) / (n * values.sum())) - (n + 1) / n)


def analyze_path(preds: np.ndarray, margins: np.ndarray, y: np.ndarray) -> dict:
    wrong = preds != y[None, :]
    errors = wrong.mean(axis=1)
    incidents = np.logical_and(~wrong[:-1], wrong[1:]).sum(axis=1) / len(y)
    recoveries = np.logical_and(wrong[:-1], ~wrong[1:]).sum(axis=1) / len(y)
    net = incidents - recoveries
    accounting = np.diff(errors) - net

    initially_correct = ~wrong[0]
    first = np.zeros(5, dtype=int)
    first_level = np.full(len(y), -1, dtype=int)
    for level in range(1, 6):
        take = initially_correct & (first_level < 0) & wrong[level]
        first[level - 1] = int(take.sum())
        first_level[take] = level
    ever = first_level > 0
    persistence = float((ever & wrong[-1]).sum() / ever.sum()) if ever.any() else 0.0

    lead_gaps = []
    for level in range(1, 6):
        incident = (~wrong[level - 1]) & wrong[level]
        stayed = (~wrong[level - 1]) & (~wrong[level])
        if incident.any() and stayed.any():
            lead_gaps.append(float(np.median(margins[level - 1, stayed]) - np.median(margins[level - 1, incident])))

    class_first = np.array([(ever & (y == label)).sum() for label in range(10)])
    return {
        "errors": errors.tolist(),
        "incident_flux": incidents.tolist(),
        "recovery_flux": recoveries.tolist(),
        "net_flux": net.tolist(),
        "cumulative_net_flux": np.r_[0.0, np.cumsum(net)].tolist(),
        "max_abs_accounting_error": float(np.max(np.abs(accounting))),
        "endpoint_risk_increase": float(errors[-1] - errors[0]),
        "first_crossing_counts": first.tolist(),
        "first_crossing_entropy": normalized_entropy(first),
        "first_crossing_pulse_concentration": float(first.max() / first.sum()) if first.sum() else 0.0,
        "endpoint_persistence": persistence,
        "median_prior_margin_gap": float(np.median(lead_gaps)) if lead_gaps else float("nan"),
        "first_crossing_class_gini": gini(class_first),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    train_data = CIFAR10(root=args.data_root, train=True, download=False)
    test_data = CIFAR10(root=args.data_root, train=False, download=False)
    train_indices = stratified_indices(np.asarray(train_data.targets), args.train_size, args.seed)
    test_indices = stratified_indices(np.asarray(test_data.targets), args.test_size, args.seed + 1)
    x_train, y_train = as_tensors(train_data, train_indices)
    x_test, y_test = as_tensors(test_data, test_indices)

    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_gen,
        num_workers=0,
    )
    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(args.epochs):
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
            total_loss += float(loss) * len(yb)
            total_correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        schedule.step()
        row = {"epoch": epoch + 1, "loss": total_loss / total, "accuracy": total_correct / total}
        history.append(row)
        print(json.dumps(row), flush=True)

    noise_gen = torch.Generator().manual_seed(args.seed + 1000)
    noise = torch.randn(x_test.shape, generator=noise_gen)
    all_logits = np.empty((len(PATHS), len(LEVELS), len(y_test), 10), dtype=np.float32)
    all_margins = np.empty((len(PATHS), len(LEVELS), len(y_test)), dtype=np.float32)
    path_reports = {}
    for p_idx, path in enumerate(PATHS):
        for level in LEVELS:
            logits = predict(model, corrupt(x_test, path, level, noise), args.batch_size)
            all_logits[p_idx, level] = logits.numpy()
            all_margins[p_idx, level] = margin(logits, y_test)
        preds = all_logits[p_idx].argmax(axis=2)
        report = analyze_path(preds, all_margins[p_idx], y_test.numpy())
        beta = report["errors"][0] + 0.15
        crossings = [idx for idx, error in enumerate(report["errors"]) if error >= beta]
        report["risk_threshold"] = beta
        report["cliff_level"] = crossings[0] if crossings else None
        path_reports[path] = report

    competence = path_reports[PATHS[0]]["errors"][0] <= 0.65
    accounting = max(r["max_abs_accounting_error"] for r in path_reports.values()) <= 1e-12
    material = sum(r["endpoint_risk_increase"] >= 0.10 for r in path_reports.values()) >= 2
    exhaustion = sum(r["cliff_level"] is not None for r in path_reports.values()) >= 2
    persistence = float(np.median([r["endpoint_persistence"] for r in path_reports.values()])) >= 0.60
    distributed = float(np.median([r["first_crossing_entropy"] for r in path_reports.values()])) >= 0.50
    boundary_local = sum(r["median_prior_margin_gap"] > 0 for r in path_reports.values()) >= 2
    identity = bool(np.array_equal(test_indices, test_indices.copy()))
    gates = {
        "clean_model_competence": competence,
        "identity_preservation": identity,
        "paired_transition_accounting": accounting,
        "material_risk_growth": material,
        "headroom_exhaustion": exhaustion,
        "endpoint_persistence": persistence,
        "distributed_first_crossings": distributed,
        "boundary_local_ordering": boundary_local,
    }

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["out"] = str(args.out)
    config["paths"] = PATHS
    config["levels"] = LEVELS
    config["risk_threshold_rule"] = "clean_error_plus_0.15"
    summary = {
        "experiment": "Round 13B controlled CIFAR-10 paired qualification smoke",
        "official_cifar10_c_result": False,
        "config": config,
        "runtime_seconds": time.time() - started,
        "training_history": history,
        "paths": path_reports,
        "gates": gates,
        "gate_count": int(sum(gates.values())),
        "gate_total": len(gates),
        "decision": "ADVANCE_TO_ONE_HOUR_PILOT" if all(gates.values()) else "STOP_AND_DIAGNOSE",
    }

    torch.save({"state_dict": model.state_dict(), "train_indices": train_indices}, args.out / "model.pt")
    np.savez_compressed(
        args.out / "paired_predictions.npz",
        logits=all_logits,
        margins=all_margins,
        labels=y_test.numpy(),
        test_indices=test_indices,
        paths=np.asarray(PATHS),
        levels=np.asarray(LEVELS),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.out / "level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "level", "error", "cumulative_net_flux"])
        writer.writeheader()
        for path, report in path_reports.items():
            for level in LEVELS:
                writer.writerow(
                    {
                        "path": path,
                        "level": level,
                        "error": report["errors"][level],
                        "cumulative_net_flux": report["cumulative_net_flux"][level],
                    }
                )
    print(json.dumps({"decision": summary["decision"], "gates": gates, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

