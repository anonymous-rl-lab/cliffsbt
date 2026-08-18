#!/usr/bin/env python3
"""Evaluate frozen CNNs on official CIFAR-10-C with paired flux accounting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10


BASE_PATH = Path(__file__).with_name("run_cifar10_paired_smoke.py")
SPEC = importlib.util.spec_from_file_location("round13_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SEEDS = (31, 47, 61)
CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
EXPECTED_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_bootstrap(values: np.ndarray, draws: int, seed: int) -> dict:
    seed_means = np.nanmean(values, axis=1)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(seed_means), size=(draws, len(seed_means)))
    dist = seed_means[sampled].mean(axis=1)
    return {
        "estimate": float(seed_means.mean()),
        "ci95": [float(np.quantile(dist, 0.025)), float(np.quantile(dist, 0.975))],
        "seed_cluster_means": seed_means.tolist(),
    }


def tensor_from_uint8(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(array, copy=True)).permute(0, 3, 1, 2).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    archive_md5 = md5(args.archive)

    labels_path = args.data_root / "labels.npy"
    labels_all = np.load(labels_path)
    labels = labels_all[:10000]
    file_shapes = {}
    arrays = {}
    for corruption in CORRUPTIONS:
        array = np.load(args.data_root / f"{corruption}.npy", mmap_mode="r")
        arrays[corruption] = array
        file_shapes[corruption] = list(array.shape)
    identity_layout = bool(
        labels_all.shape == (50000,)
        and all(np.array_equal(labels_all[:10000], labels_all[level * 10000 : (level + 1) * 10000]) for level in range(5))
        and all(tuple(array.shape) == (50000, 32, 32, 3) for array in arrays.values())
    )

    clean_data = CIFAR10(root=args.data_root.parent, train=False, download=False)
    clean_x = tensor_from_uint8(clean_data.data)
    clean_y = np.asarray(clean_data.targets, dtype=np.int64)
    if not np.array_equal(clean_y, labels):
        raise RuntimeError("Official CIFAR-10 and CIFAR-10-C labels are not aligned")
    y = torch.from_numpy(clean_y)
    rng = np.random.default_rng(1313)
    pseudo_y = (clean_y + rng.integers(1, 10, size=len(clean_y))) % 10

    predictions = np.empty((len(SEEDS), len(CORRUPTIONS), 6, len(y)), dtype=np.int16)
    margins = np.empty_like(predictions, dtype=np.float32)
    cells = []
    clean_accuracy = []

    for s_idx, seed in enumerate(SEEDS):
        checkpoint = torch.load(
            args.checkpoint_dir / f"model_seed{seed}.pt", map_location="cpu", weights_only=False
        )
        model = BASE.SmallCNN()
        model.load_state_dict(checkpoint["state_dict"])
        clean_logits = BASE.predict(model, clean_x, args.batch_size)
        clean_pred = clean_logits.argmax(1).numpy()
        clean_margin = BASE.margin(clean_logits, y)
        clean_accuracy.append(float((clean_pred == clean_y).mean()))
        print(json.dumps({"seed": seed, "clean_accuracy": clean_accuracy[-1]}), flush=True)

        for c_idx, corruption in enumerate(CORRUPTIONS):
            predictions[s_idx, c_idx, 0] = clean_pred
            margins[s_idx, c_idx, 0] = clean_margin
            for severity in range(1, 6):
                start = (severity - 1) * 10000
                stop = severity * 10000
                x = tensor_from_uint8(arrays[corruption][start:stop])
                logits = BASE.predict(model, x, args.batch_size)
                predictions[s_idx, c_idx, severity] = logits.argmax(1).numpy()
                margins[s_idx, c_idx, severity] = BASE.margin(logits, y)

            report = BASE.analyze_path(predictions[s_idx, c_idx], margins[s_idx, c_idx], clean_y)
            beta = report["errors"][0] + 0.15
            crossing = [level for level, error in enumerate(report["errors"]) if error >= beta]
            report["risk_threshold"] = beta
            report["cliff_level"] = crossing[0] if crossing else None
            if crossing:
                level = crossing[0]
                headroom = beta - report["errors"][0]
                cumulative = report["cumulative_net_flux"]
                report["first_crossing_correct"] = bool(
                    cumulative[level] + 1e-12 >= headroom
                    and (level == 0 or cumulative[level - 1] < headroom + 1e-12)
                )
            else:
                report["first_crossing_correct"] = True

            pseudo_wrong = predictions[s_idx, c_idx] != pseudo_y[None, :]
            pseudo_incident = np.logical_and(~pseudo_wrong[:-1], pseudo_wrong[1:]).mean(axis=1)
            pseudo_recovery = np.logical_and(pseudo_wrong[:-1], ~pseudo_wrong[1:]).mean(axis=1)
            placebo_net = pseudo_incident - pseudo_recovery
            true_delta = np.diff(np.asarray(report["errors"]))
            true_rmse = float(np.sqrt(np.mean((true_delta - np.asarray(report["net_flux"])) ** 2)))
            placebo_rmse = float(np.sqrt(np.mean((true_delta - placebo_net) ** 2)))
            report.update(
                {
                    "true_accounting_rmse": true_rmse,
                    "placebo_accounting_rmse": placebo_rmse,
                    "placebo_minus_true_rmse": placebo_rmse - true_rmse,
                    "seed": seed,
                    "corruption": corruption,
                }
            )
            cells.append(report)

    shape = (len(SEEDS), len(CORRUPTIONS))
    def matrix(key: str) -> np.ndarray:
        return np.asarray([cell[key] for cell in cells], dtype=float).reshape(shape)

    crossing_mask = np.asarray([cell["cliff_level"] is not None for cell in cells]).reshape(shape)
    def crossing_matrix(key: str) -> np.ndarray:
        values = matrix(key)
        return np.where(crossing_mask, values, np.nan)

    summaries = {
        "endpoint_risk_increase": cluster_bootstrap(matrix("endpoint_risk_increase"), args.bootstrap_draws, 1321),
        "endpoint_persistence_crossing": cluster_bootstrap(crossing_matrix("endpoint_persistence"), args.bootstrap_draws, 1322),
        "first_crossing_entropy_crossing": cluster_bootstrap(crossing_matrix("first_crossing_entropy"), args.bootstrap_draws, 1323),
        "median_prior_margin_gap_crossing": cluster_bootstrap(crossing_matrix("median_prior_margin_gap"), args.bootstrap_draws, 1324),
        "placebo_minus_true_rmse": cluster_bootstrap(matrix("placebo_minus_true_rmse"), args.bootstrap_draws, 1325),
    }
    all_seed_crossing_families = int(np.sum(np.all(crossing_mask, axis=0)))
    positive_all_seed_families = int(np.sum(np.all(matrix("endpoint_risk_increase") > 0, axis=0)))
    cliff_medians = []
    for c_idx in range(len(CORRUPTIONS)):
        levels = [cells[s_idx * len(CORRUPTIONS) + c_idx]["cliff_level"] for s_idx in range(len(SEEDS))]
        finite = [level for level in levels if level is not None]
        if finite:
            cliff_medians.append(float(np.median(finite)))
    distinct_median_levels = len(set(cliff_medians))

    gates = {
        "archive_integrity": archive_md5 == EXPECTED_MD5,
        "official_identity_layout": identity_layout,
        "model_competence": bool(np.all(np.asarray(clean_accuracy) >= 0.45)),
        "exact_task_boundary_accounting": bool(max(c["max_abs_accounting_error"] for c in cells) <= 1e-12),
        "reproducible_headroom_exhaustion": all_seed_crossing_families >= 10,
        "first_crossing_correctness": bool(all(c["first_crossing_correct"] for c in cells)),
        "persistent_crossings": summaries["endpoint_persistence_crossing"]["ci95"][0] > 0.70,
        "distributed_crossings": summaries["first_crossing_entropy_crossing"]["ci95"][0] > 0.50,
        "boundary_local_ordering": summaries["median_prior_margin_gap_crossing"]["ci95"][0] > 0.0,
        "boundary_specificity": summaries["placebo_minus_true_rmse"]["ci95"][0] > 0.01,
        "seed_direction_consistency": positive_all_seed_families >= 12,
        "cross_family_heterogeneity": distinct_median_levels >= 2,
    }
    summary = {
        "experiment": "Round 13D official CIFAR-10-C paired mechanism benchmark",
        "doi": "10.5281/zenodo.2535967",
        "archive_md5": archive_md5,
        "expected_archive_md5": EXPECTED_MD5,
        "config": {
            "seeds": SEEDS,
            "corruptions": CORRUPTIONS,
            "levels": [0, 1, 2, 3, 4, 5],
            "test_size": len(labels),
            "risk_threshold_rule": "clean_error_plus_0.15",
            "bootstrap_draws": args.bootstrap_draws,
        },
        "runtime_seconds": time.time() - started,
        "file_shapes": file_shapes,
        "labels_file_shape": list(labels_all.shape),
        "labels_block_shape": list(labels.shape),
        "clean_accuracy_by_seed": clean_accuracy,
        "all_seed_crossing_families": all_seed_crossing_families,
        "positive_all_seed_families": positive_all_seed_families,
        "distinct_median_cliff_levels": distinct_median_levels,
        "cluster_summaries": summaries,
        "cells": cells,
        "gates": gates,
        "gate_count": int(sum(gates.values())),
        "gate_total": len(gates),
        "decision": "MECHANISM_DOMAIN_CONFIRMED" if all(gates.values()) else "STOP_AND_DIAGNOSE",
        "claim_boundary": (
            "This confirms paired boundary transport in a controlled common-corruption CNN domain; "
            "it does not replace the data-gated real-world CURE-TSR replication."
        ),
    }
    np.savez_compressed(
        args.out / "paired_outputs.npz",
        predictions=predictions,
        margins=margins,
        labels=clean_y,
        pseudo_labels=pseudo_y,
        seeds=np.asarray(SEEDS),
        corruptions=np.asarray(CORRUPTIONS),
        levels=np.arange(6),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "gates": gates, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
