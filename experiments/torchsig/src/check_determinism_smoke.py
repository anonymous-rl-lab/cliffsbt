#!/usr/bin/env python3
"""Cross-process determinism smoke test for the TorchSig Cliff runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_pilot import calibration_panel, train_deployment_model


ROOT = Path(__file__).resolve().parents[1]
FROZEN_THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def digest_array(digest: hashlib._Hash, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


def worker(config_path: Path) -> str:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = copy.deepcopy(cfg)
    smoke["training"].update(
        {"samples_per_class": 40, "n_estimators": 30, "n_jobs": 1}
    )
    smoke["calibration"].update(
        {"theta_offset_levels": [-0.06, 0.0, 0.06], "samples_per_environment": 32}
    )
    model, training = train_deployment_model(smoke)
    environment, store = calibration_panel(smoke, model)
    digest = hashlib.sha256()
    digest.update(json.dumps(training, sort_keys=True).encode())
    for column in sorted(environment.columns):
        digest.update(column.encode())
        digest_array(digest, environment[column].to_numpy())
    for key in sorted(store):
        digest.update(key.encode())
        digest_array(digest, np.asarray(store[key]))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/formal_precursor_source_v3_deterministic.json",
    )
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if args.worker:
        print(worker(config_path))
        return
    env = os.environ.copy()
    env.update(FROZEN_THREADS)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config",
               str(config_path)]
    hashes = [subprocess.check_output(command, env=env, text=True).strip() for _ in range(2)]
    result = {
        "config": str(config_path.relative_to(ROOT)),
        "frozen_environment": FROZEN_THREADS,
        "process_hashes": hashes,
        "all_passed": hashes[0] == hashes[1],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
