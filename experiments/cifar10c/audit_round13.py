#!/usr/bin/env python3
"""Independent audit of Round 13 frozen gates and paired raw outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        if sha256(ROOT / rel) != expected:
            return False
    return True


def main() -> None:
    smoke = json.loads((ROOT / "results/cifar10_paired_smoke_seed13/summary.json").read_text())
    pilot = json.loads((ROOT / "results/cifar10_multiseed_pilot_v1/summary.json").read_text())
    official = json.loads((ROOT / "results/cifar10c_official_v1/summary.json").read_text())
    raw = np.load(ROOT / "results/cifar10c_official_v1/paired_outputs.npz")
    predictions = raw["predictions"]
    labels = raw["labels"]
    wrong = predictions != labels[None, None, None, :]
    errors = wrong.mean(axis=3)
    incident = np.logical_and(~wrong[:, :, :-1], wrong[:, :, 1:]).mean(axis=3)
    recovery = np.logical_and(wrong[:, :, :-1], ~wrong[:, :, 1:]).mean(axis=3)
    accounting_error = float(np.max(np.abs(np.diff(errors, axis=2) - (incident - recovery))))

    checks = {
        "round13b_8_of_8": smoke["gate_count"] == smoke["gate_total"] == 8,
        "round13c_9_of_9": pilot["gate_count"] == pilot["gate_total"] == 9,
        "round13d_12_of_12": official["gate_count"] == official["gate_total"] == 12,
        "round13d_decision": official["decision"] == "MECHANISM_DOMAIN_CONFIRMED",
        "official_output_shape": predictions.shape == (3, 15, 6, 10000),
        "independent_exact_accounting": accounting_error <= 1e-12,
        "official_archive_hash_recorded": official["archive_md5"] == "56bf5dcef84df0e2308c6dcbcbbd8499",
        "round13c_pretarget_hashes": verify_manifest(ROOT / "ROUND13C_PRETARGET_SHA256.txt"),
        "round13d_v2_pretarget_hashes": verify_manifest(ROOT / "ROUND13D_PRETARGET_V2_SHA256.txt"),
        "cure_claim_withheld": json.loads((ROOT / "results/cure_pairing_eligibility.json").read_text())["status"] == "DATA_UNAVAILABLE",
    }
    report = {
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "max_independent_accounting_error": accounting_error,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

