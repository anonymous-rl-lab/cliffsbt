#!/usr/bin/env python3
"""Read-only integrity and scientific-result audit for the v2 package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "PACKAGE_MANIFEST.sha256"
REPORT = PACKAGE / "PACKAGE_AUDIT.json"
BLIND_COMMIT = "4a888493531dfa797efa765adc2057e73564eacc59a9e42c86b8ef2b27c1b237"
EXPECTED_SEEDS = [113, 127, 139, 151, 163]
EXPECTED_SCHEDULES = [211, 223, 227]
EXPECTED_FAMILIES = [2, 6, 11, 12, 13, 14, 15, 16, 17, 18]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str):
    return json.loads((PACKAGE / relative).read_text(encoding="utf-8"))


def manifest_valid() -> bool:
    if not MANIFEST.is_file():
        return False
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = PACKAGE / relative
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


def evidence_manifest_valid(relative: str) -> bool:
    manifest = PACKAGE / "raw_outputs" / relative
    if not manifest.is_file():
        return False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = manifest.parent / name
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


def residual_scan_clean() -> bool:
    forbidden = [
        re.compile(r"(?i)(?<![a-z0-9])" + "v" + r"[._ -]?" + "1" + r"(?![a-z0-9])"),
        re.compile("(?i)" + "dp" + "nx4"),
        re.compile("(?i)" + "first" + " " + "registration"),
        re.compile("(?i)" + "prior" + " " + "registration"),
        re.compile("(?i)" + "previous" + " " + "registration"),
        re.compile("(?i)" + "development" + "_" + "disclosure"),
        re.compile("(?i)" + "original" + " " + "registered model seeds"),
    ]
    text_suffixes = {".py", ".md", ".json", ".csv", ".txt", ".cff", ".lock", ".sha256", ".log"}
    for path in PACKAGE.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if any(pattern.search(relative) for pattern in forbidden):
            return False
        if path.suffix.lower() in text_suffixes:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in forbidden):
                return False
    return True


def csv_rows(relative: str) -> list[dict[str, str]]:
    with (PACKAGE / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_checks(strict: bool) -> dict[str, bool]:
    required = [
        "README.md", "PROVENANCE.json", "docs/EXPERIMENT_PROTOCOL.md",
        "docs/DATA_DICTIONARY.md", "docs/RESULTS_AND_CONCLUSIONS.md",
        "docs/REPRODUCIBILITY.md", "code/run_phase1.py", "code/run_phase2.py",
        "data/TARGET_STREAMS_FROZEN.json", "raw_outputs/features.npz",
        "raw_outputs/blind_predictions.npz", "raw_outputs/results.json",
        "raw_outputs/path_level_results.csv", "raw_outputs/repair_path_results.csv",
        "raw_outputs/seed_level_results.csv",
    ]
    result = read_json("raw_outputs/results.json")
    arrays = np.load(PACKAGE / "raw_outputs" / "blind_predictions.npz", allow_pickle=False)
    features = np.load(PACKAGE / "raw_outputs" / "features.npz", allow_pickle=False)
    path_rows = csv_rows("raw_outputs/path_level_results.csv")
    repair_rows = csv_rows("raw_outputs/repair_path_results.csv")
    seed_rows = csv_rows("raw_outputs/seed_level_results.csv")
    checks = {
        "required_files_present": all((PACKAGE / item).is_file() for item in required),
        "package_manifest_valid": manifest_valid(),
        "phase1_blind_manifest_valid": evidence_manifest_valid("BLIND_OUTPUTS.sha256"),
        "phase1_blind_commit_matches_osf": sha256(PACKAGE / "raw_outputs" / "BLIND_OUTPUTS.sha256") == BLIND_COMMIT,
        "postreveal_manifest_valid": evidence_manifest_valid("POSTREVEAL.sha256"),
        "evidence_manifest_valid": evidence_manifest_valid("EVIDENCE.sha256"),
        "no_retired_residuals": residual_scan_clean(),
        "overall_decision": result.get("overall_decision") == "FORMATION_WARNING_REPAIR_CONFIRMED",
        "h1_pass": result["h1_formation"]["pass"] is True,
        "h1_counts": result["h1_formation"]["persistent_cliffs"] == 72 and result["h1_formation"]["noncliff_controls"] == 78,
        "h1_flow_closure": result["h1_formation"]["maximum_flow_closure_error"] <= 1e-12,
        "h2_pass": result["h2_warning"]["pass"] is True,
        "h2_counts": result["h2_warning"]["timely"] == 71 and result["h2_warning"]["false"] == 3,
        "h2_lead": result["h2_warning"]["median_lead"] == 3.0,
        "h3_pass": result["h3_repair"]["pass"] is True,
        "h3_cliffs": result["h3_repair"]["unique_model_family_cliffs_removed"] == 5 and result["h3_repair"]["new_model_family_cliffs"] == 0,
        "h3_risk_interval_positive": result["h3_repair"]["mean_risk_gain"]["ci95"][0] > 0,
        "path_table_rows": len(path_rows) == 150,
        "repair_table_rows": len(repair_rows) == 150,
        "seed_table_rows": len(seed_rows) == 5,
        "prediction_axes": arrays["seeds"].tolist() == EXPECTED_SEEDS and arrays["schedule_ids"].tolist() == EXPECTED_SCHEDULES and arrays["families"].tolist() == EXPECTED_FAMILIES,
        "prediction_shape": arrays["baseline_predicted"].shape == (5, 3, 10, 13, 50),
        "hybrid25_shape": arrays["hybrid25_sensors"].shape == (5, 3, 10, 13, 25),
        "test_feature_shape": features["test_features"].shape == (4200, 768),
        "train_feature_shapes": all(features[f"train_features_seed{seed}"].shape == (1650, 768) for seed in EXPECTED_SEEDS),
    }
    if strict:
        proc = subprocess.run(
            [sys.executable, str(PACKAGE / "code" / "freeze_design.py"), "--verify"],
            check=False,
            capture_output=True,
            text=True,
        )
        checks["frozen_design_verify"] = proc.returncode == 0
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    checks = build_checks(args.strict)
    report = {
        "all_pass": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "overall_decision": read_json("raw_outputs/results.json")["overall_decision"],
        "protocol_version": "2.0",
        "registration": "https://osf.io/c6ygf",
    }
    if args.write_report:
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
