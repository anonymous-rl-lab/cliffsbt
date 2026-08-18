#!/usr/bin/env python3
"""Verify that the v2 frozen design is internally consistent and outcome blind."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict

from common import CONFIG, DATA, MODEL_SEEDS, PACKAGE, SCHEDULE_IDS, TARGET_FAMILIES


def verify() -> None:
    experiment = json.loads((CONFIG / "experiment.json").read_text(encoding="utf-8"))
    warning = json.loads((CONFIG / "warning_protocol.json").read_text(encoding="utf-8"))
    repair = json.loads((CONFIG / "repair_protocol.json").read_text(encoding="utf-8"))
    claims = json.loads((PACKAGE / "CLAIMS_AND_GATES.json").read_text(encoding="utf-8"))
    model = json.loads((CONFIG / "hybrid25_warning_model_frozen.json").read_text(encoding="utf-8"))
    if experiment["model_seeds"] != MODEL_SEEDS or experiment["schedule_ids"] != SCHEDULE_IDS:
        raise RuntimeError("experiment model/schedule constants differ from code")
    if experiment["target_families"] != TARGET_FAMILIES:
        raise RuntimeError("target family constants differ from code")
    if len(set(MODEL_SEEDS)) != 5 or len(set(SCHEDULE_IDS)) != 3:
        raise RuntimeError("expected five unique model seeds and three unique schedules")
    if warning["telemetry"]["sensor_channels"] != 25 or warning["telemetry"]["active_channels"] != 11:
        raise RuntimeError("Hybrid25 channel declaration mismatch")
    if len(model["coefficient"]) != 34 or len(model["scaler_mean"]) != 34 or len(model["scaler_scale"]) != 34:
        raise RuntimeError("frozen warning readout must have 34 temporal inputs")
    if float(model["alarm_threshold"]) != float(warning["readout"]["threshold"]):
        raise RuntimeError("frozen warning threshold mismatch")
    if repair["samples_per_fragment"] * 100 != repair["total_budget"]:
        raise RuntimeError("repair budget is not full-fragment coverage")
    with (DATA / "REPAIR_CANDIDATES_FROZEN.csv").open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    fragments = defaultdict(list)
    for row in candidates:
        fragments[row["fragment"]].append(row)
    if len(fragments) != 100 or min(len(rows) for rows in fragments.values()) < repair["samples_per_fragment"]:
        raise RuntimeError("repair pool cannot supply the frozen full-coverage budget")
    streams = json.loads((DATA / "TARGET_STREAMS_FROZEN.json").read_text(encoding="utf-8"))
    for role in ("calibration", "confirmation"):
        counts = Counter(int(row["family"]) for row in streams[role])
        if counts != Counter({family: 50 for family in TARGET_FAMILIES}):
            raise RuntimeError(f"{role} stream identity counts differ from 50 per family")
    phase1_tree = ast.parse((PACKAGE / "code" / "run_phase1.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(phase1_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    if "run_phase2" in imports:
        raise RuntimeError("phase 1 imports phase 2")
    if claims["h3_repair"]["unique_model_family_cliffs_removed_min"] < 1:
        raise RuntimeError("repair gate does not require actual cliff disappearance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        raise SystemExit("this package is already frozen; use --verify")
    verify()
    print("FROZEN DESIGN VERIFY PASS")


if __name__ == "__main__":
    main()
