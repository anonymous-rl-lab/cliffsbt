#!/usr/bin/env python3
"""Audit whether delivered CURE-TSR files support cross-level pairing.

This script audits filename structure only. Passing it establishes metadata
pairing, not perceptual identity; the latter still requires documentation or a
separately declared image-content audit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PATTERN = re.compile(
    r"^(?P<sequence>\d{2})_(?P<label>\d{2})_(?P<challenge>\d{2})_"
    r"(?P<level>\d{2})_(?P<index>\d+)\.bmp$",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.bmp")) if args.root.exists() else []
    parsed = []
    unparsed = []
    for path in files:
        match = PATTERN.match(path.name)
        if match is None:
            unparsed.append(str(path))
            continue
        row = {key: int(value) for key, value in match.groupdict().items()}
        row["path"] = str(path)
        parsed.append(row)

    exact_keys = [
        (r["sequence"], r["label"], r["challenge"], r["level"], r["index"])
        for r in parsed
    ]
    duplicates = sum(count - 1 for count in Counter(exact_keys).values() if count > 1)

    groups: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    labels_by_identity: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for row in parsed:
        if row["challenge"] == 0:
            continue
        groups[(row["sequence"], row["label"], row["challenge"], row["index"])].add(
            row["level"]
        )
        labels_by_identity[(row["sequence"], row["challenge"], row["index"])].add(
            row["label"]
        )

    complete = sum(levels == {1, 2, 3, 4, 5} for levels in groups.values())
    label_conflicts = sum(len(labels) > 1 for labels in labels_by_identity.values())
    report = {
        "status": "DATA_UNAVAILABLE" if not files else "AUDITED",
        "root": str(args.root.resolve()),
        "n_bmp": len(files),
        "n_parsed": len(parsed),
        "n_unparsed": len(unparsed),
        "n_duplicate_exact_keys": duplicates,
        "n_cross_level_groups": len(groups),
        "n_complete_level_1_to_5_groups": complete,
        "complete_group_fraction": complete / len(groups) if groups else 0.0,
        "n_label_conflicts": label_conflicts,
        "metadata_pairing_gate": bool(
            files
            and not unparsed
            and duplicates == 0
            and groups
            and complete == len(groups)
            and label_conflicts == 0
        ),
        "claim_boundary": (
            "A metadata pass does not by itself prove that equal indices across levels "
            "are transformations of the same underlying image."
        ),
        "unparsed_examples": unparsed[:20],
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

