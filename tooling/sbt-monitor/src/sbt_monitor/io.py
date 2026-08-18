"""Small, dependency-free CSV helpers."""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .ledger import TaskTransportLedger


def ledger_from_csv(
    path: str | Path,
    *,
    window_col: str = "window",
    id_col: str = "identity",
    correct_col: str | None = "correct",
    truth_col: str | None = None,
    prediction_col: str | None = None,
    model_fingerprint: str | None = None,
) -> TaskTransportLedger:
    """Load a complete-panel ledger from a long-form CSV file."""

    if correct_col is None and (truth_col is None or prediction_col is None):
        raise ValueError("provide correct_col or both truth_col and prediction_col")
    grouped: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {window_col, id_col}
        if correct_col is not None:
            required.add(correct_col)
        else:
            required.update({truth_col, prediction_col})  # type: ignore[arg-type]
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            grouped.setdefault(row[window_col], []).append(row)
    ledger = TaskTransportLedger(model_fingerprint=model_fingerprint)
    for window, rows in grouped.items():
        ids = [row[id_col] for row in rows]
        if correct_col is not None:
            correct = [_parse_bool(row[correct_col]) for row in rows]
            ledger.update(
                window=window,
                ids=ids,
                correct_mask=correct,
                model_fingerprint=model_fingerprint,
            )
        else:
            ledger.update(
                window=window,
                ids=ids,
                y_true=[row[truth_col] for row in rows],  # type: ignore[index]
                y_pred=[row[prediction_col] for row in rows],  # type: ignore[index]
                model_fingerprint=model_fingerprint,
            )
    return ledger


def _parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")
