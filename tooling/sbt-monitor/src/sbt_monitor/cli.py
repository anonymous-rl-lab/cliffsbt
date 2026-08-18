"""Command-line interface for :mod:`sbt_monitor`."""
from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Sequence

from . import __version__
from .diagnostics import threshold_sweep
from .events import ConsecutiveWindows, PersistentCliffRule, RemainAboveThereafter
from .io import ledger_from_csv
from .report import write_monitor_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbt-monitor",
        description="Ledger-first signed boundary transport accounting and diagnostics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the installed version")
    sub.add_parser("spec", help="print the frozen v0.1 scientific-scope specification")

    ledger = sub.add_parser("ledger", help="compute an exact identity-paired transport ledger from CSV")
    ledger.add_argument("input", type=Path)
    ledger.add_argument("--window-col", default="window")
    ledger.add_argument("--id-col", default="identity")
    outcome = ledger.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--correct-col")
    outcome.add_argument("--truth-pred-cols", nargs=2, metavar=("TRUTH", "PRED"))
    ledger.add_argument("--model-fingerprint")
    ledger.add_argument("--boundary", type=float)
    ledger.add_argument("--persistence", default="2", help="integer consecutive windows or 'thereafter'")
    ledger.add_argument("--json-out", type=Path)
    ledger.add_argument("--html-report", type=Path)

    sweep = sub.add_parser("threshold-sweep", help="re-evaluate event rules without changing the ledger")
    sweep.add_argument("input", type=Path)
    sweep.add_argument("--window-col", default="window")
    sweep.add_argument("--id-col", default="identity")
    outcome_s = sweep.add_mutually_exclusive_group(required=True)
    outcome_s.add_argument("--correct-col")
    outcome_s.add_argument("--truth-pred-cols", nargs=2, metavar=("TRUTH", "PRED"))
    sweep.add_argument("--betas", type=float, nargs="+", required=True)
    sweep.add_argument("--persistence", default="2")
    sweep.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "spec":
        print(_load_spec())
        return 0
    if args.command in {"ledger", "threshold-sweep"}:
        truth_col = prediction_col = None
        correct_col = args.correct_col
        if args.truth_pred_cols is not None:
            truth_col, prediction_col = args.truth_pred_cols
            correct_col = None
        ledger = ledger_from_csv(
            args.input,
            window_col=args.window_col,
            id_col=args.id_col,
            correct_col=correct_col,
            truth_col=truth_col,
            prediction_col=prediction_col,
            model_fingerprint=getattr(args, "model_fingerprint", None),
        )
        persistence = _parse_persistence(args.persistence)
        if args.command == "threshold-sweep":
            payload = threshold_sweep(ledger, boundaries=args.betas, persistence=persistence)
            _emit_json(payload, args.json_out)
            return 0

        event = None
        if args.boundary is not None:
            risk, windows = ledger.risk_series()
            event = PersistentCliffRule(args.boundary, persistence).evaluate(risk, windows)
        payload = ledger.to_dict()
        if event is not None:
            payload["event"] = event.to_dict()
        _emit_json(payload, args.json_out)
        if args.html_report is not None:
            write_monitor_report(args.html_report, ledger, event=event)
            print(f"wrote HTML report: {args.html_report}", file=sys.stderr)
        return 0
    raise RuntimeError(f"unhandled command {args.command!r}")


def _parse_persistence(value: str):
    if value.lower() == "thereafter":
        return RemainAboveThereafter()
    try:
        count = int(value)
    except ValueError as exc:
        raise SystemExit("--persistence must be an integer or 'thereafter'") from exc
    return ConsecutiveWindows(count)


def _load_spec() -> str:
    return resources.files("sbt_monitor.data").joinpath("API_SCIENTIFIC_SCOPE_v0.1.md").read_text(encoding="utf-8")


def _emit_json(payload, destination: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if destination is None:
        print(text)
    else:
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"wrote JSON: {destination}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
