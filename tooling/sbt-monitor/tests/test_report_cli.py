from __future__ import annotations

import json
from pathlib import Path

from sbt_monitor import ConsecutiveWindows, PersistentCliffRule, TaskTransportLedger
from sbt_monitor.cli import main
from sbt_monitor.report import build_monitor_report


def test_html_report_contains_scope_boundaries() -> None:
    ledger = TaskTransportLedger.fixed_panel(["a", "b"])
    ledger.update(window=0, ids=["a", "b"], correct_mask=[True, True])
    ledger.update(window=1, ids=["a", "b"], correct_mask=[False, True])
    risk, windows = ledger.risk_series()
    event = PersistentCliffRule(0.5, ConsecutiveWindows(1)).evaluate(risk, windows)
    report = build_monitor_report(ledger, event=event)
    assert "not a universal alarm" in report.lower()
    assert "Safety certification" in report
    assert "Unsupported interpretation" in report


def test_cli_ledger_and_spec(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "window,identity,correct\n"
        "0,a,1\n0,b,1\n"
        "1,a,0\n1,b,1\n"
        "2,a,0\n2,b,0\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "ledger.json"
    html_path = tmp_path / "ledger.html"
    assert main(
        [
            "ledger",
            str(csv_path),
            "--correct-col",
            "correct",
            "--boundary",
            "0.5",
            "--persistence",
            "2",
            "--json-out",
            str(json_path),
            "--html-report",
            str(html_path),
        ]
    ) == 0
    payload = json.loads(json_path.read_text())
    assert payload["closure"]["all_close"]
    assert html_path.exists()
    assert main(["spec"]) == 0
    captured = capsys.readouterr()
    assert "Frozen API and Scientific-Scope Specification" in captured.out
