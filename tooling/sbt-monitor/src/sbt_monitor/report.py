"""Provenance-rich HTML reports for ledgers and calibrated readouts."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .ledger import TaskTransportLedger
from .schema import CliffEvent


def build_monitor_report(
    ledger: TaskTransportLedger,
    *,
    event: CliffEvent | None = None,
    warning_metadata: Mapping[str, Any] | None = None,
    title: str = "SBT Monitor report",
) -> str:
    """Return a self-contained HTML evidence card.

    The report deliberately states both supported and unsupported
    interpretations.  It is not a safety certificate.
    """

    closure = ledger.closure_report()
    steps = ledger.steps()
    pairing_coverages = [step.pairing.pairing_coverage for step in steps]
    min_coverage = min(pairing_coverages, default=1.0)
    rows = "".join(
        "<tr>"
        f"<td>{_h(step.window_from)}</td>"
        f"<td>{_h(step.window_to)}</td>"
        f"<td>{step.risk_from:.6g}</td>"
        f"<td>{step.risk_to:.6g}</td>"
        f"<td>{step.incident:.6g}</td>"
        f"<td>{step.recovery:.6g}</td>"
        f"<td>{step.sbt:.6g}</td>"
        f"<td>{step.turnover:.6g}</td>"
        f"<td>{step.closure_error:.3e}</td>"
        "</tr>"
        for step in steps
    )

    event_html = "<p>No operational event rule was supplied.</p>"
    if event is not None:
        event_html = f"""
        <dl>
          <dt>User-declared boundary</dt><dd>{event.boundary:.6g}</dd>
          <dt>First crossing</dt><dd>{_h(event.first_crossing_time)}</dd>
          <dt>Persistence-confirmed cliff</dt><dd>{_h(event.persistent_cliff_time)}</dd>
          <dt>Persistence rule</dt><dd>{_h(event.persistence_rule)}</dd>
          <dt>Safety certification</dt><dd>Not provided by this package</dd>
        </dl>
        """

    warning_html = "<p>No calibrated warning readout metadata was supplied.</p>"
    if warning_metadata is not None:
        warning_html = "<pre>" + html.escape(json.dumps(dict(warning_metadata), indent=2, sort_keys=True)) + "</pre>"

    supported = [
        "Adjacent fixed-model paired risk change is accounted for by incident minus recovery mass.",
        "Pairing coverage, closure error and telescopeability are explicitly reported.",
    ]
    if event is not None:
        supported.append("The supplied user rule separates first crossing from persistence-confirmed operational cliff.")
    unsupported = [
        "This report does not choose or certify a safety boundary.",
        "Exact closure alone is not causal attribution or advance warning.",
        "Prediction-state transport proxies are not task-error SBT.",
        "Warning scores do not authorize automatic shutdown, retraining or safety action.",
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1080px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
h1, h2 {{ line-height: 1.15; }}
.card {{ border: 1px solid #bbb; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.45rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
dt {{ font-weight: 650; }} dd {{ margin-bottom: 0.5rem; }}
.warning {{ border-left: 5px solid #555; padding-left: 1rem; }}
code, pre {{ white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="warning"><strong>Scope:</strong> ledger-first monitoring report; not a universal alarm or safety certificate.</p>
<section class="card">
<h2>Exact accounting</h2>
<dl>
  <dt>Model fingerprint</dt><dd>{_h(ledger.model_fingerprint)}</dd>
  <dt>Declared panel size</dt><dd>{len(ledger.panel_ids) if ledger.n_windows else 0}</dd>
  <dt>Windows</dt><dd>{ledger.n_windows}</dd>
  <dt>Minimum pairing coverage</dt><dd>{min_coverage:.4f}</dd>
  <dt>Maximum closure error</dt><dd>{closure.max_abs_error:.3e}</dd>
  <dt>Closure tolerance</dt><dd>{closure.tolerance:.3e}</dd>
  <dt>Complete cumulative ledger</dt><dd>{'Yes' if closure.telescopeable else 'No'}</dd>
</dl>
</section>
<section class="card">
<h2>Operational event</h2>
{event_html}
</section>
<section class="card">
<h2>Resolved transport ledger</h2>
<table>
<thead><tr><th>From</th><th>To</th><th>Risk from</th><th>Risk to</th><th>Incident</th><th>Recovery</th><th>SBT</th><th>Turnover</th><th>Closure error</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</section>
<section class="card"><h2>Warning provenance</h2>{warning_html}</section>
<section class="card"><h2>Supported interpretation</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in supported)}</ul></section>
<section class="card"><h2>Unsupported interpretation</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in unsupported)}</ul></section>
</body>
</html>"""


def write_monitor_report(
    path: str | Path,
    ledger: TaskTransportLedger,
    *,
    event: CliffEvent | None = None,
    warning_metadata: Mapping[str, Any] | None = None,
    title: str = "SBT Monitor report",
) -> Path:
    """Write :func:`build_monitor_report` to disk."""

    destination = Path(path)
    destination.write_text(
        build_monitor_report(
            ledger,
            event=event,
            warning_metadata=warning_metadata,
            title=title,
        ),
        encoding="utf-8",
    )
    return destination


def _h(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and np.isnan(value):
        return "—"
    return html.escape(str(value))
