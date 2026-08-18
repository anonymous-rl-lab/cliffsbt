"""Ledger-first signed boundary transport monitoring tools.

The stable top-level API intentionally excludes warning calibration.  Import the
experimental workflow from :mod:`sbt_monitor.experimental` after installing the
``warning`` extra.
"""
from . import diagnostics
from .events import (
    ConsecutiveWindows,
    FirstCrossingRule,
    PersistentCliffRule,
    RemainAboveThereafter,
)
from .ledger import TaskTransportLedger
from .proxy import PredictionStateTransport, TransportAwareStateBuilder
from .report import build_monitor_report, write_monitor_report

__version__ = "0.1.0"

__all__ = [
    "TaskTransportLedger",
    "FirstCrossingRule",
    "PersistentCliffRule",
    "ConsecutiveWindows",
    "RemainAboveThereafter",
    "PredictionStateTransport",
    "TransportAwareStateBuilder",
    "build_monitor_report",
    "write_monitor_report",
    "diagnostics",
    "__version__",
]
