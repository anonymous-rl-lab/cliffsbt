"""Shared data structures and exceptions for :mod:`sbt_monitor`.

The module intentionally keeps the stable layer free of optional machine-learning
libraries.  Only NumPy and the Python standard library are used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


class SBTMonitorError(Exception):
    """Base exception for all package-specific errors."""


class PairingError(SBTMonitorError):
    """Raised when identity-pairing requirements are violated."""


class DuplicateIdentityError(PairingError):
    """Raised when a window contains duplicate identity IDs."""


class MissingIdentityError(PairingError):
    """Raised when a fixed identity panel is incomplete."""


class ModelFingerprintError(SBTMonitorError):
    """Raised when one ledger is updated with different model fingerprints."""


class LeakageError(SBTMonitorError):
    """Raised when calibration and evaluation identifiers overlap."""


class OptionalDependencyError(SBTMonitorError, ImportError):
    """Raised when an optional package extra is required but unavailable."""


class InvalidEventRuleError(SBTMonitorError, ValueError):
    """Raised when an operational event rule is malformed."""


@dataclass(frozen=True)
class WindowRecord:
    """One outcome window aligned to the ledger's declared identity panel."""

    window: Hashable
    ids: tuple[Hashable, ...]
    errors: np.ndarray
    weights: np.ndarray
    model_fingerprint: str | None = None
    margins: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", np.asarray(self.errors, dtype=bool))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        if self.margins is not None:
            object.__setattr__(self, "margins", np.asarray(self.margins, dtype=float))
        if len(self.ids) != self.errors.size or self.errors.size != self.weights.size:
            raise ValueError("ids, errors and weights must have the same length")
        if self.margins is not None and self.margins.size != self.errors.size:
            raise ValueError("margins must match ids length")


@dataclass(frozen=True)
class PairingReport:
    """Identity alignment diagnostics for one adjacent transition."""

    window_from: Hashable
    window_to: Hashable
    n_from: int
    n_to: int
    n_pairs: int
    pairing_coverage: float
    missing_from: tuple[Hashable, ...] = ()
    missing_to: tuple[Hashable, ...] = ()
    unexpected_from: tuple[Hashable, ...] = ()
    unexpected_to: tuple[Hashable, ...] = ()
    denominator_changed: bool = False
    telescopeable: bool = True


@dataclass(frozen=True)
class TransportStep:
    """Resolved task-error transport between two consecutive windows."""

    window_from: Hashable
    window_to: Hashable
    risk_from: float
    risk_to: float
    incident: float
    recovery: float
    sbt: float
    turnover: float
    closure_error: float
    pairing: PairingReport
    incident_ids: tuple[Hashable, ...] = ()
    recovery_ids: tuple[Hashable, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_from": self.window_from,
            "window_to": self.window_to,
            "risk_from": self.risk_from,
            "risk_to": self.risk_to,
            "incident": self.incident,
            "recovery": self.recovery,
            "sbt": self.sbt,
            "turnover": self.turnover,
            "closure_error": self.closure_error,
            "incident_ids": list(self.incident_ids),
            "recovery_ids": list(self.recovery_ids),
            "pairing": {
                "n_from": self.pairing.n_from,
                "n_to": self.pairing.n_to,
                "n_pairs": self.pairing.n_pairs,
                "pairing_coverage": self.pairing.pairing_coverage,
                "missing_from": list(self.pairing.missing_from),
                "missing_to": list(self.pairing.missing_to),
                "unexpected_from": list(self.pairing.unexpected_from),
                "unexpected_to": list(self.pairing.unexpected_to),
                "denominator_changed": self.pairing.denominator_changed,
                "telescopeable": self.pairing.telescopeable,
            },
        }


@dataclass(frozen=True)
class ClosureReport:
    """Numerical closure and cumulative-eligibility summary."""

    n_steps: int
    max_abs_error: float
    mean_abs_error: float
    all_close: bool
    tolerance: float
    telescopeable: bool
    non_telescopeable_steps: tuple[tuple[Hashable, Hashable], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "all_close": self.all_close,
            "tolerance": self.tolerance,
            "telescopeable": self.telescopeable,
            "non_telescopeable_steps": [list(x) for x in self.non_telescopeable_steps],
        }


@dataclass(frozen=True)
class CliffEvent:
    """Boundary-relative first crossing and persistence-confirmed event."""

    boundary: float
    first_crossing_time: Hashable | None
    persistent_cliff_time: Hashable | None
    first_crossing_index: int | None
    persistent_cliff_index: int | None
    headroom_at_start: float
    persistence_rule: str | None = None

    @property
    def crossed(self) -> bool:
        return self.first_crossing_index is not None

    @property
    def persistent(self) -> bool:
        return self.persistent_cliff_index is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "first_crossing_time": self.first_crossing_time,
            "persistent_cliff_time": self.persistent_cliff_time,
            "first_crossing_index": self.first_crossing_index,
            "persistent_cliff_index": self.persistent_cliff_index,
            "headroom_at_start": self.headroom_at_start,
            "persistence_rule": self.persistence_rule,
            "crossed": self.crossed,
            "persistent": self.persistent,
        }


@dataclass(frozen=True)
class ProxyState:
    """One outcome-blind prediction-state transport summary."""

    window: Hashable
    values: Mapping[str, float]
    n_pairs: int
    pairing_coverage: float
    identity_anchored: bool = True
    is_task_sbt: bool = False

    def vector(self, feature_names: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray([self.values[name] for name in feature_names], dtype=float)
        except KeyError as exc:
            raise KeyError(f"proxy state does not contain feature {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class WarningEpisode:
    """A calibrated-warning episode.

    ``event_time`` is an integer position in the state sequence at which a
    persistence-confirmed event begins.  ``None`` denotes a control episode.
    """

    episode_id: str
    states: np.ndarray
    feature_names: tuple[str, ...]
    event_time: int | None
    group: Hashable | None = None
    identity_set_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        if states.ndim != 2:
            raise ValueError("states must have shape (time, features)")
        if states.shape[1] != len(self.feature_names):
            raise ValueError("feature_names length must match states columns")
        if states.shape[0] == 0:
            raise ValueError("episode must contain at least one state")
        if not np.isfinite(states).all():
            raise ValueError("states must contain only finite values")
        if self.event_time is not None and not (0 <= self.event_time < states.shape[0]):
            raise ValueError("event_time must index the episode or be None")
        object.__setattr__(self, "states", states)

    @property
    def is_control(self) -> bool:
        return self.event_time is None


@dataclass(frozen=True)
class WarningEvaluation:
    """Episode-level warning performance."""

    n_events: int
    n_controls: int
    timely_events: int
    false_controls: int
    timely_rate: float
    false_alarm_rate: float
    median_lead: float | None
    premature_events: int
    late_or_missed_events: int
    first_alarm_by_episode: Mapping[str, int | None]
    score_by_episode: Mapping[str, tuple[float, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_events": self.n_events,
            "n_controls": self.n_controls,
            "timely_events": self.timely_events,
            "false_controls": self.false_controls,
            "timely_rate": self.timely_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "median_lead": self.median_lead,
            "premature_events": self.premature_events,
            "late_or_missed_events": self.late_or_missed_events,
            "first_alarm_by_episode": dict(self.first_alarm_by_episode),
            "score_by_episode": {k: list(v) for k, v in self.score_by_episode.items()},
        }
