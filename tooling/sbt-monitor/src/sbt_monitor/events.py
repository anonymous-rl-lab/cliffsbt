"""Boundary-relative operational event rules.

This module is deliberately separate from :mod:`sbt_monitor.ledger`: changing a
boundary or persistence rule must never change the underlying transport ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence

import numpy as np

from .schema import CliffEvent, InvalidEventRuleError


class PersistenceProtocol(Protocol):
    """Protocol implemented by persistence-confirmation rules."""

    @property
    def name(self) -> str: ...

    def holds(self, risks: np.ndarray, start: int, boundary: float) -> bool: ...


@dataclass(frozen=True)
class ConsecutiveWindows:
    """Require ``n`` consecutive windows at or above the boundary."""

    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise InvalidEventRuleError("ConsecutiveWindows.n must be at least one")

    @property
    def name(self) -> str:
        return f"consecutive_windows:{self.n}"

    def holds(self, risks: np.ndarray, start: int, boundary: float) -> bool:
        end = start + self.n
        return end <= risks.size and bool(np.all(risks[start:end] >= boundary))


@dataclass(frozen=True)
class RemainAboveThereafter:
    """Require every observed window from the candidate onward to stay above."""

    @property
    def name(self) -> str:
        return "remain_above_thereafter"

    def holds(self, risks: np.ndarray, start: int, boundary: float) -> bool:
        return bool(start < risks.size and np.all(risks[start:] >= boundary))


@dataclass(frozen=True)
class FirstCrossingRule:
    """Ordinary first crossing of a user-declared operational boundary."""

    boundary: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.boundary):
            raise InvalidEventRuleError("boundary must be finite")

    def evaluate(
        self,
        risk: Sequence[float] | np.ndarray,
        windows: Sequence[Hashable] | None = None,
    ) -> CliffEvent:
        values = _validate_risk(risk)
        labels = _window_labels(values.size, windows)
        idx = _first_index(values >= self.boundary)
        return CliffEvent(
            boundary=float(self.boundary),
            first_crossing_time=None if idx is None else labels[idx],
            persistent_cliff_time=None,
            first_crossing_index=idx,
            persistent_cliff_index=None,
            headroom_at_start=float(self.boundary - values[0]) if values.size else float("nan"),
            persistence_rule=None,
        )


@dataclass(frozen=True)
class PersistentCliffRule:
    """Persistence-confirmed first passage.

    The first crossing and the confirmed event are reported separately.  This
    prevents an ordinary threshold crossing from being silently relabelled as a
    persistent operational cliff.
    """

    boundary: float
    persistence: PersistenceProtocol

    def __post_init__(self) -> None:
        if not np.isfinite(self.boundary):
            raise InvalidEventRuleError("boundary must be finite")
        if not hasattr(self.persistence, "holds") or not hasattr(self.persistence, "name"):
            raise InvalidEventRuleError("persistence must implement the persistence protocol")

    def evaluate(
        self,
        risk: Sequence[float] | np.ndarray,
        windows: Sequence[Hashable] | None = None,
    ) -> CliffEvent:
        values = _validate_risk(risk)
        labels = _window_labels(values.size, windows)
        first_idx = _first_index(values >= self.boundary)
        persistent_idx: int | None = None
        if first_idx is not None:
            for idx in range(first_idx, values.size):
                if values[idx] >= self.boundary and self.persistence.holds(values, idx, self.boundary):
                    persistent_idx = idx
                    break
        return CliffEvent(
            boundary=float(self.boundary),
            first_crossing_time=None if first_idx is None else labels[first_idx],
            persistent_cliff_time=None if persistent_idx is None else labels[persistent_idx],
            first_crossing_index=first_idx,
            persistent_cliff_index=persistent_idx,
            headroom_at_start=float(self.boundary - values[0]) if values.size else float("nan"),
            persistence_rule=self.persistence.name,
        )


def _validate_risk(risk: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(risk, dtype=float)
    if values.ndim != 1:
        raise ValueError("risk must be one-dimensional")
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError("risk values must be finite")
    return values


def _window_labels(n: int, windows: Sequence[Hashable] | None) -> tuple[Hashable, ...]:
    if windows is None:
        return tuple(range(n))
    labels = tuple(windows)
    if len(labels) != n:
        raise ValueError("windows length must match risk length")
    return labels


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return None if indices.size == 0 else int(indices[0])
