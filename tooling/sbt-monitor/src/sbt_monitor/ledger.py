"""Exact identity-paired task transport ledgers.

The stable ledger layer implements a deliberately narrow scientific contract:
for one fixed model and a fixed weighted identity panel, adjacent risk change is
exactly incident mass minus recovery mass.  Operational thresholds are handled
separately in :mod:`sbt_monitor.events`.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .schema import (
    ClosureReport,
    DuplicateIdentityError,
    MissingIdentityError,
    ModelFingerprintError,
    PairingError,
    PairingReport,
    TransportStep,
    WindowRecord,
)

MissingPolicy = Literal["raise", "intersection"]


def _as_tuple(values: Sequence[Hashable]) -> tuple[Hashable, ...]:
    return tuple(values)


def _ensure_unique(ids: Sequence[Hashable]) -> tuple[Hashable, ...]:
    ids_tuple = _as_tuple(ids)
    if len(set(ids_tuple)) != len(ids_tuple):
        seen: set[Hashable] = set()
        dupes: list[Hashable] = []
        for item in ids_tuple:
            if item in seen and item not in dupes:
                dupes.append(item)
            seen.add(item)
        raise DuplicateIdentityError(f"duplicate identity IDs: {dupes!r}")
    return ids_tuple


def _normalize_weight_mapping(
    ids: tuple[Hashable, ...],
    weights: Mapping[Hashable, float] | Sequence[float] | None,
) -> dict[Hashable, float]:
    if weights is None:
        raw = np.ones(len(ids), dtype=float)
    elif isinstance(weights, Mapping):
        missing = [identity for identity in ids if identity not in weights]
        extras = [identity for identity in weights if identity not in set(ids)]
        if missing or extras:
            raise PairingError(
                f"weight mapping does not match panel; missing={missing!r}, extras={extras!r}"
            )
        raw = np.asarray([weights[identity] for identity in ids], dtype=float)
    else:
        raw = np.asarray(weights, dtype=float)
        if raw.ndim != 1 or raw.size != len(ids):
            raise ValueError("weights must be one-dimensional and match panel length")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("at least one identity weight must be positive")
    raw = raw / total
    return {identity: float(weight) for identity, weight in zip(ids, raw, strict=True)}


class TaskTransportLedger:
    """Identity-paired ledger for a fixed deterministic decision rule.

    Parameters
    ----------
    panel_ids:
        Optional fixed identity panel.  If omitted, the first update freezes it.
    weights:
        Optional fixed non-negative identity weights.  Weights are normalized to
        sum to one over the declared panel.
    model_fingerprint:
        Optional immutable model identifier.  Once set, a different fingerprint
        in a later update raises :class:`ModelFingerprintError`.
    missing:
        ``"raise"`` (default) requires the complete panel at every window.
        ``"intersection"`` computes exact *adjacent-step* closure on the valid
        identity intersection but marks the cumulative ledger non-telescopeable.

    Notes
    -----
    The class does not store an operational risk boundary.  Event semantics are
    deliberately delegated to :class:`sbt_monitor.events.PersistentCliffRule`.
    """

    def __init__(
        self,
        *,
        panel_ids: Sequence[Hashable] | None = None,
        weights: Mapping[Hashable, float] | Sequence[float] | None = None,
        model_fingerprint: str | None = None,
        missing: MissingPolicy = "raise",
    ) -> None:
        if missing not in {"raise", "intersection"}:
            raise ValueError("missing must be 'raise' or 'intersection'")
        self._missing = missing
        self._panel_ids: tuple[Hashable, ...] | None = None
        self._weight_map: dict[Hashable, float] | None = None
        self._pending_weights = weights
        self._model_fingerprint = model_fingerprint
        self._records: "OrderedDict[Hashable, WindowRecord]" = OrderedDict()
        if panel_ids is not None:
            panel = _ensure_unique(panel_ids)
            if not panel:
                raise ValueError("panel_ids cannot be empty")
            self._panel_ids = panel
            self._weight_map = _normalize_weight_mapping(panel, weights)
            self._pending_weights = None

    @classmethod
    def fixed_panel(
        cls,
        ids: Sequence[Hashable],
        *,
        weights: Mapping[Hashable, float] | Sequence[float] | None = None,
        model_fingerprint: str | None = None,
    ) -> "TaskTransportLedger":
        """Construct a strict complete-panel ledger."""

        return cls(
            panel_ids=ids,
            weights=weights,
            model_fingerprint=model_fingerprint,
            missing="raise",
        )

    @property
    def panel_ids(self) -> tuple[Hashable, ...]:
        if self._panel_ids is None:
            raise PairingError("the identity panel is frozen by the first update")
        return self._panel_ids

    @property
    def model_fingerprint(self) -> str | None:
        return self._model_fingerprint

    @property
    def missing_policy(self) -> MissingPolicy:
        return self._missing

    @property
    def windows(self) -> tuple[Hashable, ...]:
        return tuple(self._records.keys())

    @property
    def n_windows(self) -> int:
        return len(self._records)

    @property
    def is_complete_panel(self) -> bool:
        if self._panel_ids is None:
            return False
        panel = set(self._panel_ids)
        return bool(self._records) and all(set(record.ids) == panel for record in self._records.values())

    @property
    def telescopeable(self) -> bool:
        if self.n_windows < 2:
            return self.is_complete_panel
        return all(step.pairing.telescopeable for step in self.steps())

    def update(
        self,
        *,
        window: Hashable,
        ids: Sequence[Hashable],
        y_true: Sequence[Any] | np.ndarray | None = None,
        y_pred: Sequence[Any] | np.ndarray | None = None,
        correct_mask: Sequence[bool] | np.ndarray | None = None,
        margins: Sequence[float] | np.ndarray | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        """Append one identity-aligned outcome window.

        Exactly one of ``correct_mask``, ``(y_true, y_pred)`` or ``margins`` must
        define correctness.  For margins, values ``> 0`` are treated as correct.
        """

        if window in self._records:
            raise ValueError(f"window {window!r} already exists")
        ids_tuple = _ensure_unique(ids)
        if not ids_tuple:
            raise ValueError("a window cannot be empty")

        supplied = int(correct_mask is not None) + int(y_true is not None or y_pred is not None) + int(margins is not None)
        if supplied != 1:
            raise ValueError(
                "provide exactly one of correct_mask, (y_true and y_pred), or margins"
            )
        if (y_true is None) ^ (y_pred is None):
            raise ValueError("y_true and y_pred must be supplied together")

        if correct_mask is not None:
            correct = np.asarray(correct_mask, dtype=bool)
            margins_arr = None
        elif margins is not None:
            margins_arr = np.asarray(margins, dtype=float)
            if not np.isfinite(margins_arr).all():
                raise ValueError("margins must be finite")
            correct = margins_arr > 0
        else:
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            if y_true_arr.shape != y_pred_arr.shape:
                raise ValueError("y_true and y_pred must have identical shapes")
            correct = y_true_arr == y_pred_arr
            margins_arr = None

        correct = np.asarray(correct, dtype=bool)
        if correct.ndim != 1 or correct.size != len(ids_tuple):
            raise ValueError("outcome arrays must be one-dimensional and match ids")
        if margins_arr is not None and (margins_arr.ndim != 1 or margins_arr.size != len(ids_tuple)):
            raise ValueError("margins must be one-dimensional and match ids")

        self._check_fingerprint(model_fingerprint)
        self._freeze_or_validate_panel(ids_tuple)
        assert self._weight_map is not None

        panel_set = set(self.panel_ids)
        current_set = set(ids_tuple)
        missing_ids = tuple(identity for identity in self.panel_ids if identity not in current_set)
        unexpected_ids = tuple(identity for identity in ids_tuple if identity not in panel_set)
        if unexpected_ids:
            raise PairingError(f"window contains identities outside the frozen panel: {unexpected_ids!r}")
        if missing_ids and self._missing == "raise":
            raise MissingIdentityError(
                f"window {window!r} is missing identities from the frozen panel: {missing_ids!r}"
            )

        weights_arr = np.asarray([self._weight_map[identity] for identity in ids_tuple], dtype=float)
        record = WindowRecord(
            window=window,
            ids=ids_tuple,
            errors=~correct,
            weights=weights_arr,
            model_fingerprint=self._model_fingerprint,
            margins=margins_arr,
        )
        self._records[window] = record

    def update_margins(
        self,
        *,
        window: Hashable,
        ids: Sequence[Hashable],
        margins: Sequence[float] | np.ndarray,
        model_fingerprint: str | None = None,
    ) -> None:
        """Convenience wrapper for true-class margins."""

        self.update(
            window=window,
            ids=ids,
            margins=margins,
            model_fingerprint=model_fingerprint,
        )

    def _check_fingerprint(self, supplied: str | None) -> None:
        if self._model_fingerprint is None and supplied is not None:
            self._model_fingerprint = supplied
            return
        if supplied is not None and supplied != self._model_fingerprint:
            raise ModelFingerprintError(
                f"ledger fingerprint {self._model_fingerprint!r} does not match {supplied!r}"
            )

    def _freeze_or_validate_panel(self, ids: tuple[Hashable, ...]) -> None:
        if self._panel_ids is None:
            self._panel_ids = ids
            self._weight_map = _normalize_weight_mapping(ids, self._pending_weights)
            self._pending_weights = None

    def _record_map(self, record: WindowRecord) -> dict[Hashable, tuple[bool, float]]:
        return {
            identity: (bool(error), float(weight))
            for identity, error, weight in zip(record.ids, record.errors, record.weights, strict=True)
        }

    def steps(self) -> tuple[TransportStep, ...]:
        """Return all adjacent resolved transport steps."""

        records = list(self._records.values())
        if len(records) < 2:
            return ()
        return tuple(self._compute_step(left, right) for left, right in zip(records[:-1], records[1:], strict=True))

    def _compute_step(self, left: WindowRecord, right: WindowRecord) -> TransportStep:
        left_map = self._record_map(left)
        right_map = self._record_map(right)
        panel = self.panel_ids
        panel_set = set(panel)
        left_set = set(left.ids)
        right_set = set(right.ids)
        common = tuple(identity for identity in panel if identity in left_set and identity in right_set)
        if not common:
            raise PairingError(f"windows {left.window!r} and {right.window!r} share no identities")

        missing_from = tuple(identity for identity in panel if identity not in left_set)
        missing_to = tuple(identity for identity in panel if identity not in right_set)
        unexpected_from = tuple(identity for identity in left.ids if identity not in panel_set)
        unexpected_to = tuple(identity for identity in right.ids if identity not in panel_set)
        if (missing_from or missing_to) and self._missing == "raise":
            raise MissingIdentityError(
                f"incomplete panel across {left.window!r}->{right.window!r}; "
                f"missing_from={missing_from!r}, missing_to={missing_to!r}"
            )

        assert self._weight_map is not None
        raw_weights = np.asarray([self._weight_map[identity] for identity in common], dtype=float)
        paired_weight = float(raw_weights.sum())
        if paired_weight <= 0:
            raise PairingError("paired identities have zero total weight")
        weights = raw_weights / paired_weight
        err_left = np.asarray([left_map[identity][0] for identity in common], dtype=bool)
        err_right = np.asarray([right_map[identity][0] for identity in common], dtype=bool)

        risk_left = float(np.dot(weights, err_left.astype(float)))
        risk_right = float(np.dot(weights, err_right.astype(float)))
        incident_mask = (~err_left) & err_right
        recovery_mask = err_left & (~err_right)
        incident = float(np.dot(weights, incident_mask.astype(float)))
        recovery = float(np.dot(weights, recovery_mask.astype(float)))
        sbt = incident - recovery
        turnover = incident + recovery
        closure_error = (risk_right - risk_left) - sbt
        complete = left_set == panel_set and right_set == panel_set
        pairing = PairingReport(
            window_from=left.window,
            window_to=right.window,
            n_from=len(left.ids),
            n_to=len(right.ids),
            n_pairs=len(common),
            pairing_coverage=paired_weight,
            missing_from=missing_from,
            missing_to=missing_to,
            unexpected_from=unexpected_from,
            unexpected_to=unexpected_to,
            denominator_changed=not complete,
            telescopeable=complete,
        )
        return TransportStep(
            window_from=left.window,
            window_to=right.window,
            risk_from=risk_left,
            risk_to=risk_right,
            incident=incident,
            recovery=recovery,
            sbt=sbt,
            turnover=turnover,
            closure_error=float(closure_error),
            pairing=pairing,
            incident_ids=tuple(identity for identity, flag in zip(common, incident_mask, strict=True) if flag),
            recovery_ids=tuple(identity for identity, flag in zip(common, recovery_mask, strict=True) if flag),
        )

    def closure_report(self, *, tolerance: float = 1e-12) -> ClosureReport:
        steps = self.steps()
        errors = np.asarray([abs(step.closure_error) for step in steps], dtype=float)
        max_error = float(errors.max()) if errors.size else 0.0
        mean_error = float(errors.mean()) if errors.size else 0.0
        non_tel = tuple(
            (step.window_from, step.window_to)
            for step in steps
            if not step.pairing.telescopeable
        )
        return ClosureReport(
            n_steps=len(steps),
            max_abs_error=max_error,
            mean_abs_error=mean_error,
            all_close=bool(max_error <= tolerance),
            tolerance=float(tolerance),
            telescopeable=not non_tel and self.is_complete_panel,
            non_telescopeable_steps=non_tel,
        )

    def risk_series(self) -> tuple[np.ndarray, tuple[Hashable, ...]]:
        """Return exact fixed-panel risk and window labels.

        Raises
        ------
        PairingError
            If any window is missing a declared identity.  In that case adjacent
            closure may still be available through :meth:`steps`, but cumulative
            headroom accounting is deliberately refused.
        """

        if not self._records:
            return np.asarray([], dtype=float), ()
        if not self.is_complete_panel:
            raise PairingError(
                "risk_series requires a complete fixed panel; this ledger is non-telescopeable"
            )
        assert self._weight_map is not None
        weights = np.asarray([self._weight_map[identity] for identity in self.panel_ids], dtype=float)
        values: list[float] = []
        for record in self._records.values():
            record_map = self._record_map(record)
            errors = np.asarray([record_map[identity][0] for identity in self.panel_ids], dtype=float)
            values.append(float(np.dot(weights, errors)))
        return np.asarray(values, dtype=float), self.windows

    def cumulative_sbt(self) -> np.ndarray:
        """Return cumulative SBT from the first window.

        The method refuses non-telescopeable ledgers rather than silently adding
        pairwise quantities computed on changing denominators.
        """

        report = self.closure_report()
        if not report.telescopeable:
            raise PairingError("cumulative SBT is undefined for changing valid-pair denominators")
        return np.cumsum(np.asarray([step.sbt for step in self.steps()], dtype=float))

    def first_crossings(self) -> dict[Hashable, Hashable | None]:
        """First correct-to-error transition for each identity."""

        if not self.is_complete_panel:
            raise PairingError("first_crossings requires a complete fixed panel")
        result: dict[Hashable, Hashable | None] = {identity: None for identity in self.panel_ids}
        for step in self.steps():
            for identity in step.incident_ids:
                if result[identity] is None:
                    result[identity] = step.window_to
        return result

    def endpoint_persistence(self) -> float:
        """Weighted fraction of ever-incident identities still in error at endpoint."""

        if not self.is_complete_panel:
            raise PairingError("endpoint_persistence requires a complete fixed panel")
        crossed = {identity for step in self.steps() for identity in step.incident_ids}
        if not crossed:
            return float("nan")
        assert self._weight_map is not None
        endpoint = next(reversed(self._records.values()))
        endpoint_map = self._record_map(endpoint)
        numerator = sum(self._weight_map[i] for i in crossed if endpoint_map[i][0])
        denominator = sum(self._weight_map[i] for i in crossed)
        return float(numerator / denominator)

    def transition_rows(self) -> list[dict[str, Any]]:
        """Return one row per paired identity and adjacent step."""

        rows: list[dict[str, Any]] = []
        records = list(self._records.values())
        for left, right in zip(records[:-1], records[1:], strict=True):
            left_map = self._record_map(left)
            right_map = self._record_map(right)
            common = [identity for identity in self.panel_ids if identity in left_map and identity in right_map]
            for identity in common:
                err_left = int(left_map[identity][0])
                err_right = int(right_map[identity][0])
                rows.append(
                    {
                        "window_from": left.window,
                        "window_to": right.window,
                        "identity": identity,
                        "error_from": err_left,
                        "error_to": err_right,
                        "incident": int(err_left == 0 and err_right == 1),
                        "recovery": int(err_left == 1 and err_right == 0),
                        "weight": self._weight_map[identity] if self._weight_map else None,
                    }
                )
        return rows

    def to_dict(self) -> dict[str, Any]:
        risk: list[float] | None
        try:
            risk_arr, windows = self.risk_series()
            risk = risk_arr.tolist()
        except PairingError:
            windows = self.windows
            risk = None
        return {
            "scientific_object": "task_error_transport_ledger",
            "model_fingerprint": self._model_fingerprint,
            "panel_size": len(self.panel_ids) if self._panel_ids is not None else 0,
            "windows": list(windows),
            "risk": risk,
            "steps": [step.to_dict() for step in self.steps()],
            "closure": self.closure_report().to_dict(),
            "telescopeable": self.telescopeable,
            "boundary": None,
        }
