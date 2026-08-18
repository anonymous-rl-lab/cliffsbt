"""Outcome-blind prediction-state transport proxies.

These classes never claim exact task-risk closure.  They track departures from and
returns to a baseline *prediction*, not correct-to-error and error-to-correct
transitions.  The distinction is central to the package's scientific scope.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .schema import DuplicateIdentityError, MissingIdentityError, PairingError, ProxyState


STATIC_FEATURES: tuple[str, ...] = (
    "departure_mass",
    "current_margin_q10",
    "current_margin_q50",
    "current_margin_q90",
    "margin_change_q10",
    "margin_change_q50",
    "margin_change_q90",
    "near_boundary_occupancy",
    "mean_representation_norm",
)
NET_FEATURE = "net_prediction_transport"
PERSISTENCE_FEATURE = "persistent_departure"

FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "static": STATIC_FEATURES,
    "static+net_prediction_transport": STATIC_FEATURES + (NET_FEATURE,),
    "static+persistent_departure": STATIC_FEATURES + (PERSISTENCE_FEATURE,),
    "static+both": STATIC_FEATURES + (NET_FEATURE, PERSISTENCE_FEATURE),
    "transport_aware_current": STATIC_FEATURES + (NET_FEATURE, PERSISTENCE_FEATURE),
}


def _unique(ids: Sequence[Hashable]) -> tuple[Hashable, ...]:
    ids_tuple = tuple(ids)
    if len(set(ids_tuple)) != len(ids_tuple):
        raise DuplicateIdentityError("prediction-state window contains duplicate identity IDs")
    if not ids_tuple:
        raise ValueError("identity panel cannot be empty")
    return ids_tuple


def _weights(ids: tuple[Hashable, ...], weights: Sequence[float] | Mapping[Hashable, float] | None) -> dict[Hashable, float]:
    if weights is None:
        raw = np.ones(len(ids), dtype=float)
    elif isinstance(weights, Mapping):
        if set(weights) != set(ids):
            raise PairingError("weight mapping must match the baseline identity panel")
        raw = np.asarray([weights[i] for i in ids], dtype=float)
    else:
        raw = np.asarray(weights, dtype=float)
        if raw.ndim != 1 or raw.size != len(ids):
            raise ValueError("weights must match identity panel length")
    if not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and have positive sum")
    raw = raw / raw.sum()
    return {identity: float(weight) for identity, weight in zip(ids, raw, strict=True)}


@dataclass(frozen=True)
class _PredictionRecord:
    window: Hashable
    ids: tuple[Hashable, ...]
    predictions: np.ndarray
    margins: np.ndarray | None
    representation_norm: np.ndarray | None


class PredictionStateTransport:
    """Identity-anchored, outcome-blind prediction-state transport.

    Parameters
    ----------
    baseline_ids, baseline_predictions:
        The recurring identity panel and its frozen baseline predictions.
    baseline_margins:
        Optional outcome-blind top-one minus top-two prediction margins.
    weights:
        Fixed identity weights.
    near_boundary_threshold:
        Threshold applied to the supplied outcome-blind prediction margins.

    Notes
    -----
    ``net_prediction_transport`` is departure from minus return to each identity's
    baseline predicted class.  It is a calibrated proxy candidate and is not
    task-error SBT.
    """

    def __init__(
        self,
        *,
        baseline_ids: Sequence[Hashable],
        baseline_predictions: Sequence[object] | np.ndarray,
        baseline_margins: Sequence[float] | np.ndarray | None = None,
        weights: Sequence[float] | Mapping[Hashable, float] | None = None,
        near_boundary_threshold: float = 0.10,
        missing: Literal["raise", "intersection"] = "raise",
    ) -> None:
        if missing not in {"raise", "intersection"}:
            raise ValueError("missing must be 'raise' or 'intersection'")
        ids = _unique(baseline_ids)
        predictions = np.asarray(baseline_predictions)
        if predictions.ndim != 1 or predictions.size != len(ids):
            raise ValueError("baseline_predictions must be one-dimensional and match IDs")
        margins_arr: np.ndarray | None = None
        if baseline_margins is not None:
            margins_arr = np.asarray(baseline_margins, dtype=float)
            if margins_arr.ndim != 1 or margins_arr.size != len(ids) or not np.isfinite(margins_arr).all():
                raise ValueError("baseline_margins must be finite and match IDs")
        if not np.isfinite(near_boundary_threshold):
            raise ValueError("near_boundary_threshold must be finite")
        self._panel = ids
        self._panel_set = set(ids)
        self._baseline_pred = {identity: prediction for identity, prediction in zip(ids, predictions, strict=True)}
        self._baseline_margin = (
            None
            if margins_arr is None
            else {identity: float(value) for identity, value in zip(ids, margins_arr, strict=True)}
        )
        self._weight_map = _weights(ids, weights)
        self._threshold = float(near_boundary_threshold)
        self._missing = missing
        self._records: "OrderedDict[Hashable, _PredictionRecord]" = OrderedDict()
        # Baseline itself is a legitimate first record for transition construction.
        self._baseline_record = _PredictionRecord(
            window="__baseline__",
            ids=ids,
            predictions=predictions.copy(),
            margins=None if margins_arr is None else margins_arr.copy(),
            representation_norm=None,
        )

    @classmethod
    def from_baseline(
        cls,
        *,
        ids: Sequence[Hashable],
        predictions: Sequence[object] | np.ndarray,
        margins: Sequence[float] | np.ndarray | None = None,
        weights: Sequence[float] | Mapping[Hashable, float] | None = None,
        near_boundary_threshold: float = 0.10,
    ) -> "PredictionStateTransport":
        return cls(
            baseline_ids=ids,
            baseline_predictions=predictions,
            baseline_margins=margins,
            weights=weights,
            near_boundary_threshold=near_boundary_threshold,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return STATIC_FEATURES + (NET_FEATURE, PERSISTENCE_FEATURE)

    @property
    def windows(self) -> tuple[Hashable, ...]:
        return tuple(self._records.keys())

    def update(
        self,
        *,
        window: Hashable,
        ids: Sequence[Hashable],
        predictions: Sequence[object] | np.ndarray,
        prediction_margins: Sequence[float] | np.ndarray | None = None,
        representation_norm: Sequence[float] | np.ndarray | float | None = None,
    ) -> ProxyState:
        if window in self._records:
            raise ValueError(f"window {window!r} already exists")
        ids_tuple = _unique(ids)
        current_set = set(ids_tuple)
        extras = tuple(identity for identity in ids_tuple if identity not in self._panel_set)
        missing = tuple(identity for identity in self._panel if identity not in current_set)
        if extras:
            raise PairingError(f"prediction window contains identities outside baseline panel: {extras!r}")
        if missing and self._missing == "raise":
            raise MissingIdentityError(f"prediction window missing baseline identities: {missing!r}")

        pred = np.asarray(predictions)
        if pred.ndim != 1 or pred.size != len(ids_tuple):
            raise ValueError("predictions must be one-dimensional and match IDs")
        margins: np.ndarray | None = None
        if prediction_margins is not None:
            margins = np.asarray(prediction_margins, dtype=float)
            if margins.ndim != 1 or margins.size != len(ids_tuple) or not np.isfinite(margins).all():
                raise ValueError("prediction_margins must be finite and match IDs")
        rep: np.ndarray | None = None
        if representation_norm is not None:
            if np.isscalar(representation_norm):
                rep = np.full(len(ids_tuple), float(representation_norm), dtype=float)
            else:
                rep = np.asarray(representation_norm, dtype=float)
                if rep.ndim != 1 or rep.size != len(ids_tuple):
                    raise ValueError("representation_norm must be scalar or match IDs")
            if not np.isfinite(rep).all():
                raise ValueError("representation_norm must be finite")

        record = _PredictionRecord(window, ids_tuple, pred, margins, rep)
        previous = next(reversed(self._records.values())) if self._records else self._baseline_record
        state = self._build_state(record, previous)
        self._records[window] = record
        return state

    def _build_state(self, current: _PredictionRecord, previous: _PredictionRecord) -> ProxyState:
        prev_map = {identity: value for identity, value in zip(previous.ids, previous.predictions, strict=True)}
        curr_map = {identity: value for identity, value in zip(current.ids, current.predictions, strict=True)}
        common = tuple(identity for identity in self._panel if identity in prev_map and identity in curr_map)
        if not common:
            raise PairingError("adjacent prediction windows share no identities")
        raw_weights = np.asarray([self._weight_map[i] for i in common], dtype=float)
        paired_weight = float(raw_weights.sum())
        weights = raw_weights / paired_weight
        baseline = np.asarray([self._baseline_pred[i] for i in common], dtype=object)
        prev = np.asarray([prev_map[i] for i in common], dtype=object)
        curr = np.asarray([curr_map[i] for i in common], dtype=object)

        prev_departed = prev != baseline
        curr_departed = curr != baseline
        new_departure = (~prev_departed) & curr_departed
        returned = prev_departed & (~curr_departed)
        persistent = prev_departed & curr_departed

        values: dict[str, float] = {
            "departure_mass": float(np.dot(weights, curr_departed.astype(float))),
            NET_FEATURE: float(np.dot(weights, new_departure.astype(float)) - np.dot(weights, returned.astype(float))),
            PERSISTENCE_FEATURE: float(np.dot(weights, persistent.astype(float))),
        }

        margin_map = None if current.margins is None else {
            identity: float(value) for identity, value in zip(current.ids, current.margins, strict=True)
        }
        if margin_map is None:
            margin_values = np.full(len(common), np.nan)
        else:
            margin_values = np.asarray([margin_map[i] for i in common], dtype=float)
        if np.isfinite(margin_values).all():
            q10, q50, q90 = np.quantile(margin_values, [0.10, 0.50, 0.90])
            values.update(
                {
                    "current_margin_q10": float(q10),
                    "current_margin_q50": float(q50),
                    "current_margin_q90": float(q90),
                    "near_boundary_occupancy": float(np.dot(weights, (margin_values < self._threshold).astype(float))),
                }
            )
            if self._baseline_margin is not None:
                delta = margin_values - np.asarray([self._baseline_margin[i] for i in common], dtype=float)
                dq10, dq50, dq90 = np.quantile(delta, [0.10, 0.50, 0.90])
                values.update(
                    {
                        "margin_change_q10": float(dq10),
                        "margin_change_q50": float(dq50),
                        "margin_change_q90": float(dq90),
                    }
                )
            else:
                values.update({name: float("nan") for name in ("margin_change_q10", "margin_change_q50", "margin_change_q90")})
        else:
            values.update(
                {
                    "current_margin_q10": float("nan"),
                    "current_margin_q50": float("nan"),
                    "current_margin_q90": float("nan"),
                    "margin_change_q10": float("nan"),
                    "margin_change_q50": float("nan"),
                    "margin_change_q90": float("nan"),
                    "near_boundary_occupancy": float("nan"),
                }
            )

        if current.representation_norm is None:
            values["mean_representation_norm"] = float("nan")
        else:
            rep_map = {
                identity: float(value)
                for identity, value in zip(current.ids, current.representation_norm, strict=True)
            }
            rep_values = np.asarray([rep_map[i] for i in common], dtype=float)
            values["mean_representation_norm"] = float(np.dot(weights, rep_values))

        return ProxyState(
            window=current.window,
            values=values,
            n_pairs=len(common),
            pairing_coverage=paired_weight,
            identity_anchored=True,
            is_task_sbt=False,
        )


class TransportAwareStateBuilder:
    """Select a declared outcome-blind current-state recipe.

    The builder refuses missing or non-finite features instead of silently
    imputing them.  Users should declare a feature set supported by their own
    telemetry channel and calibration domain.
    """

    def __init__(self, feature_set: str | Sequence[str] = "static+net_prediction_transport") -> None:
        if isinstance(feature_set, str):
            if feature_set not in FEATURE_SETS:
                raise ValueError(f"unknown feature_set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
            self.feature_set = feature_set
            self.feature_names = FEATURE_SETS[feature_set]
        else:
            names = tuple(feature_set)
            if not names:
                raise ValueError("a custom feature set cannot be empty")
            allowed = set(STATIC_FEATURES + (NET_FEATURE, PERSISTENCE_FEATURE))
            unknown = [name for name in names if name not in allowed]
            if unknown:
                raise ValueError(f"unknown custom proxy features: {unknown!r}")
            if len(set(names)) != len(names):
                raise ValueError("custom feature names must be unique")
            self.feature_set = "custom"
            self.feature_names = names

    def transform(self, state: ProxyState) -> np.ndarray:
        vector = state.vector(self.feature_names)
        if not np.isfinite(vector).all():
            missing = [name for name, value in zip(self.feature_names, vector, strict=True) if not np.isfinite(value)]
            raise ValueError(
                "selected feature set contains unavailable telemetry: " + ", ".join(missing)
            )
        return vector

    def metadata(self) -> dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "feature_names": list(self.feature_names),
            "scientific_object": "outcome_blind_prediction_state_transport_proxy",
            "is_task_sbt": False,
        }
