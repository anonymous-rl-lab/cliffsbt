"""Diagnostics that protect against common SBT interpretation errors."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .events import ConsecutiveWindows, PersistenceProtocol, PersistentCliffRule
from .ledger import TaskTransportLedger
from .schema import OptionalDependencyError, PairingError


@dataclass(frozen=True)
class PeerBoundaryResult:
    """Effect-size comparison between focal risk increments and a peer ledger."""

    n_steps: int
    rmse: float
    mae: float
    nrmse_rms_increment: float
    nmae_mean_abs_increment: float
    anchor_risk_gap: float
    focal_increment_rms: float
    focal_increment_mean_abs: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_steps": self.n_steps,
            "rmse": self.rmse,
            "mae": self.mae,
            "nrmse_rms_increment": self.nrmse_rms_increment,
            "nmae_mean_abs_increment": self.nmae_mean_abs_increment,
            "anchor_risk_gap": self.anchor_risk_gap,
            "focal_increment_rms": self.focal_increment_rms,
            "focal_increment_mean_abs": self.focal_increment_mean_abs,
        }


@dataclass(frozen=True)
class IdentityPermutationResult:
    """Observed identity-continuity statistics and a permutation null."""

    observed_endpoint_persistence: float
    observed_mean_turnover: float
    null_endpoint_persistence: np.ndarray
    null_mean_turnover: np.ndarray
    n_permutations: int

    def summary(self, quantiles: Sequence[float] = (0.025, 0.5, 0.975)) -> dict[str, Any]:
        q = np.asarray(quantiles, dtype=float)
        return {
            "observed_endpoint_persistence": self.observed_endpoint_persistence,
            "observed_mean_turnover": self.observed_mean_turnover,
            "null_endpoint_persistence_quantiles": dict(
                zip([str(x) for x in q], np.quantile(self.null_endpoint_persistence, q).tolist(), strict=True)
            ),
            "null_mean_turnover_quantiles": dict(
                zip([str(x) for x in q], np.quantile(self.null_mean_turnover, q).tolist(), strict=True)
            ),
            "n_permutations": self.n_permutations,
        }


def pairing_audit(ledger: TaskTransportLedger) -> dict[str, Any]:
    """Summarize pairing coverage and cumulative eligibility."""

    steps = ledger.steps()
    return {
        "n_windows": ledger.n_windows,
        "panel_size": len(ledger.panel_ids) if ledger.n_windows else 0,
        "missing_policy": ledger.missing_policy,
        "telescopeable": ledger.telescopeable,
        "minimum_pairing_coverage": min((s.pairing.pairing_coverage for s in steps), default=1.0),
        "minimum_n_pairs": min((s.pairing.n_pairs for s in steps), default=0),
        "denominator_changed_steps": [
            [s.window_from, s.window_to]
            for s in steps
            if s.pairing.denominator_changed
        ],
    }


def closure_audit(ledger: TaskTransportLedger, *, tolerance: float = 1e-12) -> dict[str, Any]:
    """Return exact closure and pairing diagnostics."""

    return {
        "closure": ledger.closure_report(tolerance=tolerance).to_dict(),
        "pairing": pairing_audit(ledger),
    }


def peer_boundary(focal: TaskTransportLedger, peer: TaskTransportLedger) -> PeerBoundaryResult:
    """Quantify how well a peer ledger reconstructs focal risk increments.

    No pass/fail label is returned.  Focal self-closure is an identity; the
    informative quantities are peer error magnitude, normalization and anchor
    risk separation.
    """

    focal_risk, focal_windows = focal.risk_series()
    peer_risk, peer_windows = peer.risk_series()
    if focal_windows != peer_windows:
        raise PairingError("focal and peer ledgers must have identical window labels")
    focal_delta = np.diff(focal_risk)
    peer_sbt = np.asarray([step.sbt for step in peer.steps()], dtype=float)
    if focal_delta.size != peer_sbt.size:
        raise PairingError("focal and peer ledgers have different step counts")
    residual = peer_sbt - focal_delta
    rmse = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    mae = float(np.mean(np.abs(residual))) if residual.size else 0.0
    rms = float(np.sqrt(np.mean(focal_delta**2))) if focal_delta.size else 0.0
    mean_abs = float(np.mean(np.abs(focal_delta))) if focal_delta.size else 0.0
    return PeerBoundaryResult(
        n_steps=int(focal_delta.size),
        rmse=rmse,
        mae=mae,
        nrmse_rms_increment=float(rmse / rms) if rms > 0 else float("nan"),
        nmae_mean_abs_increment=float(mae / mean_abs) if mean_abs > 0 else float("nan"),
        anchor_risk_gap=float(abs(peer_risk[0] - focal_risk[0])) if focal_risk.size else float("nan"),
        focal_increment_rms=rms,
        focal_increment_mean_abs=mean_abs,
    )


def threshold_sweep(
    ledger: TaskTransportLedger,
    *,
    boundaries: Sequence[float],
    persistence: PersistenceProtocol | None = None,
) -> list[dict[str, Any]]:
    """Re-evaluate boundary-relative events without changing the ledger."""

    risk, windows = ledger.risk_series()
    rule_template = persistence or ConsecutiveWindows(1)
    ledger_signature = [step.sbt for step in ledger.steps()]
    results: list[dict[str, Any]] = []
    for boundary in boundaries:
        event = PersistentCliffRule(float(boundary), rule_template).evaluate(risk, windows)
        row = event.to_dict()
        row["ledger_sbt"] = list(ledger_signature)
        results.append(row)
    return results


def identity_permutation(
    ledger: TaskTransportLedger,
    *,
    n: int = 1000,
    seed: int = 0,
) -> IdentityPermutationResult:
    """Break identity continuity while preserving each window's error count.

    The diagnostic currently requires uniform weights because independently
    permuting error assignments does not preserve weighted risk for unequal
    weights.
    """

    if n < 1:
        raise ValueError("n must be positive")
    risk, _ = ledger.risk_series()  # also enforces complete fixed panel
    del risk
    records = list(ledger._records.values())  # package-internal diagnostic access
    if not records:
        raise ValueError("ledger is empty")
    first_weights = records[0].weights
    if not np.allclose(first_weights, np.full_like(first_weights, first_weights[0])):
        raise ValueError("identity_permutation currently requires uniform weights")

    observed_persistence = ledger.endpoint_persistence()
    observed_turnover = float(np.mean([step.turnover for step in ledger.steps()])) if ledger.steps() else 0.0
    rng = np.random.default_rng(seed)
    null_persistence = np.empty(n, dtype=float)
    null_turnover = np.empty(n, dtype=float)
    ids = ledger.panel_ids
    for rep in range(n):
        permuted = TaskTransportLedger.fixed_panel(ids, model_fingerprint=ledger.model_fingerprint)
        for record in records:
            errors = rng.permutation(record.errors)
            permuted.update(
                window=record.window,
                ids=record.ids,
                correct_mask=~errors,
                model_fingerprint=ledger.model_fingerprint,
            )
        null_persistence[rep] = permuted.endpoint_persistence()
        null_turnover[rep] = float(np.mean([step.turnover for step in permuted.steps()])) if permuted.steps() else 0.0
    return IdentityPermutationResult(
        observed_endpoint_persistence=observed_persistence,
        observed_mean_turnover=observed_turnover,
        null_endpoint_persistence=null_persistence,
        null_mean_turnover=null_turnover,
        n_permutations=n,
    )


def stationary_noise_floor(
    sequences: Sequence[Sequence[float] | np.ndarray],
    *,
    quantile: float = 0.95,
    use_maximum: bool = True,
) -> float:
    """Estimate a stationary reference using a maximum-window statistic."""

    if not (0 < quantile < 1):
        raise ValueError("quantile must lie strictly between zero and one")
    values: list[float] = []
    for sequence in sequences:
        arr = np.asarray(sequence, dtype=float)
        if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("every stationary sequence must be finite and one-dimensional")
        values.append(float(np.max(arr) if use_maximum else np.mean(arr)))
    if not values:
        raise ValueError("at least one stationary sequence is required")
    return float(np.quantile(np.asarray(values), quantile))


def nested_state_ablation(*args: Any, **kwargs: Any) -> Any:
    """Run the experimental warning nested-state workflow.

    Install ``sbt-monitor[warning]``.  The function is a convenience wrapper and
    remains outside the stable warning API in v0.1.
    """

    try:
        from .experimental.warning import nested_state_ablation as _impl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OptionalDependencyError(
            "nested_state_ablation requires the 'warning' extra: pip install sbt-monitor[warning]"
        ) from exc
    return _impl(*args, **kwargs)


def false_alarm_budget_curve(*args: Any, **kwargs: Any) -> Any:
    """Experimental wrapper for a fixed-score false-alarm budget curve."""

    try:
        from .experimental.warning import false_alarm_budget_curve as _impl
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "false_alarm_budget_curve requires the 'warning' extra"
        ) from exc
    return _impl(*args, **kwargs)
