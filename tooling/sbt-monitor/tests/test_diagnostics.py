from __future__ import annotations

import numpy as np
import pytest

from sbt_monitor import ConsecutiveWindows, TaskTransportLedger, diagnostics


def _ledger(states: list[list[bool]], fingerprint: str = "m") -> TaskTransportLedger:
    ids = [f"i{i}" for i in range(len(states[0]))]
    ledger = TaskTransportLedger.fixed_panel(ids, model_fingerprint=fingerprint)
    for t, errors in enumerate(states):
        ledger.update(window=t, ids=ids, correct_mask=np.logical_not(errors), model_fingerprint=fingerprint)
    return ledger


def test_peer_boundary_returns_effect_size_not_pass_fail() -> None:
    focal = _ledger([[False, False, False], [True, False, False], [True, True, False]], "f")
    peer = _ledger([[False, False, False], [False, True, False], [False, True, True]], "p")
    result = diagnostics.peer_boundary(focal, peer)
    payload = result.to_dict()
    assert "rmse" in payload
    assert "nrmse_rms_increment" in payload
    assert "pass" not in payload
    assert result.anchor_risk_gap == pytest.approx(0.0)


def test_threshold_sweep_does_not_mutate_ledger() -> None:
    ledger = _ledger([[False, False], [True, False], [True, True]])
    before = [step.sbt for step in ledger.steps()]
    rows = diagnostics.threshold_sweep(
        ledger,
        boundaries=[0.2, 0.6, 0.9],
        persistence=ConsecutiveWindows(1),
    )
    assert len(rows) == 3
    assert [step.sbt for step in ledger.steps()] == before
    assert all(row["ledger_sbt"] == before for row in rows)


def test_identity_permutation_runs() -> None:
    ledger = _ledger(
        [
            [False, False, False, False],
            [True, True, False, False],
            [True, False, True, False],
        ]
    )
    result = diagnostics.identity_permutation(ledger, n=20, seed=3)
    assert result.null_endpoint_persistence.shape == (20,)
    assert result.n_permutations == 20


def test_stationary_noise_floor_uses_sequence_maxima() -> None:
    value = diagnostics.stationary_noise_floor([[0.1, 0.2], [0.3, 0.05]], quantile=0.5)
    assert value == pytest.approx(0.25)
