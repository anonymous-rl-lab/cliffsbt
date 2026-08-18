from __future__ import annotations

import numpy as np
import pytest

from sbt_monitor import TaskTransportLedger
from sbt_monitor.schema import (
    DuplicateIdentityError,
    MissingIdentityError,
    ModelFingerprintError,
    PairingError,
)


def test_random_binary_closure_property() -> None:
    rng = np.random.default_rng(42)
    for n_ids in [2, 3, 7, 32]:
        ids = [f"id-{i}" for i in range(n_ids)]
        for _ in range(20):
            states = rng.integers(0, 2, size=(6, n_ids), dtype=np.int8).astype(bool)
            ledger = TaskTransportLedger.fixed_panel(ids, model_fingerprint="m")
            for t, errors in enumerate(states):
                order = rng.permutation(n_ids)
                ledger.update(
                    window=t,
                    ids=[ids[i] for i in order],
                    correct_mask=(~errors[order]),
                    model_fingerprint="m",
                )
            report = ledger.closure_report(tolerance=1e-14)
            assert report.all_close
            assert report.telescopeable
            risk, _ = ledger.risk_series()
            np.testing.assert_allclose(np.diff(risk), [s.sbt for s in ledger.steps()], atol=1e-14)


def test_weighted_closure() -> None:
    ids = ["a", "b", "c"]
    ledger = TaskTransportLedger.fixed_panel(ids, weights=[1, 2, 7])
    ledger.update(window=0, ids=ids, correct_mask=[True, False, True])
    ledger.update(window=1, ids=ids, correct_mask=[False, True, True])
    step = ledger.steps()[0]
    assert step.incident == pytest.approx(0.1)
    assert step.recovery == pytest.approx(0.2)
    assert step.sbt == pytest.approx(-0.1)
    assert step.risk_to - step.risk_from == pytest.approx(step.sbt)


def test_reordering_is_aligned_by_identity() -> None:
    ids = ["a", "b", "c"]
    ledger = TaskTransportLedger.fixed_panel(ids)
    ledger.update(window=0, ids=ids, correct_mask=[True, True, False])
    ledger.update(window=1, ids=["c", "a", "b"], correct_mask=[True, False, True])
    step = ledger.steps()[0]
    assert step.incident_ids == ("a",)
    assert step.recovery_ids == ("c",)
    assert step.sbt == pytest.approx(0.0)


def test_missing_default_raises() -> None:
    ledger = TaskTransportLedger.fixed_panel(["a", "b", "c"])
    ledger.update(window=0, ids=["a", "b", "c"], correct_mask=[True, True, True])
    with pytest.raises(MissingIdentityError):
        ledger.update(window=1, ids=["a", "b"], correct_mask=[True, False])


def test_intersection_is_exact_but_non_telescopeable() -> None:
    ledger = TaskTransportLedger(panel_ids=["a", "b", "c"], missing="intersection")
    ledger.update(window=0, ids=["a", "b", "c"], correct_mask=[True, True, False])
    ledger.update(window=1, ids=["a", "b"], correct_mask=[False, True])
    step = ledger.steps()[0]
    assert step.closure_error == pytest.approx(0.0)
    assert not step.pairing.telescopeable
    assert not ledger.closure_report().telescopeable
    with pytest.raises(PairingError):
        ledger.risk_series()
    with pytest.raises(PairingError):
        ledger.cumulative_sbt()


def test_duplicate_identity_raises() -> None:
    ledger = TaskTransportLedger()
    with pytest.raises(DuplicateIdentityError):
        ledger.update(window=0, ids=["a", "a"], correct_mask=[True, False])


def test_model_fingerprint_change_raises() -> None:
    ledger = TaskTransportLedger.fixed_panel(["a", "b"], model_fingerprint="v1")
    ledger.update(window=0, ids=["a", "b"], correct_mask=[True, True], model_fingerprint="v1")
    with pytest.raises(ModelFingerprintError):
        ledger.update(window=1, ids=["a", "b"], correct_mask=[True, False], model_fingerprint="v2")


def test_margin_update_uses_nonpositive_as_error() -> None:
    ledger = TaskTransportLedger.fixed_panel(["a", "b", "c"])
    ledger.update_margins(window=0, ids=["a", "b", "c"], margins=[1.0, 0.0, -1.0])
    risk, _ = ledger.risk_series()
    assert risk[0] == pytest.approx(2 / 3)


def test_first_crossings_and_endpoint_persistence() -> None:
    ids = ["a", "b", "c", "d"]
    ledger = TaskTransportLedger.fixed_panel(ids)
    ledger.update(window=0, ids=ids, correct_mask=[True, True, True, True])
    ledger.update(window=1, ids=ids, correct_mask=[False, True, False, True])
    ledger.update(window=2, ids=ids, correct_mask=[False, True, True, True])
    crossings = ledger.first_crossings()
    assert crossings["a"] == 1
    assert crossings["c"] == 1
    assert ledger.endpoint_persistence() == pytest.approx(0.5)
