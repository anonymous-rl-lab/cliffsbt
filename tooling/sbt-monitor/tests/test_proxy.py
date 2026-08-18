from __future__ import annotations

import numpy as np
import pytest

from sbt_monitor import PredictionStateTransport, TransportAwareStateBuilder


def test_prediction_state_proxy_is_not_task_sbt() -> None:
    ids = ["a", "b", "c", "d"]
    proxy = PredictionStateTransport.from_baseline(
        ids=ids,
        predictions=[0, 0, 1, 1],
        margins=[0.8, 0.6, 0.7, 0.9],
    )
    state = proxy.update(
        window=1,
        ids=ids,
        predictions=[0, 1, 1, 1],
        prediction_margins=[0.7, 0.1, 0.5, 0.8],
        representation_norm=[1.0, 1.1, 0.9, 1.0],
    )
    assert not state.is_task_sbt
    assert state.values["net_prediction_transport"] == pytest.approx(0.25)
    assert state.values["persistent_departure"] == pytest.approx(0.0)
    vector = TransportAwareStateBuilder("static+net_prediction_transport").transform(state)
    assert vector.shape == (10,)
    assert np.isfinite(vector).all()
    custom = TransportAwareStateBuilder(["departure_mass", "net_prediction_transport"]).transform(state)
    assert custom.shape == (2,)


def test_return_to_baseline_makes_proxy_negative() -> None:
    ids = ["a", "b"]
    proxy = PredictionStateTransport.from_baseline(ids=ids, predictions=[0, 0])
    proxy.update(window=1, ids=ids, predictions=[1, 0])
    state = proxy.update(window=2, ids=ids, predictions=[0, 0])
    assert state.values["net_prediction_transport"] == pytest.approx(-0.5)
