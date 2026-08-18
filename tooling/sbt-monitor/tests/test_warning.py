from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sbt_monitor.schema import LeakageError, WarningEpisode
from sbt_monitor.experimental import (
    FrozenReadout,
    WarningCalibrator,
    false_alarm_budget_curve,
    nested_state_ablation,
)

FEATURES = ("position", "net_prediction_transport", "persistent_departure")


def _episodes(prefix: str, identity_offset: int) -> list[WarningEpisode]:
    episodes: list[WarningEpisode] = []
    for i in range(12):
        event = i < 6
        position = np.linspace(0.0, 1.0 if event else 0.25, 8) + 0.005 * i
        net = np.r_[np.zeros(3), np.linspace(0.0, 1.0 if event else 0.05, 5)]
        persistence = np.r_[np.zeros(4), np.linspace(0.0, 0.5 if event else 0.02, 4)]
        states = np.column_stack([position, net, persistence])
        episodes.append(
            WarningEpisode(
                episode_id=f"{prefix}-{i}",
                states=states,
                feature_names=FEATURES,
                event_time=6 if event else None,
                group=f"g{i % 3}",
                identity_set_id=f"ids-{identity_offset+i}",
            )
        )
    return episodes


def test_warning_calibration_serialization_and_overlap(tmp_path: Path) -> None:
    calibration = _episodes("cal", 0)
    evaluation = _episodes("eval", 100)
    result = WarningCalibrator(
        feature_names=FEATURES,
        horizon=3,
        false_alarm_budget=0.25,
        event_rule_name="test persistent event",
        model_scope_fingerprint="test-model-scope",
        c_grid=(0.1, 1.0),
    ).fit(calibration)
    evaluation_result = result.readout.evaluate(evaluation)
    assert evaluation_result.n_events == 6
    assert 0 <= evaluation_result.false_alarm_rate <= 1
    path = tmp_path / "readout.json"
    result.readout.save(path)
    loaded = FrozenReadout.load(path)
    np.testing.assert_allclose(
        loaded.score_many(evaluation[0].states),
        result.readout.score_many(evaluation[0].states),
    )
    assert loaded.event_rule_name == "test persistent event"
    assert loaded.model_scope_fingerprint == "test-model-scope"
    assert loaded.package_version == "0.1.0"
    with pytest.raises(LeakageError):
        loaded.evaluate(calibration)


def test_nested_ablation_and_budget_curve() -> None:
    calibration = _episodes("cal", 0)
    evaluation = _episodes("eval", 100)
    result = nested_state_ablation(
        calibration_episodes=calibration,
        evaluation_episodes=evaluation,
        feature_sets={
            "position": ("position",),
            "position+net": ("position", "net_prediction_transport"),
            "all": FEATURES,
        },
        horizon=3,
        false_alarm_budget=0.25,
        event_rule_name="test persistent event",
        model_scope_fingerprint="test-model-scope",
        c_grid=(0.1, 1.0),
    )
    assert set(result) == {"position", "position+net", "all"}
    curve = false_alarm_budget_curve(
        calibration_episodes=calibration,
        evaluation_episodes=evaluation,
        feature_names=("position", "net_prediction_transport"),
        horizon=3,
        budgets=(0.0, 0.25, 0.5),
        event_rule_name="test persistent event",
        model_scope_fingerprint="test-model-scope",
        c_grid=(0.1, 1.0),
    )
    assert [row["false_alarm_budget"] for row in curve] == [0.0, 0.25, 0.5]
