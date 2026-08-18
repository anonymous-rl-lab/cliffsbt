from __future__ import annotations

import numpy as np

from sbt_monitor import ConsecutiveWindows, FirstCrossingRule, PersistentCliffRule, RemainAboveThereafter


def test_first_crossing_separate_from_persistent_cliff() -> None:
    risk = np.array([0.1, 0.35, 0.2, 0.4, 0.45])
    windows = [0, 1, 2, 3, 4]
    first = FirstCrossingRule(0.3).evaluate(risk, windows)
    persistent = PersistentCliffRule(0.3, ConsecutiveWindows(2)).evaluate(risk, windows)
    assert first.first_crossing_time == 1
    assert first.persistent_cliff_time is None
    assert persistent.first_crossing_time == 1
    assert persistent.persistent_cliff_time == 3


def test_remain_above_thereafter() -> None:
    risk = [0.1, 0.4, 0.2, 0.35, 0.36]
    event = PersistentCliffRule(0.3, RemainAboveThereafter()).evaluate(risk)
    assert event.first_crossing_index == 1
    assert event.persistent_cliff_index == 3
