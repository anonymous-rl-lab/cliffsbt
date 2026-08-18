"""Small synthetic calibration example; requires sbt-monitor[warning]."""
import numpy as np

from sbt_monitor.experimental import WarningCalibrator, WarningEpisode

features = ("position", "net_prediction_transport")
calibration = []
evaluation = []
for split, destination, offset in [("cal", calibration, 0), ("eval", evaluation, 100)]:
    for i in range(8):
        event = i < 4
        states = np.column_stack(
            [
                np.linspace(0, 1, 8) + 0.02 * i,
                np.r_[np.zeros(3), np.linspace(0.0, 0.8 if event else 0.1, 5)],
            ]
        )
        destination.append(
            WarningEpisode(
                episode_id=f"{split}-{i}",
                states=states,
                feature_names=features,
                event_time=6 if event else None,
                group=f"g{i % 2}",
                identity_set_id=f"ids-{offset+i}",
            )
        )

result = WarningCalibrator(
    feature_names=features,
    horizon=3,
    false_alarm_budget=0.25,
    event_rule_name="synthetic persistent event",
    model_scope_fingerprint="synthetic-model-scope",
).fit(calibration)
print(result.calibration_evaluation.to_dict())
print(result.readout.evaluate(evaluation).to_dict())
result.readout.save("synthetic_readout.json")
