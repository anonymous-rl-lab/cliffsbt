# Experimental warning calibration

Install:

```bash
pip install "sbt-monitor[warning]"
```

The workflow is intentionally domain specific:

1. define the operational boundary and persistence-confirmed event outside the package;
2. construct outcome-blind telemetry states;
3. create labelled calibration episodes;
4. select regularization by leave-one-group-out calibration;
5. choose a threshold subject to the user-declared calibration false-alarm budget;
6. freeze and serialize the readout;
7. evaluate on disjoint identities without threshold tuning.

`WarningCalibrator` requires an explicit event-rule name and model-scope fingerprint. `FrozenReadout` stores those fields together with the package version, feature schema, scaler, coefficients, intercept, threshold, horizon, budget, calibration IDs, identity-set IDs and a SHA-256 calibration hash.

The workflow does not prove dynamic sufficiency, threshold invariance, or cross-domain transfer. Use `nested_state_ablation` and `false_alarm_budget_curve` as diagnostics, then seek prospective replication.


Minimal constructor:

```python
from sbt_monitor.experimental import WarningCalibrator, WarningEpisode

calibrator = WarningCalibrator(
    feature_names=("departure_mass", "net_prediction_transport"),
    horizon=3,
    false_alarm_budget=0.075,
    event_rule_name="risk>=0.30 for 2 consecutive windows",
    model_scope_fingerprint="sha256:my-calibration-model-scope",
)
```
