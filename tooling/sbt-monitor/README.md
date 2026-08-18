# sbt-monitor

**Ledger-first tools for signed boundary transport accounting, domain-calibrated warning, and deployment-cliff diagnostics.**

> [!IMPORTANT]
> 1. Exact sample-resolved ledgers require recurring identity IDs or an explicitly justified coupling.
> 2. One ledger assumes one fixed model fingerprint; model updates are not silently mixed with deployment transport.
> 3. The operational boundary is supplied by the user. `sbt-monitor` never chooses or certifies a safety threshold.
> 4. No pretrained or universal warning readout is shipped. Warning must be calibrated in the user's own labelled domain.
> 5. Warning scores are monitoring outputs, not authorization for shutdown, retraining, or any safety action.

## What this package is

`sbt-monitor` separates four objects that should not be conflated:

| Object | What it measures | Exact? | Stable in v0.1? |
|---|---|---:|---:|
| `TaskTransportLedger` | Correct-to-error and error-to-correct mass for paired identities | Yes, for a fixed model and fixed weights | Yes |
| `PersistentCliffRule` | User-declared first crossing plus persistence confirmation | Event semantics | Yes |
| `PredictionStateTransport` | Outcome-blind departure from and return to a baseline prediction | No; proxy only | Yes |
| `experimental.WarningCalibrator` | A readout fitted and thresholded on user calibration episodes | Domain conditional | Experimental |

The package does **not** contain the CURE-OR coefficients, a universal 7.5% false-alarm budget, or an automatic repair policy.

## Install

```bash
pip install sbt-monitor
```

Install the experimental calibration workflow only when needed:

```bash
pip install "sbt-monitor[warning]"
```

## Exact task ledger

```python
from sbt_monitor import TaskTransportLedger

ids = ["a", "b", "c", "d"]
ledger = TaskTransportLedger.fixed_panel(
    ids,
    model_fingerprint="sha256:model-v1",
)

ledger.update(
    window=0,
    ids=ids,
    y_true=[0, 0, 1, 1],
    y_pred=[0, 0, 1, 1],
    model_fingerprint="sha256:model-v1",
)
ledger.update(
    window=1,
    ids=["d", "b", "a", "c"],  # reordering is aligned by identity
    y_true=[1, 0, 0, 1],
    y_pred=[0, 0, 0, 1],
    model_fingerprint="sha256:model-v1",
)

step = ledger.steps()[0]
print(step.incident, step.recovery, step.sbt, step.turnover)
print(ledger.closure_report())
```

For every complete adjacent pair,

\[
R_{t+1}-R_t = J_t^+ - J_t^-.
\]

The scalar identity is deliberately modest. The resolved ledger adds incident and recovery separately, turnover, identity sets, first-crossing timing, and path-conditioned persistence.

## First crossing is not the same as a persistent cliff

```python
from sbt_monitor import ConsecutiveWindows, PersistentCliffRule

risk, windows = ledger.risk_series()
rule = PersistentCliffRule(
    boundary=0.30,
    persistence=ConsecutiveWindows(2),
)
event = rule.evaluate(risk, windows)
print(event.first_crossing_time)
print(event.persistent_cliff_time)
```

Changing `boundary` changes event labels, not the transport ledger.

## Outcome-blind prediction-state proxy

```python
from sbt_monitor import PredictionStateTransport, TransportAwareStateBuilder

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
vector = TransportAwareStateBuilder(
    "static+net_prediction_transport"
).transform(state)
# A custom subset is also allowed:
minimal = TransportAwareStateBuilder(["departure_mass", "net_prediction_transport"]).transform(state)
```

`net_prediction_transport` is a baseline-prediction transition proxy. It is **not** task-error SBT and does not satisfy task-risk closure.

## Experimental domain calibration

```python
from sbt_monitor.experimental import WarningCalibrator

calibrator = WarningCalibrator(
    feature_names=feature_names,
    horizon=3,
    false_alarm_budget=0.075,  # user-declared; not a package default
    event_rule_name="risk>=0.30 for 2 consecutive windows",
    model_scope_fingerprint="sha256:my-model-scope",
)
result = calibrator.fit(calibration_episodes)
result.readout.save("readout.json")
```

The frozen JSON records feature schema, scaler, coefficients, threshold, horizon, event-rule name, model-scope fingerprint, package version, calibration hash, and scope warnings. Evaluation identifiers that overlap calibration raise `LeakageError` by default.

## Diagnostics

```python
from sbt_monitor import diagnostics

diagnostics.closure_audit(ledger)
diagnostics.identity_permutation(ledger, n=1000, seed=7)
diagnostics.peer_boundary(focal_ledger, peer_ledger)
diagnostics.threshold_sweep(
    ledger,
    boundaries=[0.20, 0.25, 0.30, 0.35],
    persistence=ConsecutiveWindows(2),
)
```

Peer-boundary diagnostics return RMSE, normalized error, and anchor-risk separation. They intentionally do not produce a tautological `PASS/FAIL` verdict.

## CLI

```bash
sbt-monitor ledger paired_predictions.csv \
  --correct-col correct \
  --boundary 0.30 \
  --persistence 2 \
  --json-out ledger.json \
  --html-report ledger.html

sbt-monitor spec
```

The CSV must be long-form with one row per `(window, identity)`.

## Scientific scope

The frozen v0.1 contract is in [`API_SCIENTIFIC_SCOPE_v0.1.md`](API_SCIENTIFIC_SCOPE_v0.1.md). The short form is:

- exact accounting is fixed-model and identity-paired;
- operational first passage is boundary relative;
- outcome-blind prediction-state transport is a proxy;
- warning is domain calibrated and experimental;
- the package never certifies safety or triggers automatic intervention.

## Development

```bash
python -m pip install -e ".[test]"
pytest
```

Build and validate release artifacts:

```bash
python -m build
python -m twine check dist/*
```

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The scientific manuscript and permanent archive DOI should be added before the public release associated with publication.
