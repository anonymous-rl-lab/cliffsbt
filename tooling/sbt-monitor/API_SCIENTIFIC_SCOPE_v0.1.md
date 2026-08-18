# `sbt-monitor` v0.1.0 — Frozen API and Scientific-Scope Specification

**Status:** FROZEN for the v0.1.0 implementation  
**PyPI distribution:** `sbt-monitor`  
**Python import:** `sbt_monitor`  
**License:** Apache-2.0  
**Scientific posture:** ledger first; no universal alarm; no automatic intervention.

## 1. Objects that must remain separate

| Layer | Stable object | Labels needed online? | Exact statement | v0.1 status |
|---|---|---:|---|---|
| Task accounting | `TaskTransportLedger` | Yes | For a fixed model and identity-paired outcomes, weighted risk change equals incident minus recovery mass. | Stable |
| Operational event | `FirstCrossingRule`, `PersistentCliffRule` | Risk sequence required | The user-declared boundary selects a first crossing; a separately declared persistence rule confirms an operational cliff. | Stable |
| Outcome-blind proxy | `PredictionStateTransport`, `TransportAwareStateBuilder` | No | Departure from and return to a baseline prediction are proxies for task transport, not task-error SBT. | Stable |
| Warning | `experimental.WarningCalibrator` | Labels only in calibration | A readout is fitted and thresholded in the user's calibration domain; no bundled readout or universal threshold is provided. | Experimental |

The ledger is boundary independent. Changing an operational boundary may change first-passage labels and warning performance, but must not change incident, recovery, turnover or task SBT.

## 2. Stable public API

```python
from sbt_monitor import (
    TaskTransportLedger,
    FirstCrossingRule,
    PersistentCliffRule,
    ConsecutiveWindows,
    PredictionStateTransport,
    TransportAwareStateBuilder,
    diagnostics,
)
```

Required ledger update contract:

```python
ledger.update(
    window=t,
    ids=identity_ids,
    y_true=labels,
    y_pred=predictions,
    model_fingerprint=model_hash,
)
```

Identity IDs are mandatory. Reordering is aligned by ID. Duplicate IDs, a changed model fingerprint, or missing identities raise by default. `missing="intersection"` is allowed only as a declared downgrade: each adjacent-step closure is computed on the valid pair set and the cumulative ledger is marked non-telescopeable.

Operational events are applied after accounting:

```python
event_rule = PersistentCliffRule(
    boundary=0.30,
    persistence=ConsecutiveWindows(2),
)
event = event_rule.evaluate(ledger.risk_series())
```

Outcome-blind state construction uses explicitly different names:

```python
proxy = PredictionStateTransport.from_baseline(...)
state = TransportAwareStateBuilder("static+net_prediction_transport")
```

The string `net_sbt` is prohibited for online proxy features.

## 3. Non-negotiable safeguards

1. **Identity pairing:** exact sample-resolved ledgers require recurring identities or an explicitly justified coupling.
2. **Fixed model:** one ledger has one model fingerprint. Model changes are not silently combined with deployment transport.
3. **User-declared boundary:** the package never chooses, certifies or recommends a safety threshold.
4. **Domain calibration:** no pretrained warning weights or universal false-alarm budget are shipped.
5. **No automatic action:** warning scores are monitoring outputs, not authorization for shutdown, retraining or safety intervention.
6. **Evidence labels:** reports distinguish exact, calibrated, post hoc and unsupported interpretations.

## 4. Supported and unsupported claims

**Supported in v0.1:** exact fixed-model paired accounting; weighted accounting with fixed non-negative identity weights; first-crossing versus persistence-confirmed event semantics; pairing and closure audits; identity-permutation, peer-boundary effect sizes, threshold sensitivity and nested-state diagnostics; generation of a provenance-rich HTML report.

**Not supported in v0.1:** universal early warning; label-free calibration; safety certification; causal attribution from closure alone; path-independent persistence; automatic repair selection; inference across changing models without an expanded boundary-motion analysis; exact cumulative accounting when the paired panel changes across steps.

## 5. Release gates

A v0.1.0 artifact may be published only if: all tests pass; wheel and sdist build cleanly; both artifacts install in a clean environment; metadata validation passes; README displays the five safeguards above before the quickstart; package and artifact SHA-256 manifests are generated; and the experimental warning namespace is not re-exported at the package top level.

## 6. Version roadmap

- **0.1.x:** stable accounting, event, proxy and diagnostic APIs; warning remains experimental.
- **0.2.0:** warning calibration may become stable only after prospective replication outside CURE-OR, including the frozen physical-robot nested state comparison.
- **1.0.0:** reserved for documented multi-domain, independent-model and physical-system validation with a stable serialization contract.
