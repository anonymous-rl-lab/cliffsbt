# Scientific scope

The normative v0.1 contract is [`../API_SCIENTIFIC_SCOPE_v0.1.md`](../API_SCIENTIFIC_SCOPE_v0.1.md).

## Exact layer

`TaskTransportLedger` uses a fixed deterministic decision rule and recurring identity IDs. For fixed non-negative identity weights,

\[
\Delta R_t = J_t^+ - J_t^-.
\]

Closure is an accounting identity, not a causal mechanism or warning guarantee.

## Event layer

A first threshold crossing and a persistence-confirmed operational cliff are different objects. The user supplies both the boundary and persistence rule. The package never interprets the boundary as a safety limit.

## Proxy layer

`PredictionStateTransport` tracks departures from and returns to a baseline predicted class. It is outcome blind and identity anchored, but it is not task-error SBT. Its usefulness must be established by user-domain calibration.

## Warning layer

The v0.1 warning workflow is experimental. It fits coefficients and an alarm threshold using labelled calibration episodes and records the calibration hash. No pretrained readout is bundled.
