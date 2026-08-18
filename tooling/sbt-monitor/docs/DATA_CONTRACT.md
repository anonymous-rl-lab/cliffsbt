# Data contract

## Task ledger

Every update requires:

- a unique window identifier;
- unique recurring identity IDs;
- either `(y_true, y_pred)`, `correct_mask`, or true-class margins;
- the same model fingerprint used by all other windows in the ledger.

The default `missing="raise"` requires the complete panel. `missing="intersection"` permits exact adjacent-step accounting on the valid pair set, but the cumulative ledger becomes non-telescopeable and `risk_series()` is refused.

## Prediction-state proxy

The proxy requires:

- baseline predicted class per identity;
- current predicted class per identity;
- optional outcome-blind top-one minus top-two margins;
- optional representation norms.

Labels are neither required nor accepted by the proxy builder.

## Warning episodes

`WarningEpisode.states` has shape `(time, features)`. `event_time` indexes a persistence-confirmed event in labelled calibration data or is `None` for a control. Calibration and evaluation episode or identity-set IDs must not overlap unless the caller explicitly opts into an in-sample diagnostic.
