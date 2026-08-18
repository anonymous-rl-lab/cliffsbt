# Diagnostics

- `closure_audit`: numerical closure and pairing coverage.
- `identity_permutation`: preserves each window's error count while breaking identity continuity; uniform weights only in v0.1.
- `peer_boundary`: RMSE, normalized error and anchor-risk gap; never a tautological pass/fail.
- `threshold_sweep`: moves the operational boundary while keeping the ledger fixed.
- `stationary_noise_floor`: maximum-window reference quantile from stationary sequences.
- `nested_state_ablation`: independently calibrates declared feature subsets under one false-alarm budget.
- `false_alarm_budget_curve`: keeps the fitted score model fixed and varies only threshold selection budget.
