# Contributing

Contributions are welcome when they preserve the frozen scientific distinctions in `API_SCIENTIFIC_SCOPE_v0.1.md`.

Before opening a pull request:

1. add or update tests;
2. run `pytest`;
3. do not expose `WarningCalibrator` at the package top level in the 0.1 series;
4. do not call prediction-state departure/return features task-error SBT;
5. do not add a universal warning threshold or automatic action;
6. document whether a result is exact, calibrated, exploratory, or post hoc.
