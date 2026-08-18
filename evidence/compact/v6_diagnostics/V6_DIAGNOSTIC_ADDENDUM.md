# Cliff NMI v6 diagnostic addendum

All analyses below reuse committed outputs. The nested warning comparison refits each feature set only on calibration identities under the same false-alarm budget. The trained-peer normalization is arithmetic on frozen risk and prediction arrays.

## Strictly nested current-state channel ablation

| method                                          |   cal_timely |   cal_false |   conf_timely |   conf_false |   conf_median_lead |   delta_timely_vs_static |   delta_false_vs_static |
|:------------------------------------------------|-------------:|------------:|--------------:|-------------:|-------------------:|-------------------------:|------------------------:|
| Static current telemetry (refit)                |           61 |           6 |            62 |            3 |                  3 |                        0 |                       0 |
| Static current + net departure-recovery (refit) |           59 |           6 |            71 |            3 |                  3 |                        9 |                       0 |
| Static current + persistent departure (refit)   |           61 |           6 |            62 |            3 |                  3 |                        0 |                       0 |
| Current active-state telemetry (refit)          |           59 |           6 |            70 |            3 |                  3 |                        8 |                       0 |

## Domain-specific trained-peer normalization

| domain     | subset   |   n |   self_rmse_mean |   peer_rmse_mean |   peer_rmse_median |   incorrect_rmse_mean |   incorrect_rmse_median |   focal_delta_rms_mean |   focal_delta_rms_median |   peer_nrmse_rms_median |   peer_rmse_over_mae_median |   baseline_risk_gap_median |
|:-----------|:---------|----:|-----------------:|-----------------:|-------------------:|----------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|----------------------------:|---------------------------:|
| CIFAR-10-C | all      |  45 |         0.000000 |         0.015389 |           0.009711 |              0.074870 |                0.069808 |               0.066870 |                 0.063067 |                0.165213 |                    0.188196 |                   0.000200 |
| CIFAR-10-C | active   |  45 |         0.000000 |         0.015389 |           0.009711 |              0.074870 |                0.069808 |               0.066870 |                 0.063067 |                0.165213 |                    0.188196 |                   0.000200 |
| CIFAR-10-C | cliff    |  35 |         0.000000 |         0.017377 |           0.012147 |              0.087107 |                0.084183 |               0.077910 |                 0.076021 |                0.164113 |                    0.179733 |                   0.000200 |
| CURE-OR    | all      | 150 |         0.000000 |         0.025030 |           0.024495 |            —        |              —        |               0.035466 |                 0.027073 |                0.857969 |                    1.105542 |                   0.020000 |
| CURE-OR    | active   | 135 |         0.000000 |         0.027502 |           0.026458 |            —        |              —        |               0.039407 |                 0.038730 |                0.857969 |                    1.105542 |                   0.020000 |
| CURE-OR    | cliff    |  72 |         0.000000 |         0.034334 |           0.032913 |            —        |              —        |               0.057927 |                 0.058020 |                0.707107 |                    0.782231 |                   0.020000 |

## CIFAR boundary-specificity gradient

| domain     | subset   |   incorrect_partition_rmse |   trained_peer_rmse |   focal_self_rmse |   peer_nrmse_rms_median |   peer_rmse_over_mae_median |
|:-----------|:---------|---------------------------:|--------------------:|------------------:|------------------------:|----------------------------:|
| CIFAR-10-C | all      |                   0.074870 |            0.015389 |          0.000000 |                0.165213 |                    0.188196 |
| CIFAR-10-C | cliff    |                   0.087107 |            0.017377 |          0.000000 |                0.164113 |                    0.179733 |
