# Cliff NMI v6 committed-output diagnostic report

All analyses reuse frozen outputs and fixed identity splits. No classifier, challenge path or preregistered H1–H3 decision is altered. The comparison specification was written after confirmation reveal and is therefore post hoc, although model fitting and threshold selection are restricted to calibration data.

## Fair refitted warning baselines

| method                                          | type       |   threshold |   cal_timely_rate |   cal_false_rate |   cal_median_lead |   cal_timely |   cal_false |   cal_cliffs |   cal_controls |   conf_timely_rate |   conf_false_rate |   conf_median_lead |   conf_timely |   conf_false |   conf_cliffs |   conf_controls |   risk_ridge_alpha |
|:------------------------------------------------|:-----------|------------:|------------------:|-----------------:|------------------:|-------------:|------------:|-------------:|---------------:|-------------------:|------------------:|-------------------:|--------------:|-------------:|--------------:|----------------:|-------------------:|
| Registered Hybrid25                             | registered |      0.8700 |            0.8551 |           0.0741 |            4.0000 |           59 |           6 |           69 |             81 |             0.9861 |            0.0385 |             3.0000 |            71 |            3 |            72 |              78 |           —      |
| Time only (refit)                               | refit      |      0.7164 |            0.0000 |           0.0000 |          —      |            0 |           0 |           69 |             81 |             0.0000 |            0.0000 |           —      |             0 |            0 |            72 |              78 |           —      |
| Static current telemetry (refit)                | refit      |      0.5076 |            0.8841 |           0.0741 |            5.0000 |           61 |           6 |           69 |             81 |             0.8611 |            0.0385 |             3.0000 |            62 |            3 |            72 |              78 |           —      |
| Static current + net departure-recovery (refit) | refit      |      0.6488 |            0.8551 |           0.0741 |            5.0000 |           59 |           6 |           69 |             81 |             0.9861 |            0.0385 |             3.0000 |            71 |            3 |            72 |              78 |           —      |
| Static current + persistent departure (refit)   | refit      |      0.5149 |            0.8841 |           0.0741 |            5.0000 |           61 |           6 |           69 |             81 |             0.8611 |            0.0385 |             3.0000 |            62 |            3 |            72 |              78 |           —      |
| Current active-state telemetry (refit)          | refit      |      0.6522 |            0.8551 |           0.0741 |            5.0000 |           59 |           6 |           69 |             81 |             0.9722 |            0.0385 |             3.0000 |            70 |            3 |            72 |              78 |           —      |
| Entropy-margin trend (refit)                    | refit      |      0.7121 |            0.6522 |           0.0741 |            5.0000 |           45 |           6 |           69 |             81 |             0.7917 |            0.2564 |             2.0000 |            57 |           20 |            72 |              78 |           —      |
| Unsigned shift (refit)                          | refit      |      0.7636 |            0.7536 |           0.0741 |            3.5000 |           52 |           6 |           69 |             81 |             0.8889 |            0.2692 |             3.0000 |            64 |           21 |            72 |              78 |           —      |
| Hybrid25 full temporal (refit)                  | refit      |      0.7171 |            0.8406 |           0.0741 |            5.0000 |           58 |           6 |           69 |             81 |             0.8472 |            0.0128 |             3.0000 |            61 |            1 |            72 |              78 |           —      |
| Estimated current risk                          | refit      |      0.5219 |            0.3913 |           0.0494 |            3.0000 |           27 |           4 |           69 |             81 |             0.3889 |            0.0769 |             2.0000 |            28 |            6 |            72 |              78 |             0.0100 |
| Estimated risk + slope                          | refit      |      0.9676 |            0.3188 |           0.0741 |            6.0000 |           22 |           6 |           69 |             81 |             0.7917 |            0.1795 |             2.0000 |            57 |           14 |            72 |              78 |             0.0100 |
| Risk-proxy CUSUM                                | refit      |      0.2028 |            0.4203 |           0.0741 |            2.0000 |           29 |           6 |           69 |             81 |             0.6806 |            0.3462 |             3.0000 |            49 |           27 |            72 |              78 |             0.0100 |

## Registered false alarms by classifier-head seed

|   seed |   false |   controls |
|-------:|--------:|-----------:|
|    113 |       0 |         15 |
|    127 |       3 |         18 |
|    139 |       0 |         15 |
|    151 |       0 |         15 |
|    163 |       0 |         15 |

Descriptive complete-seed bootstrap range: [0.0000, 0.1071].

## Cliff-difficulty strata

Calibration cutpoints: `{"endpoint_overshoot": [0.12, 0.38], "endpoint_delta": [0.26, 0.52], "initial_headroom": [0.14, 0.14], "event": [3.0, 7.0], "last_pre_event_headroom": [0.03333333333333321, 0.06], "pre_event_slope": [0.01999999999999999, 0.05000000000000002]}`

| role         | stratifier              | stratum   |   n |   timely |   timely_rate |   median_lead |   median_value |
|:-------------|:------------------------|:----------|----:|---------:|--------------:|--------------:|---------------:|
| calibration  | endpoint_overshoot      | low       |  24 |       22 |        0.9167 |        5.0000 |         0.0500 |
| calibration  | endpoint_overshoot      | mid       |  39 |       34 |        0.8718 |        3.5000 |         0.3600 |
| calibration  | endpoint_overshoot      | high      |   6 |        3 |        0.5000 |        1.0000 |         0.4000 |
| calibration  | endpoint_delta          | low       |  24 |       22 |        0.9167 |        5.0000 |         0.1900 |
| calibration  | endpoint_delta          | mid       |  36 |       31 |        0.8611 |        5.0000 |         0.5000 |
| calibration  | endpoint_delta          | high      |   9 |        6 |        0.6667 |        2.0000 |         0.5400 |
| calibration  | initial_headroom        | low       |  54 |       46 |        0.8519 |        4.0000 |         0.1400 |
| calibration  | initial_headroom        | high      |  15 |       13 |        0.8667 |        5.0000 |         0.1600 |
| calibration  | event                   | low       |  26 |       18 |        0.6923 |        1.0000 |         2.0000 |
| calibration  | event                   | mid       |  27 |       25 |        0.9259 |        5.0000 |         6.0000 |
| calibration  | event                   | high      |  16 |       16 |        1.0000 |        8.0000 |         9.0000 |
| calibration  | last_pre_event_headroom | low       |  23 |       23 |        1.0000 |        4.0000 |         0.0200 |
| calibration  | last_pre_event_headroom | mid       |  33 |       31 |        0.9394 |        5.0000 |         0.0400 |
| calibration  | last_pre_event_headroom | high      |  13 |        5 |        0.3846 |        1.0000 |         0.1400 |
| calibration  | pre_event_slope         | low       |  25 |       17 |        0.6800 |        7.0000 |         0.0000 |
| calibration  | pre_event_slope         | mid       |  23 |       22 |        0.9565 |        3.0000 |         0.0300 |
| calibration  | pre_event_slope         | high      |  21 |       20 |        0.9524 |        2.0000 |         0.0700 |
| confirmation | endpoint_overshoot      | low       |  27 |       27 |        1.0000 |        3.0000 |         0.0600 |
| confirmation | endpoint_overshoot      | mid       |  33 |       32 |        0.9697 |        2.0000 |         0.3600 |
| confirmation | endpoint_overshoot      | high      |  12 |       12 |        1.0000 |        2.0000 |         0.4000 |
| confirmation | endpoint_delta          | low       |   9 |        9 |        1.0000 |        3.0000 |         0.2400 |
| confirmation | endpoint_delta          | mid       |  24 |       24 |        1.0000 |        5.0000 |         0.3100 |
| confirmation | endpoint_delta          | high      |  39 |       38 |        0.9744 |        1.0000 |         0.6000 |
| confirmation | initial_headroom        | high      |  72 |       71 |        0.9861 |        3.0000 |         0.2200 |
| confirmation | event                   | low       |  29 |       28 |        0.9655 |        1.0000 |         2.0000 |
| confirmation | event                   | mid       |  25 |       25 |        1.0000 |        3.0000 |         6.0000 |
| confirmation | event                   | high      |  18 |       18 |        1.0000 |        7.0000 |         8.5000 |
| confirmation | last_pre_event_headroom | low       |  16 |       16 |        1.0000 |        4.5000 |         0.0200 |
| confirmation | last_pre_event_headroom | mid       |  39 |       38 |        0.9744 |        3.0000 |         0.0400 |
| confirmation | last_pre_event_headroom | high      |  17 |       17 |        1.0000 |        1.0000 |         0.1200 |
| confirmation | pre_event_slope         | low       |  10 |       10 |        1.0000 |        5.0000 |         0.0100 |
| confirmation | pre_event_slope         | mid       |  24 |       24 |        1.0000 |        5.0000 |         0.0400 |
| confirmation | pre_event_slope         | high      |  38 |       37 |        0.9737 |        1.0000 |         0.1050 |

## Trained peer-boundary placebo

| domain     | subset   |   n |   self_rmse_max |   peer_rmse_mean |   peer_rmse_median |   peer_rmse_min |   peer_rmse_max |   baseline_risk_gap_median |
|:-----------|:---------|----:|----------------:|-----------------:|-------------------:|----------------:|----------------:|---------------------------:|
| CIFAR-10-C | all      |  45 |        0.000000 |         0.015389 |           0.009711 |        0.002528 |        0.046373 |                   0.000200 |
| CIFAR-10-C | active   |  45 |        0.000000 |         0.015389 |           0.009711 |        0.002528 |        0.046373 |                   0.000200 |
| CIFAR-10-C | cliff    |  35 |        0.000000 |         0.017377 |           0.012147 |        0.004892 |        0.046373 |                   0.000200 |
| CURE-OR    | all      | 150 |        0.000000 |         0.025030 |           0.024495 |        0.000000 |        0.061644 |                   0.020000 |
| CURE-OR    | active   | 135 |        0.000000 |         0.027502 |           0.026458 |        0.005774 |        0.061644 |                   0.020000 |
| CURE-OR    | cliff    |  72 |        0.000000 |         0.034334 |           0.032913 |        0.012910 |        0.061644 |                   0.020000 |
