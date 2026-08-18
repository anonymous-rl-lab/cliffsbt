# Cliff NMI v7 post hoc diagnostic addendum

All analyses reuse frozen outputs. They were specified after confirmation reveal and do not modify the preregistered CURE-OR H1--H3 decision.

## Nested prediction-state transport proxy by classifier-head seed

|   seed |   current_state |   static_plus_net |   static_plus_persistence |   static_state |
|-------:|----------------:|------------------:|--------------------------:|---------------:|
|    113 |              14 |                15 |                        12 |             12 |
|    127 |              12 |                12 |                        11 |             11 |
|    139 |              15 |                15 |                        15 |             15 |
|    151 |              14 |                14 |                        13 |             13 |
|    163 |              15 |                15 |                        11 |             11 |

## Complete-seed descriptive ranges

| variant                 | method                                  |   observed_timely_rate_difference |   timely_difference_p2_5 |   timely_difference_p97_5 |   observed_false_rate_difference |   false_difference_p2_5 |   false_difference_p97_5 |   positive_seed_directions |   zero_seed_directions |   negative_seed_directions |
|:------------------------|:----------------------------------------|----------------------------------:|-------------------------:|--------------------------:|---------------------------------:|------------------------:|-------------------------:|---------------------------:|-----------------------:|---------------------------:|
| static_plus_net         | Static + net prediction-state transport |                            0.1250 |                   0.0417 |                    0.2133 |                           0.0000 |                  0.0000 |                   0.0000 |                          4 |                      1 |                          0 |
| static_plus_persistence | Static + persistent departure           |                            0.0000 |                   0.0000 |                    0.0000 |                           0.0000 |                  0.0000 |                   0.0000 |                          0 |                      5 |                          0 |
| current_state           | Static + both proxy channels            |                            0.1111 |                   0.0400 |                    0.2000 |                           0.0000 |                  0.0000 |                   0.0000 |                          4 |                      1 |                          0 |

## False-alarm-budget sensitivity

| kind                    | method                                  |   false_budget |   threshold |   cal_timely |   cal_false |   cal_cliffs |   cal_controls |   cal_median_lead |   conf_timely |   conf_false |   conf_cliffs |   conf_controls |   conf_median_lead |
|:------------------------|:----------------------------------------|---------------:|------------:|-------------:|------------:|-------------:|---------------:|------------------:|--------------:|-------------:|--------------:|----------------:|-------------------:|
| static_state            | Static current telemetry                |         0.0500 |      0.9034 |           27 |           3 |           69 |             81 |            3.0000 |            20 |            0 |            72 |              78 |             2.0000 |
| static_state            | Static current telemetry                |         0.0750 |      0.5061 |           61 |           6 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_state            | Static current telemetry                |         0.1000 |      0.5061 |           61 |           6 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_state            | Static current telemetry                |         0.1500 |      0.5044 |           62 |           9 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_state            | Static current telemetry                |         0.2000 |      0.5044 |           62 |           9 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_plus_net         | Static + net prediction-state transport |         0.0500 |      0.9462 |           28 |           4 |           69 |             81 |            4.0000 |            31 |            1 |            72 |              78 |             4.0000 |
| static_plus_net         | Static + net prediction-state transport |         0.0750 |      0.6481 |           59 |           6 |           69 |             81 |            5.0000 |            71 |            3 |            72 |              78 |             3.0000 |
| static_plus_net         | Static + net prediction-state transport |         0.1000 |      0.6054 |           61 |           7 |           69 |             81 |            5.0000 |            72 |            3 |            72 |              78 |             3.0000 |
| static_plus_net         | Static + net prediction-state transport |         0.1500 |      0.5368 |           62 |           9 |           69 |             81 |            5.0000 |            72 |            5 |            72 |              78 |             3.0000 |
| static_plus_net         | Static + net prediction-state transport |         0.2000 |      0.5368 |           62 |           9 |           69 |             81 |            5.0000 |            72 |            5 |            72 |              78 |             3.0000 |
| static_plus_persistence | Static + persistent departure           |         0.0500 |      0.9079 |           26 |           3 |           69 |             81 |            3.0000 |            20 |            0 |            72 |              78 |             2.0000 |
| static_plus_persistence | Static + persistent departure           |         0.0750 |      0.5121 |           61 |           6 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_plus_persistence | Static + persistent departure           |         0.1000 |      0.5073 |           62 |           8 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_plus_persistence | Static + persistent departure           |         0.1500 |      0.5073 |           62 |           8 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| static_plus_persistence | Static + persistent departure           |         0.2000 |      0.5073 |           62 |           8 |           69 |             81 |            5.0000 |            62 |            3 |            72 |              78 |             3.0000 |
| current_state           | Static + both proxy channels            |         0.0500 |      0.9459 |           28 |           4 |           69 |             81 |            4.0000 |            32 |            1 |            72 |              78 |             3.5000 |
| current_state           | Static + both proxy channels            |         0.0750 |      0.6516 |           59 |           6 |           69 |             81 |            5.0000 |            70 |            3 |            72 |              78 |             3.0000 |
| current_state           | Static + both proxy channels            |         0.1000 |      0.6062 |           61 |           7 |           69 |             81 |            5.0000 |            72 |            3 |            72 |              78 |             3.0000 |
| current_state           | Static + both proxy channels            |         0.1500 |      0.5152 |           62 |          12 |           69 |             81 |            5.0000 |            72 |            6 |            72 |              78 |             3.0000 |
| current_state           | Static + both proxy channels            |         0.2000 |      0.5152 |           62 |          12 |           69 |             81 |            5.0000 |            72 |            6 |            72 |              78 |             3.0000 |
| full34                  | Full temporal chart                     |         0.0500 |      0.9392 |           46 |           4 |           69 |             81 |            5.0000 |            36 |            0 |            72 |              78 |             3.0000 |
| full34                  | Full temporal chart                     |         0.0750 |      0.7158 |           58 |           6 |           69 |             81 |            5.0000 |            61 |            1 |            72 |              78 |             3.0000 |
| full34                  | Full temporal chart                     |         0.1000 |      0.5109 |           60 |           8 |           69 |             81 |            5.0000 |            70 |            3 |            72 |              78 |             3.0000 |
| full34                  | Full temporal chart                     |         0.1500 |      0.4818 |           61 |          10 |           69 |             81 |            5.0000 |            70 |            3 |            72 |              78 |             3.0000 |
| full34                  | Full temporal chart                     |         0.2000 |      0.4818 |           61 |          10 |           69 |             81 |            5.0000 |            70 |            3 |            72 |              78 |             3.0000 |

## Trained-peer mismatch by focal seed

| domain     |   focal_seed | subset   |   n |   peer_rmse_mean |   peer_rmse_median |   peer_nrmse_rms_median |   anchor_risk_gap_median |
|:-----------|-------------:|:---------|----:|-----------------:|-------------------:|------------------------:|-------------------------:|
| CIFAR-10-C |           31 | all      |  15 |           0.0121 |             0.0088 |                  0.1373 |                   0.0109 |
| CIFAR-10-C |           31 | active   |  15 |           0.0121 |             0.0088 |                  0.1373 |                   0.0109 |
| CIFAR-10-C |           31 | cliff    |  12 |           0.0133 |             0.0109 |                  0.1353 |                   0.0109 |
| CIFAR-10-C |           47 | all      |  15 |           0.0170 |             0.0097 |                  0.2508 |                   0.0002 |
| CIFAR-10-C |           47 | active   |  15 |           0.0170 |             0.0097 |                  0.2508 |                   0.0002 |
| CIFAR-10-C |           47 | cliff    |  11 |           0.0192 |             0.0121 |                  0.1641 |                   0.0002 |
| CIFAR-10-C |           61 | all      |  15 |           0.0170 |             0.0097 |                  0.2095 |                   0.0002 |
| CIFAR-10-C |           61 | active   |  15 |           0.0170 |             0.0097 |                  0.2095 |                   0.0002 |
| CIFAR-10-C |           61 | cliff    |  12 |           0.0198 |             0.0136 |                  0.1870 |                   0.0002 |
| CURE-OR    |          113 | all      |  30 |           0.0263 |             0.0274 |                  1.0000 |                   0.0200 |
| CURE-OR    |          113 | active   |  27 |           0.0292 |             0.0283 |                  1.0000 |                   0.0200 |
| CURE-OR    |          113 | cliff    |  15 |           0.0360 |             0.0327 |                  0.7188 |                   0.0200 |
| CURE-OR    |          127 | all      |  30 |           0.0253 |             0.0241 |                  0.8751 |                   0.0200 |
| CURE-OR    |          127 | active   |  27 |           0.0275 |             0.0245 |                  0.8751 |                   0.0200 |
| CURE-OR    |          127 | cliff    |  12 |           0.0327 |             0.0271 |                  0.7071 |                   0.0200 |
| CURE-OR    |          139 | all      |  30 |           0.0252 |             0.0255 |                  0.8434 |                   0.0400 |
| CURE-OR    |          139 | active   |  30 |           0.0252 |             0.0255 |                  0.8434 |                   0.0400 |
| CURE-OR    |          139 | cliff    |  15 |           0.0341 |             0.0346 |                  0.5898 |                   0.0400 |
| CURE-OR    |          151 | all      |  30 |           0.0256 |             0.0254 |                  0.8094 |                   0.0200 |
| CURE-OR    |          151 | active   |  27 |           0.0284 |             0.0283 |                  0.8094 |                   0.0200 |
| CURE-OR    |          151 | cliff    |  15 |           0.0360 |             0.0327 |                  0.7348 |                   0.0200 |
| CURE-OR    |          163 | all      |  30 |           0.0228 |             0.0220 |                  0.8004 |                   0.0200 |
| CURE-OR    |          163 | active   |  24 |           0.0274 |             0.0270 |                  0.8004 |                   0.0200 |
| CURE-OR    |          163 | cliff    |  15 |           0.0326 |             0.0337 |                  0.6102 |                   0.0200 |
