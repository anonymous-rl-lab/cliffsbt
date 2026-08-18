# Results and conclusions

## Executive conclusion / 核心结论

The registered CURE-OR v2 experiment closed the complete
formation–warning–repair loop. Exact signed transition counts reconstructed all
paired risk increments; the frozen Hybrid25 chart warned 71 of 72 persistent
cliffs with three false alarms among 78 controls and a median lead of three
windows; and the calibration-gated update removed five model–family cliffs,
introduced none, and reduced mean deployment risk by 0.0620.

中文统一表述：在冻结的 CURE-OR v2 两阶段确认实验中，同一个边界输运对象贯通了
Cliff 的形成、提前观测与训练控制；三个注册模块均通过，机器结论为
`FORMATION_WARNING_REPAIR_CONFIRMED`。

## Integrity

- Package audit: PASS.
- Phase 1 independent audit: 22/22 PASS.
- Phase 2 independent audit: 16/16 PASS.
- Phase 1 confirmation labels scored: false.
- Blind commitment preserved: true.
- Phase order preserved: true.

## H1 — formation

- Module pass: true.
- Eligible operational paths: 150.
- Persistent cliffs: 72.
- Non-cliff controls: 78.
- Maximum absolute flow-closure error:
  `1.1102230246251565e-16`.
- Median endpoint-minus-baseline risk on cliff paths: `0.54`.

The paired risk increment closed numerically to machine precision under forward
minus recovery transitions. This is direct empirical verification of the exact
signed boundary-transport accounting on every observed path.

## H2 — outcome-blind warning

- Module pass: true.
- Timely warnings: `71/72 = 0.9861111111`.
- False alarms: `3/78 = 0.0384615385`.
- Median timely lead: 3 windows.

| Seed | Timely/cliffs | False/non-cliffs | Median lead |
|---:|---:|---:|---:|
| 113 | 15/15 | 0/15 | 2 |
| 127 | 12/12 | 3/18 | 2 |
| 139 | 15/15 | 0/15 | 3 |
| 151 | 15/15 | 0/15 | 3 |
| 163 | 14/15 | 0/15 | 4 |

The allowed calibration diagnostic was 59/69 timely, 6/81 false, and median
lead 4. Confirmation alarms and prediction arrays were committed before
confirmation labels were scored.

## H3 — guarded repair

- Module pass: true.
- Eligible models: 4/5 (`113, 139, 151, 163`).
- Abstaining model: seed 127; its deployed model remained the unchanged
  baseline under the fixed safety gate.
- Unique removed model–family cliffs: 5
  (`113:17`, `139:17`, `151:13`, `151:17`, `163:17`).
- Newly introduced model–family cliffs: 0.
- Models with at least one removed cliff: 4.
- Mean deployment-risk gain: `0.0620307692`.
- Model-seed cluster-bootstrap 95% interval:
  `[0.0298051282, 0.0852410256]`.
- Mean event-time gain: `1.14` windows.
- Event-time-gain 95% interval: `[0.4933333333, 1.7]`.

| Seed | Eligible | Mean risk gain | Mean event-time gain |
|---:|:---:|---:|---:|
| 113 | yes | 0.0904615 | 2.0667 |
| 127 | no | 0 | 0 |
| 139 | yes | 0.0597949 | 1.0667 |
| 151 | yes | 0.0745128 | 1.3333 |
| 163 | yes | 0.0853846 | 1.2333 |

The repair result supports this fixed 500-example full-coverage,
calibration-gated anchored update in this benchmark. It is not a comparative
test of coverage versus pressure, hazard, or another allocation policy.

## Scientific interpretation

The experiment supports four distinct levels of statement:

1. **Exact accounting:** for paired identities and a fixed deterministic
   classifier, signed forward-minus-recovery transport exactly reconstructs
   every risk increment. This identity is architecture-independent.
2. **Formation:** persistent cliffs occur when cumulative harmful transport
   exhausts the available risk headroom along an ordered deployment path.
3. **Warning:** under target calibration, fixed Hybrid25 telemetry, sufficient
   population scale/coherence, and the registered schedules, future persistent
   transport was warned before onset with the reported rates.
4. **Control:** the fixed guarded update reshaped the future risk trajectory,
   reduced deployment risk, and removed or delayed observed cliffs.

## Claim boundary

- The inferential units are five model seeds; 150 paths are not 150 independent
  replicates.
- The confirmation uses the same public mini CURE-OR benchmark and fixed
  identities/families. It is not fresh-data, fresh-identity, cross-dataset,
  cross-domain, or natural longitudinal-drift confirmation.
- Hybrid25 is outcome-blind at deployment, but labels are used for calibration
  and repair. The complete system is not fully label-free.
- `tau = 0.50` is the registered benchmark operating boundary, not an externally
  validated safety threshold.
- The experiment does not establish a universal 25-channel warning sensor or a
  universal repair rule.
- The exact boundary-transport identity is architecture-independent; the
  warning and repair findings are conditional empirical results.
- The experiment does not directly estimate local geometric quantities such as
  a risk-direction vector, curvature matrix, or observable-geometry matrix.

## Paper-ready result paragraph

> In a preregistered two-phase CURE-OR confirmation, five classifier heads and
> three frozen deployment schedules closed the formation–warning–repair loop on
> the same benchmark field: exact signed transition counts reconstructed all
> risk increments; a frozen outcome-blind Hybrid25 chart warned 71 of 72
> persistent cliffs with 3 false alarms among 78 controls and a three-window
> median lead; and a calibration-gated training update removed five
> model–family cliffs, introduced none, and reduced mean deployment risk by
> 0.0620 (model-seed-clustered 95% CI, 0.0298–0.0852).
