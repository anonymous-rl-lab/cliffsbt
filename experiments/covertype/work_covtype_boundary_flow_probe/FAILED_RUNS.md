# Failed runs retained

## 2026-08-15: pretarget SHA 3cc80e36c224695fe00d09b058db3504fe7a3058b7e1a4c30e48447185774d26

The first smoke execution stopped with `KeyError: 'raw_no_elevation'`. The frozen
configuration named the second transport space `raw_no_elevation`, while the in-memory
window record used the mechanical key `raw`. Target arrays had been loaded before the
exception, but no result table or summary was emitted. The repair changes only the two
record keys (`raw` -> `raw_no_elevation` and corresponding position/distance keys). No
sample, source, model, metric, threshold, gate, or analysis rule changed.
