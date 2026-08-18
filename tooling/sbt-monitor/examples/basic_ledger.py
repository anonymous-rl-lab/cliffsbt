"""Minimal exact-ledger example."""
from sbt_monitor import ConsecutiveWindows, PersistentCliffRule, TaskTransportLedger

ids = ["A", "B", "C", "D"]
ledger = TaskTransportLedger.fixed_panel(ids, model_fingerprint="example-model-v1")
ledger.update(window=0, ids=ids, correct_mask=[True, True, False, True], model_fingerprint="example-model-v1")
ledger.update(window=1, ids=["D", "B", "A", "C"], correct_mask=[True, False, True, True], model_fingerprint="example-model-v1")
ledger.update(window=2, ids=ids, correct_mask=[False, False, True, True], model_fingerprint="example-model-v1")

for step in ledger.steps():
    print(step.to_dict())

risk, windows = ledger.risk_series()
event = PersistentCliffRule(0.50, ConsecutiveWindows(2)).evaluate(risk, windows)
print(event.to_dict())
