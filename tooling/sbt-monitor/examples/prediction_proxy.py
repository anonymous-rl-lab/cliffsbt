"""Outcome-blind prediction-state proxy example."""
from sbt_monitor import PredictionStateTransport, TransportAwareStateBuilder

ids = ["A", "B", "C", "D"]
proxy = PredictionStateTransport.from_baseline(
    ids=ids,
    predictions=[0, 0, 1, 1],
    margins=[0.8, 0.6, 0.7, 0.9],
)
state = proxy.update(
    window=1,
    ids=ids,
    predictions=[0, 1, 1, 1],
    prediction_margins=[0.7, 0.1, 0.5, 0.8],
    representation_norm=[1.0, 1.1, 0.9, 1.0],
)
builder = TransportAwareStateBuilder("static+net_prediction_transport")
print(builder.feature_names)
print(builder.transform(state))
print("is task SBT:", state.is_task_sbt)
