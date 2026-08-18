# Examples

```bash
python examples/basic_ledger.py
python examples/prediction_proxy.py
pip install -e ".[warning]"
python examples/calibrate_warning.py

sbt-monitor ledger examples/paired_predictions.csv \
  --correct-col correct --boundary 0.5 --persistence 2 \
  --html-report /tmp/sbt-report.html
```
