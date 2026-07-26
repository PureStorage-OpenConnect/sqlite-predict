# sqlite-predict (Python)

Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
`predict()`, with replayable receipts. The wheel bundles the zero-dependency
loadable extension for your platform.

```python
import sqlite3, sqlite_predict

db = sqlite3.connect(":memory:")
sqlite_predict.load(db)

db.execute("CREATE TABLE readings(ts TEXT, value REAL)")
# ... insert rows ...
for row in db.execute(
    "SELECT * FROM forecast('SELECT ts, value FROM readings', 24)"
):
    print(row)
```

See the [project README](https://github.com/PureStorage-OpenConnect/sqlite-predict)
for the full SQL surface, models, and benchmarks.
