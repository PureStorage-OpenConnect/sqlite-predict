# sqlite-predict

Prediction as a SQL primitive for SQLite: `forecast()`, `detect_anomalies()`,
`predict()`. The wheel bundles the zero-dependency loadable extension for your
platform.

```python
import sqlite3
import sqlite_predict

db = sqlite3.connect(":memory:")
sqlite_predict.load(db)

db.execute("CREATE TABLE readings(ts TEXT, value REAL)")
# ... insert rows ...

# forecast() is an aggregate: your query supplies the rows, and each
# group returns one JSON document
doc = db.execute("SELECT forecast(ts, value, 24) FROM readings").fetchone()[0]
print(doc)
```

See the [project README](https://github.com/PureStorage-OpenConnect/sqlite-predict)
for the full SQL surface, models, and benchmarks.
