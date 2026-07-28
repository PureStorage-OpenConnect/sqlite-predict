---
title: Python
description: Use sqlite-predict from Python in 60 seconds.
---

Install from PyPI. The wheels are prebuilt per platform, so there's no compiler
step:

```sh
pip install sqlite-predict
```

Load it into any `sqlite3` connection and call the primitives over your rows:

```python
import sqlite3
import sqlite_predict

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_predict.load(db)
db.enable_load_extension(False)

# did it load?
print(db.execute("select predict_version()").fetchone()[0])

# a small hourly series
db.execute("create table readings(ts text, value real)")
db.executemany(
    "insert into readings values (?, ?)",
    [(f"2024-01-01T{h:02d}:00:00", 50 + h) for h in range(24)],
)

# forecast 6 steps ahead, with prediction intervals. forecast() is an
# aggregate like sum(): your statement supplies the rows, and each
# group returns one JSON document.
import json

doc = json.loads(db.execute(
    "select forecast(ts, value, 6) from readings").fetchone()[0])
print(doc["status"])
for row in doc["rows"]:
    print(row["step"], round(row["forecast"], 1),
          round(row["lower_bound"], 1), round(row["upper_bound"], 1))
```

Because it is an aggregate, `WHERE`, joins, and bound parameters compose, and
`GROUP BY city` returns one document per city. This is the form SQLAlchemy and
friends build naturally.

`sqlite_predict.loadable_path()` returns the path to the loadable if you need to
load it into another connection library yourself.

Next: [Operations](../../guides/operations/) for what else you can call,
[Using with ORMs](../../guides/orms/) for SQLAlchemy composition, or
[Auto-selection & conformal intervals](../../guides/auto-and-conformal/).
