---
title: Python
description: Use sqlite-predict from Python in 60 seconds.
---

Install from PyPI. The wheels are prebuilt per platform, so there's no compiler
step:

```sh
pip install sqlite-predict
```

Load it into any `sqlite3` connection and call the primitives over a `SELECT`:

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

# forecast 6 steps ahead, with prediction intervals
for step, fc, lo, hi in db.execute(
    "select step, forecast, lower_bound, upper_bound "
    "from forecast('select ts, value from readings', 6)"
):
    print(step, round(fc, 1), round(lo, 1), round(hi, 1))
```

`sqlite_predict.loadable_path()` returns the path to the loadable if you need to
load it into another connection library yourself.

Next: [Operations](/guides/operations/) for what else you can call, or
[Auto-selection & conformal intervals](/guides/auto-and-conformal/).
