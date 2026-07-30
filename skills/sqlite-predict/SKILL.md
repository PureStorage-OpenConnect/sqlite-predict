---
name: sqlite-predict
description: >-
  Forecast, detect anomalies, and predict on tabular data directly in a
  SQLite database using the sqlite-predict extension. Use when an agent
  holds time series or tabular data in SQLite and needs predictions
  without exporting data, calling a cloud model, or building a pipeline.
license: MIT OR Apache-2.0
metadata:
  version: "0.1.0"
---

# Using sqlite-predict

The extension turns prediction into SQL. Load `predict0` (pip/npm/cargo
package `sqlite-predict`), then call functions over your own rows. Serving
is pure: no writes, works on read-only databases and inside views.

## Which operation for which question

| Question | Call |
| --- | --- |
| "What will this metric do next?" | `SELECT forecast(ts, value, horizon) FROM t` |
| "Which points are anomalous?" | `SELECT detect_anomalies(ts, value) FROM t` |
| "Predict a label/value for these rows" | `SELECT * FROM predict(train_sql, apply_sql, options)` |
| "How accurate would this be on my data?" | `SELECT * FROM backtest(series_sql, horizon)` |
| "Make serving instant and self-contained" | `distill_predict` / `distill_forecast` (see the distill-lifecycle skill) |

`forecast` and `detect_anomalies` are aggregates: plain SQL supplies the
rows, `GROUP BY` splits series, `WHERE` and joins compose, ORMs work
unchanged. `predict`, `backtest`, and the distillers are table-valued
functions that take a query string.

## Reading results

Each aggregate group returns one JSON document: `{"model", "status",
"rows"}`. Expand to typed rows in SQL when needed:

```sql
SELECT r.* FROM forecast_rows((SELECT forecast(ts, value, 24) FROM t)) AS r;
```

- `model` reports the model that actually served. With no model named,
  `auto` selects the best available per series and reports the winner's
  id, never the string "auto".
- `status` is `ok`, or a degraded status with empty rows:
  `insufficient_history` (fewer than 8 points) or `non_numeric` (a
  timestamp or value cell poisoned the series). Degraded statuses are
  data conditions, not errors; check `status` before using `rows`.
- Zero input rows return SQL NULL, the aggregate convention.

## Timestamps and values

Timestamps: ISO-8601 text (`YYYY-MM-DD[ T]HH:MM[:SS][Z]`) or integer
epoch seconds/milliseconds. Anything else poisons the series as
`non_numeric`. Values must be numeric. The aggregate floors timestamps
to whole seconds.

## Errors are self-explaining

Every call error is one string from a closed set, formatted
`PREDICT_ERR_<NAME>: detail` (options, schema, horizon, licensing,
model-not-found, and so on). There are no silent fallbacks: an unknown
option key, a non-read-only inner query, or tampered model weights all
fail loudly. When a call errors, read the message; it names the exact
problem and often the fix.

## Defaults an agent should know

- No model named means `auto`: bundled statistical models plus any
  eligible distilled students compete per series. Pass
  `'{"candidates":["theta-classic","tsb"]}'` to narrow the pool.
- Prediction intervals default to a Gaussian band that is overconfident
  on smooth data; pass `'{"interval_method":"conformal"}'` for
  calibrated coverage (statistical models only) and verify with
  `backtest` (see the interpret-backtest skill).
- `predict` with `NULL` train_query serves a distilled student; with a
  train query it runs the in-context `knn5-incontext` model zero-shot.

## Checking the environment

`SELECT predict_version()` returns the extension version, compiled
runtimes, and bundled model ids as JSON. Use it to confirm the extension
is loaded and what is available before planning work.
