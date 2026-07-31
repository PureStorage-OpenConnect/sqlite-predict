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
package `sqlite-predict`), then call functions over your own rows. Serving is
pure: `forecast`, `detect_anomalies`, `backtest`, `predict`, and `fit` without
`register` do not write, so they run on read-only databases and inside views.
Registering a model (`fit` with `'{"register":"m"}'`, and the distillers)
persists to `_predict_models` and needs a writable database.

## Which operation for which question

| Question | Call |
| --- | --- |
| "What will this metric do next?" | `SELECT forecast(ts, value, horizon) FROM t` |
| "Which points are anomalous?" | `SELECT detect_anomalies(ts, value) FROM t` |
| "Train a model on labeled rows" | `SELECT fit(f1, ..., fN, label, '{"register":"m"}') FROM t` |
| "Predict a label/value per row" | `SELECT predict('m', f1, ..., fN) FROM t` |
| "How accurate would this be on my data?" | `SELECT backtest(ts, value, horizon) FROM t` |
| "Make serving instant and self-contained" | `distill_predict` / `distill_forecast` (see the distill-lifecycle skill) |

`forecast`, `detect_anomalies`, `backtest`, and `fit` are aggregates:
plain SQL supplies the rows, `GROUP BY` splits series or segments, `WHERE`
and joins compose, ORMs work unchanged. `predict` is a scalar, one
prediction per row, so it drops into any `SELECT` beside your other
columns. The distillers and `predict_batch` are table-valued functions
that take a query string.

## Reading results

Each aggregate group returns one JSON document: `{"model", "status",
"rows"}`. Expand to typed rows in SQL when needed:

```sql
SELECT r.* FROM forecast_rows((SELECT forecast(ts, value, 24) FROM t)) AS r;
```

`anomaly_rows(...)` and `backtest_rows(...)` expand anomaly and backtest
documents the same way. `predict` returns its value directly, no
expansion.

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
- `fit(f1, ..., fN, label)` trains over your rows; the label is the last
  argument, there is no `target` option. `'{"register":"churn-v1"}'`
  registers the model and returns its id; otherwise it returns a model
  blob you can pass to `predict`. Default kind is `gbt`; `'{"kind":"tree"}'`
  for a single tree.
- `predict(model, f1, ..., fN)` serves a native student per row; the
  model is a registered id or a `fit()` blob, and features are positional,
  so a count mismatch fails loud. Pass `'{"proba":true}'` for a
  `{"prediction": "1", "confidence": 0.98}` document. For in-context `knn5-incontext`
  (a train query, no fit step) or ONNX serving, use `predict_batch`.

## Checking the environment

`SELECT predict_version()` returns the extension version, compiled
runtimes, and bundled model ids as JSON. Use it to confirm the extension
is loaded and what is available before planning work.
