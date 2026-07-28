---
title: Functions
description: Every SQL function sqlite-predict registers, with signatures.
---

## Aggregate functions

`forecast` and `detect_anomalies` are **aggregate functions**, like `sum()`:
your statement supplies the rows, so filtering, joins, bound parameters, and
`GROUP BY` series-splitting are ordinary SQL. Both are pure functions: nothing
is written, so they work on read-only databases and inside views.

### `forecast(ts, value, horizon [, options])`

One `(ts, value)` observation per input row; rows are sorted by `ts`
internally, so input order never matters. `horizon` is 1 to 1000, and it and
`options` must be constant within a group.

`ts` accepts ISO-8601 UTC text (`YYYY-MM-DD`, optionally followed by
`[T ]HH:MM[:SS][Z]`) or a non-negative integer epoch, read as seconds, or as
milliseconds when the value is at or above 100000000000. Every timestamp is
floored to the whole second before modeling. A statement may feed at most
2,097,152 input rows across its groups; more raises `PREDICT_ERR_RESOURCE`.

Returns one JSON document per group:

```json
{"model": "theta-classic",
 "status": "ok",
 "rows": [{"step": 1, "forecast_timestamp": "…", "forecast": 1.0,
           "lower_bound": 0.5, "upper_bound": 1.5}, …]}
```

```sql
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;
```

`model` is the resolved model id (useful with `'{"model":"auto"}'`); `status`
is `ok`, `truncated`, `insufficient_history` (fewer than 8 points), or
`non_numeric`. A degraded
series returns a status document with empty `rows`; it does not fail the
statement. An aggregate over zero rows returns `NULL` (like `sum()`).

Passing a BigQuery-style query string, `forecast('SELECT …', 24)`, raises an
error explaining that `forecast` is an aggregate over your rows.

### `detect_anomalies(ts, value [, options])`

Same document pattern and the same accepted timestamp forms; each element of
`rows` carries `ts`, `value`,
`forecast`, `lower_bound`, `upper_bound`, `is_anomaly`,
`anomaly_probability`. The interval fields are null during warmup and for
model `sub-pca`.

### `forecast_rows(doc)` / `anomaly_rows(doc)`

Table-valued expansion of a document back into typed rows (the row fields
above plus `status`), for SQL-side consumption (app-side `JSON.parse` is the
other, equally supported route):

```sql
SELECT r.* FROM forecast_rows(
  (SELECT forecast(ts, value, 24) FROM readings)) r;
```

`forecast_rows(NULL)` yields zero rows; a document with empty `rows` yields a
single status row.

## Table-valued functions

Evaluation and training stay query-shaped: they take a read-only `SELECT`
string, because they need to re-run it across folds or split it into train and
apply sets.

### `backtest(query, horizon [, options])`

Rolling-origin evaluation of a forecast model. `query` is a read-only `SELECT`
of `(time, value)`, optionally with grouping columns. Columns: `series_key`,
`fold`, `cutoff_timestamp`, `model`, `n`, `mae`, `rmse`, `mase`, `smape`,
`coverage`, `mean_interval_width`, `status`.

### `predict(train_query, apply_query [, options])`

Learns from `train_query` (features plus a `target` column) and predicts the
target for every row of `apply_query`. Columns: `row_ref`, `prediction`,
`confidence`, `status`.

The first column of `apply_query` is the row reference: it is echoed back as
`row_ref` so you can join predictions to your rows, and it is **never a
feature**. Every column after it is a feature and must match a `train_query`
feature column **by name**; an apply column with no matching train column is
an error. So `predict('SELECT f1, f2, label FROM t', 'SELECT id, f1, f2 FROM
u', '{"target":"label"}')` trains on `f1, f2` and predicts one row per `u`
row, keyed by `id`.

To serve a distilled student, pass its id as the
`model` option and `NULL` as `train_query` (the student already learned):
`predict(NULL, 'SELECT id, f1, f2 FROM t', '{"model":"churn-v1"}')`.

The default in-context model caps training at 100,000 rows and 64 feature
columns; more raises `PREDICT_ERR_CONTEXT_TOO_LARGE` / `PREDICT_ERR_SCHEMA`.

### `distill_predict(train_query [, options])`

Fits a native tabular student and registers it. Requires the `target` and
`student_id` options; feature columns must be numeric. Columns: `model_id`,
`content_hash`, `train_rows`, `holdout_metric`.

### `distill_forecast(train_query [, options])`

Fits a native forecast student and registers it. Requires the `context`,
`horizon`, and `student_id` options. Two input modes: with a `teacher`
(ONNX build), `train_query` yields `(series_key, value)` rows and the teacher
labels sliding windows; without one, each `train_query` row is a training
window laid out by position as `context` input columns followed by the
`horizon` continuation columns (or `horizon * nquant` quantile columns when
`quantiles` is passed), and the column count must match exactly. Columns:
`model_id`, `content_hash`, `train_rows`, `train_rmse`.

## Scalar utilities

| Function | Returns |
| --- | --- |
| `predict_version()` | the extension version string |
| `predict_ulid(ts)` | the smallest ULID for an ISO-8601 UTC timestamp: deterministic (the time component is set, the entropy bits are zero), useful as a range key |
| `predict_debug()` | build info: profile, compiled features, available runtimes |
| `predict_register(model_id, config)` | registers an external model in `_predict_models` and returns its content hash; `config` is a bare weights path or a JSON object (`runtime`, `kind`, `license`, `weights_uri`, `io_spec`, `target`). Registration is metadata only: executing an `onnx` model still needs the ONNX build |

See [Options](../options/) for the per-function option set and
[Models](../models/) for the `model` values.
