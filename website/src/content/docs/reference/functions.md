---
title: Functions
description: Every SQL function sqlite-predict registers, with signatures.
---

## Aggregate functions

`forecast`, `detect_anomalies`, `backtest`, and `fit` are **aggregate
functions**, like `sum()`: your statement supplies the rows, so filtering,
joins, bound parameters, and `GROUP BY` splitting are ordinary SQL. The first
three are pure (nothing is written, so they work on read-only databases and
inside views); `fit` optionally registers a model.

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
`non_numeric`. A degraded series returns a status document with empty `rows`;
it does not fail the statement. An aggregate over zero rows returns `NULL`
(like `sum()`).

Passing a BigQuery-style query string, `forecast('SELECT …', 24)`, raises an
error explaining that `forecast` is an aggregate over your rows.

### `detect_anomalies(ts, value [, options])`

Same document pattern and the same accepted timestamp forms; each element of
`rows` carries `ts`, `value`, `forecast`, `lower_bound`, `upper_bound`,
`is_anomaly`, `anomaly_probability`. The interval fields are null during warmup
and for model `sub-pca`.

### `backtest(ts, value, horizon [, options])`

Rolling-origin evaluation of a forecast model over your rows: it repeatedly
hides the tail, forecasts it, and scores against the held-out actuals. Returns
one JSON document per group of per-fold results; expand with `backtest_rows()`.
The row fields are `series_key`, `fold`, `cutoff_timestamp`, `model`, `n`,
`mae`, `rmse`, `mase`, `smape`, `coverage`, `mean_interval_width`, `status`.

```sql
SELECT avg(mae) FROM backtest_rows(
  (SELECT backtest(ts, value, 6, '{"folds":20}') FROM readings));
```

### `fit(f1, ..., fN, label [, options])`

Trains a native tabular student over your rows. The features are the leading
positional arguments and the **label is the last argument** (there is no
`target` option). The optional trailing `options` is a TEXT JSON object, so a
trailing `{...}` argument is always read as options: a class **label must not be
a JSON-object string** (it would be consumed as options). Returns the model:
with `'{"register":"churn-v1"}'` it registers into `_predict_models` and returns
the id, otherwise it returns a model blob you can pass to `predict`. Options:
`kind` (`gbt` default, or `tree`), `task` (`classify`/`regress`, inferred from
the label), `register`.

```sql
SELECT fit(tenure, spend, churned, '{"kind":"gbt","register":"churn-v1"}') FROM history;
```

### `forecast_rows(doc)` / `anomaly_rows(doc)` / `backtest_rows(doc)`

Table-valued expansion of a forecast, anomaly, or backtest document back into
typed rows (the row fields above plus `status`), for SQL-side consumption
(app-side `JSON.parse` is the other, equally supported route):

```sql
SELECT r.* FROM forecast_rows(
  (SELECT forecast(ts, value, 24) FROM readings)) r;
```

`forecast_rows(NULL)` yields zero rows; a document with empty `rows` yields a
single status row.

## Scalar prediction

### `predict(model, f1, ..., fN [, options])`

Serves a native student one prediction per row, so it drops into any `SELECT`
beside your other columns and composes with `WHERE` and joins. `model` is a
registered id (from `fit` or `distill_predict`) or a `fit()` blob. Features are
positional and must match the model's feature count; a mismatch raises
`PREDICT_ERR_SCHEMA`. Returns the prediction: a class label for a `classify`
model, a numeric value for a `regress` model. For a classifier,
`'{"proba":true}'` returns a `{"prediction": "1", "confidence": 0.98}` JSON
document.

```sql
SELECT id, predict('churn-v1', tenure, spend) AS churn FROM active;

-- one-shot train + apply, no registration and no query strings:
WITH m(model) AS (SELECT fit(tenure, spend, churned) FROM history)
SELECT a.id, predict((SELECT model FROM m), a.tenure, a.spend) FROM active a;
```

For in-context models (`knn5-incontext`) or batched ONNX serving, use
`predict_batch`.

## Table-valued functions

Training and batched serving stay query-shaped: they take read-only `SELECT`
strings, because they re-run across folds or split into train and apply sets.

### `predict_batch(train_query, apply_query [, options])`

The batched serving and in-context path (the former `predict` TVF). With a
`train_query` it runs the in-context `knn5-incontext` model zero-shot; with
`NULL` and a `'{"model":…}'` option it serves a registered ONNX model over the
`apply_query` rows in one batch. The first `apply_query` column is the
`row_ref`, echoed back and never a feature; the rest are features. Columns:
`row_ref`, `prediction`, `confidence`, `status`.

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
| `predict_sha256(x)` | lowercase hex SHA-256 of a TEXT or BLOB value. NULL passes through; numeric arguments are rejected with `PREDICT_ERR_SCHEMA` rather than hashed through SQLite's lossy number-to-text conversion. Used by provenance workflows like the prediction-receipts skill to hash result documents in pure SQL |
| `predict_debug()` | build info: profile, compiled features, available runtimes |
| `predict_register(model_id, config)` | registers an external model in `_predict_models` and returns its content hash; `config` is a bare weights path or a JSON object (`runtime`, `kind`, `license`, `weights_uri`, `io_spec`, `target`). Registration is metadata only: executing an `onnx` model still needs the ONNX build |

See [Options](../options/) for the per-function option set and
[Models](../models/) for the `model` values.
