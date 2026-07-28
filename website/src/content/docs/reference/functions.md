---
title: Functions
description: Every SQL function sqlite-predict registers, with signatures.
---

## Table-valued functions

### `forecast(query, horizon [, options])`

Forecasts each series `horizon` steps into the future. `query` is a read-only
`SELECT` of `(time, value)`, optionally with grouping columns. Columns per row:
`series_key`, `step`, `forecast_timestamp`, `forecast`, `lower_bound`,
`upper_bound`, `status`, `receipt_id`.

### `detect_anomalies(query [, options])`

Scores each input point for abnormality. Columns: `series_key`, `ts`, `value`,
`forecast`, `lower_bound`, `upper_bound`, `is_anomaly`, `anomaly_probability`,
`status`, `receipt_id`.

### `predict(train_query, apply_query [, options])`

Learns from `train_query` (features plus a `target` column) and predicts the
target for every row of `apply_query`. Columns: `row_ref`, `prediction`,
`confidence`, `status`, `receipt_id`.

### `backtest(query, horizon [, options])`

Rolling-origin evaluation of a forecast model. Columns: `series_key`, `fold`,
`cutoff_timestamp`, `model`, `n`, `mae`, `rmse`, `mase`, `smape`, `coverage`,
`mean_interval_width`, `status`, `receipt_id`.

### `distill_predict(train_query [, options])`

Fits a native tabular student and registers it. Columns: `model_id`,
`content_hash`, `train_rows`, `holdout_metric`, `receipt_id`.

### `distill_forecast(train_query [, options])`

Fits a native forecast student and registers it. Columns: `model_id`,
`content_hash`, `train_rows`, `train_rmse`, `receipt_id`.

### `predict_replay(receipt_id)`

Re-executes a recorded query-form call against its anchored data. Columns:
`match`, `result_hash`, `original_hash`, `detail`. Aggregate-form (commitment)
receipts are rejected with a pointer to `predict_verify`.

### `predict_verify(receipt_id, query)`

Verifies an aggregate-form commitment receipt against caller-supplied rows:
`query` is a read-only `SELECT` of one `(ts, value)` series; its digest is
checked against the receipt's committed input digest, then the recorded call
re-runs on those rows and result hashes are compared. A digest mismatch is a
finding (`match = 0`), not an error. Columns as `predict_replay`.

## Aggregate forms

`forecast` and `detect_anomalies` are also **aggregate functions** under the
same names. SQLite resolves the form by position: FROM clause = table-valued
(above), expression position = aggregate. The aggregate form is the one to use
from ORMs and query builders: the statement supplies the rows, so filtering,
joins, parameter binding, and `GROUP BY` series-splitting are ordinary SQL
instead of text inside a string. See the [ORM guide](../../guides/orms/).

### `forecast(ts, value, horizon [, options])` — aggregate

One `(ts, value)` observation per input row; rows are sorted by `ts`
internally, so input order never matters. Returns one JSON document per group:

```json
{"model": "theta-classic", "receipt_id": "01J…", "status": "ok",
 "rows": [{"step": 1, "forecast_timestamp": "…", "forecast": 1.0,
           "lower_bound": 0.5, "upper_bound": 1.5}, …]}
```

```sql
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;
```

An aggregate over zero rows returns `NULL` (like `sum()`). A degraded series
(`insufficient_history`, `non_numeric`) returns a status document with empty
`rows` and `receipt_id` null; it does not fail the statement.

### `detect_anomalies(ts, value [, options])` — aggregate

Same shape; each element of `rows` carries `ts`, `value`, `forecast`,
`lower_bound`, `upper_bound`, `is_anomaly`, `anomaly_probability`.

Aggregate options are the query form's set **minus** the query-shape keys:
`time_col`, `value_col`, and `group_cols` are rejected, because the argument
positions and `GROUP BY` carry that information. `horizon` and `options` must
be constant within a group.

### `forecast_rows(doc)` / `anomaly_rows(doc)`

Table-valued expansion of an aggregate-form document back into typed rows,
named exactly like the query form's columns, for SQL-side consumption
(app-side `JSON.parse` is the other, equally supported route):

```sql
SELECT g.city, r.*
FROM (SELECT city, forecast(ts, value, 24) AS doc
        FROM readings GROUP BY city) AS g,
     forecast_rows(g.doc) AS r;
```

`forecast_rows(NULL)` yields zero rows; a document with empty `rows` yields a
single status row.

## Scalar utilities

| Function | Returns |
| --- | --- |
| `predict_version()` | the extension version string |
| `predict_ulid([ts])` | a fresh ULID, optionally seeded with an epoch-ms timestamp |
| `predict_debug()` | build info: profile, compiled features, available runtimes |

See [Options](../options/) for the per-function option set and
[Models](../models/) for the `model` values.
