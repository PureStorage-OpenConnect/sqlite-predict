---
title: Functions
description: Every SQL function sqlite-predict registers, with signatures.
---

## Table-valued functions

### `forecast(query, horizon [, options])`

Forecasts each series `horizon` steps into the future. `query` is a read-only
`SELECT` of `(time, value)`, optionally with grouping columns. Columns per row:
`series_key`, `step`, `forecast`, `lower_bound`, `upper_bound`, `model`,
`status`, `receipt_id`.

### `detect_anomalies(query [, options])`

Scores each input point for abnormality. Columns: `series_key`, the original
time and value, `expected`, `anomaly_score`, `anomaly_prob`, `is_anomaly`,
`model`, `status`, `receipt_id`.

### `predict(train_query, apply_query [, options])`

Learns from `train_query` (features plus a `target` column) and predicts the
target for every row of `apply_query`. Columns: `row_ref`, `prediction`,
`confidence`, `model`, `receipt_id`.

### `backtest(query, horizon [, options])`

Rolling-origin evaluation of a forecast model. Columns: `series_key`, `fold`,
`model`, `mae`, `rmse`, `mase`, `smape`, `coverage`, `mean_interval_width`,
`receipt_id`.

### `distill_predict(train_query [, options])`

Fits a native tabular student and registers it. Columns: `model_id`,
`student_kind`, `n_train`, `holdout_metric`, `receipt_id`.

### `distill_forecast(train_query [, options])`

Fits a native forecast student and registers it. Columns: `model_id`,
`n_series`, `train_rmse`, `receipt_id`.

### `predict_replay(receipt_id)`

Re-executes a recorded call against its anchored data. Columns: `match`,
`detail`.

## Scalar utilities

| Function | Returns |
| --- | --- |
| `predict_version()` | the extension version string |
| `predict_ulid([ts])` | a fresh ULID, optionally seeded with an epoch-ms timestamp |
| `predict_debug()` | build info: profile, compiled features, available runtimes |

See [Options](/reference/options/) for the per-function option set and
[Models](/reference/models/) for the `model` values.
