---
title: Operations
description: The table-valued functions sqlite-predict adds, and how they compose.
---

Every operation is a table-valued function you call over a read-only `SELECT`.
Results are ordinary rows you can join, filter, and materialize. Options are a
trailing JSON object, e.g. `'{"group_cols":["region"],"confidence_level":0.9}'`.

| Function | Question | Returns |
| --- | --- | --- |
| `forecast(query, horizon [, options])` | Where is this metric going? | future rows with prediction intervals and per-series status |
| `detect_anomalies(query [, options])` | Which points are abnormal? | anomaly-scored rows with expected value and probability |
| `predict(train_query, apply_query [, options])` | Classify or regress unseen rows | a prediction and confidence per row |
| `backtest(query, horizon [, options])` | How accurate is the model here? | per-fold MAE / RMSE / MASE / sMAPE and interval coverage |
| `distill_predict(train_query [, options])` | Compress a teacher into a fast student | a registered native tabular model |
| `distill_forecast(train_query [, options])` | Compress a forecast model into a student | a registered native forecast model |
| `predict_replay(receipt_id)` | Did this prediction reproduce? | a match flag from re-running the recorded call |

## Column inference

`forecast()` and `detect_anomalies()` infer the time column (an integer epoch or
an ISO-8601 string) and the value column from the query's output, or you can name
them explicitly with `time_col` / `value_col`. Pass `group_cols` to split one
query into many series keyed by `series_key`:

```sql
SELECT * FROM forecast(
  'SELECT created_at, confidence, hypothesis_id FROM hypotheses',
  12, '{"group_cols":["hypothesis_id"]}');
```

## Per-series status

Each series carries a `status`: `ok`, `truncated` (the context was capped by
`context_limit`), `insufficient_history`, or `non_numeric`. Forecasting needs a
minimum of 8 points; series below that come back as a single status row rather
than a guess.

See the [Functions reference](/reference/functions/) for every signature and the
[Options reference](/reference/options/) for the full option set.
