---
title: Operations
description: The SQL functions sqlite-predict adds, and how they compose.
---

Serving is an **aggregate**: `forecast` and `detect_anomalies` are aggregate
functions over your own rows, like `sum()`. Evaluation and training are
**table-valued functions** over a read-only `SELECT`: `backtest`, `predict`,
and the distillers. Options are a trailing JSON object, e.g.
`'{"confidence_level":0.9}'`.

| Function | Question | Returns |
| --- | --- | --- |
| `forecast(ts, value, horizon [, options])` | Where is this metric going? | one JSON document per group: future rows with prediction intervals and a status |
| `detect_anomalies(ts, value [, options])` | Which points are abnormal? | one JSON document per group: anomaly-scored rows with expected value and probability |
| `predict(train_query, apply_query [, options])` | Classify or regress unseen rows | a prediction and confidence per row |
| `backtest(query, horizon [, options])` | How accurate is the model here? | per-fold MAE / RMSE / MASE / sMAPE and interval coverage |
| `distill_predict(train_query [, options])` | Compress a teacher into a fast student | a registered native tabular model |
| `distill_forecast(train_query [, options])` | Compress a forecast model into a student | a registered native forecast model |
| `forecast_rows(doc)` / `anomaly_rows(doc)` | Expand a document to typed rows | one row per forecast step / scored point |

## One convention: aggregates serve, query TVFs evaluate

The serving calls take your rows directly, so filtering, joins, bound
parameters, and `GROUP BY` series-splitting are ordinary SQL:

```sql
-- one series
SELECT forecast(ts, value, 24) FROM readings;

-- many series, one document each
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;
```

Rows are sorted by `ts` internally, so input order never matters. The
aggregate is a **pure function**: nothing is written, so it works on read-only
databases and inside views. Each group returns one JSON document; parse it in
your app or expand it with `forecast_rows()` / `anomaly_rows()`. See
[Using with ORMs](../orms/).

The evaluation and training calls (`backtest`, `predict`, `distill_*`) stay
query-shaped: they take a read-only `SELECT` string, because they need to
re-run it across folds or split it into train and apply sets.

## Column inference

`backtest()` infers the time column (an integer epoch or an ISO-8601 string)
and the value column from its query's output, or you can name them explicitly
with `time_col` / `value_col`. Pass `group_cols` to split one query into many
series keyed by `series_key`:

```sql
SELECT * FROM backtest(
  'SELECT created_at, confidence, hypothesis_id FROM hypotheses',
  12, '{"group_cols":["hypothesis_id"]}');
```

The aggregates need none of this: argument positions carry the columns and
`GROUP BY` carries the series split.

## Per-series status

Each series carries a `status`: `ok`, `truncated` (the context was capped by
`context_limit`), `insufficient_history`, or `non_numeric`. For the aggregates
it lives in the returned document (a degraded series comes back as a status
document with empty `rows`, not an error); for `backtest` it is the `status`
column. Forecasting needs a minimum of 8 points.

See the [Functions reference](../../reference/functions/) for every signature and the
[Options reference](../../reference/options/) for the full option set.
