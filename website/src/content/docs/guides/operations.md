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
| `predict_verify(receipt, query)` | Are these the rows behind this prediction? | a match flag from checking supplied rows against a receipt document |

## Two forms: table-valued and aggregate

`forecast` and `detect_anomalies` each have two forms under one name, resolved
by where the call appears:

```sql
-- table-valued (FROM position): quick sessions, query-anchored receipts
SELECT * FROM forecast('SELECT ts, value FROM readings', 24);

-- aggregate (expression position): the statement supplies the rows
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;
```

**Use the table-valued form** for CLI one-liners and when you want the receipt
anchored to a re-runnable query against database state. **Use the aggregate
form** from application code and ORMs: rows flow in through ordinary SQL (with
`WHERE`, joins, and bound parameters), `GROUP BY` replaces `group_cols`, input
order never matters (it sorts by `ts` internally), and it is a **pure
function**: nothing is written, so it works on read-only databases and inside
views, and its [receipt comes back inside the result
document](../receipts/#document-receipts-the-aggregate-form) for you to store
wherever provenance lives. The aggregate returns one JSON document per group;
parse it in your app or expand it with `forecast_rows()` / `anomaly_rows()`.
See [Using with ORMs](../orms/).

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

See the [Functions reference](../../reference/functions/) for every signature and the
[Options reference](../../reference/options/) for the full option set.
