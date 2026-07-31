---
title: Operations
description: The SQL functions sqlite-predict adds, and how they compose.
---

One rule covers the surface. The calls that work over your existing rows are
**aggregate functions**, like `sum()`: `forecast`, `detect_anomalies`,
`backtest`, and `fit`. Serving a trained model is a **scalar**, `predict`, one
prediction per row. Only the calls that run their own queries stay
**table-valued** over a read-only `SELECT`: `distill_predict`,
`distill_forecast`, and `predict_batch`. Options are a trailing JSON object,
e.g. `'{"confidence_level":0.9}'`.

| Function | Question | Returns |
| --- | --- | --- |
| `forecast(ts, value, horizon [, options])` | Where is this metric going? | one JSON document per group: future rows with prediction intervals and a status |
| `detect_anomalies(ts, value [, options])` | Which points are abnormal? | one JSON document per group: anomaly-scored rows with expected value and probability |
| `backtest(ts, value, horizon [, options])` | How accurate is the model here? | one JSON document per group: per-fold MAE / RMSE / MASE / sMAPE and coverage |
| `fit(f1, ..., fN, label [, options])` | Train a model on labeled rows | a registered model id, or a model blob |
| `predict(model, f1, ..., fN [, options])` | Classify or regress a row | a prediction, or a `{prediction, confidence}` document with `proba` |
| `distill_predict(train_query [, options])` | Compress a teacher into a fast student | a registered native tabular model |
| `distill_forecast(train_query [, options])` | Compress a forecast model into a student | a registered native forecast model |
| `predict_batch(train_query, apply_query [, options])` | Batched or in-context serving | a prediction and confidence per row |
| `forecast_rows(doc)` / `anomaly_rows(doc)` / `backtest_rows(doc)` | Expand a document to typed rows | one row per step / point / fold |

## One convention: aggregates compute over your rows, predict serves per row

The aggregates take your rows directly, so filtering, joins, bound parameters,
and `GROUP BY` splitting are ordinary SQL:

```sql
-- one series
SELECT forecast(ts, value, 24) FROM readings;

-- many series, one document each
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;

-- a model per segment falls out of GROUP BY
SELECT region, fit(tenure, spend, churned, '{"kind":"gbt"}') FROM history GROUP BY region;
```

Rows are sorted by `ts` internally where it applies, so input order never
matters. `forecast`, `detect_anomalies`, and `backtest` are **pure functions**:
nothing is written, so they work on read-only databases and inside views. Each
returns one JSON document; parse it in your app or expand it with
`forecast_rows()` / `anomaly_rows()` / `backtest_rows()`.

Serving a trained model is the `predict` scalar, so it drops into any `SELECT`
beside your other columns:

```sql
SELECT id, predict('churn-v1', tenure, spend) AS churn FROM active;
```

See the language [getting-started guides](../../getting-started/python/), which
show the pattern through SQLAlchemy, Drizzle, and Diesel.

The remaining calls (`distill_predict`, `distill_forecast`, `predict_batch`)
stay query-shaped: they take read-only `SELECT` strings, because they re-run
across folds or split into train and apply sets.

## Per-series status

Each series carries a `status`: `ok`, `truncated` (the context was capped by
`context_limit`), `insufficient_history`, or `non_numeric`. For `forecast` and
`detect_anomalies` it lives in the returned document (a degraded series comes
back as a status document with empty `rows`, not an error); `backtest` carries a
`status` per fold in its expanded rows. (`fit` is an aggregate too, but it
returns a model id or blob, not a status document.) Forecasting needs a minimum
of 8 points.

See the [Functions reference](../../reference/functions/) for every signature and the
[Options reference](../../reference/options/) for the full option set.
