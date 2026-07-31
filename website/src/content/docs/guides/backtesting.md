---
title: Backtesting
description: Score a model's accuracy on your own data with rolling-origin evaluation.
---

`backtest(ts, value, horizon [, options])` runs a rolling-origin evaluation: for
each of several past cutoffs it fits the model, forecasts, and compares to the
held-out actuals. It is an aggregate over your rows, like `forecast`, so `WHERE`,
joins, and `GROUP BY` compose. Each group returns one JSON document of per-fold
results; expand it to typed rows with `backtest_rows()`:

```sql
SELECT r.fold, r.model, r.mae, r.rmse, r.mase, r.smape, r.coverage, r.mean_interval_width
FROM backtest_rows((SELECT backtest(ts, value, 6, '{"folds":20}') FROM readings)) AS r;
```

Aggregate over the folds in plain SQL:

```sql
SELECT avg(mase) FROM backtest_rows((SELECT backtest(ts, value, 6,
  '{"model":"auto","folds":20}') FROM readings));
```

## Options

- `folds` sets the number of rolling origins (default 20).
- `gap` inserts a leakage guard: `gap` points are skipped between the end of each
  training window and the first scored target.
- `interval_method` (`residual` or `conformal`) selects which band's `coverage`
  and `mean_interval_width` are reported. Conformal coverage is evaluated
  leave-fold-out, so it is a genuine out-of-sample estimate.

## Validate conformal coverage

Because `backtest` reports coverage for the chosen interval method, you can
confirm the conformal band actually covers at the nominal level on your data:

```sql
SELECT avg(coverage) FROM backtest_rows((SELECT backtest(ts, value, 6,
  '{"interval_method":"conformal","confidence_level":0.9,"folds":25}') FROM readings));
```
