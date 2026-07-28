---
title: Backtesting
description: Score a model's accuracy on your own data with rolling-origin evaluation.
---

`backtest(query, horizon [, options])` runs a rolling-origin evaluation: for each
of several past cutoffs it fits the model, forecasts, and compares to the held-out
actuals. It returns per-fold accuracy and interval quality, so a caller (say, an
agent auditing its own forecasts) can measure quality locally.

```sql
SELECT fold, model, mae, rmse, mase, smape, coverage, mean_interval_width
FROM backtest('SELECT ts, value FROM readings', 6, '{"folds":20}');
```

Aggregate in plain SQL:

```sql
SELECT avg(mase) FROM backtest('SELECT ts, value FROM readings', 6,
  '{"model":"auto","folds":20}');
```

## Options

- `folds` sets the number of rolling origins (default 20).
- `gap` inserts a leakage guard: `gap` points are skipped between the end of each
  training window and the first scored target.
- `interval_method` (`residual` or `conformal`) selects which band's `coverage`
  and `mean_interval_width` are reported. Conformal coverage is evaluated
  leave-fold-out, so it is a genuine out-of-sample estimate.

## Validate conformal coverage

Because `backtest()` reports coverage for the chosen interval method, you can
confirm the conformal band actually covers at the nominal level on your data:

```sql
SELECT avg(coverage) FROM backtest('SELECT ts, value FROM readings', 6,
  '{"interval_method":"conformal","confidence_level":0.9,"folds":25}');
```
