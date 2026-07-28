---
title: Auto-selection & conformal intervals
description: Let sqlite-predict pick the model, and get calibrated prediction intervals.
---

## Auto-selection

You don't have to hand-pick a model. `'{"model":"auto"}'` runs a rolling-origin
backtest of each candidate on the series and forecasts with the lowest-error one.
The choice is deterministic: the same rows pick the same winner.

```sql
SELECT forecast(ts, value, 12, '{"model":"auto"}') FROM readings;
```

The default pool is the bundled statistical models (`theta-classic`,
`stub-seasonal-naive`, `tsb` for intermittent demand). Pass `candidates` to set
the pool explicitly, and a candidate may be a **distilled forecast student**, so
your own compressed foundation model competes with the cheap baselines per
series:

```sql
SELECT forecast(ts, value, 12,
  '{"model":"auto","candidates":["theta-classic","tsb","my-student"]}')
FROM readings;
```

The winning model's id comes back in the result document's `model` field.

## Conformal intervals

The default prediction band is Gaussian, sized from the model's in-sample error.
On smooth data that band is overconfident. `'{"interval_method":"conformal"}'`
replaces it with a distribution-free band calibrated on the model's
**out-of-sample** rolling residuals:

```sql
SELECT forecast(ts, value, 6, '{"interval_method":"conformal"}')
FROM readings;
```

On a smooth synthetic series the default band covered only 57% of points at a
nominal 90% level; the conformal band landed at 90%. You can check coverage on
your own data with [`backtest()`](../backtesting/).

Conformal applies to the statistical models. A foundation-model student already
emits its own quantile band, so asking for conformal on one is rejected rather
than silently ignored. A series too short to calibrate returns
`insufficient_history` instead of a bogus interval.
