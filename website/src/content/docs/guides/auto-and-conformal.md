---
title: Auto-selection & conformal intervals
description: Let sqlite-predict pick the model, and get calibrated prediction intervals.
---

## Auto-selection

You don't have to hand-pick a model. `'{"model":"auto"}'` runs a rolling-origin
backtest of each candidate on the series and forecasts with the lowest-error one.
The choice is deterministic, so it replays to the same result.

```sql
SELECT step, forecast FROM forecast(
  'SELECT ts, value FROM readings', 12, '{"model":"auto"}');
```

The default pool is the bundled statistical models (`theta-classic`,
`stub-seasonal-naive`, `tsb` for intermittent demand). Pass `candidates` to set
the pool explicitly, and a candidate may be a **distilled forecast student**, so
your own compressed foundation model competes with the cheap baselines per
series:

```sql
SELECT step, forecast FROM forecast('SELECT ts, value FROM readings', 12,
  '{"model":"auto","candidates":["theta-classic","tsb","my-student"]}');
```

The candidate set is recorded in the receipt, so an auto forecast replays to the
same winner.

## Conformal intervals

The default prediction band is Gaussian, sized from the model's in-sample error.
On smooth data that band is overconfident. `'{"interval_method":"conformal"}'`
replaces it with a distribution-free band calibrated on the model's
**out-of-sample** rolling residuals:

```sql
SELECT step, forecast, lower_bound, upper_bound FROM forecast(
  'SELECT ts, value FROM readings', 6, '{"interval_method":"conformal"}');
```

On a smooth synthetic series the default band covered only 57% of points at a
nominal 90% level; the conformal band landed at 90%. You can check coverage on
your own data with [`backtest()`](../backtesting/).

Conformal applies to the statistical models. A foundation-model student already
emits its own quantile band, so asking for conformal on one is rejected rather
than silently ignored. A series too short to calibrate returns
`insufficient_history` instead of a bogus interval.
