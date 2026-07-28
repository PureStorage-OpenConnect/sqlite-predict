---
title: Auto-selection & conformal intervals
description: Let sqlite-predict pick the model, and get calibrated prediction intervals.
---

## Auto-selection

You don't have to pick a model at all: **auto is the default**. A call with no
`model` option runs a rolling-origin backtest of each candidate on the series
and forecasts with the lowest-error one. The choice is deterministic: the same
rows pick the same winner.

```sql
SELECT forecast(ts, value, 12) FROM readings;   -- auto, implicitly
```

Selection backtests every candidate per call, which costs single-digit
milliseconds on typical series and tens of milliseconds at the 4096-point
context cap. On a latency-critical path over long series, pin a `model` or
lower `context_limit`.

The default pool is the bundled statistical models (`theta-classic`,
`stub-seasonal-naive`, `tsb` for intermittent demand) **plus every eligible
registered forecast student**: distill a model once and `auto` competes it
against the baselines automatically, no call-site changes. Eligibility is
per call: a student trained for a shorter horizon than requested, or any
student when `interval_method` is `conformal`, sits that call out quietly.

Pass `candidates` to narrow the pool explicitly, for example when many
students are registered and you only want one backtested per call:

```sql
SELECT forecast(ts, value, 12,
  '{"model":"auto","candidates":["theta-classic","tsb","my-student"]}')
FROM readings;
```

The winning model's id comes back in the result document's `model` field, so
you can see when your student beat the baselines.

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
