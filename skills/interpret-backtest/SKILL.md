---
name: interpret-backtest
description: >-
  Read sqlite-predict backtest() output to choose models, trust or
  distrust prediction intervals, and decide when auto-selection needs
  narrowing. Use before serving forecasts that feed decisions, when
  intervals matter, or when choosing between models on a specific series.
license: MIT OR Apache-2.0
metadata:
  version: "0.1.0"
---

# Interpreting backtest()

`backtest` answers "how would this model have done on my series" with
rolling-origin evaluation: it repeatedly hides the tail of the series,
forecasts it, and scores against what actually happened. It is an
aggregate over your rows, like `forecast`, and returns one JSON document;
expand it with `backtest_rows(...)`.

```sql
SELECT * FROM backtest_rows((SELECT backtest(ts, value, 24,
       '{"model":"theta-classic","folds":5}') FROM readings));
```

## The metrics that matter

- **MASE** is the headline: error relative to the seasonal-naive floor.
  Below 1.0 beats naive; above 1.0 means the model is losing to
  last-season carry-forward and you should not serve it on this series.
  Compare models by MASE on the same series and folds.
- **Coverage** is the fraction of actuals that landed inside the
  prediction interval. Compare it to the confidence level you asked for:
  a 0.90 band with 0.60 measured coverage is lying to you.
- Per-fold rows expose stability. A model that wins on average but
  swings wildly across folds is riskier than a slightly worse, steady
  one.

## Interval judgment

The default Gaussian band can be overconfident on smooth series: the
sqlite-predict benchmarks measured 0.57 coverage at a nominal 0.90 for
theta-class models on smooth gluonts series (`benchmarks/results/` in
the repo). When interval truth matters, request conformal intervals in
the same call:

```sql
SELECT * FROM backtest_rows((SELECT backtest(ts, value, 24,
       '{"model":"theta-classic","interval_method":"conformal"}')
       FROM readings));
```

Conformal intervals calibrate to measured out-of-sample residuals and
substantially improve empirical coverage (they reached the nominal level
in our benchmarks, but finite folds carry no guarantee). They need
enough history to calibrate and apply to the statistical models only; a
short series makes the call refuse rather than fabricate. Always verify
with `backtest` coverage on your own series rather than trusting either
method's reputation, ours included.

## Auto-selection judgment

Bare `forecast(ts, value, h)` lets `auto` pick per series by rolling-
origin MASE. Use `backtest` when you want to see what auto sees:

- If one model wins consistently across your series, pin it
  (`'{"model":"theta-classic"}'`) and save the selection cost.
- If the pool is polluted (a distilled student trained for a different
  regime keeps winning on stale patterns), narrow it:
  `'{"candidates":["theta-classic","tsb"]}'`.
- Intermittent series (many zeros, sporadic demand) are `tsb` territory;
  if auto is not picking it, check whether the series reaches it and
  consider pinning.

## Reading degraded outcomes

`backtest` needs enough history for its folds: expect loud errors or
degraded statuses on short series rather than fabricated confidence.
That is the tool working. A series too short to backtest is a series too
short to trust a model on; fall back to wider intervals and say so in
whatever the forecast feeds.
