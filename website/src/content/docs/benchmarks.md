---
title: Benchmarks
description: Measured, reproducible accuracy numbers for the bundled models.
---

These are the numbers we hold ourselves to. Every one is reproducible from the
harness in [`benchmarks/`](https://github.com/PureStorage-OpenConnect/sqlite-predict/tree/main/benchmarks);
lower is better for error metrics unless noted. This is pre-alpha, so treat them
as a spike baseline, not a leaderboard claim.

## Forecasting (MASE, lower is better)

Standard synthetic suites (trend + seasonality + noise), scored against a
seasonal-naive floor of 1.0.

| Model | MASE | Notes |
| --- | --- | --- |
| seasonal-naive | 1.11 | the floor |
| theta-classic | 1.03 | bundled statistical model, zero dependencies |
| distilled Chronos student | 0.89 | native blob, no runtime at serve time |
| Chronos (teacher, ONNX) | ~0.80 | for reference; needs the ONNX build |

The point of the middle two rows: distilling the foundation model into a native
student recovers most of the teacher's accuracy (0.89 vs 0.80) while serving in
the zero-dependency core. See [Distillation](../guides/distillation/).

## Anomaly detection (VUS-PR, higher is better)

Univariate track of the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD)
benchmark.

| Detector | median VUS-PR |
| --- | --- |
| `sub-pca` (bundled) | ~0.44 |

`sub-pca` (subsequence PCA, Jacobi eigensolver in C) lands at or above the
published SOTA band on the univariate suite, from a dependency-free build.

## Tabular prediction

On [TabArena](https://tabarena.ai)-style tasks, a distilled `gbt` student
matches or beats a tuned XGBoost baseline, while running inside SQLite with no
external library. Full per-task tables live in the harness.

## Interval calibration

On a smooth synthetic series at a nominal 90% level:

| Interval method | empirical coverage |
| --- | --- |
| residual (Gaussian) | 0.57 |
| conformal | 0.90 |

The default Gaussian band is overconfident on smooth data; the conformal band
hits nominal coverage. You can reproduce this on your own data with
[`backtest()`](../guides/backtesting/). See
[Auto-selection & conformal intervals](../guides/auto-and-conformal/).

## Reproducing

```sh
git clone https://github.com/PureStorage-OpenConnect/sqlite-predict
cd sqlite-predict && make loadable
cd benchmarks && uv run python run.py
```

Results are written to `benchmarks/results/`. If your numbers differ, open an
issue with your platform and the generated table.
