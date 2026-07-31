---
title: Models
description: The bundled models and the runtimes that back them.
---

Pick a model with the `model` option; leave it off for the default. Every
bundled model ships in the zero-dependency core, no ONNX runtime required.

## Forecasting

| `model` | Kind | Notes |
| --- | --- | --- |
| `auto` | meta | The default when no `model` is named. Rolling-origin backtests each candidate per series and forecasts with the lowest-error one. See [Auto-selection](../../guides/auto-and-conformal/). |
| `theta-classic` | statistical | The Theta method (`default-ts` aliases to it). Strong, cheap trend + seasonality baseline. |
| `stub-seasonal-naive` | statistical | Seasonal-naive with drift. The benchmark floor every forecasting paper reports against. |
| `tsb` | statistical | Teunter-Syntetos-Babai, for intermittent demand (many zeros). Forecast-only. |

## Anomaly detection

| `model` | Kind | Notes |
| --- | --- | --- |
| `theta-classic` | statistical | The default detector: flags points that fall outside the one-step-ahead forecast interval. |
| `stub-seasonal-naive` | statistical | The same residual detector over the seasonal-naive forecaster. |
| `sub-pca` | statistical | Subsequence PCA (Jacobi eigensolver in C); competitive with SOTA on the TSB-AD-U benchmark. Pick it with `'{"model":"sub-pca"}'`; its rows carry no interval fields. |

## Tabular prediction

| `model` | Kind | Notes |
| --- | --- | --- |
| native students | native | Trained by `fit` (`gbt` or `tree`) or `distill_predict` (`tree`, `gbt`, or `mlp`). `fit` returns a model blob by default, or registers under your id when you pass `'{"register":"..."}'`; either way the `predict` scalar serves it per row. |
| `knn5-incontext` | statistical | In-context k-nearest-neighbors (k=5), a zero-setup baseline with no training step. Serve it through `predict_batch(train_sql, apply_sql)`. |

## Distilled forecast students

`distill_forecast` registers a DLinear/TiDE-style native student under the
`student_id` you choose. Serve it through `forecast()` with `'{"model":"<id>"}'`,
or enter it as a candidate in `auto`. These are portable blobs: copy the model
row to another database and it works there. See [Distillation](../../guides/distillation/).

## Foundation-model teachers (optional ONNX build)

The default build is pure C. `make loadable-onnx` adds a `runtime='onnx'` path so
`distill_forecast`/`distill_predict` can distill *from* a live foundation-model
teacher (Chronos for forecasting, a tabular FM for the tabular path). Serving the
distilled student never needs that build. License-tagged teachers enforce
`PREDICT_ERR_LICENSE` unless you pass `accept_license`.
