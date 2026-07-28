---
title: Models
description: The bundled models and the runtimes that back them.
---

Pick a model with the `model` option; leave it off for the default. Every
bundled model ships in the zero-dependency core, no ONNX runtime required.

## Forecasting

| `model` | Kind | Notes |
| --- | --- | --- |
| `theta-classic` | statistical | The Theta method. Default (`default-ts` aliases to it). Strong, cheap trend + seasonality baseline. |
| `stub-seasonal-naive` | statistical | Seasonal-naive with drift. The benchmark floor every forecasting paper reports against. |
| `tsb` | statistical | Teunter-Syntetos-Babai, for intermittent demand (many zeros). Forecast-only. |
| `auto` | meta | Rolling-origin backtests each candidate per series and forecasts with the lowest-error one. See [Auto-selection](../../guides/auto-and-conformal/). |

## Anomaly detection

| `model` | Kind | Notes |
| --- | --- | --- |
| `sub-pca` | statistical | Subsequence PCA (Jacobi eigensolver in C). Default detector; competitive with SOTA on the TSB-AD-U benchmark. |

## Tabular prediction

| `model` | Kind | Notes |
| --- | --- | --- |
| `knn5-incontext` | statistical | In-context k-nearest-neighbors (k=5). Default zero-setup baseline for `predict()`. |
| distilled students | native | Registered by `distill_predict` as `tree`, `gbt`, or `mlp`. Call by the `model_id` you gave them. |

## Distilled forecast students

`distill_forecast` registers a DLinear/TiDE-style native student under the
`student_id` you choose. Serve it through `forecast()` with `'{"model":"<id>"}'`,
or enter it as a candidate in `auto`. These are portable blobs: copy the model
row to another database and it works there. See [Distillation](../../guides/distillation/).

## Foundation-model teachers (optional ONNX build)

The default build is pure C. `make loadable-onnx` adds a `runtime='onnx'` path so
`distill_forecast`/`distill_predict` can distill *from* a live foundation-model
teacher (Chronos for forecasting, a tabular FM for `predict()`). Serving the
distilled student never needs that build. License-tagged teachers enforce
`PREDICT_ERR_LICENSE` unless you pass `accept_license`.
