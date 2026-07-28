---
title: Distillation
description: Compress a slow teacher into a tiny native student that runs anywhere.
---

Foundation models are accurate but too heavy to call per query on CPU. The
answer is distillation: run the teacher once to label your data, then fit a small
**native student** that runs in the zero-dependency core with no runtime.

## Tabular: `distill_predict`

```sql
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT f1, f2, label FROM training',
  '{"target":"label","student_kind":"gbt"}');
```

`student_kind` is `tree` (a single CART), `gbt` (a gradient-boosted forest with
second-order leaves, which matches or beats tuned XGBoost on most tasks), or
`mlp` (a one-hidden-layer net for boundaries a tree renders poorly). With `proba`
and `classes` the student learns the teacher's full probability distribution
(soft-label distillation), not just its argmax. The result registers a model you
call with [`predict()`](../../reference/functions/).

## Forecasting: `distill_forecast`

```sql
SELECT model_id, train_rmse FROM distill_forecast(
  'SELECT series_key, value FROM history ORDER BY series_key, t',
  '{"teacher":"chronos-onnx","context":48,"horizon":12,"student_id":"fc"}');
```

The student is a DLinear/TiDE-style net (a linear skip plus a small residual). It
serves through [`forecast()`](../../reference/functions/) like any other model, and
can compete in [`auto`](../auto-and-conformal/) selection.

## Distribute the student

A distilled student is a small native row in `_predict_models` (kilobytes,
`runtime='tree'`). It serves in the zero-dependency build with no ONNX runtime,
travels with a snapshot or fork of the database, and can be copied to another
database and used there. So you can **distill once on a capable machine and
predict everywhere SQLite runs**. Only distilling *from a live foundation-model
teacher* needs the optional ONNX build; a student distilled from its own target
columns needs nothing extra.
