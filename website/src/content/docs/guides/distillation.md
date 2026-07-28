---
title: Distillation
description: Compress a slow teacher into a tiny native student, then serve it like any other model.
---

A strong teacher is usually too heavy to call per query: your production
model needs its Python stack, and a foundation model needs seconds of CPU or
a GPU. The answer is distillation: run the teacher once to label your data,
then fit a small **native student** that runs in the zero-dependency core
with no runtime. The teacher can be your own model (put its predictions in
the target column and you have compressed it into the database), a
permissively licensed foundation model, or your plain labels. Every distilled
student registers under the `student_id` you give it, and you serve it by
passing that id as the `model` option of the normal serving call. There is no
separate "serve a student" API.

## Tabular: distill with `distill_predict`, serve with `predict`

```sql
-- once: fit and register the student
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT f1, f2, label FROM training',
  '{"target":"label","student_id":"churn-v1","student_kind":"gbt"}');

-- forever: serve it. train_query is NULL because the student already
-- learned; only the rows to predict are needed.
SELECT row_ref, prediction, confidence FROM predict(NULL,
  'SELECT id, f1, f2 FROM customers',
  '{"model":"churn-v1"}');
```

The `NULL` first argument is the signature of serving a student: `predict`
normally takes a training query for in-context models, but a student carries
its training inside its blob, so passing a query alongside a student is
rejected (`PREDICT_ERR_OPTIONS`) rather than silently ignored.

`student_kind` is `tree` (a single CART), `gbt` (a gradient-boosted forest
with second-order leaves, which matches or beats tuned XGBoost on most
tasks), or `mlp` (a one-hidden-layer net for boundaries a tree renders
poorly). With `proba` and `classes` the student learns the teacher's full
probability distribution (soft-label distillation), not just its argmax.
Feature columns must be numeric: encode categorical text before distilling
(the in-context `knn5-incontext` handles text features itself; the distiller
does not).

## Forecasting: distill with `distill_forecast`, serve with `forecast`

```sql
-- once: an onnx teacher labels sliding windows in-database (needs the
-- opt-in onnx build; window mode below needs nothing extra)
SELECT model_id, train_rmse FROM distill_forecast(
  'SELECT series_key, value FROM history ORDER BY series_key, t',
  '{"teacher":"chronos-onnx","context":48,"horizon":12,"student_id":"fc"}');

-- forever: serve it through the ordinary aggregate
SELECT forecast(ts, value, 12, '{"model":"fc"}') FROM readings;

-- or let auto decide: a registered student competes against the
-- statistical baselines automatically, and the document's "model"
-- field tells you when it won
SELECT forecast(ts, value, 12, '{"model":"auto"}') FROM readings;
```

The student is a DLinear/TiDE-style net (a linear skip plus a small
residual). Without an onnx teacher, `distill_forecast` also trains directly
from windowed rows: each training row is, by position, `context` window
columns followed by `horizon` continuation columns (your own teacher's
forecasts, computed anywhere and stored as columns), and the column count
must match exactly. Pass `quantiles` to distill a teacher's quantile fan
instead of a point forecast: the continuation part then holds
`horizon * nquant` columns, one block of quantile levels per step.

## Distribute the student

A distilled student is a small native row in `_predict_models` (kilobytes,
`runtime='tree'`). It serves in the zero-dependency build with no ONNX
runtime, travels with a snapshot or fork of the database, and can be copied
to another database and used there. So you can **distill once on a capable
machine and predict everywhere SQLite runs**. Only distilling *from a live
foundation-model teacher* needs the optional ONNX build; a student distilled
from its own target columns needs nothing extra.

One caveat that is not mechanical: **the teacher's license governs what you
may distill**. A student is derived from its teacher, and some model
licenses restrict distillation or commercial use of outputs. TabFM's
non-commercial license does both; TabPFN-2's Prior Labs License permits
distillation, commercially too, with attribution when you distribute the
student (the later TabPFN-2.5/2.6/3 weights are non-commercial); Chronos
is Apache-2.0; your own labels and models carry no such limits. The `accept_license` option makes a
restrictively-licensed teacher an explicit opt-in.

:::caution[Not legal advice]
The license descriptions here are lay summaries as of mid-2026. Model
licenses change between versions and over time, and how they apply
depends on your use. Read the license that shipped with the exact
weights you use, and consult your own counsel where it matters.
Compliance is your responsibility; neither this extension nor its
documentation can determine it for you.
:::
