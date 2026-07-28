---
title: Options
description: Every option each function accepts, passed as a trailing JSON object.
---

Options are a single JSON object in the last argument. Unknown keys and
wrong-typed values are rejected with `PREDICT_ERR_OPTIONS`, so a typo fails loud
rather than being ignored.

## forecast

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | string | `auto` | A model id to pin, or `auto` (the default: best of the stats plus eligible registered students, per series). |
| `confidence_level` | number (0,1) | 0.9 | Nominal coverage of the interval. |
| `interval_method` | `residual` \| `conformal` | `residual` | Interval construction. |
| `folds` | integer | 20 | Rolling origins used by `auto` / conformal. |
| `gap` | integer | 0 | Leakage guard between train and target. |
| `candidates` | string[] | stat models + eligible registered students | Narrows the `auto` pool. |
| `context_limit` | integer | 4096 | Cap on points fed to the model (the most recent are kept; a capped series reports status `truncated`). |

Aggregate rules: the `options` argument and `horizon` must be the same value
on every row of a group, and a `SELECT` string passed as the first argument is
rejected with an error explaining that `forecast` is an aggregate over your
rows. There are no column-naming keys: the argument positions carry the
columns and `GROUP BY` carries the series split.

## detect_anomalies

`model` and `context_limit` as above, except that `model` defaults to
`theta-classic` (the residual detector) and `context_limit` defaults to 0:
the detector scores the whole series unless you cap it. Plus:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `anomaly_prob_threshold` | number (0,1) | 0.99 | Probability above which a point is flagged. |

The same aggregate rules apply.

## backtest

`confidence_level`, `interval_method`, `folds`, `gap`, and `context_limit` as
for `forecast`. `model` differs: it must be a statistical model id or `auto`,
and it defaults to `theta-classic`, not `auto`; a distilled student competes
inside `auto` but cannot be pinned here. Plus the query-shape keys (backtest
takes its data as a query, so columns must be resolvable):

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `time_col` | string | inferred | Name of the time column. |
| `value_col` | string | inferred | Name of the value column. |
| `group_cols` | string[] | none | Split into series keyed by these columns. |

## predict

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `target` | string | required for in-context models | Column in `train_query` to learn. A served student (`train_query` = `NULL`) needs no target. |
| `task` | `classify` \| `regress` | inferred | Prediction task. |
| `model` | string | `knn5-incontext` | Model or distilled student id. |
| `device` | `cpu` \| `gpu` | `cpu` | Inference device (onnx build). |
| `precision` | string | model default | Inference precision (onnx build). |
| `accept_license` | 0 \| 1 | 0 | Accept a license-tagged model. |

## distill_predict

`target` and `student_id` are required. Feature columns must be numeric:
encode categorical text before distilling (the in-context `knn5-incontext`
handles text features itself; the distiller does not).

| Key | Type | Meaning |
| --- | --- | --- |
| `target` | string | Target column. Required. |
| `task` | `classify` \| `regress` | Task. Inferred when omitted. |
| `student_id` | string | Id to register the student under. Required. |
| `student_kind` | `tree` \| `gbt` \| `mlp` | Student architecture. Default `tree` (soft-label distillation implies `gbt`). |
| `teacher` | string | A registered model that relabels the training rows first. |
| `proba`, `classes` | string[] | Soft-label distillation: per-class probability columns and their class labels. |

## distill_forecast

`context`, `horizon`, and `student_id` are required. Without a `teacher`,
each `train_query` row is one training window laid out by position:
`context` input columns, then the `horizon` continuation columns (or
`horizon * nquant` quantile columns when `quantiles` is passed); the column
count must match exactly.

| Key | Type | Meaning |
| --- | --- | --- |
| `student_id` | string | Id to register the student under. Required. |
| `teacher` | string | Teacher forecast model (onnx build): `train_query` then yields `(series_key, value)` rows and the teacher labels sliding windows. |
| `context` | integer | Input window length. Required; max 4096. |
| `horizon` | integer | Forecast length the student is trained for. Required. |
| `quantiles` | number[] | Quantile levels to distill from a teacher's quantile fan. Default: a point (median-only) student. |
| `hidden` | integer | Residual-net hidden width, 0 to 2048 (0 = pure linear student). Default 256. |
| `epochs`, `lr` | | Training epochs (default 1500) and learning rate (default 0.005). |
