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
| `model` | string | `theta-classic` | Forecast model, or `auto`. |
| `confidence_level` | number (0,1) | 0.9 | Nominal coverage of the interval. |
| `interval_method` | `residual` \| `conformal` | `residual` | Interval construction. |
| `folds` | integer | 20 | Rolling origins used by `auto` / conformal. |
| `gap` | integer | 0 | Leakage guard between train and target. |
| `candidates` | string[] | stat models + eligible registered students | Narrows the `auto` pool. |
| `context_limit` | integer | model default | Cap on points fed to the model. |

Aggregate rules: the `options` argument and `horizon` must be the same value
on every row of a group, and a `SELECT` string passed as the first argument is
rejected with an error explaining that `forecast` is an aggregate over your
rows. There are no column-naming keys: the argument positions carry the
columns and `GROUP BY` carries the series split.

## detect_anomalies

`model` and `context_limit` as above, plus:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `anomaly_prob_threshold` | number (0,1) | 0.99 | Probability above which a point is flagged. |

The same aggregate rules apply.

## backtest

`confidence_level`, `interval_method`, `folds`, `gap`, `context_limit`, and
`model` as for `forecast`, plus the query-shape keys (backtest takes its data
as a query, so columns must be resolvable):

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `time_col` | string | inferred | Name of the time column. |
| `value_col` | string | inferred | Name of the value column. |
| `group_cols` | string[] | none | Split into series keyed by these columns. |

## predict

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `target` | string | required | Column in `train_query` to learn. |
| `task` | `classify` \| `regress` | inferred | Prediction task. |
| `model` | string | `knn5-incontext` | Model or distilled student id. |
| `device` | `cpu` \| `gpu` | `cpu` | Inference device (onnx build). |
| `precision` | string | model default | Inference precision (onnx build). |
| `accept_license` | 0 \| 1 | 0 | Accept a license-tagged model. |

## distill_predict

| Key | Type | Meaning |
| --- | --- | --- |
| `target` | string | Target column. |
| `task` | `classify` \| `regress` | Task. |
| `student_id` | string | Id to register the student under. |
| `student_kind` | `tree` \| `gbt` \| `mlp` | Student architecture. |
| `teacher` | string | Teacher model (onnx build) for soft labels. |
| `proba`, `classes` | | Soft-label distillation of a probability distribution. |

## distill_forecast

| Key | Type | Meaning |
| --- | --- | --- |
| `student_id` | string | Id to register the student under. |
| `teacher` | string | Teacher forecast model (onnx build). |
| `context` | integer | Input window length. |
| `horizon` | integer | Forecast length the student is trained for. |
| `hidden` | integer | Residual-net hidden width. |
| `epochs`, `lr` | | Training epochs and learning rate. |
