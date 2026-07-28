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
| `time_col` | string | inferred | Name of the time column. |
| `value_col` | string | inferred | Name of the value column. |
| `group_cols` | string[] | none | Split into series keyed by these columns. |
| `model` | string | `theta-classic` | Forecast model, or `auto`. |
| `confidence_level` | number (0,1) | 0.9 | Nominal coverage of the interval. |
| `interval_method` | `residual` \| `conformal` | `residual` | Interval construction. |
| `folds` | integer | 20 | Rolling origins used by `auto` / conformal. |
| `gap` | integer | 0 | Leakage guard between train and target. |
| `candidates` | string[] | bundled stat models | Candidate pool for `auto`. |
| `context_limit` | integer | model default | Cap on points fed to the model. |
| `receipt` | 0 \| 1 | 1 | Write a receipt. |

## detect_anomalies

`time_col`, `value_col`, `group_cols`, `context_limit`, `model`, `receipt` as
above, plus:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `anomaly_prob_threshold` | number (0,1) | 0.99 | Probability above which a point is flagged. |

## The aggregate forms

The [aggregate forms](../functions/#aggregate-forms) of `forecast` and
`detect_anomalies` take the same options **minus the query-shape keys**:
`time_col`, `value_col`, and `group_cols` are rejected with
`PREDICT_ERR_OPTIONS`, because the argument positions carry the columns and
`GROUP BY` carries the series split. Everything else applies unchanged
(`model`, `confidence_level` / `anomaly_prob_threshold`, `interval_method`,
`folds`, `gap`, `candidates`, `context_limit`, `receipt`).

Two aggregate-only rules: the `options` argument (and `horizon`, for
`forecast`) must be the same value on every row of a group, and a `SELECT`
string passed as the first argument is redirected with an error pointing at
the table-valued form.

## backtest

`time_col`, `value_col`, `group_cols`, `confidence_level`, `context_limit`,
`model`, `folds`, `gap`, `interval_method`, `receipt` as for `forecast`.

## predict

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `target` | string | required | Column in `train_query` to learn. |
| `task` | `classify` \| `regress` | inferred | Prediction task. |
| `model` | string | `knn5-incontext` | Model or distilled student id. |
| `device` | `cpu` \| `gpu` | `cpu` | Inference device (onnx build). |
| `precision` | string | model default | Inference precision (onnx build). |
| `accept_license` | 0 \| 1 | 0 | Accept a license-tagged model. |
| `receipt` | 0 \| 1 | 1 | Write a receipt. |

## distill_predict

| Key | Type | Meaning |
| --- | --- | --- |
| `target` | string | Target column. |
| `task` | `classify` \| `regress` | Task. |
| `student_id` | string | Id to register the student under. |
| `student_kind` | `tree` \| `gbt` \| `mlp` | Student architecture. |
| `teacher` | string | Teacher model (onnx build) for soft labels. |
| `proba`, `classes` | | Soft-label distillation of a probability distribution. |
| `receipt` | 0 \| 1 | Write a receipt. |

## distill_forecast

| Key | Type | Meaning |
| --- | --- | --- |
| `student_id` | string | Id to register the student under. |
| `teacher` | string | Teacher forecast model (onnx build). |
| `context` | integer | Input window length. |
| `horizon` | integer | Forecast length the student is trained for. |
| `hidden` | integer | Residual-net hidden width. |
| `epochs`, `lr` | | Training epochs and learning rate. |
| `receipt` | 0 \| 1 | Write a receipt. |
