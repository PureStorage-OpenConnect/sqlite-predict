---
title: Errors
description: The closed set of error codes sqlite-predict raises.
---

Errors surface as ordinary SQLite errors whose message begins with a stable
code, e.g. `PREDICT_ERR_OPTIONS: unknown option key 'horzon'`. The set is closed:
a call raises one of these or succeeds. Match on the prefix to handle a class
programmatically.

## Input and options

| Code | Raised when |
| --- | --- |
| `PREDICT_ERR_OPTIONS` | Unknown option key, wrong type, out-of-range value, or (aggregate form) options varying within a group / a query-shape key. |
| `PREDICT_ERR_QUERY_NOT_READONLY` | The inner query is not a read-only `SELECT`. |
| `PREDICT_ERR_SCHEMA` | The query's columns can't be resolved to time/value/target, an expansion function got a malformed document, or a query string was passed to an aggregate form. |
| `PREDICT_ERR_HORIZON` | Horizon is 0, negative, above the maximum, or (aggregate form) varying within a group. |
| `PREDICT_ERR_THRESHOLD` | An anomaly threshold is outside (0,1). |
| `PREDICT_ERR_TARGET` | The `target` column is missing or invalid. |
| `PREDICT_ERR_TASK` | The task can't be inferred or conflicts with the data. |
| `PREDICT_ERR_CONTEXT_TOO_LARGE` | The context exceeds the model's maximum. |

## Models and runtimes

| Code | Raised when |
| --- | --- |
| `PREDICT_ERR_MODEL_NOT_FOUND` | The named model or student isn't registered. |
| `PREDICT_ERR_MODEL_EXISTS` | Registering an id that already exists. |
| `PREDICT_ERR_STUDENT_EXISTS` | A distilled student id is already taken. |
| `PREDICT_ERR_MODEL_HASH` | A model's content hash doesn't match its row. |
| `PREDICT_ERR_LICENSE` | A license-tagged model used without `accept_license`. |
| `PREDICT_ERR_RUNTIME_UNAVAILABLE` | An `onnx` model on a core (non-onnx) build. |
| `PREDICT_ERR_IO_SPEC` | A model's declared input/output spec is malformed. |
| `PREDICT_ERR_INFERENCE` | The backing runtime failed during inference. |

## Receipts and replay

| Code | Raised when |
| --- | --- |
| `PREDICT_ERR_RECEIPT_NOT_FOUND` | `predict_replay` given an unknown receipt id. |
| `PREDICT_ERR_ANCHOR_UNAVAILABLE` | The anchored data has changed, or the wrong verifier was used (`predict_replay` on a commitment receipt, `predict_verify` on a query-anchored one). |
| `PREDICT_ERR_REPLAY_MISMATCH` | The replayed result hash differs from the original. |

## Resources

| Code | Raised when |
| --- | --- |
| `PREDICT_ERR_RESOURCE` | An internal resource (DDL, allocation) failed. |
