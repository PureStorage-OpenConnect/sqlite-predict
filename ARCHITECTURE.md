# Architecture

A short map of how `sqlite-predict` is put together, for contributors.

## Shape

The extension is pure C99, built as a loadable SQLite/libSQL extension
(`predict0.{dylib,so,dll}`) with a single entry point,
`sqlite3_predict_init`. It has no third-party runtime dependencies; the
SQLite amalgamation headers are fetched at build time and never
redistributed.

Source layout:

| File | Responsibility |
| --- | --- |
| `sqlite-predict.c` | entry point, function registration, shared helpers (timestamp parse/format, ULID, options parsing, read-only query prepare, JSON string-array parse, normal-quantile) |
| `predict-forecast.c` | the `forecast()` and `detect_anomalies()` aggregates, `backtest()`, the expansion TVFs, the statistical models, the native forecast-student serving path, and the `collect_series()` helper `backtest()` uses |
| `predict-tabular.c` | `predict()` vtab, the in-context k-NN model, and dispatch to a runtime backend for registered models |
| `predict-registry.c` | the model registry (`_predict_models`) and content hashing |
| `predict-onnx.c` | ONNX runtime backend (opt-in build only); the only file that links onnxruntime |
| `predict-student.c` | the native student **serving** runtime: blob (de)serialization and tree / forest / MLP inference (`predict0_tree_run`). The serve side of the train/serve boundary |
| `predict-student.h` | the shared student-model **format**: the tree/forest/MLP structs and the runtime entry points both sides agree on |
| `predict-distill.c` | the **training** side: the CART / gradient-boosting / MLP trainers, the `distill_predict()` vtab, and the `distill_forecast()` vtab that fits the forecast student. Builds student blobs that `predict-student.c` serves |
| `predict-internal.h` | shared types, contract constants, error codes, internal prototypes |
| `vendor/sha256.c` | a self-contained FIPS 180-4 SHA-256 |

## Two calling conventions: aggregates and TVFs

The series operations are plain SQL aggregate functions.
`forecast(ts, value, horizon[, options])` and
`detect_anomalies(ts, value[, options])` aggregate over the caller's own
rows: each input row contributes one (ts, value) observation, each
aggregate evaluation (one `GROUP BY` group, or the whole statement) is
one series, and the result is a single JSON document. The
`forecast_rows(doc)` and `anomaly_rows(doc)` table-valued functions
expand that document back into typed rows.

The query-shaped operations (`backtest()`, `predict()`,
`distill_predict()`, `distill_forecast()`) are [eponymous virtual table
modules][tvf] instead. Arguments arrive as hidden columns (a `query`
string plus operation-specific columns such as `horizon` and `options`),
resolved in `xBestIndex` and consumed in `xFilter`, which does all the
work and materializes result rows the cursor then walks. This is the
same mechanism SQLite's own `generate_series` uses.

`collect_series()` (prepare and validate an inner query as read-only and
single-statement, resolve or infer the time/value/group columns, collect
rows into per-series buffers) now serves only `backtest()`, the one
series operation that still takes a query string.

## Models

Models are looked up in a registry table (`_predict_models`) by id, with a
content hash, a license tag, and a `runtime`. The bundled models are pure-C
statistical methods (`runtime='bundled'`). `predict()` dispatches by
runtime: the default `knn5-incontext` path is unchanged and never touches
the registry (so it still works read-only), while a named `onnx` model
routes through the backend in `predict-onnx.c`.

That backend (compiled only into `make loadable-onnx`, `-DSQLITE_PREDICT_ONNX`)
serves three `io_spec` layouts. The **vector** layout is a self-contained model
mapping a feature vector to a prediction. The **in_context** layout is a
teacher (TabFM-shaped): it ingests the `train_query` rows as three tensors
(`x_train`, `y_train`, `x_query`) each call and labels the query rows against
that context. The **sequence** layout is a forecast foundation model served by
`forecast()` (not `predict()`): `predict0_onnx_forecast` feeds a context window
as a `[1, ctx]` tensor (truncated to a multiple of the model's patch size) and
reads a `[1, Q, H]` quantile fan back, from which the point and the interval at
the requested confidence are interpolated over the io_spec's quantile levels.
Chronos-Bolt exports cleanly to this shape (`scripts/export_chronos_onnx.py`),
and the served MASE matches the Python model.

The `io_spec` is usually derived, not written. `predict_register` reads the
model's input/output tensors (`predict0_onnx_introspect`) to fill in the
layout, tensor names, output kind, and class count, so a bare weights path is
a complete registration for the vector case (in-context adds only `target`,
which is a SQL column introspection can't see). An explicit `io_spec`
overrides the derivation. Feature columns map positionally by default (apply
column order); a `features` list switches to name-based mapping. All the JSON
handling (reading the `io_spec`, building the derived one) goes through
SQLite's JSON1 (`json_extract`/`json_each`/`json_object`), never a hand-rolled
parser. Both cache one onnxruntime session per (weights, device,
precision), run query rows in batches, and select the execution provider
explicitly, erasing no failure into a silent CPU fallback. Weights are pinned
by content hash, verified before deserialization, so exactly the registered
bytes run. Both run on CPU. A GPU build (`make loadable-onnx-gpu`,
`-DSQLITE_PREDICT_ONNX_GPU`) wires the CUDA and TensorRT providers and
fp16/int8 precision; the provider-options symbols are in every onnxruntime
C API, so it compiles and links against the CPU onnxruntime for a CI
compile-check, while real GPU execution is validated on a dedicated GPU job.
GPU results stay distinguishable from the deterministic CPU path
(provider and precision are explicit options, never silent fallbacks).
Even so, the `benchmarks/` numbers (and the TabFM→ONNX eval in
`benchmarks/results/tabfm-onnx.md`) are why the default answer for a
teacher's accuracy is usually distillation to a small student.

`distill_predict()` (`predict-distill.c`) is that path, and it lives in the
zero-dependency core. It fits a native student on a training signal,
evaluates it on a held-out fraction, and writes the student into
`_predict_models` as an inline BLOB (`runtime='tree'`, `kind='student'`). By
default the signal is the `target` column of the training query: your
labels, or a strong teacher's predictions computed offline and stored in that
column, which is how a 30-second TabFM run becomes a microsecond student. A
`teacher` argument names a registered `predict()` model, which `distill`
re-runs over the rows (aligned by row number) to relabel them first, the way
to compress the in-context knn5 into a standalone tree.

Three `student_kind`s exist. `'tree'` and `'gbt'` share the CART trainer;
`'tree'` is a single depth-8 tree. `'gbt'` is a gradient-boosted forest of
shallow trees, and it is the go-to when accuracy matters: it fits each tree to
the loss
gradient but sets each leaf to the **second-order (Newton) step**
`Σg / (Σh + λ)` using the softmax Hessian (the same thing that lifts
XGBoost above a vanilla gradient booster), with shrinkage (a small learning
rate over many rounds) doing the regularizing. It is deterministic by
construction: no bootstrap, no feature-sampling, no early-stopping split, so
the student stays reproducible and exactly replayable. Given `proba` and
`classes`, the same forest distills the teacher's per-class probability
distribution (soft-label distillation): the softmax cross-entropy target
becomes the teacher's probability rather than a hard one-hot, so the student
inherits a foundation model's calibration instead of only its argmax, while
the holdout is still scored against the true `target` labels. `'mlp'` is the
third kind: a one-hidden-layer softmax net (classification only) trained with
deterministic full-batch Adam, for warped boundaries an axis-aligned tree
ensemble cannot render however good the targets are. It also consumes soft
targets, so a smooth student learns a smooth teacher's distribution directly.
`predict()` dispatches
a `tree`-runtime model to the native runtime in `predict-student.c`, which tells a
single tree (`PSTREE` blob) from a forest (`PSGBT` blob) from a net (`PSMLP`
blob) by magic and needs
no onnxruntime. The blob formats are little-endian and normatively specified
and versioned (`PSTREE01` / `PSGBT01` / `PSMLP01`), so a stored student stays
servable across upgrades, snapshots, and forks, and a serving-only module can
execute a blob it never trained. They are rigorously bounds-checked on read,
because the registry is
writable by any SQL caller: a hand-crafted blob is rejected,
never crashed on. The student is native and deterministic, like the stat
models.

Forecasting has its own student. `distill_forecast()` (`predict-distill.c`)
fits a regression MLP that maps an instance-normalized context window of `nfeat`
recent values to `nquant` quantiles per horizon step, distilled from a teacher's
forecasts over sliding windows. It is stored as a fourth student blob (`PSFCST`,
the same registry format) and served not by `predict()` but by `forecast()`
(`predict-forecast.c`), which extracts the most recent window, normalizes it by
its own mean and standard deviation (so one student serves series of any
magnitude), applies the net, and de-normalizes. When the student carries a
quantile fan (`nquant>1`, distilled from a teacher's quantiles), the point and a
calibrated interval are read straight off the fan by interpolating over its
levels; a point student (`nquant=1`) instead gets its interval from a refit-free
backtest over the series' own history. This is why a zero-dependency build can
approach foundation-model forecast accuracy: a strong teacher's forecasts,
distilled once offline, become a microsecond native student. A tree cannot
represent that temporal function, so `distill_forecast` uses the MLP trainer
specifically.

## Determinism

Statistical inference is deterministic on a given machine: same rows in,
same document out, byte for byte (canonical `%.17g` float formatting in
the JSON documents). The ONNX CPU-fp32 path is deterministic too. GPU and
fp16 inference is not bit-reproducible across machines (different hardware
and kernels round differently). Cross-machine, cross-backend bit-equality
is never claimed (see `benchmarks/notes.md`).

[tvf]: https://www.sqlite.org/vtab.html#tabfunc2
