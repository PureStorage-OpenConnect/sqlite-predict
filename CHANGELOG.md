# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `forecast()`, `detect_anomalies()`, `predict()`, and `distill()`
  table-valued functions with a trailing JSON options argument.
- `distill()` trains a native student, stored as an inline BLOB and executed
  by the zero-dependency core with no onnxruntime. By default it trains on the
  target column — your labels, or a strong teacher's predictions computed
  offline (run TabFM once, store its output, distill it into a student that
  runs anywhere). A `teacher` argument names a registered `predict()` model to
  relabel the rows first (e.g. compress the in-context knn5 into a tree).
  `student_kind='tree'` is a single CART; `'gbt'` is a gradient-boosted forest
  with second-order (Newton) leaves that matches or beats tuned XGBoost on most
  tasks (see `benchmarks/results/tabarena-full.md`); `'mlp'` is a
  one-hidden-layer neural net (classification only) for boundaries an
  axis-aligned tree ensemble renders poorly. With `proba` and `classes` the
  gbt and mlp students distill the teacher's full probability distribution
  (soft-label distillation) instead of its hard argmax, preserving the
  calibration a foundation-model teacher provides. Both serve in microseconds,
  are deterministic, and carry the same exact-replay receipts as the stat
  models. The student blob is bounds-checked on read (caller-writable
  registry).
- `distill_forecast()` trains a native **forecast student**: a multi-output
  regression MLP that maps an instance-normalized context window to a horizon of
  future values, distilled from a teacher's forecasts (a foundation model run
  offline, or any per-window teacher) over sliding windows. It registers a
  `PSFCST` inline BLOB (RFC 0005 §4.1.6) that `forecast()` serves natively, with
  no teacher and no onnxruntime, deterministically and in microseconds. On
  m4_hourly it reaches ~0.89 MASE versus the Chronos teacher's ~0.79 and the
  seasonal-naive floor of ~1.0, where a gradient-boosted tree student stalls at
  ~1.18: the gap trees left was an architecture-capacity gap, not a
  distillation failure. `context`, `horizon`, `hidden`, `epochs`, and `lr` are
  options; the prediction interval comes from a refit-free backtest of the
  student over the series' own history.
- Replayable receipts on every prediction, and `predict_replay()` to verify
  a recorded call reproduces against its anchored data state.
- Bundled zero-dependency models: `theta-classic`, `stub-seasonal-naive`
  (time series) and `knn5-incontext` (tabular). Forecast prediction intervals
  are calibrated from a per-horizon in-sample backtest of each model, rather
  than a lag-1 residual grown as `sigma*sqrt(h)`; on real gluonts data this
  brings 95% coverage toward nominal and roughly halves the interval score.
- `predict_ulid()` and `predict_version()` utility functions.
- Opt-in ONNX runtime serving path for `predict()` (`make loadable-onnx`):
  runs exported models through onnxruntime in two `io_spec` layouts, with a
  process-global session cache, batched inference, and explicit fail-loud
  execution-provider selection. The `vector` layout serves a self-contained
  model (a distilled student or classifier/regressor); the `in_context`
  layout serves a teacher that ingests the `train_query` rows as context
  each call (TabFM-shaped), anchoring those rows in the receipt. A GPU build
  (`make loadable-onnx-gpu`) adds the CUDA and TensorRT execution providers
  and fp16/int8 precision, compile-checked in CI and validated on a
  dedicated GPU job. The default build stays zero-dependency.
- `predict_register(model_id, config)` to register an external model,
  pinning its weights by content hash. `_predict_models` gains `weights_uri`
  and `io_spec`; receipts record the execution provider and precision. The
  config can be a bare weights path: the io_spec is read off the model
  (tensor names, output kind, class count) and feature columns map by
  position, so the common case is one line. An explicit io_spec overrides.
- A license gate: a non-permissive model requires `accept_license` to match
  before it will run.
- Test suite, AddressSanitizer/UBSan and valgrind soak targets, a libFuzzer
  harness, and Windows, WebAssembly, and ONNX CI jobs.

_This project is pre-alpha; everything above is subject to change before a
tagged release._
