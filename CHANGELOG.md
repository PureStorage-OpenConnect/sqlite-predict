# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `forecast()`, `detect_anomalies()`, `predict()`, `distill_predict()`, and
  `backtest()` table-valued functions with a trailing JSON options argument.
- **Auto model selection, conformal intervals, and `backtest()`.**
  `forecast('...', h, '{"model":"auto"}')` selects the statistical model with the
  lowest rolling-origin MASE per series (deterministic and replayable).
  `'{"interval_method":"conformal"}'` swaps the default Gaussian prediction band
  for a distribution-free one calibrated on out-of-sample residuals: on smooth
  series the in-sample band is overconfident (measured 0.57 empirical coverage at
  a nominal 0.90), while conformal lands at the nominal level. `backtest(query,
  horizon, options)` reports per-fold MAE/RMSE/MASE/sMAPE and interval coverage
  from a rolling-origin evaluation, with `folds` and a `gap` leakage guard, so a
  caller can score forecasts and validate conformal coverage on-device
  (conformal via leave-fold-out calibration). All three share one rolling-origin
  backtest core, carry exact-replay receipts, and are covered by ASan/UBSan.
- **Intermittent-demand model + explicit auto candidates.** `tsb`
  (Teunter-Syntetos-Babai) joins the bundled statistical models and the `auto`
  pool, for the sparse, mostly-zero series (rare events: errors, retries) that
  theta and seasonal-naive model poorly; it is forecast-only (rejected by
  `detect_anomalies`). `'{"model":"auto","candidates":[...]}'` sets the auto
  pool explicitly, and a candidate may be a distilled forecast student, so an
  agent's own foundation-model student competes with the cheap baselines per
  series. Candidates are recorded canonically in the receipt for deterministic
  replay; `conformal` and an over-long horizon are rejected for a student
  candidate.
- `distill_predict()` trains a native student, stored as an inline BLOB and executed
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
- `distill_forecast()` trains a native **forecast student**: a regression net
  that maps an instance-normalized context window to future values, distilled
  from a teacher's forecasts (a foundation model run offline, or any per-window
  teacher) over sliding windows. It registers a `PSFCST` inline BLOB (RFC 0005
  §4.1.6) that `forecast()` serves natively, with no teacher and no onnxruntime,
  deterministically and in microseconds. The student is a **DLinear/TiDE-style
  net**: a direct linear map from the context to the fan (which carries the
  seasonal-naive + trend structure) plus a small, scaled nonlinear residual. A
  raw-lag MLP was the wrong inductive bias here; the linear skip is what makes
  the student competitive on strongly seasonal data. `hidden=0` drops the
  residual for a pure-linear student (a smaller, convex, fast-converging model);
  the default keeps the residual (TiDE) at a benchmark-chosen width of 256. On
  m4_hourly, distilling an onnx Chronos teacher (~0.80 MASE), the TiDE student
  reaches ~0.89 MASE and pure-linear ~0.98, both beating the best classical
  method (AutoTheta ~1.03) and the seasonal-naive floor (~1.11), where the old
  raw-lag MLP stalled at ~1.03. Notably the residual *width* is the lever the
  raw MLP lacked: with the linear skip stabilizing it, 256 hidden units help
  (128 and 512 are both worse), whereas widening the skip-less MLP only hurt.
  (More diverse training series past the ones being forecast dilute a single
  global student, so distilling on the target series is best.) With a `quantiles`
  option the student distills the teacher's full quantile fan and `forecast()`
  reads a **calibrated** interval straight off it (78% coverage of an 80% band
  on m4_hourly, matching Chronos, and CRPS-competitive); with no `quantiles` it
  is a point forecaster whose interval comes from a refit-free backtest over the
  series' own history. `context`, `horizon`, `hidden`, `epochs`, and `lr` are
  also options. Distilling the well-calibrated teacher fan beat native pinball
  quantile regression, which under-covers on limited data. With a `teacher`
  option naming a registered onnx forecast model (the `loadable-onnx` build),
  the whole loop runs in one SQL call with no Python: the train_query returns a
  raw series (`value`, or `series_key, value` for several), the onnx teacher
  labels sliding context windows in-DB, and the student is fit on them (the
  forecast analog of `distill_predict`'s `teacher=`). Its quantile levels come
  from the teacher's fan.
- `detect_anomalies(model='sub-pca')`: a **subsequence-reconstruction** detector,
  the method family that leads the TSB-AD-U benchmark (where one-step-residual
  detectors sit mid-pack). It embeds the series into sliding windows of one
  period, standardizes each phase, fits PCA over the windows with a deterministic
  Jacobi eigensolver, and scores each window by its reconstruction error in the
  top-variance subspace (the top 30% of components) -- a window off the "normal"
  manifold reconstructs poorly. `anomaly_probability` is the score's percentile
  rank; there is no forecast/interval. On a broad 200-series TSB-AD-U sample it
  scores 0.44 median VUS-PR (at the published SOTA level) versus the theta
  residual detector's 0.23 -- roughly 2x on the typical series, all in zero-
  dependency C with exact-replay receipts. Two honest caveats: the *mean* gap is
  much smaller (0.41 vs 0.38) because sub-pca fails badly on weakly-periodic or
  very long series where the fixed-window PCA does not fit, so the two are partly
  complementary; and on a subset filtered to moderate-length periodic series the
  gap is far larger (0.63 vs 0.25). Either way the forecast-residual detectors do
  NOT close it: a stronger forecaster masks anomalies by predicting them (verified
  against a Chronos one-step baseline on TSB-AD-U, which only tied the theta
  z-score), so the win comes from the detector family, not a better forecaster.
- Replayable receipts on every prediction, and `predict_replay()` to verify
  a recorded call reproduces against its anchored data state.
- Bundled zero-dependency models: `theta-classic`, `stub-seasonal-naive`,
  `sub-pca` (time series) and `knn5-incontext` (tabular). Forecast prediction intervals
  are calibrated from a per-horizon in-sample backtest of each model, rather
  than a lag-1 residual grown as `sigma*sqrt(h)`; on real gluonts data this
  brings 95% coverage toward nominal and roughly halves the interval score.
- `predict_ulid()` and `predict_version()` utility functions.
- Opt-in ONNX runtime serving path (`make loadable-onnx`): runs exported
  models through onnxruntime, with a process-global session cache, batched
  inference, and explicit fail-loud execution-provider selection. Three
  `io_spec` layouts. The `vector` layout serves a self-contained model (a
  distilled student or classifier/regressor) for `predict()`; the `in_context`
  layout serves a teacher that ingests the `train_query` rows as context
  each call (TabFM-shaped), anchoring those rows in the receipt. The `sequence`
  layout serves a **forecast foundation model** for `forecast()` and as an in-DB
  `distill_forecast` teacher: it feeds a context window as a tensor and reads the
  point + interval off the model's quantile fan. Two exporters ship:
  `scripts/export_chronos_onnx.py` (chronos-bolt, a clean single-graph export;
  on m4_hourly the onnx-served Chronos reproduces its ~0.80 MASE) and
  `scripts/export_timesfm_onnx.py` (timesfm-2.5-200m). TimesFM is a much harder
  export: two of its refinements (flip-invariance and the continuous quantile
  head) will not survive torch.export, because a second decode introduces data-
  dependent symints and the head mutates the output in place. Neither is a
  property of the weights, so instead of shipping a weaker single-decode model,
  the export emits the raw two-head core (point + quantile fans) and the
  extension **reconstructs the full model** outside the graph: a second run on
  the reflected context (flip-invariance), the continuous-head blend, crossing
  repair, and instance denorm. These are declared as independent io_spec flags
  (`flip_invariance`, `continuous_head`, `quantile_crossing_repair`,
  `denormalize`, `fixed_context`, `outputs.point`/`outputs.quantile`), not keyed
  on a model name, so any two-head quantile core exported the same way is served.
  The in-DB reconstruction matches the reference `timesfm.forecast` pipeline to
  float32 precision. Its context is fixed at 512. Both register as teachers, so
  `distill_forecast(teacher='chronos-onnx')` or `teacher='timesfm-onnx'` runs the
  whole distillation in one SQL call. A GPU build
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
- Distribution: a single-file **amalgamation** (`make amalgamation` ->
  `sqlite-predict.c`) you compile in one `cc` invocation with no build system;
  a **Python package** (`pip install sqlite-predict`, `sqlite_predict.load(conn)`)
  whose wheels bundle the loadable per platform; a release workflow that ships
  Linux/macOS/Windows binaries plus the amalgamation with `SHA256SUMS` on a `v*`
  tag; and `examples/quickstart.sql`. CI regenerates and compiles the
  amalgamation on every push so it cannot drift.

_This project is pre-alpha; everything above is subject to change before a
tagged release._
