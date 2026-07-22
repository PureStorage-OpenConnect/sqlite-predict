# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `forecast()`, `detect_anomalies()`, and `predict()` table-valued
  functions with a trailing JSON options argument.
- Replayable receipts on every prediction, and `predict_replay()` to verify
  a recorded call reproduces against its anchored data state.
- Bundled zero-dependency models: `theta-classic`, `stub-seasonal-naive`
  (time series) and `knn5-incontext` (tabular).
- `predict_ulid()` and `predict_version()` utility functions.
- Opt-in ONNX runtime serving path for `predict()` (`make loadable-onnx`):
  runs an exported tabular model (a distilled student or classifier/
  regressor) through onnxruntime, with a process-global session cache,
  batched inference, and explicit fail-loud execution-provider selection.
  The default build stays zero-dependency.
- `predict_register(model_id, config_json)` to register an external model,
  pinning its weights by content hash. `_predict_models` gains `weights_uri`
  and `io_spec`; receipts record the execution provider and precision.
- A license gate: a non-permissive model requires `accept_license` to match
  before it will run.
- Test suite, AddressSanitizer/UBSan and valgrind soak targets, a libFuzzer
  harness, and Windows, WebAssembly, and ONNX CI jobs.

_This project is pre-alpha; everything above is subject to change before a
tagged release._
