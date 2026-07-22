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
- Test suite, AddressSanitizer/UBSan and valgrind soak targets, and a
  libFuzzer harness.

_This project is pre-alpha; everything above is subject to change before a
tagged release._
