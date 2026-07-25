# sqlite-predict

An extremely small prediction extension for SQLite and libSQL that runs
anywhere. It gives you zero-shot `forecast()`, `detect_anomalies()`, and
`predict()` as SQL functions, and **every result carries a replayable
receipt**.

```sql
.load ./predict0

-- forecast a metric 24 steps ahead
SELECT * FROM forecast('SELECT ts, value FROM readings', 24);
-- ┌────────────┬──────┬──────────────────────┬──────────┬───────┬───────┬────────┬──────────────┐
-- │ series_key │ step │  forecast_timestamp  │ forecast │  lo   │  hi   │ status │  receipt_id  │
-- └────────────┴──────┴──────────────────────┴──────────┴───────┴───────┴────────┴──────────────┘
```

> [!WARNING]
> **Pre-alpha and unreleased.** The API, the on-disk receipt format, and
> the SQL surface are all unstable and may change without notice. Not yet
> recommended for production.

## Why

Prediction is becoming a query primitive. BigQuery has `AI.FORECAST`,
Databricks has `ai_forecast()`, Snowflake has ML functions. All of them run
in the cloud, with your data shipped to the model. `sqlite-predict` brings
the same shape to the database that runs everywhere else: on the edge, in
the browser, on a phone, and inside the local-first databases that AI
agents increasingly sit on.

The cloud versions stop at the prediction. This one keeps going: every
result carries a **receipt** that pins down the model and the exact data it
read, so an agent can cite the number and an auditor can reproduce it.

## Operations

| Function | Question | Returns |
| --- | --- | --- |
| `forecast(query, horizon [, options])` | Where is this metric going? | future rows with prediction intervals and per-series status |
| `detect_anomalies(query [, options])` | Which points are abnormal? | anomaly-scored rows with expected value and probability |
| `predict(train_query, apply_query [, options])` | Classify/regress unseen rows | a prediction and confidence per row, zero-shot from in-context examples |
| `distill_predict(train_query [, options])` | Compress a slow teacher into a fast student | a tiny native model (decision tree or gradient-boosted forest), registered and ready for `predict()` |
| `distill_forecast(train_query [, options])` | Compress a forecast foundation model into a fast student | a native forecast MLP, registered and ready for `forecast()` |
| `predict_replay(receipt_id)` | Did this prediction reproduce? | a match flag by re-running the recorded call against its anchored data state |

`query` is any read-only `SELECT`; results are ordinary rows you can join,
filter, and materialize. Options are a trailing JSON object
(`'{"group_cols":["region"],"confidence_level":0.9}'`). The interface
deliberately mirrors BigQuery's `AI.FORECAST`: rows in, rows out.

### Receipts and replay

Every prediction is bound to a receipt: the model identity and content
hash, an anchor for the exact data state it read, the call parameters, and
a canonical hash of the result. `predict_replay()` re-executes the recorded
call against that anchored state and confirms the result reproduces
byte-for-byte.

```sql
SELECT receipt_id FROM forecast('SELECT ts, value FROM readings', 12);
-- 01J...  (a ULID, stamped on every result row)

SELECT match, detail FROM predict_replay('01J...');
-- 1 | reproduced (12 rows)
```

Pass `'{"receipt": 0}'` to skip receipt writing on hot paths.

## Models

The default build is **pure C with no dependencies**. It ships small,
honest statistical models:

- `theta-classic` (the Theta method) and `stub-seasonal-naive` for
  `forecast()` / `detect_anomalies()`
- `knn5-incontext` (z-scored 5-NN) for `predict()`

Foundation models are treated as *teachers*, not serving paths: in
benchmarking they were far too slow to call per query on CPU. The path to
their accuracy is distillation into a small native student that runs in the
zero-dependency core with no onnxruntime, serves in microseconds, and carries
the same receipts. `distill_predict()` does this for the tabular side (the
teacher we benchmark is **TabFM**, see
[`benchmarks/results/tabarena-full.md`](benchmarks/results/tabarena-full.md)),
and `distill_forecast()` does it for time series: a **Chronos** teacher
distilled into a native forecast student recovers most of the FM's edge (see
[`benchmarks/results/forecast.md`](benchmarks/results/forecast.md)).

By default `distill_predict()` trains directly on the target column. That column can
hold your labels, or a strong teacher's predictions computed offline: run
TabFM once over your training rows on a GPU box, store what it predicts, and
distill compresses it into a student that runs anywhere.

```sql
-- distill whatever the target column holds (labels, or a teacher's
-- precomputed predictions) into a fast native student, once
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT tenure, spend, plan, churned FROM customers',
  '{"target":"churned","student_id":"churn-v1","student_kind":"gbt"}');

-- then serve it per row, forever
SELECT * FROM predict(NULL, 'SELECT id, tenure, spend, plan FROM customers',
                      '{"model":"churn-v1"}');
```

Pass a `teacher` to instead relabel the rows with a registered model first
(for example, `'{"teacher":"knn5-incontext", ...}'` compresses the in-context
k-NN into a standalone tree).

Three student kinds: `'tree'` is a single decision tree (a few kilobytes,
interpretable); `'gbt'` is a gradient-boosted forest with second-order
(Newton) leaves that on the [TabArena benchmark](benchmarks/results/tabarena-full.md)
matches or beats tuned XGBoost on most tasks; `'mlp'` is a one-hidden-layer
neural net (classification only) for boundaries an axis-aligned tree ensemble
renders poorly. All three run in the zero-dependency core, are deterministic,
and replay exactly.

When the teacher gives calibrated probabilities (as a foundation model does),
distill its whole distribution instead of its hard label with `proba` and
`classes`: name the per-class probability columns, and the gbt student matches
the teacher's soft targets rather than its argmax, keeping the calibration a
hard label throws away.

```sql
-- distill TabFM's predicted probabilities (columns p_stay, p_churn), while the
-- true `churned` label scores the holdout
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT tenure, spend, plan, p_stay, p_churn, churned FROM scored',
  '{"target":"churned","proba":["p_stay","p_churn"],
    "classes":["stay","churn"],"student_id":"churn-soft"}');
```

An opt-in ONNX build (`make loadable-onnx`) runs exported models through
onnxruntime. Point `predict_register()` at a model file and call it by name;
the io_spec is read off the model (input/output tensors, output kind, class
count), so the common case is one line:

```sql
SELECT predict_register('churn', '/models/churn.onnx');
SELECT * FROM predict(NULL, 'SELECT id, tenure, spend FROM customers',
                      '{"model":"churn"}');
```

Feature columns map by position (apply-query order); pass an explicit
`io_spec` only to override — real class labels, a named-feature mapping, or a
model whose tensors introspection can't disambiguate. It serves two shapes,
with a cached session and batched inference:

- **vector** — a self-contained model (a distilled student, or any exported
  tabular classifier/regressor) mapping a feature vector to a prediction.
  A bare weights path is enough.
- **in_context** — a teacher that ingests the `train_query` rows as context
  on each call and labels the `apply_query` rows against them, the way TabFM
  works. Register it with the weights path plus a `target` (the training
  label column, which introspection can't infer): `predict_register('t',
  '{"weights_uri":"/m.onnx","target":"label"}')`.

Both run on CPU today. A GPU build (`make loadable-onnx-gpu`) adds the CUDA
and TensorRT execution providers and fp16/int8 precision; it needs an
onnxruntime-gpu install, is compile-checked in CI, and its GPU execution is
validated on a dedicated GPU job. Provider selection is explicit and fails
loud: asking for `cuda` on a build without it errors, never a silent drop to
CPU. The default build never links onnxruntime at all.

## Installing

Pre-alpha: build from source (prebuilt binaries will come with releases).

```sh
make loadable        # builds dist/predict0.{dylib,so,dll}
```

`make` fetches the SQLite amalgamation headers into `vendor/` on first
build. Requires a C99 compiler. Then, from any SQLite client:

```sql
.load ./dist/predict0
```

For the ONNX serving path, install onnxruntime (macOS: `brew install
onnxruntime`; Linux: extract an [onnxruntime release][ort] and point
`ONNXRUNTIME_PREFIX` at it) and build `make loadable-onnx`. For GPU
execution, point `ONNXRUNTIME_PREFIX` at an onnxruntime-gpu install and
build `make loadable-onnx-gpu`.

[ort]: https://github.com/microsoft/onnxruntime/releases

## How it works

Each operation is an [eponymous table-valued function][tvf] (a virtual
table module) over the rows of an inner query. Statistical models run
in-process on CPU. Receipts and the logical-digest anchor live in
`_predict_receipts` / `_predict_models` tables the extension manages;
canonical result hashing (type-tagged fields, big-endian float bit
patterns) makes replay deterministic across runs on the same machine.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design and
[`SECURITY.md`](SECURITY.md) for the trust model.

## Development

```sh
make loadable            # build the loadable extension
make test                # pytest suite (uses uv)
make test-asan           # AddressSanitizer + UBSan over the C soak driver
make test-valgrind       # real valgrind in a Linux container (needs Docker)
make fuzz                # libFuzzer (needs an LLVM with the fuzzer runtime)
make fuzz-docker         # libFuzzer + ASan in a Linux container
```

CI builds and tests on Linux and macOS (the full pytest suite, plus
AddressSanitizer, UBSan, valgrind, and a libFuzzer smoke run), builds the
loadable DLL on Windows through MinGW, and compiles to WebAssembly with
emscripten and runs it under node. The Windows and wasm checks drive a
standalone C soak driver that exercises every operation, so "runs
anywhere" is a thing CI proves rather than a thing the README claims.

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option (`SPDX-License-Identifier: MIT
OR Apache-2.0`). Unless you state otherwise, any contribution you
intentionally submit for inclusion shall be dual-licensed as above,
without additional terms.

## Acknowledgements

Structure and spirit follow Alex Garcia's
[`sqlite-vec`](https://github.com/asg017/sqlite-vec). Built on
[SQLite](https://sqlite.org). Benchmarks distill and compare against Google's
TabFM (tabular) and compare against
[Chronos](https://github.com/amazon-science/chronos-forecasting) (time series);
the [TabPFN](https://github.com/PriorLabs/TabPFN) line is the intellectual
lineage for in-context tabular models.

[tvf]: https://www.sqlite.org/vtab.html#tabfunc2
