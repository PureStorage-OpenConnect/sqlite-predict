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

Foundation models (Chronos, TimesFM, TabPFN/TabFM) are treated as
*teachers*, not serving paths. In benchmarking they were far too slow to
call per query on CPU. The intended path for their accuracy is `distill()`,
which compresses a teacher into a compact model that serves in
microseconds; an optional ONNX build for running teachers directly is on
the roadmap.

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
[SQLite](https://sqlite.org). Benchmarks compare against
[Chronos](https://github.com/amazon-science/chronos-forecasting) and
[TabPFN](https://github.com/PriorLabs/TabPFN)/TabFM.

[tvf]: https://www.sqlite.org/vtab.html#tabfunc2
