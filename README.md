# sqlite-predict

[![CI](https://github.com/PureStorage-OpenConnect/sqlite-predict/actions/workflows/ci.yml/badge.svg)](https://github.com/PureStorage-OpenConnect/sqlite-predict/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![C99](https://img.shields.io/badge/C99-zero%20dependencies-brightgreen.svg)

**Prediction as a SQL primitive.** `forecast()`, `detect_anomalies()`, and
`predict()` become SQL functions. The core is one small C99 file with no
dependencies, so it runs wherever SQLite already does: in your app, on a
phone, in the browser, in the per-database state an AI agent keeps.

```sql
.load ./predict0

-- forecast() is an aggregate: your statement supplies the rows, WHERE and
-- joins and bound parameters compose, and GROUP BY splits the series.
-- Each group returns one JSON document: {"model", "status", "rows"}.
SELECT city, forecast(ts, value, 24) FROM readings GROUP BY city;

-- want typed rows instead? expand the document in SQL:
SELECT r.* FROM forecast_rows((SELECT forecast(ts, value, 24)
                               FROM readings)) AS r;
-- ┌──────┬──────────────────────┬──────────┬─────────────┬─────────────┬────────┐
-- │ step │  forecast_timestamp  │ forecast │ lower_bound │ upper_bound │ status │
-- └──────┴──────────────────────┴──────────┴─────────────┴─────────────┴────────┘
```

None of this ships your data to a cloud model. It runs in-process, on CPU, in
microseconds. And where a foundation model would be too heavy to call per
query, you [**distill it once into a tiny native student**](#distill-a-foundation-model-into-your-database)
that lives inside your database and runs anywhere.

## Is it accurate?

Yes, and the numbers are honest and reproducible (`benchmarks/`), not a demo on
a toy series.

- **Forecasting.** A Chronos foundation model distilled into a zero-dependency
  DLinear/TiDE student reaches **0.89 MASE on m4_hourly**, closing ~70% of the
  gap from the seasonal-naive floor (1.11) to the foundation model itself
  (~0.80), and beating every classical method (Theta 1.03). The student is a few
  kilobytes and serves in microseconds. ([details](benchmarks/results/forecast.md))
- **Anomaly detection.** The `sub-pca` detector scores at the **published
  state-of-the-art level (~0.44 median VUS-PR)** on TSB-AD-U, the reliable
  univariate benchmark, about 2x a forecast-residual detector on the typical
  series.
- **Tabular.** Distilling TabFM into a gradient-boosted student **matches or
  beats tuned XGBoost** on most [TabArena](benchmarks/results/tabarena-full.md)
  tasks, at a couple of kilobytes and microseconds per row.
- **Calibrated uncertainty.** On smooth data the default Gaussian prediction
  band is overconfident (measured **0.57 coverage at a nominal 0.90**); the
  `conformal` option lands at the nominal level, and `backtest()` lets a caller
  verify coverage on its own data locally.

All of it runs on CPU, in-process, with no network and no GPU.

> [!WARNING]
> **Pre-alpha.** The API and the SQL surface are
> unstable and may change without notice. Not yet recommended for production.

## Distill a foundation model into your database

Foundation models are the accuracy ceiling and the deployment problem: too
slow to call per query on CPU, too heavy to ship inside an app. Distillation
is sqlite-predict's answer, and it is a SQL primitive like everything else:
run the big model once as a teacher, compress what it learned into a native
student a few kilobytes big, and serve that student in microseconds from the
zero-dependency core.

The student is not a file on the side. **It is a row in your database**, so it
snapshots, forks, branches, and syncs with the data it predicts, and it
travels wherever the database file goes. An agent that distills a model owns
that model the same way it owns its tables.

```sql
-- once: distill (your labels, or a teacher's precomputed predictions)
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT tenure, spend, churned FROM customers',
  '{"target":"churned","student_id":"churn-v1","student_kind":"gbt"}');

-- forever: serve the student per row, in microseconds, no runtime attached
SELECT * FROM predict(NULL, 'SELECT id, tenure, spend FROM customers',
                      '{"model":"churn-v1"}');

-- the same move for time series: distill a Chronos-class teacher's forecasts
-- into a DLinear/TiDE student, then serve it through forecast()
SELECT forecast(ts, value, 24, '{"model":"traffic-v1"}') FROM readings;
```

The numbers above are this mechanism measured: the distilled forecast student
reaches **0.89 MASE** where the teacher sits at ~0.80 and every classical
method is above 1.0, and the distilled `gbt` student **matches or beats tuned
XGBoost** on most TabArena tasks. The [Models](#models) section covers the
student architectures and soft-label distillation; the
[distillation guide](https://purestorage-openconnect.github.io/sqlite-predict/guides/distillation/)
walks the whole flow.

## Why

Prediction is becoming a query primitive. BigQuery has `AI.FORECAST`,
Databricks has `ai_forecast()`, Snowflake has ML functions. All of them run in
the cloud, with your data shipped to the model. `sqlite-predict` brings the same
shape to the database that already runs everywhere else: in apps, on phones, in
the browser, and in the per-database state AI agents increasingly keep.

That last one is the point. Agents are getting a database each. This is the
layer that lets an agent forecast its own metrics, flag anomalies in what it
observes, and predict outcomes, right where its state already sits.

Two things follow from being a SQL primitive instead of a service. It is
permissively licensed (MIT/Apache-2.0), so you ship it inside your product
rather than rent it. And it composes with the rest of the in-database AI
toolbox: sqlite-vec gave SQLite vector search; this gives it prediction.

## Operations

| Function | Question | Returns |
| --- | --- | --- |
| `forecast(ts, value, horizon [, options])` | Where is this metric going? | an aggregate over your rows: forecast steps with prediction intervals, one JSON document per group |
| `detect_anomalies(ts, value [, options])` | Which points are abnormal? | an aggregate over your rows: per-point anomaly probability and interval, one JSON document per group |
| `predict(train_query, apply_query [, options])` | Classify/regress unseen rows | a prediction and confidence per row, zero-shot from in-context examples |
| `distill_predict(train_query [, options])` | Compress a slow teacher into a fast student | a tiny native model (decision tree or gradient-boosted forest), registered and ready for `predict()` |
| `distill_forecast(train_query [, options])` | Compress a forecast foundation model into a fast student | a native DLinear/TiDE forecast net (a linear skip plus a small residual), registered and ready for `forecast()` |
| `backtest(query, horizon [, options])` | How accurate is the model here? | per-fold accuracy (MAE, RMSE, MASE, sMAPE) and interval coverage from a rolling-origin evaluation |

One calling convention per operation. `forecast` and `detect_anomalies` are
**aggregates**: your statement supplies the rows, so `WHERE`, joins, bound
parameters, and `GROUP BY` all compose naturally, from the CLI or from an ORM
(Drizzle, SQLAlchemy, Diesel). They are **pure functions**: nothing is ever
written, so they run on read-only databases and inside views. Each group
returns one JSON document; expand it with `forecast_rows()` /
`anomaly_rows()`, or `JSON.parse` it in your app. Each language's
[getting-started guide](https://purestorage-openconnect.github.io/sqlite-predict/getting-started/python/)
shows the pattern through its native ORM (SQLAlchemy, Drizzle, Diesel).

`backtest`, `predict`, and the `distill_*` operations take read-only SELECT
queries (training needs labeled or windowed row sets); their results are
ordinary rows you can join, filter, and materialize. Options everywhere are a
trailing JSON object (`'{"confidence_level":0.9}'`).

### Calibrated intervals, auto-selection, and backtesting

- **Auto-selection is the default.** A call with no `model` option picks the
  model with the lowest rolling-origin error per series, deterministically,
  over the bundled statistical models plus every eligible registered forecast
  student: distill once and your model competes with the baselines
  automatically, at every call site. `'{"candidates":[...]}'` narrows the
  pool; `'{"model":"theta-classic"}'` pins one. The result document's
  `model` field reports the winner.
- **Conformal intervals.** `'{"interval_method":"conformal"}'` replaces the
  default Gaussian band with a distribution-free one calibrated on out-of-sample
  residuals. On smooth data the default band is overconfident; conformal lands
  at the nominal coverage. (Statistical models only; foundation-model students
  already emit their own quantile band.)
- **`backtest()`.** Rolling-origin evaluation with per-fold MAE/RMSE/MASE/sMAPE
  and interval coverage, so a caller can score its own forecasts and validate
  conformal coverage locally. `folds` sets the number of origins; `gap` inserts
  a leakage guard between train and test.

```sql
-- confirm the conformal band actually covers at the nominal 90%
SELECT avg(coverage) FROM backtest('SELECT ts, value FROM readings', 6,
  '{"interval_method":"conformal","confidence_level":0.9,"folds":25}');
```

## Models

The default build is **pure C with no dependencies**. It ships small,
honest statistical models:

- `theta-classic` (the Theta method) and `stub-seasonal-naive` for `forecast()`
  and `detect_anomalies()`; `tsb` (Teunter-Syntetos-Babai) for
  intermittent / sparse-demand `forecast()` (rare events: errors, retries).
  `auto` picks per series by rolling-origin error over these plus every
  eligible registered forecast student; a `candidates` list narrows the
  pool
- `sub-pca` for `detect_anomalies(ts, value, '{"model":"sub-pca"}')`: a
  subsequence-reconstruction detector (windowed PCA reconstruction error), the
  method family that leads the TSB-AD-U benchmark; ~2x the residual detector on
  the typical series there
- `knn5-incontext` (z-scored 5-NN) for `predict()`

Foundation models are treated as *teachers*, not serving paths: in
benchmarking they were far too slow to call per query on CPU. The path to
their accuracy is distillation into a small native student that runs in the
zero-dependency core with no onnxruntime and serves in microseconds.
`distill_predict()` does this for the tabular side (the
teacher we benchmark is **TabFM**, see
[`benchmarks/results/tabarena-full.md`](benchmarks/results/tabarena-full.md)),
and `distill_forecast()` does it for time series: a **Chronos** teacher
distilled into a native **DLinear/TiDE** student (a direct linear map plus a
small nonlinear residual, the architecture that actually fits seasonal data
where a plain MLP does not) closes most of the gap to the FM (see
[`benchmarks/results/forecast.md`](benchmarks/results/forecast.md)).

By default `distill_predict()` trains directly on the target column. That
column can hold your labels, or a strong teacher's predictions computed
offline: run TabFM once over your training rows on a GPU box, store what it
predicts, and distill compresses it into a student that runs anywhere
([example above](#distill-a-foundation-model-into-your-database)).
Pass a `teacher` to instead relabel the rows with a registered model first
(for example, `'{"teacher":"knn5-incontext", ...}'` compresses the in-context
k-NN into a standalone tree).

Three student kinds: `'tree'` is a single decision tree (a few kilobytes,
interpretable); `'gbt'` is a gradient-boosted forest with second-order
(Newton) leaves that on the [TabArena benchmark](benchmarks/results/tabarena-full.md)
matches or beats tuned XGBoost on most tasks; `'mlp'` is a one-hidden-layer
neural net (classification only) for boundaries an axis-aligned tree ensemble
renders poorly. All three run in the zero-dependency core and are
deterministic. Distillation feature columns must be numeric: encode
categorical text before distilling (the in-context `knn5-incontext` handles
text features itself; the distillers do not).

When the teacher gives calibrated probabilities (as a foundation model does),
distill its whole distribution instead of its hard label with `proba` and
`classes`: name the per-class probability columns, and the gbt student matches
the teacher's soft targets rather than its argmax, keeping the calibration a
hard label throws away.

```sql
-- distill TabFM's predicted probabilities (columns p_stay, p_churn), while the
-- true `churned` label scores the holdout
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT tenure, spend, p_stay, p_churn, churned FROM scored',
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
`io_spec` only to override: real class labels, a named-feature mapping, or a
model whose tensors introspection can't disambiguate. It serves two shapes,
with a cached session and batched inference:

- **vector**: a self-contained model (a distilled student, or any exported
  tabular classifier/regressor) mapping a feature vector to a prediction.
  A bare weights path is enough.
- **in_context**: a teacher that ingests the `train_query` rows as context
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

**Python** (the fastest way to try it):

```sh
pip install sqlite-predict          # also: npm install sqlite-predict, cargo add sqlite-predict
```

```python
import sqlite3, sqlite_predict

db = sqlite3.connect("app.db")
sqlite_predict.load(db)
db.execute("SELECT forecast(ts, value, 24) FROM readings")
```

Or drop the extension into any SQLite client directly. Three ways, in order of
least effort, all producing a loadable you open with `.load`:

**1. Precompiled binary.** Download `predict0-<os>-<arch>.{so,dylib,dll}` for
your platform from a [release][rel] (checksums in `SHA256SUMS`), then:

```sql
.load ./predict0
SELECT predict_version();
```

**2. Single-file amalgamation.** Grab `sqlite-predict.c` from a [release][rel]
(or `make amalgamation`) and compile the whole zero-dependency core from one
file, no build system or vendored headers:

```sh
cc -O3 -fPIC -shared sqlite-predict.c -o predict0.so   # loadable, or
cc -c -DSQLITE_CORE sqlite-predict.c                   # static, into your app
```

**3. From source.**

```sh
make loadable        # builds dist/predict0.{dylib,so,dll}
```

`make` fetches the SQLite amalgamation headers into `vendor/` on first build;
it needs a C99 compiler. From any SQLite client:

```sql
.load ./dist/predict0
```

[rel]: https://github.com/PureStorage-OpenConnect/sqlite-predict/releases

For the ONNX serving path, install onnxruntime (macOS: `brew install
onnxruntime`; Linux: extract an [onnxruntime release][ort] and point
`ONNXRUNTIME_PREFIX` at it) and build `make loadable-onnx`. For GPU
execution, point `ONNXRUNTIME_PREFIX` at an onnxruntime-gpu install and
build `make loadable-onnx-gpu`.

[ort]: https://github.com/microsoft/onnxruntime/releases

## How it works

`forecast` and `detect_anomalies` are SQL aggregate functions; `backtest`,
`predict`, and the `distill_*` operations are [eponymous table-valued
functions][tvf] (virtual table modules) over the rows of an inner query.
Statistical models run in-process on CPU and are deterministic. Registered
models and distilled students live in one `_predict_models` table the
extension manages, content-addressed so the exact bytes are pinned and
verified before deserialization.

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
