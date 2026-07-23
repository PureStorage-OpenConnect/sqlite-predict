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
| `sqlite-predict.c` | entry point, function registration, shared helpers (timestamp parse/format, ULID, options parsing, normal-quantile) |
| `predict-forecast.c` | `forecast()` and `detect_anomalies()` vtabs, the statistical models, and the shared `collect_series()` helper |
| `predict-tabular.c` | `predict()` vtab, the in-context k-NN model, and dispatch to a runtime backend for registered models |
| `predict-receipts.c` | model registry, receipts, canonical hashing, the logical-digest anchor, and `predict_replay()` |
| `predict-onnx.c` | ONNX runtime backend (opt-in build only); the only file that links onnxruntime |
| `predict-internal.h` | shared types, contract constants, error codes, internal prototypes |
| `vendor/sha256.c` | a self-contained FIPS 180-4 SHA-256 |

## Operations are table-valued functions

Each operation is an [eponymous virtual table module][tvf]. Arguments
arrive as hidden columns (`query`, `horizon`, `options`), resolved in
`xBestIndex` and consumed in `xFilter`, which does all the work and
materializes result rows the cursor then walks. This is the same mechanism
SQLite's own `generate_series` uses.

`forecast()` and `detect_anomalies()` share `collect_series()`: prepare and
validate the inner query (read-only, single statement), resolve or infer
the time/value/group columns, and collect rows into per-series buffers.

## Models

Models are looked up in a registry table (`_predict_models`) by id, with a
content hash, a license tag, and a `runtime`. The bundled models are pure-C
statistical methods (`runtime='bundled'`). `predict()` dispatches by
runtime: the default `knn5-incontext` path is unchanged and never touches
the registry (so it still works read-only), while a named `onnx` model
routes through the backend in `predict-onnx.c`.

That backend (compiled only into `make loadable-onnx`, `-DSQLITE_PREDICT_ONNX`)
serves two `io_spec` layouts. The **vector** layout is a self-contained model
mapping a feature vector to a prediction. The **in_context** layout is a
teacher (TabFM-shaped): it ingests the `train_query` rows as three tensors
(`x_train`, `y_train`, `x_query`) each call and labels the query rows against
that context. Both cache one onnxruntime session per (weights, device,
precision), run query rows in batches, and select the execution provider
explicitly, erasing no failure into a silent CPU fallback. Weights are pinned
by content hash, so a receipt records exactly which bytes ran, and the
in-context receipt anchors the training rows too, so mutating the context
breaks replay. Both run on CPU. A GPU build (`make loadable-onnx-gpu`,
`-DSQLITE_PREDICT_ONNX_GPU`) wires the CUDA and TensorRT providers and
fp16/int8 precision; the provider-options symbols are in every onnxruntime
C API, so it compiles and links against the CPU onnxruntime for a CI
compile-check, while real GPU execution is validated on a dedicated GPU job.
The receipt records the execution provider and precision, which is what
makes GPU results honestly distinguishable from the deterministic CPU path.
Even so, the `benchmarks/` numbers (and the TabFM→ONNX eval in
`benchmarks/results/tabfm-onnx.md`) are why the default answer for a
teacher's accuracy is usually distillation to a small vector student.

## Receipts, anchoring, and replay

Every operation writes a row to `_predict_receipts` (unless
`'{"receipt":0}'`): the model id and hash, an anchor for the data state
read, the canonical call parameters, the inner SQL, and a hash of the
result set.

- **Canonical result hash** (`predict-receipts.c`): rows are hashed in a
  defined order with type-tagged fields, separated by `0x1F` between fields
  and `0x1E` between rows. Integers hash as decimal text, reals as their
  big-endian IEEE-754 bit pattern, and text as UTF-8. This keeps the hash
  stable across runs on the same machine.
- **Logical-digest anchor**: a hash of the user tables' schema and rows
  (excluding `_predict_%` and `sqlite_%`). A page-level file digest can
  never replay-match, because writing the receipt itself changes the file;
  the logical digest is indifferent to receipt writes and to `VACUUM`.
- **Replay** re-executes the recorded call read-only against the anchored
  state and compares result hashes. It refuses to run if the current state
  no longer matches the anchor.

## Determinism

Statistical inference is deterministic on a given machine, which is what
makes replay meaningful. The ONNX CPU-fp32 path is deterministic too, so
its receipts replay bit-exact; the receipt records the execution provider
and precision. GPU and fp16 inference is not bit-reproducible across
machines (different hardware and kernels round differently), so that path
carries a `nondeterministic` receipt and replays within a tolerance,
reporting a `match_kind` rather than a bit-exact hash — that tiered replay
lands with the GPU backend. Cross-machine, cross-backend bit-equality is
never claimed (see `benchmarks/notes.md`).

## Deviations from the spec

The implementation surfaced amendments queued for the design spec: the
`ts-stat`/`tabular-stat` model kinds, the `logical-digest` anchor kind, a
`{"train","apply"}` JSON `input_sql` for two-query operations, and null
option values meaning "key omitted." These are noted at the top of the
files that introduce them.

[tvf]: https://www.sqlite.org/vtab.html#tabfunc2
