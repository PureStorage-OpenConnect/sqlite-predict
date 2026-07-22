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
| `predict-tabular.c` | `predict()` vtab and the in-context k-NN model |
| `predict-receipts.c` | model registry, receipts, canonical hashing, the logical-digest anchor, and `predict_replay()` |
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
content hash and a license tag. The bundled models are pure-C statistical
methods. Foundation models are out-of-process teachers reached via
distillation (roadmap), not per-query serving paths. The benchmarks in
`benchmarks/` drove that decision.

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
makes replay meaningful. Cross-machine and cross-backend determinism is
not guaranteed (see the notes on GPU backends in `benchmarks/notes.md`);
replay verification is same-machine.

## Deviations from the spec

The implementation surfaced amendments queued for the design spec: the
`ts-stat`/`tabular-stat` model kinds, the `logical-digest` anchor kind, a
`{"train","apply"}` JSON `input_sql` for two-query operations, and null
option values meaning "key omitted." These are noted at the top of the
files that introduce them.

[tvf]: https://www.sqlite.org/vtab.html#tabfunc2
