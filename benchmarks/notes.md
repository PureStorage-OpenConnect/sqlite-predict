# Spike results (M0-M5), 2026-07-22

The numbers RFC 0005's open questions asked for, measured.

## The artifact

- `predict0.dylib`: **104 KB**, zero dependencies, C99. 56 tests.
- Operations: `forecast()`, `detect_anomalies()`, `predict()`,
  `predict_replay()`, `predict_ulid()`, `predict_version()`; receipts on
  all three prediction ops; replay verifies all three.
- Bundled models: theta-classic, stub-seasonal-naive (ts-stat),
  knn5-incontext (tabular-stat).

## Latency (measured)

| call | latency |
|---|---|
| forecast, 57-point daily series, horizon 7, receipt off | 1 ms |
| forecast, same, receipt on (158 MB real db) | **657 ms** |
| detect_anomalies, 38-point series | 18 ms |
| predict, 1,045 apply rows x 4 features, 4k train | 50 ms |
| replay (real db) | 652 ms |
| reference: chronos-bolt-small per forecast | 13-33 ms + torch |
| reference: TabFM per predict call (CPU) | 32-42 s |
| reference: TabFM-distilled student serve | 1 µs/row |

## Model comparison summaries (full tables in results/)

- **Time series** (`comparison.md`): chronos median MASE 0.744 vs theta
  1.118; chronos wins intermittent (0.279) and level shifts, loses
  random walks (3.55) and trend+season; chronos intervals under-cover
  (54-96%), ours over-cover (100% — needs calibration work).
- **Tabular** (`comparison-tabular.md`): at 300 labels knn5 ties TabFM
  on classification (0.99/0.98); TabFM near Bayes-optimal on friedman1
  regression (1.184 vs knn 2.990, noise floor 1.0).
- **Distillation** (`distill-viability.md`): distilled students retain
  91-101% of TabFM across suites/budgets at 1 µs and 71 KB. Decisive at
  20-50 labels, where direct training falls below the majority floor.
  Convergence at 300 labels marks the honest boundary.

Model policy these numbers fix: cheap statistical/in-context models as
the zero-dependency defaults; distill_predict() as the FM value path; raw FMs
as opt-in teachers (ONNX build), never default serving.

## Dogfood (gestalt ai-ml-trends thread cache, 5,223 observations)

- All three operations ran unmodified on real data; replay match=1.
- Anomaly scan flagged the two real ingest-spike days (Apr 3, Apr 15)
  blind.
- predict() on intrigue-from-metadata: RMSE 0.121 vs 0.139 floor — weak
  by design (intrigue lives in content, not metadata), independently
  consistent with the engine's own scorer findings.
- Schema drift bit immediately (ingested_at/intrigue_score vs the
  spec's example names): named column overrides are load-bearing.

## Open-question answers for the RFC revision

- **OQ1 (runtime):** deferred correctly. The 33 s CPU TabFM calls and
  the distillation results mean the FM path is distill_predict()-first; a GGML
  or ONNX teacher build is about batch distillation, not serving. The
  ONNX serving path shipped (vector + in_context layouts, CPU); the
  TabFM→ONNX eval (`results/tabfm-onnx.md`) then characterized TabFM itself.
  It can be exported and run (one-line model patch + external-data format,
  ~798 ms/forward, 100% label agreement with PyTorch; see the results doc)
  but is not directly servable: the patch is upstream model surgery
  (`repeat_interleave` with no `dim`), its fp32 weights (~6.5 GB) exceed
  ONNX's 2 GB single-file limit, and its packed signature +
  quantile/ensemble preprocessing do not fit the `in_context` contract.
  The extension serves *distilled* students; TabFM stays the offline
  teacher.
- **OQ2 (in-context latency):** measured. TabFM 32 s at n=20 context
  growing to 144 s at n=300; knn5-incontext 50 ms per thousand rows.
- **OQ4 (determinism):** all replay round-trips bitwise-reproduce on
  same-machine CPU. Cross-backend: MPS produced coin-flip outputs at
  3.6x speed — silent numerical divergence is real, and replay hashing
  is the mechanism that catches it.
- **OQ7 (anchor cost): the spike's biggest finding.** Logical digest of
  a real 158 MB database costs ~650 ms per receipted call because it
  hashes every user table (including embedding BLOBs irrelevant to the
  query). Receipts-by-default at machine frequency need either digest
  scoping (hash only the tables the inner query reads — resolvable via
  sqlite3_set_authorizer during prepare) or platform generation anchors.

## Spec amendments queued for RFC 0005 rev 2

1. `kind` vocabulary: add `ts-stat`, `tabular-stat`.
2. `anchor_kind`: add `logical-digest`; define file-digest as
   unreplayable-by-construction (the receipt write changes the file).
3. Null option values mean key-omitted (params round-trip as options).
4. Two-query operations store `input_sql` as a `{"train","apply"}` JSON
   object.
5. Timestep inference: specify median inter-arrival, not span/(n-1)
   (gappy real data exposed the difference).
6. Anchor scoping (from OQ7): digest SHOULD cover only tables the inner
   query references.
7. Interval calibration: coverage tolerance needs a normative bound
   (both our over-coverage and chronos's under-coverage would fail a
   ±5% band).

## Hardening (M6, 2026-07-22)

- Audit found and fixed three real bugs: silent feature drop past 64
  cols (now PREDICT_ERR_SCHEMA), 512-byte group-key truncation silently
  merging distinct series (now dynamic keys), and epoch-ms integers
  misread as seconds (13-digit heuristic; pre-1970 rejected). Plus one
  earlier use-after-free in an error path caught by the adversarial
  suite. All have regression tests.
- Transaction semantics decided and pinned: receipts participate in the
  enclosing transaction; rollback discards the receipt with the rows.
- ASan+UBSan (non-recoverable): clean over the full C soak.
- valgrind (Linux/gcc container): 83,736 allocs = 83,736 frees, zero
  errors, "no leaks are possible". First Linux build, passing.
- libFuzzer+ASan (Linux container): 194,114 runs / 121s over options
  JSON, query SQL, and ulid/receipt inputs — zero crashes; corpus kept
  in fuzz/corpus/.
- pytest leak soak: bounded RSS over 1,200 mixed calls.

## Code-quality pass (Fowler review, 2026-07-22)

Reviewed for smells after the user flagged hardcoding. Done:
- **Date handling rewritten** (was genuinely broken): strict digit parser
  (rejects embedded signs, :60, leading whitespace); Hinnant O(1)
  civil<->days replacing an O(years) loop that was a 294k-iteration DoS
  + buffer overflow past year 9999; format clamps to [0001,9999] so it's
  total for any i64; integer-epoch columns reject negatives/out-of-domain
  as non_numeric instead of silent garbage. 14 new date tests.
- **Contract constants lifted** to predict-internal.h: the 0x1f/0x1e
  separators (shared by key builder + hasher + spec — the one magic
  number that could silently corrupt output), buffer-size invariants,
  epoch domain bounds.
- **predict0_emit_receipt extracted**: the ensure->hash->digest->insert
  tail was 3x-duplicated; now one helper owning the ordering contract.

Deferred (recommended next, own focused session): extract
collect_series() — the ~140-line series-collection loop is duplicated
between forecast and detect_anomalies (it already forced the epoch-ms
fix to be applied twice). Not rushed at session end because the two
call sites have divergent error-path cleanup and a botched extraction
would hide a memory bug in an error path. The replay-hash oracle +
sanitizers make it safe to do carefully.

Not changed (deliberate, not smells): the 4 vtab module structs
(idiomatic C, sqlite-vec does the same); vendored SHA-256 / Acklam
quantile / base32 (no C stdlib equivalent); hand-rolled calendar math
(libc timegm is non-portable to Windows/wasm — the reason sqlite itself
hand-rolls its date code; now commented as such).
