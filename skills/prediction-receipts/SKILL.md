---
name: prediction-receipts
description: >-
  Record verifiable provenance for predictions made with sqlite-predict:
  what was asked, which model answered, and a hash that lets the result
  be replayed and checked later. Use when a prediction feeds a decision
  worth auditing, when results are shared with other agents or humans,
  or when you need to prove a result was not altered after the fact.
license: MIT OR Apache-2.0
metadata:
  version: "0.1.0"
---

# Prediction receipts

sqlite-predict serving is deterministic and pure: the same rows, options,
and model produce byte-identical result documents, and registered model
weights are pinned by content hash. That makes provenance a recording
problem the agent can own, with no support needed from the extension.

Record receipts for predictions that matter (decisions, reports, handoffs),
not for every exploratory call.

## The receipts table

Create it in your own database (or a sidecar). Convention, so receipts
from different agents interoperate:

```sql
CREATE TABLE IF NOT EXISTS _predict_receipts (
  id           INTEGER PRIMARY KEY,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  operation    TEXT NOT NULL,   -- 'forecast' | 'detect_anomalies' | 'predict' | 'backtest'
  input_sql    TEXT NOT NULL,   -- the exact SQL that supplied the rows
  options      TEXT,            -- the options JSON as passed, or NULL
  model_id     TEXT NOT NULL,   -- the "model" field of the result document
  content_hash TEXT,            -- registry hash for a registered model, else NULL
  extension    TEXT NOT NULL,   -- "extension" field of predict_version()
  result_sha256 TEXT NOT NULL   -- SHA-256 of the exact result document text
);
```

## The reference script

Prefer the bundled implementation over hand-rolling the steps below, so
receipts stay interoperable across agents:

```
scripts/receipt.py record  DB "SELECT forecast(ts, value, 24) FROM t"
scripts/receipt.py verify  DB RECEIPT_ID   # exit 0 match, 2 mismatch
scripts/receipt.py list    DB
```

It creates the table, canonicalizes the result (the aggregate document
verbatim; rows as compact JSON with shortest round-trip floats), hashes
it, resolves the model's registry pin, and on verify reports what
changed (data, extension version, or model hash). Python stdlib only;
pass the loadable with --extension or SQLITE_PREDICT_EXTENSION.

## Writing a receipt by hand

1. Run the prediction and keep the raw result document text (the JSON the
   aggregate returned, or the concatenated rows for a TVF, in a fixed
   order).
2. `model_id` comes from the document's `model` field. For a registered
   model or student, read its pin:
   `SELECT content_hash FROM _predict_models WHERE model_id = ?`.
3. Hash the exact document text with SHA-256 (any host language; SQLite
   itself has no sha256 built in).
4. Insert the row. Store the document itself wherever the result is used;
   the receipt stores its hash, not the data.

## Replaying a receipt

To verify a receipt later: re-run `input_sql` through the same operation
with the same `options`, hash the new document, and compare to
`result_sha256`. Three outcomes:

- Hashes match: the result reproduces. With an unchanged `content_hash`,
  the same model on the same data gave the same answer.
- Hashes differ and the underlying table changed: the data moved, which
  is normal; the receipt documents what was true at `created_at`.
- Hashes differ on unchanged data: investigate. Check the extension
  version (`predict_version()` vs the receipt's `extension`) and the
  model's `content_hash` (tampered weights fail loudly at load with
  `PREDICT_ERR_MODEL_HASH`, so a swap cannot hide).

## Honest limits

- Replay assumes the input rows are reproducible. If the table mutates,
  snapshot the database (or record row counts and a data hash alongside)
  when the receipt matters enough.
- Determinism is per build for the statistical models and native
  students. ONNX-served models are deterministic on the same CPU build
  but not bit-identical across machines or GPU providers; record the
  device if you serve through ONNX and intend to replay.
- A receipt proves what was computed, not that the prediction was good.
  Pair with `backtest` (see interpret-backtest) when quality claims
  matter.
