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
problem the agent can own; the only support the extension provides is
the `predict_sha256()` hash utility, so the whole workflow can run in
pure SQL.

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

## Recording in pure SQL

The extension exposes `predict_sha256()` (the same hash that pins model
weights), so the whole workflow runs in SQL with no host language:

```sql
INSERT INTO _predict_receipts (operation, input_sql, options, model_id,
                               content_hash, extension, result_sha256)
SELECT 'forecast',
       'SELECT forecast(ts, value, 24) FROM readings',
       NULL,
       json_extract(doc, '$.model'),
       (SELECT content_hash FROM _predict_models
        WHERE model_id = json_extract(doc, '$.model')),
       json_extract(predict_version(), '$.extension'),
       predict_sha256(doc)
FROM (SELECT (SELECT forecast(ts, value, 24) FROM readings) AS doc);
```

The inner `input_sql` string and the query that computes `doc` must be
the same SQL, verbatim; that is what makes the receipt replayable. If
nothing has been distilled or registered, `_predict_models` does not
exist yet: record `NULL` for `content_hash` instead of the subquery.

## Verifying in pure SQL

```sql
SELECT id,
       result_sha256 = predict_sha256(
         (SELECT forecast(ts, value, 24) FROM readings)) AS match
FROM _predict_receipts WHERE id = 1;
```

`match` is 1 when the replay reproduces the recorded result. On 0,
compare the receipt's `extension` against `predict_version()` and its
`content_hash` against the registry to find what moved; unchanged data
with an unchanged model and extension should never mismatch.

## The reference script (optional)

For row-shaped results (`predict`, `backtest`) the document needs
canonical serialization, and pure SQL float formatting is not
round-trip safe. Use the bundled script for those, or when you want
mismatch diagnostics computed for you:

```
scripts/receipt.py record  DB "SELECT forecast(ts, value, 24) FROM t"
scripts/receipt.py verify  DB RECEIPT_ID   # exit 0 match, 2 mismatch
scripts/receipt.py list    DB
```

Python stdlib only; pass the loadable with --extension or
SQLITE_PREDICT_EXTENSION. It creates the table, canonicalizes (the
aggregate document verbatim; rows as compact JSON with shortest
round-trip floats), resolves the model pin, and reports what changed on
verify.

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
