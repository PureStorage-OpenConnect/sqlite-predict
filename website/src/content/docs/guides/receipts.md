---
title: Receipts & replay
description: Every prediction is bound to a receipt you can reproduce byte-for-byte.
---

Every served result carries a **receipt**: the model identity and content
hash, an anchor for the exact inputs, the canonical call parameters, and a
canonical hash of the result. The two forms carry it differently:

- **Table-valued calls** write a receipt row to `_predict_receipts`, anchored
  to the database state the query read, and stamp its id (a ULID) on every
  result row.
- **Aggregate calls** return the receipt as a
  [document inside the result](#document-receipts-the-aggregate-form) —
  nothing is written, and there is no id: the document is its own identity.

```sql
SELECT receipt_id FROM forecast('SELECT ts, value FROM readings', 12);
-- 01J...
```

`predict_replay(receipt_id)` re-executes a recorded table-valued call against
the anchored data state and confirms the result reproduces:

```sql
SELECT match, detail FROM predict_replay('01J...');
-- 1 | reproduced (12 rows)
```

- `match = 1` means the result hash is identical to the original.
- If the data the call read has changed, replay returns
  `PREDICT_ERR_ANCHOR_UNAVAILABLE` rather than a false match.
- The result and params are canonicalized (type-tagged fields, IEEE-754 bit
  patterns for reals), so a match means bit-identical, not approximately
  equal. Reproduction across *different* platforms is bounded by the C
  library's floating-point rounding (interval bounds pass through `log`,
  `sqrt`, and `erf`, whose last bits vary between libm implementations).

This is what makes an agent's predictions **auditable**: the agent cites the
receipt id (or hands over the receipt document), and anyone can reproduce the
number later, or detect that the underlying data moved.

## Document receipts (the aggregate form)

The [aggregate form](../operations/#two-forms-table-valued-and-aggregate)
**returns its receipt instead of writing it**: a self-contained ~450-byte
document inside the result, in the mold of a supply-chain attestation — this
model (pinned by hash), these params, inputs with this digest, a result with
this hash. The database is never written; where provenance lives is your
decision, made above the database. An agent that wants the receipt kept
simply stores the document with a tool call; an app INSERTs it into its own
table; a pipeline ships it to the log system.

```json
{"op": "forecast", "model": "theta-classic", "model_hash": "75af…",
 "params": {"horizon": 24, …}, "input_digest": "…", "result_hash": "…"}
```

Verification is `predict_verify(receipt, query)` — hand it the receipt
document (or the whole result document) plus the rows, from anywhere:

```sql
SELECT match, detail FROM predict_verify(:receipt_json,
  'SELECT ts, value FROM readings WHERE city = ''SF''');
-- 1 | verified (24 rows)
```

| | query-anchored (table-valued) | document (aggregate) |
| --- | --- | --- |
| receipt lives | in `_predict_receipts`, in your DB | wherever you put it |
| verified by | `predict_replay` against anchored DB state | `predict_verify` with the document + rows |
| source table changed since | `PREDICT_ERR_ANCHOR_UNAVAILABLE` | match = 0, "inputs do not match" |
| original rows available (anywhere) | n/a | **match = 1**, even from a temp table, a backup, another database |
| works on read-only databases | no (receipts are writes) | **yes, receipts included** |
| receipt contains your data | no (query text only) | no (digests only) |

Both answer different questions: the query-anchored receipt proves "this exact
database state produced this number"; the document receipt proves "these exact
inputs produced this number" to anyone holding the document and the inputs. A
digest mismatch is a finding (match = 0), never a false match; a model-hash
mismatch is a hard error, because a match under a different model would be
meaningless.

Degraded groups (`insufficient_history`, `non_numeric`) carry `receipt` null:
there is no model execution to attest. `'{"receipt": 0}'` merely omits the
object from the document.

## Skipping receipts

Pass `'{"receipt": 0}'` to skip receipt writing on hot read paths. On a
**read-only database** (replicas, `mode=ro` connections), calls that would
write a receipt fail loudly and name this option as the fix; with it, serving
works read-only.
