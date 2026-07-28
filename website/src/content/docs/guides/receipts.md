---
title: Receipts & replay
description: Every prediction is bound to a receipt you can reproduce byte-for-byte.
---

Every result carries a **receipt**: the model identity and content hash, an
anchor for the exact data state the call read, the canonical call parameters, and
a canonical hash of the result. The receipt id (a ULID) is stamped on every
result row.

```sql
SELECT receipt_id FROM forecast('SELECT ts, value FROM readings', 12);
-- 01J...
```

`predict_replay(receipt_id)` re-executes the recorded call against the anchored
data state and confirms the result reproduces:

```sql
SELECT match, detail FROM predict_replay('01J...');
-- 1 | reproduced (12 rows)
```

- `match = 1` means the result hash is identical to the original.
- If the data the call read has changed, replay returns
  `PREDICT_ERR_ANCHOR_UNAVAILABLE` rather than a false match.
- The result and params are canonicalized (type-tagged fields, IEEE-754 bit
  patterns for reals), so replay is exact across machines.

This is what makes an agent's predictions **auditable**: the agent cites the
receipt id, and anyone can reproduce the number later, or detect that the
underlying data moved.

## Inline-series receipts (the aggregate form)

The [aggregate form](../operations/#two-forms-table-valued-and-aggregate) has
no query text to re-run, so its receipts embed the input series itself
(`input_data`), anchored by a digest of that data (`anchor_kind =
'inline-series'`). That flips the durability trade:

| | query-anchored (table-valued) | inline-series (aggregate) |
| --- | --- | --- |
| replay re-runs | the stored query against anchored DB state | the model on the embedded rows |
| source table changed since | `PREDICT_ERR_ANCHOR_UNAVAILABLE` | **still replays, match = 1** |
| receipt contains your data | no (query text only) | yes (the numeric series) |

Both behaviors are correct answers to different questions: the query-anchored
receipt proves "this exact database state produced this number"; the inline
receipt proves "these exact inputs produced this number" and keeps proving it
after the table churns. Because the receipt holds a copy of the series, treat
`_predict_receipts` at the same sensitivity as the data itself, and prune it
on the usual schedule.

A receipt is written per served group; degraded groups (`insufficient_history`,
`non_numeric`) return `receipt_id` null since there is no model execution to
attest.

## Skipping receipts

Pass `'{"receipt": 0}'` to skip receipt writing on hot read paths. On a
**read-only database** (replicas, `mode=ro` connections), calls that would
write a receipt fail loudly and name this option as the fix; with it, serving
works read-only.
