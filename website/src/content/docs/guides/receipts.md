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

## Commitment receipts (the aggregate form)

The [aggregate form](../operations/#two-forms-table-valued-and-aggregate) has
no query text to re-run, so its receipt is a **commitment**, in the mold of
supply-chain attestations: this model (pinned by hash), these params, inputs
with this digest, a result with this hash. It stores no row values and is
constant-size (~450 bytes) no matter how long the series is, so leaving
receipts on costs nothing measurable.

Verification is `predict_verify(receipt_id, query)`: you bring the rows, it
checks them against the committed digest, re-runs the recorded call, and
compares result hashes:

```sql
SELECT match, detail FROM predict_verify('01J…',
  'SELECT ts, value FROM readings WHERE city = ''SF''');
-- 1 | verified (24 rows)
```

| | query-anchored (table-valued) | commitment (aggregate) |
| --- | --- | --- |
| verified by | `predict_replay` against anchored DB state | `predict_verify` with caller-supplied rows |
| source table changed since | `PREDICT_ERR_ANCHOR_UNAVAILABLE` | match = 0, "inputs do not match" |
| original rows available (anywhere) | n/a | **match = 1**, even from a temp table or restored backup |
| receipt contains your data | no (query text only) | no (digests only) |

Both answer different questions: the query-anchored receipt proves "this exact
database state produced this number"; the commitment receipt proves "these
exact inputs produced this number" to anyone who can produce the inputs. A
digest mismatch is a finding (match = 0), never a false match.

A receipt is written per served group; degraded groups (`insufficient_history`,
`non_numeric`) return `receipt_id` null since there is no model execution to
attest.

## Skipping receipts

Pass `'{"receipt": 0}'` to skip receipt writing on hot read paths. On a
**read-only database** (replicas, `mode=ro` connections), calls that would
write a receipt fail loudly and name this option as the fix; with it, serving
works read-only.
