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

Pass `'{"receipt": 0}'` to skip receipt writing on hot paths.
