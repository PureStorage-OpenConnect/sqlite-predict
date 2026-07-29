---
name: distill-lifecycle
description: >-
  Distill a teacher model into a native sqlite-predict student and manage
  it over time: verify quality before serving, re-distill on drift, and
  respect the teacher's license. Use when per-call serving needs to be
  instant and self-contained, when a prediction runs at volume, or when a
  model must travel inside the database file.
metadata:
  version: 0.1.0
---

# The distillation lifecycle

Distillation compresses a teacher (your own labels, your existing model's
predictions, or a licensed foundation model) into a student a few
kilobytes big that serves in microseconds in the zero-dependency core.
The fit takes seconds; the lifecycle judgment is yours.

## Distill

```sql
-- tabular: target column holds labels or your model's predictions
SELECT model_id, holdout_metric FROM distill_predict(
  'SELECT f1, f2, label FROM training',
  '{"target":"label","student_id":"churn-v1","student_kind":"gbt"}');

-- time series: from windows, or with a registered onnx teacher
SELECT model_id FROM distill_forecast('SELECT series_key, value FROM obs',
  '{"context":96,"horizon":24,"student_id":"traffic-v1"}');
```

Students register in `_predict_models` as content-hashed rows; they
snapshot, fork, and sync with the database. Serve by name:
`predict(NULL, apply_sql, '{"model":"churn-v1"}')` or
`forecast(ts, value, 24, '{"model":"traffic-v1"}')`.

## Verify before serving

Never serve a student on faith:

1. Read `holdout_metric` from the distill call. It is measured on a
   holdout of your data. Compare it to a floor you trust (the majority
   class, last-value carry-forward, or your current model).
2. For forecast students, run `backtest` on the same series and compare
   the student against `theta-classic` and the naive floor (see the
   interpret-backtest skill).
3. Know the measured shape of distillation loss: on our benchmark suite
   classification students give up a median half point of accuracy
   against their teacher, but regression tails are worse. Check
   regression students more skeptically.
4. Soft-label distillation (`proba`/`classes`) preserves the teacher's
   calibration and rescues heavily imbalanced datasets, where hard labels
   can collapse to one class (the distiller refuses loudly when they do).

## Watch for drift, re-distill cheaply

A student is frozen; the world is not. When input distributions move,
quality decays silently, so schedule verification rather than assuming:

- Re-run `backtest` (forecast) or score a fresh labeled sample (tabular)
  on a cadence proportional to how fast the data changes.
- Re-distilling costs seconds. Register the new student under a
  versioned id (`churn-v2`), verify, then switch the serving call. Keep
  the old row until the new one is trusted; retire it after.

## Licensing is part of the lifecycle

A student derives from its teacher, and the teacher's license travels
with it. Your own labels and models carry no limits. Foundation-model
teachers vary by vendor and version, and restrictively licensed ones
require an explicit `accept_license` opt-in before they will run. Read
the license of the exact weights you use before distilling for anything
beyond evaluation; the project documentation's license notes are
orientation, not legal advice.
