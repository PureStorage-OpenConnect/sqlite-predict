-- sqlite-predict quickstart. Everything below is self-contained.
--   sqlite3 :memory: '.read examples/quickstart.sql'
-- (needs a SQLite CLI that allows .load; on macOS use `brew install sqlite`.)
.load ./dist/predict0
.mode box
.headers on

-- A daily-seasonal metric with slow drift and one spike, generated in pure SQL.
CREATE TABLE readings AS
WITH RECURSIVE t(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM t WHERE i < 479)
SELECT datetime('2024-01-01', '+' || i || ' hours') AS ts,
       50 + 20 * sin(i * 2 * 3.141592653589793 / 24) + i / 24.0
          + (CASE WHEN i = 300 THEN 60 ELSE 0 END) AS value
FROM t;

.print '\n== forecast the next 24 hours, with prediction intervals =='
SELECT step,
       forecast_timestamp,
       round(forecast, 1)     AS forecast,
       round(lower_bound, 1)  AS lo,
       round(upper_bound, 1)  AS hi
FROM forecast('SELECT ts, value FROM readings', 24)
LIMIT 5;

.print '\n== find the anomaly (sub-pca, a SOTA-level subsequence detector) =='
SELECT ts,
       round(value, 1)               AS value,
       round(anomaly_probability, 3) AS score
FROM detect_anomalies('SELECT ts, value FROM readings',
                      '{"model":"sub-pca"}')
WHERE is_anomaly = 1;

.print '\n== every result is auditable: reproduce a forecast byte-for-byte =='
-- The forecast stamps a receipt (in _predict_receipts, which is excluded from
-- the data anchor). Replay re-runs the recorded call against the exact data
-- state it read and confirms the result reproduces. Modifying `readings` after
-- the call -- or creating a table -- changes that state, and replay would then
-- report the mismatch instead of a match. So read the receipt back, don't write.
SELECT count(*) AS forecasted
FROM forecast('SELECT ts, value FROM readings', 12);
SELECT match, detail FROM predict_replay(
  (SELECT receipt_id FROM _predict_receipts ORDER BY created_at DESC LIMIT 1));

.print '\n== the aggregate form: plain SQL supplies the rows (the ORM path) =='
-- forecast() also works as an aggregate in expression position: WHERE,
-- joins, and GROUP BY compose, and each group returns one JSON document.
-- It is a pure function: nothing is written, so it works on read-only
-- databases and in views, and the receipt comes back INSIDE the document
-- (digests only, never your data) for you to store wherever provenance
-- lives.
SELECT json_extract(forecast(ts, value, 6), '$.status')            AS status,
       json_extract(forecast(ts, value, 6), '$.receipt.model_hash') AS model_hash
FROM readings;

-- expand the document back to typed rows in SQL
SELECT r.step, r.forecast_timestamp, round(r.forecast, 1) AS forecast
FROM forecast_rows((SELECT forecast(ts, value, 6, '{"receipt":0}')
                    FROM readings)) AS r;

.print '\n== verify a receipt document by re-supplying the rows =='
-- hand predict_verify the document (or just its receipt field) + the rows
SELECT match, detail FROM predict_verify(
  (SELECT forecast(ts, value, 6) FROM readings),
  'SELECT ts, value FROM readings');

-- change the data and the same receipt reports honestly
CREATE TEMP TABLE saved AS
  SELECT forecast(ts, value, 6) AS doc FROM readings;
DELETE FROM readings WHERE rowid % 5 = 0;
SELECT match, detail FROM predict_verify(
  (SELECT doc FROM saved), 'SELECT ts, value FROM readings');
