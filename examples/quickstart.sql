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

.print ''
.print '== forecast the next 24 hours, with prediction intervals =='
-- forecast() is an aggregate: your statement supplies the rows (WHERE,
-- joins, and GROUP BY all compose), and each group returns one JSON
-- document: {"model", "status", "rows"}. Expand it to typed rows with
-- forecast_rows(), or JSON-parse it in your app.
SELECT r.step,
       r.forecast_timestamp,
       round(r.forecast, 1)     AS forecast,
       round(r.lower_bound, 1)  AS lo,
       round(r.upper_bound, 1)  AS hi
FROM forecast_rows((SELECT forecast(ts, value, 24) FROM readings)) AS r
LIMIT 5;

.print ''
.print '== find the anomaly (sub-pca, a SOTA-level subsequence detector) =='
SELECT r.ts,
       round(r.value, 1)               AS value,
       round(r.anomaly_probability, 3) AS score
FROM anomaly_rows((SELECT detect_anomalies(ts, value, '{"model":"sub-pca"}')
                   FROM readings)) AS r
WHERE r.is_anomaly = 1;

.print ''
.print '== multi-series is just GROUP BY =='
SELECT substr(ts, 12, 2) AS hour_of_day,
       json_extract(forecast(ts, value, 4), '$.status') AS status
FROM readings GROUP BY hour_of_day LIMIT 4;

.print ''
.print '== how accurate is it here? backtest on your own data =='
SELECT fold, model, round(mase, 2) AS mase, round(coverage, 2) AS coverage
FROM backtest('SELECT ts, value FROM readings', 24, '{"folds":5}')
LIMIT 5;
