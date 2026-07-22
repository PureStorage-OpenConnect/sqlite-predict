-- Smoke test for the sqlite3 CLI (needs a CLI that allows .load;
-- on macOS use homebrew sqlite, not /usr/bin/sqlite3):
--   sqlite3 :memory: '.read test.sql'
.load ./dist/predict0
SELECT predict_version();
SELECT predict_debug();
