/* libFuzzer harness: fuzzes the three untrusted text inputs — options
 * JSON, inner query SQL, and predict_ulid's timestamp — against a fresh
 * in-memory database per iteration. Build: make fuzz-build. */
#include "predict-internal.h"

#include <stddef.h>

static int exec_ok(sqlite3 *db, const char *sql) {
  return sqlite3_exec(db, sql, NULL, NULL, NULL);
}

static void seed_tables(sqlite3 *db) {
  exec_ok(db, "CREATE TABLE series(ts TEXT, value REAL, grp TEXT);"
              "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1"
              " FROM n WHERE i < 40)"
              "INSERT INTO series SELECT"
              " strftime('%Y-%m-%dT%H:00:00Z','2026-01-01','+'||i||' hours'),"
              " 10.0 + (i % 7) + (i * 0.1), 'g' || (i % 2) FROM n;"
              "CREATE TABLE tab(id INTEGER, f1 REAL, f2 REAL, label TEXT);"
              "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1"
              " FROM n WHERE i < 30)"
              "INSERT INTO tab SELECT i, i*0.3, (i*7)%5, 'c'||(i%2) FROM n;");
}

static void run_sql_discard(sqlite3 *db, const char *sql, const char *bind1,
                            const char *bind2, const char *bind3) {
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK)
    return;
  if (bind1)
    sqlite3_bind_text(stmt, 1, bind1, -1, SQLITE_TRANSIENT);
  if (bind2)
    sqlite3_bind_text(stmt, 2, bind2, -1, SQLITE_TRANSIENT);
  if (bind3)
    sqlite3_bind_text(stmt, 3, bind3, -1, SQLITE_TRANSIENT);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    /* touch every column so xColumn paths execute */
    for (int i = 0; i < sqlite3_column_count(stmt); i++)
      (void)sqlite3_column_text(stmt, i);
  }
  sqlite3_finalize(stmt);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2)
    return 0;
  char *text = sqlite3_malloc((int)size); /* NUL-terminate a copy */
  if (!text)
    return 0;
  memcpy(text, data + 1, size - 1);
  text[size - 1] = '\0';

  sqlite3 *db = NULL;
  if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
    sqlite3_free(text);
    return 0;
  }
  sqlite3_predict_init(db, NULL, NULL);
  seed_tables(db);

  switch (data[0] % 6) {
  case 0: /* fuzz forecast aggregate options */
    run_sql_discard(db,
                    "SELECT forecast(ts, value, 3, ?1) FROM series"
                    " GROUP BY grp",
                    text, NULL, NULL);
    break;
  case 1: /* fuzz the aggregate ts argument + the misuse stub */
    run_sql_discard(db, "SELECT forecast(?1, value, 3) FROM series", text,
                    NULL, NULL);
    run_sql_discard(db, "SELECT forecast(?1, 3)", text, NULL, NULL);
    break;
  case 2: /* fuzz detect_anomalies options */
    run_sql_discard(db, "SELECT detect_anomalies(ts, value, ?1) FROM series",
                    text, NULL, NULL);
    break;
  case 3: /* fuzz predict: input as options, and as apply query */
    run_sql_discard(db,
                    "SELECT * FROM predict('SELECT f1, f2, label FROM tab',"
                    " 'SELECT id, f1, f2 FROM tab', ?1)",
                    text, NULL, NULL);
    run_sql_discard(db,
                    "SELECT * FROM predict('SELECT f1, f2, label FROM tab',"
                    " ?1, '{\"target\":\"label\"}')",
                    text, NULL, NULL);
    break;
  case 4: /* fuzz backtest: options and the inner query */
    run_sql_discard(db, "SELECT * FROM backtest("
                        "'SELECT ts, value FROM series', 3, ?1)",
                    text, NULL, NULL);
    run_sql_discard(db, "SELECT * FROM backtest(?1, 3)", text, NULL, NULL);
    break;
  default: /* fuzz the expansion functions' document parser + predict_ulid */
    run_sql_discard(db, "SELECT * FROM forecast_rows(?1)", text, NULL, NULL);
    run_sql_discard(db, "SELECT * FROM anomaly_rows(?1)", text, NULL, NULL);
    run_sql_discard(db, "SELECT predict_ulid(?1)", text, NULL, NULL);
    break;
  }

  sqlite3_close(db);
  sqlite3_free(text);
  return 0;
}
