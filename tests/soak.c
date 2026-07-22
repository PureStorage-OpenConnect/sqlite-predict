/* C soak driver for valgrind (and any harness without python): runs
 * every operation repeatedly, success and error paths, then exits.
 * Leak-checked with --leak-check=full --errors-for-leak-kinds=definite. */
#include "predict-internal.h"

static int fail(const char *what, sqlite3 *db) {
  fprintf(stderr, "FAIL %s: %s\n", what, sqlite3_errmsg(db));
  return 1;
}

static int run_discard(sqlite3 *db, const char *sql, int expect_ok) {
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK)
    return expect_ok ? fail(sql, db) : 0;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    for (int i = 0; i < sqlite3_column_count(stmt); i++)
      (void)sqlite3_column_text(stmt, i);
  }
  sqlite3_finalize(stmt);
  if (expect_ok && rc != SQLITE_DONE)
    return fail(sql, db);
  if (!expect_ok && rc == SQLITE_DONE)
    return fail("expected error but succeeded", db);
  return 0;
}

int main(void) {
  sqlite3 *db = NULL;
  if (sqlite3_open(":memory:", &db))
    return 1;
  if (sqlite3_predict_init(db, NULL, NULL) != SQLITE_OK)
    return fail("init", db);

  if (run_discard(
          db,
          "CREATE TABLE series(ts TEXT, value REAL, grp TEXT);", 1) ||
      run_discard(
          db,
          "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n"
          " WHERE i < 200) INSERT INTO series SELECT"
          " strftime('%Y-%m-%dT%H:00:00Z','2026-01-01','+'||i||' hours'),"
          " 10.0 + (i % 24) + i * 0.05, 'g' || (i % 3) FROM n",
          1) ||
      run_discard(db,
                  "CREATE TABLE tab(id INTEGER, f1 REAL, f2 REAL,"
                  " label TEXT)",
                  1) ||
      run_discard(db,
                  "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1"
                  " FROM n WHERE i < 120) INSERT INTO tab SELECT i, i*0.3,"
                  " (i*7)%5, 'c'||(i%2) FROM n",
                  1))
    goto done_fail;

  for (int i = 0; i < 50; i++) {
    if (run_discard(db,
                    "SELECT * FROM forecast('SELECT ts, value FROM series',"
                    " 6)",
                    1) ||
        run_discard(db,
                    "SELECT * FROM forecast('SELECT ts, value, grp FROM"
                    " series', 4,"
                    " '{\"group_cols\":[\"grp\"],\"receipt\":0}')",
                    1) ||
        run_discard(db,
                    "SELECT * FROM detect_anomalies("
                    "'SELECT ts, value FROM series', '{\"receipt\":0}')",
                    1) ||
        run_discard(db,
                    "SELECT * FROM predict("
                    "'SELECT f1, f2, label FROM tab WHERE id < 100',"
                    "'SELECT id, f1, f2 FROM tab WHERE id >= 100',"
                    " '{\"target\":\"label\"}')",
                    1))
      goto done_fail;

    /* error paths every iteration too */
    run_discard(db, "SELECT * FROM forecast('DELETE FROM series', 3)", 0);
    run_discard(db,
                "SELECT * FROM forecast('SELECT ts, value FROM series', 3,"
                " '{\"bogus\":1}')",
                0);
    run_discard(db,
                "SELECT * FROM predict('SELECT f1, f2, label FROM tab',"
                " 'SELECT id, f1 FROM tab', '{\"target\":\"label\"}')",
                0);
    run_discard(db, "SELECT * FROM predict_replay('01NOPE')", 0);
    run_discard(db, "SELECT predict_ulid('not a time')", 0);
  }

  /* one replay round-trip on the last receipt */
  if (run_discard(db,
                  "SELECT match FROM predict_replay("
                  "(SELECT receipt_id FROM _predict_receipts ORDER BY"
                  " receipt_id DESC LIMIT 1))",
                  1))
    goto done_fail;

  sqlite3_close(db);
  fprintf(stderr, "soak ok\n");
  return 0;

done_fail:
  sqlite3_close(db);
  return 1;
}
