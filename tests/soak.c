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

  /* meta functions — predict_version() sets the JSON result subtype, so
   * this exercises that path under -DSQLITE_STRICT_SUBTYPE (see Makefile) */
  if (run_discard(db, "SELECT predict_version()", 1) ||
      run_discard(db, "SELECT predict_debug()", 1))
    goto done_fail;

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

  /* distill a native tree student from the knn teacher, and register a
   * deliberately corrupt tree blob, so the sanitizers cover the distill
   * training path, the tree runtime, and the bounds-checked deserializer. */
  if (run_discard(
          db,
          "SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
          " '{\"target\":\"label\",\"student_id\":\"soak_student\"}')",
          1) ||
      run_discard(
          db,
          "SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
          " '{\"target\":\"label\",\"student_id\":\"soak_gbt\","
          "\"student_kind\":\"gbt\"}')",
          1) ||
      run_discard(
          db,
          "SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
          " '{\"target\":\"label\",\"teacher\":\"knn5-incontext\","
          "\"student_id\":\"soak_knn\",\"student_kind\":\"gbt\"}')",
          1) ||
      run_discard( /* soft-label distillation: proba columns synthesized */
          db,
          "SELECT * FROM distill_predict('SELECT f1, f2,"
          " CASE WHEN label=''c0'' THEN 0.75 ELSE 0.25 END AS pc0,"
          " CASE WHEN label=''c1'' THEN 0.75 ELSE 0.25 END AS pc1,"
          " label FROM tab', '{\"target\":\"label\","
          "\"proba\":[\"pc0\",\"pc1\"],\"classes\":[\"c0\",\"c1\"],"
          "\"student_id\":\"soak_soft\"}')",
          1) ||
      run_discard( /* mlp student (softmax net) */
          db,
          "SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
          " '{\"target\":\"label\",\"student_kind\":\"mlp\","
          "\"student_id\":\"soak_mlp\"}')",
          1) ||
      run_discard(db,
                  "INSERT INTO _predict_models (model_id, kind, runtime,"
                  " weights, content_hash, license) VALUES ('soak_bad',"
                  "'student','tree',x'505354524545303100000000','x',"
                  "'unspecified')",
                  1) ||
      run_discard(db,
                  "INSERT INTO _predict_models (model_id, kind, runtime,"
                  " weights, content_hash, license) VALUES ('soak_gbt_bad',"
                  "'student','tree',x'505347425430310000000000','x',"
                  "'unspecified')",
                  1) ||
      run_discard(db,
                  "INSERT INTO _predict_models (model_id, kind, runtime,"
                  " weights, content_hash, license) VALUES ('soak_mlp_bad',"
                  "'student','tree',x'50534D4C5030310000000000','x',"
                  "'unspecified')",
                  1))
    goto done_fail;

  /* distill a native forecast student (window mode, no onnx teacher) so the
   * sanitizers cover the forecast-student serve path and its use as an auto
   * candidate: resolve_candidates load/free + fcst_point + fcst_run */
  run_discard(db,
              "SELECT model_id FROM distill_forecast("
              "'WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n"
              " WHERE i<300) SELECT (i+0)%16,(i+1)%16,(i+2)%16,(i+3)%16,"
              "(i+4)%16,(i+5)%16,(i+6)%16,(i+7)%16,(i+8)%16,(i+9)%16 FROM n',"
              " '{\"context\":8,\"horizon\":2,\"student_id\":\"soak_fcst\","
              "\"epochs\":50}')",
              1);

  for (int i = 0; i < 50; i++) {
    /* serving loop: the aggregate forms (the one calling convention for
     * forecast/detect_anomalies) + predict, across models and options */
    if (run_discard(db, "SELECT forecast(ts, value, 6) FROM series", 1) ||
        run_discard(db,
                    "SELECT grp, forecast(ts, value, 4) FROM series"
                    " GROUP BY grp",
                    1) ||
        run_discard(db, "SELECT detect_anomalies(ts, value) FROM series",
                    1) ||
        run_discard(db,
                    "SELECT detect_anomalies(ts, value,"
                    " '{\"model\":\"sub-pca\"}') FROM series",
                    1) ||
        run_discard(db,
                    "SELECT * FROM predict("
                    "'SELECT f1, f2, label FROM tab WHERE id < 100',"
                    "'SELECT id, f1, f2 FROM tab WHERE id >= 100',"
                    " '{\"target\":\"label\"}')",
                    1) ||
        run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                        " '{\"model\":\"soak_student\"}')",
                    1))
      goto done_fail;

    /* gbt-student (forest runtime) + tree/forest error paths */
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_gbt\"}')",
                1);
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_knn\"}')",
                1);
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_soft\"}')",
                1);
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_mlp\"}')",
                1);
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_bad\"}')",
                0);
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_gbt_bad\"}')",
                0);
    run_discard(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
                    " '{\"model\":\"soak_mlp_bad\"}')",
                0);
    run_discard(db,
                "SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
                " '{\"target\":\"label\",\"student_id\":\"soak_student\"}')",
                0);
    run_discard(db, "SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
                    " '{\"student_id\":\"nope\"}')",
                0);

    /* auto selection, conformal intervals, backtest() (grouped, gapped) --
     * exercises the rolling-origin backtest scratch allocations */
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"model\":\"auto\"}') FROM series",
                1);
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"interval_method\":\"conformal\"}') FROM series",
                1);
    run_discard(db, "SELECT grp, forecast(ts, value, 4,"
                    " '{\"model\":\"auto\",\"interval_method\":"
                    "\"conformal\",\"folds\":8,\"gap\":2}')"
                    " FROM series GROUP BY grp",
                1);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value FROM series', 6,"
                    " '{\"folds\":10}')",
                1);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value FROM series', 6,"
                    " '{\"model\":\"auto\",\"interval_method\":\"conformal\","
                    "\"folds\":12}')",
                1);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value, grp FROM series',"
                    " 5, '{\"group_cols\":[\"grp\"],\"gap\":3}')",
                1);
    /* option error + edge paths */
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"interval_method\":\"bogus\"}') FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 6, '{\"folds\":0}')"
                    " FROM series",
                0);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value FROM series', 6,"
                    " '{\"model\":\"nope\"}')",
                0);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value FROM series', 6,"
                    " '{\"gap\":100000}')",
                1);

    /* tsb + auto candidate sets (statistical models; the student-candidate
     * path is exercised by the onnx build) */
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"model\":\"tsb\"}') FROM series",
                1);
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"model\":\"auto\",\"candidates\":[\"tsb\","
                    "\"theta-classic\"]}') FROM series",
                1);
    run_discard(db, "SELECT grp, forecast(ts, value, 4,"
                    " '{\"model\":\"auto\",\"candidates\":"
                    "[\"stub-seasonal-naive\",\"tsb\"],\"gap\":2}')"
                    " FROM series GROUP BY grp",
                1);
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"candidates\":[\"tsb\"]}') FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 6,"
                    " '{\"model\":\"auto\",\"candidates\":[\"nope\"]}')"
                    " FROM series",
                0);
    run_discard(db, "SELECT detect_anomalies(ts, value,"
                    " '{\"model\":\"tsb\"}') FROM series",
                0);

    /* forecast student: direct serve, and as an auto candidate (exercises
     * resolve_candidates student load/free + fcst_point + fcst_run) */
    run_discard(db, "SELECT forecast(ts, value, 2,"
                    " '{\"model\":\"soak_fcst\"}') FROM series",
                1);
    run_discard(db, "SELECT forecast(ts, value, 2,"
                    " '{\"model\":\"auto\",\"candidates\":[\"theta-classic\","
                    "\"soak_fcst\"]}') FROM series",
                1);
    run_discard(db, "SELECT forecast(ts, value, 2,"
                    " '{\"model\":\"auto\",\"candidates\":[\"soak_fcst\"],"
                    "\"interval_method\":\"conformal\"}') FROM series",
                0);

    /* error paths every iteration too: aggregate misuse and option
     * rejection, plus backtest's collect_series failure branches, so
     * valgrind sees the partial-series cleanup under load */
    run_discard(db, "SELECT forecast('SELECT ts FROM series', value, 3)"
                    " FROM series",
                0);
    run_discard(db, "SELECT forecast('SELECT ts, value FROM series', 3)", 0);
    run_discard(db, "SELECT forecast(ts, value, 3, '{\"bogus\":1}')"
                    " FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 3,"
                    " '{\"time_col\":\"nope\"}') FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 3,"
                    " '{\"group_cols\":[\"grp\"]}') FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 3, '{\"receipt\":0}')"
                    " FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, rowid % 2 + 1) FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 4, CASE WHEN rowid % 2"
                    " THEN '{}' ELSE NULL END) FROM series",
                0);
    run_discard(db, "SELECT forecast(ts, value, 0) FROM series", 0);
    run_discard(db, "SELECT forecast(ts, value, 4) FROM series WHERE 0", 1);
    run_discard(db, "SELECT * FROM backtest('DELETE FROM series', 3)", 0);
    run_discard(db, "SELECT * FROM backtest('NOT SQL', 3)", 0);
    run_discard(db, "SELECT * FROM backtest('SELECT ts FROM series', 3)", 0);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value FROM series', 3,"
                    " '{\"time_col\":\"nope\"}')",
                0);
    run_discard(db, "SELECT * FROM backtest('SELECT ts, value, grp FROM series',"
                    " 3, '{\"group_cols\":[\"nope\"]}')",
                0);
    /* duplicate option keys (the CI fuzzer's leak): last-wins, no leak */
    run_discard(db, "SELECT forecast(ts, value, 3,"
                    " '{\"model\":\"theta-classic\",\"model\":"
                    "\"stub-seasonal-naive\"}') FROM series",
                1);
    run_discard(db,
                "SELECT * FROM backtest('SELECT ts, value, grp FROM series',"
                " 3, '{\"model\":\"theta-classic\",\"model\":"
                "\"stub-seasonal-naive\",\"time_col\":\"ts\",\"time_col\":"
                "\"ts\",\"group_cols\":[\"grp\"],\"group_cols\":[\"grp\"]}')",
                1);
    run_discard(db,
                "SELECT * FROM predict('SELECT f1, f2, label FROM tab',"
                " 'SELECT id, f1, f2 FROM tab',"
                " '{\"target\":\"label\",\"task\":\"classify\",\"task\":"
                "\"classify\",\"model\":\"knn5-incontext\",\"model\":"
                "\"knn5-incontext\"}')",
                1);
    run_discard(db,
                "SELECT * FROM predict('SELECT f1, f2, label FROM tab',"
                " 'SELECT id, f1 FROM tab', '{\"target\":\"label\"}')",
                0);
    run_discard(db, "SELECT predict_ulid('not a time')", 0);

    /* expansion functions: round trips + garbage documents */
    run_discard(db, "SELECT * FROM forecast_rows((SELECT forecast(ts, value,"
                    " 4) FROM series))",
                1);
    run_discard(db, "SELECT * FROM anomaly_rows((SELECT detect_anomalies(ts,"
                    " value) FROM series))",
                1);
    run_discard(db, "SELECT * FROM forecast_rows('not json')", 0);
    run_discard(db, "SELECT * FROM anomaly_rows('[1,2]')", 0);
    run_discard(db, "SELECT * FROM forecast_rows(NULL)", 1);
  }

  sqlite3_close(db);
  fprintf(stderr, "soak ok\n");
  return 0;

done_fail:
  sqlite3_close(db);
  return 1;
}
