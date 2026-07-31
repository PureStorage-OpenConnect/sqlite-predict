/* ASan/valgrind soak for the onnx backend. Exercises the vector path and
 * its error branches repeatedly so a sanitizer sees every alloc/free pair
 * in predict-onnx.c. Needs a fixture model path as argv[1] (tests/fixtures/
 * logreg.onnx). Built by `make test-asan-onnx`; onnxruntime's own
 * still-reachable allocations are filtered by tests/onnx.supp. */
#include "predict-internal.h"
#include <string.h>

#define APPLY_ROWS 1501 /* the apply table below holds 1501 rows (i = 0..1500) */

/* Run one statement and check its outcome. expect_err == NULL means it must
 * succeed; when expect_rows >= 0 the success must return exactly that many rows
 * (so a zero or truncated result is caught, not silently accepted). A non-NULL
 * expect_err means it must fail at step with an error message containing that
 * PREDICT_ERR_* code, proving the intended validation fired. A prepare failure
 * is always a driver bug (the SQL here is fixed), never an expected error, so it
 * fails the driver instead of masquerading as a passed error case. Returns 1 on
 * any mismatch so main() can propagate it to the exit code. */
static int run(sqlite3 *db, const char *sql, int expect_rows,
               const char *expect_err) {
  sqlite3_stmt *st = NULL;
  if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK) {
    fprintf(stderr, "FAIL (prepare): %s: %s\n", sql, sqlite3_errmsg(db));
    return 1;
  }
  int nrows = 0, rc;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    nrows++;
    for (int i = 0; i < sqlite3_column_count(st); i++)
      (void)sqlite3_column_text(st, i);
  }
  /* capture the message before finalize, which can clear it */
  char *msg =
      rc != SQLITE_DONE ? sqlite3_mprintf("%s", sqlite3_errmsg(db)) : NULL;
  sqlite3_finalize(st);

  int bad = 0;
  if (expect_err) {
    if (rc == SQLITE_DONE) {
      fprintf(stderr, "FAIL (expected %s, got success): %s\n", expect_err, sql);
      bad = 1;
    } else if (!msg || !strstr(msg, expect_err)) {
      fprintf(stderr, "FAIL (expected %s): %s: %s\n", expect_err, sql,
              msg ? msg : "(no message)");
      bad = 1;
    }
  } else if (rc != SQLITE_DONE) {
    fprintf(stderr, "FAIL (expected ok): %s: %s\n", sql,
            msg ? msg : "(no message)");
    bad = 1;
  } else if (expect_rows >= 0 && nrows != expect_rows) {
    fprintf(stderr, "FAIL (expected %d rows, got %d): %s\n", expect_rows, nrows,
            sql);
    bad = 1;
  }
  sqlite3_free(msg);
  return bad;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: %s <vector.onnx> <incontext.onnx> [two_head.onnx]\n",
            argv[0]);
    return 2;
  }
  const char *two_head = argc >= 4 ? argv[3] : NULL;
  sqlite3 *db = NULL;
  if (sqlite3_open(":memory:", &db))
    return 1;
  if (sqlite3_predict_init(db, NULL, NULL) != SQLITE_OK)
    return 1;

  char *reg = sqlite3_mprintf(
      "SELECT predict_register('clf', json_object('runtime','onnx','kind',"
      "'student','license','MIT','weights_uri',%Q,'io_spec',json_object("
      "'layout','vector','input','float_input','features',json_array('f1',"
      "'f2'),'output',json_object('name','probabilities','kind','probs',"
      "'labels',json_array('0','1')))))",
      argv[1]);
  char *regic = sqlite3_mprintf(
      "SELECT predict_register('knn1', json_object('runtime','onnx','kind',"
      "'tabular-fm','license','MIT','weights_uri',%Q,'io_spec',json_object("
      "'layout','in_context','inputs',json_object('x_train','x_train',"
      "'y_train','y_train','x_query','x_query'),'features',json_array('f1',"
      "'f2'),'target','label','output',json_object('name','probabilities',"
      "'kind','probs','labels',json_array('0','1')))))",
      argv[2]);
  /* bare-path registration exercises the introspection + positional path */
  char *regbare = sqlite3_mprintf(
      "SELECT predict_register('bare', %Q)", argv[1]);
  if (run(db, reg, -1, NULL) || run(db, regic, -1, NULL) ||
      run(db, regbare, -1, NULL)) {
    fprintf(stderr, "register failed\n");
    sqlite3_free(reg);
    sqlite3_free(regic);
    sqlite3_free(regbare);
    sqlite3_close(db);
    return 1;
  }
  sqlite3_free(reg);
  sqlite3_free(regic);
  sqlite3_free(regbare);

  /* two-head sequence core: exercises the reconstruction path (two runs,
   * flip-invariance, continuous head, crossing repair, instance denorm) and
   * its fail-loud branch. Optional so older callers still work. */
  if (two_head) {
    char *regth = sqlite3_mprintf(
        "SELECT predict_register('th', json_object('runtime','onnx','kind',"
        "'ts-fm','license','MIT','weights_uri',%Q,'io_spec',json_object("
        "'layout','sequence','input','context','outputs',json_object('point',"
        "'point_fan','quantile','quant_fan'),'quantiles',json_array(0.1,0.3,"
        "0.5,0.7,0.9),'patch',8,'fixed_context',json('true'),'flip_invariance',"
        "json('true'),'continuous_head',json('true'),'quantile_crossing_repair',"
        "json('true'),'denormalize','instance')))",
        two_head);
    /* single-output model that wrongly declares flip_invariance -> fail loud */
    char *regbad = sqlite3_mprintf(
        "SELECT predict_register('badf', json_object('runtime','onnx','kind',"
        "'ts-fm','license','MIT','weights_uri',%Q,'io_spec',json_object("
        "'layout','sequence','input','context','output','quant_fan','quantiles',"
        "json_array(0.1,0.3,0.5,0.7,0.9),'patch',8,'flip_invariance',"
        "json('true'))))",
        two_head);
    int bad = run(db, regth, -1, NULL) || run(db, regbad, -1, NULL);
    sqlite3_free(regth);
    sqlite3_free(regbad);
    if (bad) {
      fprintf(stderr, "two-head register failed\n");
      sqlite3_close(db);
      return 1;
    }
  }

  int fails = 0;
  if (two_head) {
    fails += run(db, "CREATE TABLE ser(ts TEXT, value REAL)", -1, NULL);
    fails += run(db,
        "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n WHERE"
        " i < 40) INSERT INTO ser SELECT datetime('2020-01-01', '+' || i ||"
        " ' hours'), 10.0 + (i%9) FROM n",
        -1, NULL);
    fails += run(db, "CREATE TABLE sk(series_key INTEGER, t INTEGER,"
                     " value REAL)", -1, NULL);
    fails += run(db,
        "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n WHERE"
        " i < 60) INSERT INTO sk SELECT 0, i, 10.0 + (i%9) FROM n",
        -1, NULL);
    /* distill the two-head teacher into forecast students -- exercises the
     * skip/nhid=0 training allocations: a TiDE student (hidden default) and a
     * pure-linear student (hidden=0), point and quantile. */
    fails += run(db,
        "SELECT * FROM distill_forecast('SELECT series_key, value FROM sk"
        " ORDER BY series_key, t', json_object('teacher','th','context',8,"
        "'horizon',3,'student_id','fs_tide','epochs',60))",
        -1, NULL);
    fails += run(db,
        "SELECT * FROM distill_forecast('SELECT series_key, value FROM sk"
        " ORDER BY series_key, t', json_object('teacher','th','context',8,"
        "'horizon',3,'hidden',0,'student_id','fs_lin','epochs',60))",
        -1, NULL);
  }

  fails += run(db, "CREATE TABLE apply(id INTEGER, f1 REAL, f2 REAL)", -1, NULL);
  fails += run(db,
      "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n WHERE"
      " i < 1500) INSERT INTO apply SELECT i, (i%7)-3.0, (i%5)-2.0 FROM n",
      -1, NULL);
  fails += run(db, "CREATE TABLE tr(id INTEGER, f1 REAL, f2 REAL,"
                   " label TEXT)", -1, NULL);
  fails += run(db,
      "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n WHERE"
      " i < 40) INSERT INTO tr SELECT i, (i%5)-2.0, (i%3)-1.0,"
      " CAST(i%2 AS TEXT) FROM n",
      -1, NULL);

  for (int i = 0; i < 20; i++) {
    /* vector: success (multi-batch: APPLY_ROWS rows) */
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','clf'))", APPLY_ROWS, NULL);
    /* introspected model + positional features */
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','bare'))", APPLY_ROWS, NULL);
    /* in_context: success (train context + APPLY_ROWS-row query, multi-batch) */
    fails += run(db, "SELECT * FROM predict_batch('SELECT f1, f2, label FROM tr',"
            "'SELECT id, f1, f2 FROM apply',json_object('model','knn1'))",
            APPLY_ROWS, NULL);
    /* every error branch, both layouts: assert the specific closed-set code */
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','clf','device','banana'))",
            -1, "PREDICT_ERR_OPTIONS");
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','clf','device','cuda'))",
            -1, "PREDICT_ERR_RUNTIME_UNAVAILABLE");
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','clf','precision','fp16'))",
            -1, "PREDICT_ERR_RUNTIME_UNAVAILABLE");
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1 FROM apply',"
            "json_object('model','clf'))", -1, "PREDICT_ERR_SCHEMA");
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','ghost'))",
            -1, "PREDICT_ERR_MODEL_NOT_FOUND");
    /* in_context error branches: no train, missing target, bad label */
    fails += run(db, "SELECT * FROM predict_batch(NULL,'SELECT id, f1, f2 FROM"
            " apply',json_object('model','knn1'))", -1, "PREDICT_ERR_SCHEMA");
    fails += run(db, "SELECT * FROM predict_batch('SELECT f1, f2 FROM tr',"
            "'SELECT id, f1, f2 FROM apply',json_object('model','knn1'))",
            -1, "PREDICT_ERR_SCHEMA");
    /* two-head forecast (aggregate form): reconstruction success +
     * point/interval, then the fail-loud single-output flip declaration */
    if (two_head) {
      fails += run(db, "SELECT forecast(ts, value, 3,"
              "json_object('model','th','confidence_level',0.8)) FROM ser",
              -1, NULL);
      fails += run(db, "SELECT forecast(ts, value, 3,"
              "json_object('model','th')) FROM ser", -1, NULL);
      fails += run(db, "SELECT forecast(ts, value, 3,"
              "json_object('model','badf')) FROM ser",
              -1, "PREDICT_ERR_IO_SPEC");
      /* serve the distilled skip students (TiDE + pure linear) */
      fails += run(db, "SELECT forecast(ts, value, 3,"
              "json_object('model','fs_tide')) FROM ser", -1, NULL);
      fails += run(db, "SELECT forecast(ts, value, 3,"
              "json_object('model','fs_lin','confidence_level',0.8))"
              " FROM ser", -1, NULL);
    }
  }

  sqlite3_close(db);
  if (fails) {
    fprintf(stderr, "onnx soak: %d unexpected statement results\n", fails);
    return 1;
  }
  fprintf(stderr, "onnx soak ok\n");
  return 0;
}
