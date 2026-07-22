/* ASan/valgrind soak for the onnx backend. Exercises the vector path and
 * its error branches repeatedly so a sanitizer sees every alloc/free pair
 * in predict-onnx.c. Needs a fixture model path as argv[1] (tests/fixtures/
 * logreg.onnx). Built by `make soak-onnx`; onnxruntime's own still-reachable
 * allocations are filtered by tests/onnx.supp. */
#include "predict-internal.h"

static int run(sqlite3 *db, const char *sql, int expect_ok) {
  sqlite3_stmt *st = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, NULL);
  if (rc != SQLITE_OK)
    return expect_ok ? 1 : 0;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW)
    for (int i = 0; i < sqlite3_column_count(st); i++)
      (void)sqlite3_column_text(st, i);
  sqlite3_finalize(st);
  if (expect_ok && rc != SQLITE_DONE)
    return 1;
  if (!expect_ok && rc == SQLITE_DONE)
    return 1;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <fixture.onnx>\n", argv[0]);
    return 2;
  }
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
  if (run(db, reg, 1)) {
    fprintf(stderr, "register failed\n");
    sqlite3_free(reg);
    sqlite3_close(db);
    return 1;
  }
  sqlite3_free(reg);

  run(db, "CREATE TABLE apply(id INTEGER, f1 REAL, f2 REAL)", 1);
  run(db,
      "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM n WHERE"
      " i < 1500) INSERT INTO apply SELECT i, (i%7)-3.0, (i%5)-2.0 FROM n",
      1);

  for (int i = 0; i < 20; i++) {
    /* success, with and without receipt (multi-batch: 1501 rows) */
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            "json_object('model','clf'))", 1);
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            "json_object('model','clf','receipt',0))", 1);
    /* every error branch */
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            "json_object('model','clf','device','banana'))", 0);
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            "json_object('model','clf','device','cuda'))", 0);
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            "json_object('model','clf','precision','fp16'))", 0);
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1 FROM apply',"
            "json_object('model','clf'))", 0);
    run(db, "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            "json_object('model','ghost'))", 0);
  }

  sqlite3_close(db);
  fprintf(stderr, "onnx soak ok\n");
  return 0;
}
