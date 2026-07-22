/* Model registry, receipts, canonical hashing, logical-digest anchors,
 * and the predict_replay() table-valued function. RFC 0005 §4.1, §4.2.6.
 *
 * Two deliberate deviations from the 0005 draft, to feed its revision:
 * model kind 'ts-stat' (the draft's kind vocabulary lacks statistical
 * models), and anchor_kind 'logical-digest' (a page-level file digest
 * can never replay-match, because writing the receipt itself changes
 * the file; the logical digest hashes user tables' schema+rows and is
 * indifferent to _predict_% writes and vacuum). */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#pragma region hashing

void predict0_hash_init(predict0_hasher *h) { sha256_init(&h->sha); }

static void hash_sep(predict0_hasher *h) {
  static const u8 fs = 0x1f;
  sha256_update(&h->sha, &fs, 1);
}

void predict0_hash_null(predict0_hasher *h) {
  sha256_update(&h->sha, (const u8 *)"n", 1);
  hash_sep(h);
}

void predict0_hash_int(predict0_hasher *h, i64 v) {
  char buf[24];
  int n = snprintf(buf, sizeof(buf), "i%lld", (long long)v);
  sha256_update(&h->sha, (const u8 *)buf, (usize)n);
  hash_sep(h);
}

void predict0_hash_real(predict0_hasher *h, f64 v) {
  u64 bits;
  memcpy(&bits, &v, 8);
  u8 buf[9];
  buf[0] = 'r';
  for (int i = 0; i < 8; i++)
    buf[1 + i] = (u8)(bits >> (8 * (7 - i))); /* big-endian bit pattern */
  sha256_update(&h->sha, buf, 9);
  hash_sep(h);
}

void predict0_hash_text(predict0_hasher *h, const char *s) {
  sha256_update(&h->sha, (const u8 *)"t", 1);
  if (s)
    sha256_update(&h->sha, (const u8 *)s, strlen(s));
  hash_sep(h);
}

void predict0_hash_row_end(predict0_hasher *h) {
  static const u8 rs = 0x1e;
  sha256_update(&h->sha, &rs, 1);
}

void predict0_hash_hex(predict0_hasher *h, char hex[65]) {
  u8 digest[32];
  sha256_final(&h->sha, digest);
  static const char *hexd = "0123456789abcdef";
  for (int i = 0; i < 32; i++) {
    hex[i * 2] = hexd[digest[i] >> 4];
    hex[i * 2 + 1] = hexd[digest[i] & 0xf];
  }
  hex[64] = '\0';
}

static void hash_column_value(predict0_hasher *h, sqlite3_stmt *stmt,
                              int col) {
  switch (sqlite3_column_type(stmt, col)) {
  case SQLITE_NULL:
    predict0_hash_null(h);
    break;
  case SQLITE_INTEGER:
    predict0_hash_int(h, sqlite3_column_int64(stmt, col));
    break;
  case SQLITE_FLOAT:
    predict0_hash_real(h, sqlite3_column_double(stmt, col));
    break;
  default:
    predict0_hash_text(h, (const char *)sqlite3_column_text(stmt, col));
    break;
  }
}

#pragma endregion

#pragma region registry

static const char *DDL =
    "CREATE TABLE IF NOT EXISTS _predict_models (\n"
    "  model_id     TEXT PRIMARY KEY,\n"
    "  kind         TEXT NOT NULL CHECK (kind IN\n"
    "    ('ts-fm','ts-stat','tabular-fm','student','embedding')),\n"
    "  runtime      TEXT NOT NULL CHECK (runtime IN\n"
    "    ('onnx','ggml','tree','remote','bundled')),\n"
    "  weights      BLOB,\n"
    "  content_hash TEXT NOT NULL,\n"
    "  license      TEXT NOT NULL,\n"
    "  created_at   TEXT NOT NULL\n"
    "    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),\n"
    "  CHECK ((weights IS NULL) = (runtime IN ('remote','bundled')))\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS _predict_receipts (\n"
    "  receipt_id  TEXT PRIMARY KEY,\n"
    "  operation   TEXT NOT NULL CHECK (operation IN\n"
    "    ('forecast','detect_anomalies','predict','find_similar','distill')),\n"
    "  model_id    TEXT NOT NULL,\n"
    "  model_hash  TEXT NOT NULL,\n"
    "  anchor_kind TEXT NOT NULL CHECK (anchor_kind IN\n"
    "    ('branch','generation','file-digest','logical-digest','none')),\n"
    "  anchor      TEXT,\n"
    "  params      TEXT NOT NULL,\n"
    "  input_sql   TEXT NOT NULL,\n"
    "  result_hash TEXT NOT NULL,\n"
    "  created_at  TEXT NOT NULL\n"
    "    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),\n"
    "  CHECK ((anchor_kind = 'none') = (anchor IS NULL))\n"
    ");\n"
    "CREATE INDEX IF NOT EXISTS idx_predict_receipts_model\n"
    "  ON _predict_receipts(model_id);\n";

static int bundled_model_row(sqlite3 *db, const char *id, const char *kind) {
  /* content_hash of a bundled model = sha256("bundled:<id>:<version>") */
  predict0_hasher h;
  predict0_hash_init(&h);
  char *desc = sqlite3_mprintf("bundled:%s:%s", id, SQLITE_PREDICT_VERSION);
  if (!desc)
    return SQLITE_NOMEM;
  sha256_update(&h.sha, (const u8 *)desc, strlen(desc));
  sqlite3_free(desc);
  char hex[65];
  predict0_hash_hex(&h, hex);

  char *sql = sqlite3_mprintf(
      "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
      " content_hash, license) VALUES (%Q, %Q, 'bundled', NULL, %Q,"
      " 'MIT OR Apache-2.0') ON CONFLICT(model_id) DO NOTHING",
      id, kind, hex);
  if (!sql)
    return SQLITE_NOMEM;
  int rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
  sqlite3_free(sql);
  return rc;
}

int predict0_receipts_ensure(sqlite3 *db, char **errmsg) {
  char *emsg = NULL;
  int rc = sqlite3_exec(db, DDL, NULL, NULL, &emsg);
  if (rc != SQLITE_OK) {
    *errmsg = sqlite3_mprintf(
        "%s: cannot create receipt tables (%s); pass receipt: 0 on"
        " read-only databases",
        PREDICT_ERR_RESOURCE, emsg ? emsg : "unknown");
    sqlite3_free(emsg);
    return rc;
  }
  rc = bundled_model_row(db, "theta-classic", "ts-stat");
  if (rc == SQLITE_OK)
    rc = bundled_model_row(db, "stub-seasonal-naive", "ts-stat");
  if (rc != SQLITE_OK)
    *errmsg = sqlite3_mprintf("%s: cannot register bundled models",
                              PREDICT_ERR_RESOURCE);
  return rc;
}

char *predict0_registry_model_hash(sqlite3 *db, const char *model_id) {
  sqlite3_stmt *stmt = NULL;
  char *out = NULL;
  if (sqlite3_prepare_v2(
          db, "SELECT content_hash FROM _predict_models WHERE model_id = ?",
          -1, &stmt, NULL) != SQLITE_OK)
    return NULL;
  sqlite3_bind_text(stmt, 1, model_id, -1, SQLITE_STATIC);
  if (sqlite3_step(stmt) == SQLITE_ROW)
    out = sqlite3_mprintf("%s",
                          (const char *)sqlite3_column_text(stmt, 0));
  sqlite3_finalize(stmt);
  return out;
}

#pragma endregion

#pragma region digest

int predict0_logical_digest(sqlite3 *db, char out[65], char **errmsg) {
  predict0_hasher h;
  predict0_hash_init(&h);

  sqlite3_stmt *tables = NULL;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT name, sql FROM sqlite_master WHERE type='table'"
      " AND name NOT LIKE '\\_predict\\_%' ESCAPE '\\'"
      " AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name",
      -1, &tables, NULL);
  if (rc != SQLITE_OK) {
    *errmsg = sqlite3_mprintf("%s: digest failed (%s)", PREDICT_ERR_RESOURCE,
                              sqlite3_errmsg(db));
    return rc;
  }

  while ((rc = sqlite3_step(tables)) == SQLITE_ROW) {
    const char *name = (const char *)sqlite3_column_text(tables, 0);
    predict0_hash_text(&h, name);
    predict0_hash_text(&h, (const char *)sqlite3_column_text(tables, 1));
    predict0_hash_row_end(&h);

    char *sql = sqlite3_mprintf("SELECT * FROM \"%w\" ORDER BY rowid", name);
    if (!sql) {
      sqlite3_finalize(tables);
      return SQLITE_NOMEM;
    }
    sqlite3_stmt *rows = NULL;
    rc = sqlite3_prepare_v2(db, sql, -1, &rows, NULL);
    if (rc != SQLITE_OK) {
      /* WITHOUT ROWID tables: fall back to natural order (their PK) */
      char *sql2 = sqlite3_mprintf("SELECT * FROM \"%w\"", name);
      sqlite3_free(sql);
      sql = sql2;
      rc = sql ? sqlite3_prepare_v2(db, sql, -1, &rows, NULL) : SQLITE_NOMEM;
    }
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
      sqlite3_finalize(tables);
      *errmsg = sqlite3_mprintf("%s: digest failed on table %s",
                                PREDICT_ERR_RESOURCE, name);
      return rc;
    }
    int ncol = sqlite3_column_count(rows);
    while ((rc = sqlite3_step(rows)) == SQLITE_ROW) {
      for (int i = 0; i < ncol; i++)
        hash_column_value(&h, rows, i);
      predict0_hash_row_end(&h);
    }
    sqlite3_finalize(rows);
    if (rc != SQLITE_DONE) {
      sqlite3_finalize(tables);
      *errmsg = sqlite3_mprintf("%s: digest scan failed on table %s",
                                PREDICT_ERR_RESOURCE, name);
      return rc;
    }
  }
  sqlite3_finalize(tables);
  predict0_hash_hex(&h, out);
  return SQLITE_OK;
}

#pragma endregion

#pragma region receipt insert

int predict0_receipt_insert(sqlite3 *db, const char *operation,
                            const char *model_id, const char *model_hash,
                            const char *anchor_kind, const char *anchor,
                            const char *params, const char *input_sql,
                            const char *result_hash, char receipt_id_out[27],
                            char **errmsg) {
  i64 now_ms = 0;
  sqlite3_stmt *now = NULL;
  if (sqlite3_prepare_v2(db,
                         "SELECT CAST((julianday('now') - 2440587.5)"
                         " * 86400000 AS INTEGER)",
                         -1, &now, NULL) == SQLITE_OK) {
    if (sqlite3_step(now) == SQLITE_ROW)
      now_ms = sqlite3_column_int64(now, 0);
    sqlite3_finalize(now);
  }
  predict0_ulid_new(now_ms, receipt_id_out);

  sqlite3_stmt *ins = NULL;
  int rc = sqlite3_prepare_v2(
      db,
      "INSERT INTO _predict_receipts (receipt_id, operation, model_id,"
      " model_hash, anchor_kind, anchor, params, input_sql, result_hash)"
      " VALUES (?,?,?,?,?,?,?,?,?)",
      -1, &ins, NULL);
  if (rc != SQLITE_OK) {
    *errmsg = sqlite3_mprintf("%s: receipt insert failed (%s)",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
    return rc;
  }
  sqlite3_bind_text(ins, 1, receipt_id_out, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 2, operation, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 3, model_id, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 4, model_hash, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 5, anchor_kind, -1, SQLITE_STATIC);
  if (anchor)
    sqlite3_bind_text(ins, 6, anchor, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 7, params, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 8, input_sql, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 9, result_hash, -1, SQLITE_STATIC);
  rc = sqlite3_step(ins);
  sqlite3_finalize(ins);
  if (rc != SQLITE_DONE) {
    *errmsg = sqlite3_mprintf("%s: receipt insert failed (%s)",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

#pragma endregion

#pragma region replay

/* predict_replay(receipt_id): RETURNS TABLE
 * (match INTEGER, result_hash TEXT, original_hash TEXT, detail TEXT) */

#define RP_COL_MATCH 0
#define RP_COL_HASH 1
#define RP_COL_ORIG 2
#define RP_COL_DETAIL 3
#define RP_COL_RECEIPT 4

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} replay_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  int done;
  int match;
  char hash[65];
  char *orig;
  char *detail;
} replay_cursor;

static int rp_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  replay_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(match INTEGER, result_hash TEXT,"
          " original_hash TEXT, detail TEXT, receipt_id HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int rp_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int rp_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->iColumn == RP_COL_RECEIPT &&
        c->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen = 1;
    }
  }
  if (!seen) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: predict_replay(receipt_id) requires a receipt id",
        PREDICT_ERR_RECEIPT_NOT_FOUND);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int rp_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  replay_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static int rp_close(sqlite3_vtab_cursor *pCur) {
  replay_cursor *c = (replay_cursor *)pCur;
  sqlite3_free(c->orig);
  sqlite3_free(c->detail);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int rp_error(replay_cursor *cur, const char *code,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg = sqlite3_mprintf("%s: %s", code, detail);
  return SQLITE_ERROR;
}

/* rows fetched during replay, sorted for hashing */
typedef struct {
  char *key;
  int has_step;
  i64 step;
  char *ts;
  int has_vals;
  f64 fc, lo, hi;
} ReplayRow;

static int replay_row_cmp(const void *a, const void *b) {
  const ReplayRow *x = a, *y = b;
  int c = strcmp(x->key ? x->key : "", y->key ? y->key : "");
  if (c)
    return c;
  return x->step < y->step ? -1 : x->step > y->step ? 1 : 0;
}

static int rp_filter(sqlite3_vtab_cursor *pCur, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  replay_cursor *cur = (replay_cursor *)pCur;
  replay_vtab *vtab = (replay_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  cur->done = 0;

  if (argc < 1)
    return rp_error(cur, PREDICT_ERR_RECEIPT_NOT_FOUND, "missing receipt id");
  const char *rid = (const char *)sqlite3_value_text(argv[0]);

  /* load the receipt */
  sqlite3_stmt *get = NULL;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT operation, model_id, model_hash, anchor_kind, anchor, params,"
      " input_sql, result_hash FROM _predict_receipts WHERE receipt_id = ?",
      -1, &get, NULL);
  if (rc != SQLITE_OK)
    return rp_error(cur, PREDICT_ERR_RECEIPT_NOT_FOUND,
                    "no receipt tables in this database");
  sqlite3_bind_text(get, 1, rid, -1, SQLITE_STATIC);
  if (sqlite3_step(get) != SQLITE_ROW) {
    sqlite3_finalize(get);
    return rp_error(cur, PREDICT_ERR_RECEIPT_NOT_FOUND,
                    rid ? rid : "(null)");
  }
  char *operation =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 0));
  char *model_id =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 1));
  char *anchor_kind =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 3));
  char *anchor =
      sqlite3_column_type(get, 4) == SQLITE_NULL
          ? NULL
          : sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 4));
  char *params =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 5));
  char *input_sql =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 6));
  char *orig_hash =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(get, 7));
  sqlite3_finalize(get);

#define RP_CLEANUP()                                                          \
  do {                                                                        \
    sqlite3_free(operation);                                                  \
    sqlite3_free(model_id);                                                   \
    sqlite3_free(anchor_kind);                                                \
    sqlite3_free(anchor);                                                     \
    sqlite3_free(params);                                                     \
    sqlite3_free(input_sql);                                                  \
    sqlite3_free(orig_hash);                                                  \
  } while (0)

  if (strcmp(operation, "forecast") != 0) {
    int r = rp_error(cur, PREDICT_ERR_ANCHOR_UNAVAILABLE,
                     "replay implemented for forecast only in this build");
    RP_CLEANUP();
    return r;
  }

  /* model must still exist with the same hash */
  char *cur_hash = predict0_registry_model_hash(db, model_id);
  if (!cur_hash) {
    int r = rp_error(cur, PREDICT_ERR_MODEL_NOT_FOUND, model_id);
    RP_CLEANUP();
    return r;
  }
  sqlite3_free(cur_hash);

  /* the anchored state must be the current state */
  if (strcmp(anchor_kind, "none") == 0) {
    int r = rp_error(cur, PREDICT_ERR_ANCHOR_UNAVAILABLE,
                     "receipt was recorded without an anchor");
    RP_CLEANUP();
    return r;
  }
  char digest[65];
  char *errmsg = NULL;
  if (predict0_logical_digest(db, digest, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    RP_CLEANUP();
    return SQLITE_ERROR;
  }
  if (!anchor || strcmp(digest, anchor) != 0) {
    int r = rp_error(cur, PREDICT_ERR_ANCHOR_UNAVAILABLE,
                     "database state no longer matches the receipt anchor");
    RP_CLEANUP();
    return r;
  }

  /* re-run through the public surface with receipts off */
  sqlite3_stmt *run = NULL;
  rc = sqlite3_prepare_v2(
      db,
      "SELECT series_key, step, forecast_timestamp, forecast, lower_bound,"
      " upper_bound FROM forecast(?,"
      " CAST(json_extract(?, '$.horizon') AS INTEGER),"
      " json_set(json_remove(?, '$.horizon'), '$.receipt', 0))",
      -1, &run, NULL);
  if (rc != SQLITE_OK) {
    int r = rp_error(cur, PREDICT_ERR_REPLAY_MISMATCH,
                     "could not re-prepare the forecast");
    RP_CLEANUP();
    return r;
  }
  sqlite3_bind_text(run, 1, input_sql, -1, SQLITE_STATIC);
  sqlite3_bind_text(run, 2, params, -1, SQLITE_STATIC);
  sqlite3_bind_text(run, 3, params, -1, SQLITE_STATIC);

  ReplayRow *rows = NULL;
  int n = 0, cap = 0, oom = 0;
  while ((rc = sqlite3_step(run)) == SQLITE_ROW) {
    if (n == cap) {
      cap = cap ? cap * 2 : 64;
      ReplayRow *g = sqlite3_realloc(rows, sizeof(ReplayRow) * cap);
      if (!g) {
        oom = 1;
        break;
      }
      rows = g;
    }
    ReplayRow *r = &rows[n++];
    memset(r, 0, sizeof(*r));
    r->key = sqlite3_mprintf(
        "%s", (const char *)sqlite3_column_text(run, 0));
    r->has_step = sqlite3_column_type(run, 1) != SQLITE_NULL;
    r->step = sqlite3_column_int64(run, 1);
    r->ts = sqlite3_column_type(run, 2) == SQLITE_NULL
                ? NULL
                : sqlite3_mprintf(
                      "%s", (const char *)sqlite3_column_text(run, 2));
    r->has_vals = sqlite3_column_type(run, 3) != SQLITE_NULL;
    r->fc = sqlite3_column_double(run, 3);
    r->lo = sqlite3_column_double(run, 4);
    r->hi = sqlite3_column_double(run, 5);
  }
  int run_rc = rc;
  sqlite3_finalize(run);

  if (oom || (run_rc != SQLITE_DONE)) {
    for (int i = 0; i < n; i++) {
      sqlite3_free(rows[i].key);
      sqlite3_free(rows[i].ts);
    }
    sqlite3_free(rows);
    int r = rp_error(cur, PREDICT_ERR_REPLAY_MISMATCH,
                     oom ? "out of memory" : sqlite3_errmsg(db));
    RP_CLEANUP();
    return r;
  }

  qsort(rows, (usize)n, sizeof(ReplayRow), replay_row_cmp);
  predict0_hasher h;
  predict0_hash_init(&h);
  for (int i = 0; i < n; i++) {
    ReplayRow *r = &rows[i];
    predict0_hash_text(&h, r->key);
    if (r->has_step)
      predict0_hash_int(&h, r->step);
    else
      predict0_hash_null(&h);
    if (r->ts)
      predict0_hash_text(&h, r->ts);
    else
      predict0_hash_null(&h);
    if (r->has_vals) {
      predict0_hash_real(&h, r->fc);
      predict0_hash_real(&h, r->lo);
      predict0_hash_real(&h, r->hi);
    } else {
      predict0_hash_null(&h);
      predict0_hash_null(&h);
      predict0_hash_null(&h);
    }
    predict0_hash_row_end(&h);
  }
  predict0_hash_hex(&h, cur->hash);

  for (int i = 0; i < n; i++) {
    sqlite3_free(rows[i].key);
    sqlite3_free(rows[i].ts);
  }
  sqlite3_free(rows);

  cur->match = strcmp(cur->hash, orig_hash) == 0;
  sqlite3_free(cur->orig);
  cur->orig = sqlite3_mprintf("%s", orig_hash);
  sqlite3_free(cur->detail);
  cur->detail = cur->match
                    ? sqlite3_mprintf("reproduced (%d rows)", n)
                    : sqlite3_mprintf("result hash diverged over %d rows", n);
  RP_CLEANUP();
#undef RP_CLEANUP
  return SQLITE_OK;
}

static int rp_next(sqlite3_vtab_cursor *pCur) {
  ((replay_cursor *)pCur)->done = 1;
  return SQLITE_OK;
}

static int rp_eof(sqlite3_vtab_cursor *pCur) {
  return ((replay_cursor *)pCur)->done;
}

static int rp_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  replay_cursor *c = (replay_cursor *)pCur;
  switch (col) {
  case RP_COL_MATCH:
    sqlite3_result_int(ctx, c->match);
    break;
  case RP_COL_HASH:
    sqlite3_result_text(ctx, c->hash, -1, SQLITE_TRANSIENT);
    break;
  case RP_COL_ORIG:
    sqlite3_result_text(ctx, c->orig ? c->orig : "", -1, SQLITE_TRANSIENT);
    break;
  case RP_COL_DETAIL:
    sqlite3_result_text(ctx, c->detail ? c->detail : "", -1,
                        SQLITE_TRANSIENT);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int rp_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  UNUSED_PARAMETER(pCur);
  *pRowid = 0;
  return SQLITE_OK;
}

static sqlite3_module replayModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ rp_connect,
    /* xBestIndex  */ rp_best_index,
    /* xDisconnect */ rp_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ rp_open,
    /* xClose      */ rp_close,
    /* xFilter     */ rp_filter,
    /* xNext       */ rp_next,
    /* xEof        */ rp_eof,
    /* xColumn     */ rp_column,
    /* xRowid      */ rp_rowid,
    /* xUpdate     */ NULL,
    /* xBegin      */ NULL,
    /* xSync       */ NULL,
    /* xCommit     */ NULL,
    /* xRollback   */ NULL,
    /* xFindMethod */ NULL,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ NULL,
    /* xIntegrity  */ NULL};

#pragma endregion

int predict0_receipts_init(sqlite3 *db) {
  return sqlite3_create_module(db, "predict_replay", &replayModule, NULL);
}
