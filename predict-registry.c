/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* Model registry (_predict_models) and content hashing. RFC 0005 §4.1.1.
 * Registered models are content-addressed: content_hash pins the exact
 * weights and is verified at load, before deserialization (§6.2):
 * inline blobs in predict0_registry_lookup, URI weights in the onnx
 * session builder.
 *
 * One deliberate deviation from the 0005 draft, to feed its revision:
 * model kind 'ts-stat' (the draft's kind vocabulary lacks statistical
 * models). */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#pragma region hashing

void predict0_hash_init(predict0_hasher *h) { sha256_init(&h->sha); }

void predict0_hash_hex(predict0_hasher *h, char hex[PREDICT_HEX_BUFSIZE]) {
  u8 digest[32];
  sha256_final(&h->sha, digest);
  static const char *hexd = "0123456789abcdef";
  for (int i = 0; i < 32; i++) {
    hex[i * 2] = hexd[digest[i] >> 4];
    hex[i * 2 + 1] = hexd[digest[i] & 0xf];
  }
  hex[PREDICT_HEX_BUFSIZE - 1] = 0;
}

#pragma endregion

#pragma region registry

static const char *DDL =
    "CREATE TABLE IF NOT EXISTS _predict_models (\n"
    "  model_id     TEXT PRIMARY KEY,\n"
    "  kind         TEXT NOT NULL CHECK (kind IN\n"
    "    ('ts-fm','ts-stat','tabular-fm','tabular-stat','student')),\n"
    "  runtime      TEXT NOT NULL CHECK (runtime IN\n"
    "    ('onnx','ggml','tree','remote','bundled')),\n"
    "  weights      BLOB,\n"
    "  weights_uri  TEXT,\n"
    "  io_spec      TEXT,\n"
    "  content_hash TEXT NOT NULL,\n"
    "  license      TEXT NOT NULL,\n"
    "  created_at   TEXT NOT NULL\n"
    "    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),\n"
    /* remote/bundled models carry no local weights. Local runtimes carry
     * exactly one source: an inline BLOB (small students, travel with the
     * DB through snapshot/fork) XOR an external URI (large teachers that
     * cannot live in a BLOB). */
    "  CHECK (CASE runtime\n"
    "    WHEN 'remote'  THEN weights IS NULL AND weights_uri IS NULL\n"
    "    WHEN 'bundled' THEN weights IS NULL AND weights_uri IS NULL\n"
    "    ELSE (weights IS NULL) <> (weights_uri IS NULL) END)\n"
    ");\n";

static int bundled_model_row(sqlite3 *db, const char *id, const char *kind) {
  /* content_hash of a bundled model = sha256("bundled:<id>:<version>") */
  predict0_hasher h;
  predict0_hash_init(&h);
  char *desc = sqlite3_mprintf("bundled:%s:%s", id, SQLITE_PREDICT_VERSION);
  if (!desc)
    return SQLITE_NOMEM;
  sha256_update(&h.sha, (const u8 *)desc, strlen(desc));
  sqlite3_free(desc);
  char hex[PREDICT_HEX_BUFSIZE];
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

int predict0_registry_ensure(sqlite3 *db, char **errmsg) {
  char *emsg = NULL;
  int rc = sqlite3_exec(db, DDL, NULL, NULL, &emsg);
  if (rc != SQLITE_OK) {
    *errmsg = sqlite3_mprintf(
        "%s: cannot create the model registry: %s", PREDICT_ERR_RESOURCE,
        emsg ? emsg : "unknown");
    sqlite3_free(emsg);
    return rc;
  }
  /* Idempotent column adds for registries created before the ONNX path.
   * ADD COLUMN errors when the column already exists; that is expected and
   * ignored (the same try/catch pattern used across the engine). */
  sqlite3_exec(db, "ALTER TABLE _predict_models ADD COLUMN weights_uri TEXT",
               NULL, NULL, NULL);
  sqlite3_exec(db, "ALTER TABLE _predict_models ADD COLUMN io_spec TEXT",
               NULL, NULL, NULL);
  rc = bundled_model_row(db, "theta-classic", "ts-stat");
  if (rc == SQLITE_OK)
    rc = bundled_model_row(db, "stub-seasonal-naive", "ts-stat");
  if (rc == SQLITE_OK)
    rc = bundled_model_row(db, "sub-pca", "ts-stat");
  if (rc == SQLITE_OK)
    rc = bundled_model_row(db, "tsb", "ts-stat");
  if (rc == SQLITE_OK)
    /* 'auto' is a meta-model: it selects among the ts-stat models per
     * series by rolling-origin MASE */
    rc = bundled_model_row(db, "auto", "ts-stat");
  if (rc == SQLITE_OK)
    rc = bundled_model_row(db, "knn5-incontext", "tabular-stat");
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

static char *col_dup(sqlite3_stmt *s, int i) {
  if (sqlite3_column_type(s, i) == SQLITE_NULL)
    return NULL;
  return sqlite3_mprintf("%s", (const char *)sqlite3_column_text(s, i));
}

int predict0_registry_lookup(sqlite3 *db, const char *model_id,
                             predict0_model_row *out) {
  memset(out, 0, sizeof(*out));
  sqlite3_stmt *s = NULL;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT runtime, kind, weights_uri, io_spec, license, content_hash,"
      " weights FROM _predict_models WHERE model_id = ?",
      -1, &s, NULL);
  if (rc != SQLITE_OK)
    return rc;
  sqlite3_bind_text(s, 1, model_id, -1, SQLITE_STATIC);
  rc = sqlite3_step(s);
  if (rc != SQLITE_ROW) {
    sqlite3_finalize(s);
    return rc == SQLITE_DONE ? 1 : rc; /* 1 = absent */
  }
  out->runtime = col_dup(s, 0);
  out->kind = col_dup(s, 1);
  out->weights_uri = col_dup(s, 2);
  out->io_spec = col_dup(s, 3);
  out->license = col_dup(s, 4);
  out->content_hash = col_dup(s, 5);
  if (sqlite3_column_type(s, 6) == SQLITE_BLOB) {
    int n = sqlite3_column_bytes(s, 6);
    const void *b = sqlite3_column_blob(s, 6);
    if (n > 0 && b && (out->weights = sqlite3_malloc(n))) {
      memcpy(out->weights, b, n);
      out->weights_len = n;
    }
  }
  sqlite3_finalize(s);
  /* a required-but-missing string is a corrupt row, not a partial hit */
  if (!out->runtime || !out->kind || !out->license || !out->content_hash) {
    predict0_model_row_free(out);
    return SQLITE_CORRUPT;
  }
  /* content-address check (§6.2): inline weights must hash to the
   * registered content_hash, or the row was tampered with/corrupted */
  if (out->weights && out->weights_len > 0) {
    predict0_hasher h;
    predict0_hash_init(&h);
    sha256_update(&h.sha, (const u8 *)out->weights, (usize)out->weights_len);
    char hex[PREDICT_HEX_BUFSIZE];
    predict0_hash_hex(&h, hex);
    if (strcmp(hex, out->content_hash) != 0) {
      predict0_model_row_free(out);
      return 2; /* hash mismatch */
    }
  }
  return 0;
}

void predict0_model_row_free(predict0_model_row *m) {
  if (!m)
    return;
  sqlite3_free(m->runtime);
  sqlite3_free(m->kind);
  sqlite3_free(m->weights_uri);
  sqlite3_free(m->io_spec);
  sqlite3_free(m->license);
  sqlite3_free(m->content_hash);
  sqlite3_free(m->weights);
  memset(m, 0, sizeof(*m));
}

int predict0_hash_file(const char *path, char out[PREDICT_HEX_BUFSIZE],
                       char **errmsg) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    *errmsg = sqlite3_mprintf("%s: cannot open weights file: %s",
                              PREDICT_ERR_RESOURCE, path);
    return SQLITE_ERROR;
  }
  predict0_hasher h;
  predict0_hash_init(&h);
  u8 buf[65536];
  usize n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
    sha256_update(&h.sha, buf, n);
  int ferr = ferror(f);
  fclose(f);
  if (ferr) {
    *errmsg = sqlite3_mprintf("%s: read error on weights file: %s",
                              PREDICT_ERR_RESOURCE, path);
    return SQLITE_ERROR;
  }
  predict0_hash_hex(&h, out);
  return SQLITE_OK;
}

#pragma endregion
