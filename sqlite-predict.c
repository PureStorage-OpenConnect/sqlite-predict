/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT1
#endif

#pragma region meta

static void predict_version_fn(sqlite3_context *context, int argc,
                               sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  char *json = sqlite3_mprintf(
      "{\"extension\":\"%s\",\"spec\":\"%s\",\"runtimes\":[\"stat\",\"tree\""
#ifdef SQLITE_PREDICT_ONNX
      ",\"onnx\""
#endif
      /* the bundled models every build serves (§4.2.8: registered
       * aliases); user-registered models live in _predict_models and are
       * per-database, so a pure version function cannot list them */
      "],\"models\":[\"auto\",\"theta-classic\",\"stub-seasonal-naive\","
      "\"tsb\",\"sub-pca\",\"knn5-incontext\"]}",
      SQLITE_PREDICT_VERSION, SQLITE_PREDICT_SPEC);
  if (!json) {
    sqlite3_result_error_nomem(context);
    return;
  }
  sqlite3_result_text(context, json, -1, sqlite3_free);
  sqlite3_result_subtype(context, 'J'); /* JSON, per json1 convention */
}

static void predict_debug_fn(sqlite3_context *context, int argc,
                             sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  char *debug =
      sqlite3_mprintf("Version: %s\nDate: %s\nCommit: %s", SQLITE_PREDICT_VERSION,
                      SQLITE_PREDICT_DATE, SQLITE_PREDICT_SOURCE);
  if (!debug) {
    sqlite3_result_error_nomem(context);
    return;
  }
  sqlite3_result_text(context, debug, -1, sqlite3_free);
}

#pragma endregion

#pragma region helpers

static int days_in_month(int y, int m) {
  static const int d[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0))
    return 29;
  return d[m - 1];
}

/* Days from 1970-01-01 to a proleptic-Gregorian date, and its inverse
 * (Howard Hinnant's branch-free O(1) algorithms). Correct for the whole
 * 0001..9999 range; replaces the old O(years) accumulation loops that
 * turned a large epoch into a 294k-iteration DoS. */
static i64 days_from_civil(i64 y, unsigned m, unsigned d) {
  y -= m <= 2;
  i64 era = (y >= 0 ? y : y - 399) / 400;
  unsigned yoe = (unsigned)(y - era * 400);
  unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (i64)doe - 719468;
}

static void civil_from_days(i64 z, int *y, int *m, int *d) {
  z += 719468;
  i64 era = (z >= 0 ? z : z - 146096) / 146097;
  unsigned doe = (unsigned)(z - era * 146097);
  unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  i64 yy = (i64)yoe + era * 400;
  unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned mp = (5 * doy + 2) / 153;
  *d = (int)(doy - (153 * mp + 2) / 5 + 1);
  *m = (int)(mp < 10 ? mp + 3 : mp - 9);
  *y = (int)(yy + (*m <= 2));
}

/* Read min_d..max_d ASCII digits (no sign, no whitespace) at *pos,
 * advancing *pos. Returns the value, or -1 if fewer than min_d digits
 * are present. */
static int read_digits(const char *s, int *pos, int min_d, int max_d) {
  int v = 0, got = 0;
  while (got < max_d && s[*pos] >= '0' && s[*pos] <= '9') {
    v = v * 10 + (s[*pos] - '0');
    (*pos)++;
    got++;
  }
  return got >= min_d ? v : -1;
}

int predict0_parse_timestamp(const char *s, i64 *out_ms) {
  if (!s)
    return 1;
  int p = 0;
  int y = read_digits(s, &p, 1, 4);
  if (y < 1 || s[p] != '-')
    return 1;
  p++;
  int mo = read_digits(s, &p, 1, 2);
  if (mo < 1 || s[p] != '-')
    return 1;
  p++;
  int d = read_digits(s, &p, 1, 2);
  if (d < 1)
    return 1;

  int h = 0, mi = 0, sec = 0;
  if (s[p] == 'T' || s[p] == ' ') {
    p++;
    h = read_digits(s, &p, 1, 2);
    if (h < 0 || s[p] != ':')
      return 1;
    p++;
    mi = read_digits(s, &p, 1, 2);
    if (mi < 0)
      return 1;
    if (s[p] == ':') {
      p++;
      sec = read_digits(s, &p, 1, 2);
      if (sec < 0)
        return 1;
      if (s[p] == '.') { /* fractional seconds: parsed, not retained */
        p++;
        if (read_digits(s, &p, 1, 9) < 0)
          return 1;
      }
    }
  }
  if (s[p] == 'Z')
    p++;
  if (s[p] != '\0')
    return 1;
  if (y > 9999 || mo > 12 || d > days_in_month(y, mo) || h > 23 ||
      mi > 59 || sec > 59)
    return 1;

  i64 days = days_from_civil(y, (unsigned)mo, (unsigned)d);
  *out_ms = ((days * 24 + h) * 60 + mi) * 60 * 1000 + (i64)sec * 1000;
  return 0;
}

void predict0_format_timestamp(i64 ms, char *buf, usize bufsize) {
  if (ms < PREDICT_MS_MIN)
    ms = PREDICT_MS_MIN; /* clamp keeps the year in 0001..9999, so the */
  if (ms > PREDICT_MS_MAX)
    ms = PREDICT_MS_MAX; /* output is always 20 chars and never overflows */
  i64 secs = ms / 1000;
  i64 days = secs / 86400;
  int rem = (int)(secs - days * 86400); /* floored: rem always in [0,86400) */
  if (rem < 0) {
    rem += 86400;
    days -= 1;
  }
  int y, mo, d;
  civil_from_days(days, &y, &mo, &d);
  snprintf(buf, bufsize, "%04d-%02d-%02dT%02d:%02d:%02dZ", y, mo, d,
           rem / 3600, (rem / 60) % 60, rem % 60);
}

/* Parse a JSON array of strings into a heap array of sqlite3_mprintf'd
 * strings (caller frees each element and the array). path selects an
 * array inside a larger document ("$.features"); NULL means json itself
 * is the array (validated as one). Returns SQLITE_OK and sets out/n; on
 * failure returns an SQLITE_ code with *errmsg set. */
int predict0_json_str_array(sqlite3 *db, const char *json, const char *path,
                            char ***out, int *n, char **errmsg) {
  *out = NULL;
  *n = 0;
  int cap = 0;
  sqlite3_stmt *st = NULL;
  const char *sql = path ? "SELECT value FROM json_each(?1, ?2)"
                         : "SELECT value FROM json_each(?1) WHERE"
                           " json_type(?1) = 'array'";
  if (sqlite3_prepare_v2(db, sql, -1, &st, NULL) != SQLITE_OK) {
    *errmsg =
        sqlite3_mprintf("%s: cannot parse option array", PREDICT_ERR_OPTIONS);
    return SQLITE_ERROR;
  }
  sqlite3_bind_text(st, 1, json, -1, SQLITE_STATIC);
  if (path)
    sqlite3_bind_text(st, 2, path, -1, SQLITE_STATIC);
  int rc = SQLITE_OK;
  while (sqlite3_step(st) == SQLITE_ROW) {
    if (*n == cap) {
      cap = cap ? cap * 2 : 8;
      char **g = sqlite3_realloc(*out, sizeof(char *) * cap);
      if (!g) {
        rc = SQLITE_NOMEM;
        break;
      }
      *out = g;
    }
    (*out)[*n] =
        sqlite3_mprintf("%s", (const char *)sqlite3_column_text(st, 0));
    if (!(*out)[*n]) {
      rc = SQLITE_NOMEM;
      break;
    }
    (*n)++;
  }
  sqlite3_finalize(st);
  if (rc == SQLITE_NOMEM)
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  return rc;
}

/* Prepare a caller-supplied inner query with the safety contract every
 * operation shares (§6.1): it must parse, be a single statement, and be
 * read-only. what names the argument in error messages ("train_query").
 * Returns SQLITE_OK with *out set, or SQLITE_ERROR with *errmsg set. */
int predict0_prepare_ro(sqlite3 *db, const char *sql, const char *what,
                        sqlite3_stmt **out, char **errmsg) {
  *out = NULL;
  if (!sql) {
    *errmsg =
        sqlite3_mprintf("%s: %s must be text", PREDICT_ERR_SCHEMA, what);
    return SQLITE_ERROR;
  }
  sqlite3_stmt *stmt = NULL;
  const char *tail = NULL;
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, &tail) != SQLITE_OK || !stmt) {
    if (stmt)
      sqlite3_finalize(stmt);
    *errmsg = sqlite3_mprintf("%s: %s does not parse: %s", PREDICT_ERR_SCHEMA,
                              what, sqlite3_errmsg(db));
    return SQLITE_ERROR;
  }
  while (tail &&
         (*tail == ' ' || *tail == '\n' || *tail == ';' || *tail == '\t'))
    tail++;
  if (tail && *tail != '\0') {
    sqlite3_finalize(stmt);
    *errmsg = sqlite3_mprintf("%s: %s must be a single statement",
                              PREDICT_ERR_SCHEMA, what);
    return SQLITE_ERROR;
  }
  if (!sqlite3_stmt_readonly(stmt)) {
    sqlite3_finalize(stmt);
    *errmsg = sqlite3_mprintf("%s: %s must be a read-only SELECT",
                              PREDICT_ERR_QUERY_NOT_READONLY, what);
    return SQLITE_ERROR;
  }
  *out = stmt;
  return SQLITE_OK;
}

/* Free a runtime backend's neutral result array. */
void predict0_results_free(predict0_result *rows, int n) {
  if (!rows)
    return;
  for (int i = 0; i < n; i++) {
    sqlite3_free(rows[i].ref_t);
    sqlite3_free(rows[i].prediction);
  }
  sqlite3_free(rows);
}

f64 predict0_norm_quantile(f64 p) {
  /* Acklam's inverse normal CDF approximation, |relative error| < 1.15e-9 */
  static const f64 a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                          -2.759285104469687e+02, 1.383577518672690e+02,
                          -3.066479806614716e+01, 2.506628277459239e+00};
  static const f64 b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                          -1.556989798598866e+02, 6.680131188771972e+01,
                          -1.328068155288572e+01};
  static const f64 c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                          -2.400758277161838e+00, -2.549732539343734e+00,
                          4.374664141464968e+00,  2.938163982698783e+00};
  static const f64 dd[] = {7.784695709041462e-03, 3.224671290700398e-01,
                           2.445134137142996e+00, 3.754408661907416e+00};
  const f64 plow = 0.02425, phigh = 1 - plow;
  f64 q, r;
  if (p < plow) {
    q = sqrt(-2 * log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
            c[5]) /
           ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1);
  }
  if (p <= phigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
            a[5]) *
           q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = sqrt(-2 * log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
           c[5]) /
         ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1);
}

int predict0_options_parse(sqlite3 *db, const char *json,
                           const char *const *keys, predict0_option_cb cb,
                           void *ctx, char **errmsg) {
  if (!json || json[0] == '\0')
    return 0;

  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(
      db, "SELECT key, value FROM json_each(?) WHERE json_type(?) = 'object'",
      -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    *errmsg = sqlite3_mprintf("%s: options parsing unavailable: %s",
                              PREDICT_ERR_OPTIONS, sqlite3_errmsg(db));
    return 1;
  }
  sqlite3_bind_text(stmt, 1, json, -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt, 2, json, -1, SQLITE_STATIC);

  int seen_any = 0;
  for (;;) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE)
      break;
    if (rc != SQLITE_ROW) {
      sqlite3_finalize(stmt);
      *errmsg = sqlite3_mprintf("%s: options is not valid JSON",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    seen_any = 1;
    const char *key = (const char *)sqlite3_column_text(stmt, 0);
    int known = 0;
    for (const char *const *k = keys; *k; k++) {
      if (strcmp(*k, key) == 0) {
        known = 1;
        break;
      }
    }
    if (!known) {
      *errmsg = sqlite3_mprintf("%s: unknown option '%s'", PREDICT_ERR_OPTIONS,
                                key);
      sqlite3_finalize(stmt);
      return 1;
    }
    /* a null value is equivalent to omitting the key */
    if (sqlite3_column_type(stmt, 1) == SQLITE_NULL)
      continue;
    if (cb(ctx, key, sqlite3_column_value(stmt, 1), errmsg)) {
      sqlite3_finalize(stmt);
      return 1;
    }
  }
  sqlite3_finalize(stmt);

  if (!seen_any) {
    /* zero rows: either an empty object (fine) or not an object at all */
    sqlite3_stmt *check = NULL;
    rc = sqlite3_prepare_v2(db, "SELECT json_type(?)", -1, &check, NULL);
    if (rc == SQLITE_OK) {
      sqlite3_bind_text(check, 1, json, -1, SQLITE_STATIC);
      int bad = 1;
      if (sqlite3_step(check) == SQLITE_ROW) {
        const char *t = (const char *)sqlite3_column_text(check, 0);
        bad = !(t && strcmp(t, "object") == 0);
      }
      sqlite3_finalize(check);
      if (bad) {
        *errmsg = sqlite3_mprintf("%s: options must be a JSON object",
                                  PREDICT_ERR_OPTIONS);
        return 1;
      }
    }
  }
  return 0;
}

static const char B32[] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

static void ulid_encode(i64 ms, const u8 rand16[10], char *buf) {
  u8 bytes[16];
  for (int i = 0; i < 6; i++)
    bytes[i] = (u8)(ms >> (8 * (5 - i)));
  memcpy(bytes + 6, rand16, 10);
  /* 16 bytes = 128 bits -> 26 base32 chars (2 leading bits zero-padded) */
  int bit = -2; /* start offset so 26*5 = 130 covers 128 bits */
  for (int i = 0; i < 26; i++) {
    int v = 0;
    for (int j = 0; j < 5; j++) {
      int idx = bit + j;
      int byte = idx >> 3;
      v <<= 1;
      if (idx >= 0 && byte < 16)
        v |= (bytes[byte] >> (7 - (idx & 7))) & 1;
    }
    buf[i] = B32[v];
    bit += 5;
  }
  buf[26] = '\0';
}

static void predict0_ulid_min(i64 ms, char *buf) {
  u8 zeros[10] = {0};
  ulid_encode(ms, zeros, buf);
}

static void predict_ulid_fn(sqlite3_context *context, int argc,
                            sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  const char *ts = (const char *)sqlite3_value_text(argv[0]);
  i64 ms = 0;
  if (predict0_parse_timestamp(ts, &ms)) {
    char *e = sqlite3_mprintf("%s: unparseable timestamp '%s'",
                              PREDICT_ERR_SCHEMA, ts ? ts : "(null)");
    sqlite3_result_error(context, e, -1);
    sqlite3_free(e);
    return;
  }
  char buf[PREDICT_ULID_BUFSIZE];
  predict0_ulid_min(ms, buf);
  sqlite3_result_text(context, buf, 26, SQLITE_TRANSIENT);
}

/* predict_register(model_id, config_json): record an external model in
 * _predict_models so predict() can dispatch to it. config is a JSON object:
 *   { "runtime":"onnx", "kind":"tabular-fm"|"student"|...,
 *     "license":"<SPDX>", "weights_uri":"/path/model.onnx",
 *     "io_spec": { ...tensor mapping... } }
 * The content_hash is computed from the weights file and pins the exact
 * bytes (verified again when the weights load). Returns the content_hash.
 * Registration is metadata only: executing the model still requires the
 * matching runtime to be compiled in. */
static void predict_register_fn(sqlite3_context *context, int argc,
                                sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  sqlite3 *db = sqlite3_context_db_handle(context);
  const char *model_id = (const char *)sqlite3_value_text(argv[0]);
  const char *config = (const char *)sqlite3_value_text(argv[1]);
  if (!model_id || !model_id[0] || !config) {
    char *e = sqlite3_mprintf(
        "%s: predict_register(model_id, config_json) requires both",
        PREDICT_ERR_OPTIONS);
    sqlite3_result_error(context, e, -1);
    sqlite3_free(e);
    return;
  }
  char *emsg = NULL;
  if (predict0_registry_ensure(db, &emsg) != SQLITE_OK) {
    sqlite3_result_error(context, emsg ? emsg : "registry unavailable", -1);
    sqlite3_free(emsg);
    return;
  }

  char *runtime = NULL, *kind = NULL, *license = NULL, *uri = NULL,
       *io_spec = NULL, *target = NULL;

#define REG_FAIL(...)                                                          \
  do {                                                                        \
    char *e = sqlite3_mprintf(__VA_ARGS__);                                   \
    sqlite3_result_error(context, e, -1);                                     \
    sqlite3_free(e);                                                          \
    goto reg_cleanup;                                                         \
  } while (0)

  /* config is either a JSON object or, for the common case, a bare path to a
   * weights file. A bare path means "onnx student, derive the rest". */
  const char *p = config;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
    p++;
  if (*p != '{') {
    uri = sqlite3_mprintf("%s", config); /* bare weights path */
  } else {
    sqlite3_stmt *s = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_extract(?1,'$.runtime'), json_extract(?1,'$.kind'),"
            " json_extract(?1,'$.license'), json_extract(?1,'$.weights_uri'),"
            " json_extract(?1,'$.io_spec'), json_extract(?1,'$.target')",
            -1, &s, NULL) != SQLITE_OK)
      REG_FAIL("%s: config is not valid JSON", PREDICT_ERR_OPTIONS);
    sqlite3_bind_text(s, 1, config, -1, SQLITE_STATIC);
    if (sqlite3_step(s) != SQLITE_ROW) {
      sqlite3_finalize(s);
      REG_FAIL("%s: config is not valid JSON", PREDICT_ERR_OPTIONS);
    }
#define COL_OR_NULL(i)                                                        \
  (sqlite3_column_type(s, i) == SQLITE_NULL                                   \
       ? NULL                                                                 \
       : sqlite3_mprintf("%s", (const char *)sqlite3_column_text(s, i)))
    runtime = COL_OR_NULL(0);
    kind = COL_OR_NULL(1);
    license = COL_OR_NULL(2);
    uri = COL_OR_NULL(3);
    io_spec = COL_OR_NULL(4);
    target = COL_OR_NULL(5);
#undef COL_OR_NULL
    sqlite3_finalize(s);
  }

  /* defaults: a user registering their own model shouldn't have to restate
   * the obvious. The onnx build derives the io_spec from the model below. */
  if (!runtime)
    runtime = sqlite3_mprintf("onnx");
  if (!kind)
    kind = sqlite3_mprintf("student");
  if (!license)
    license = sqlite3_mprintf("unspecified");

  if (strcmp(runtime, "tree") == 0)
    /* tree students carry inline-BLOB weights that only distill_predict/
     * distill_forecast write; a URI-registered 'tree' row could never be
     * served, so reject it here rather than at first use */
    REG_FAIL("%s: 'tree' students are created by distill_predict/"
             "distill_forecast, not predict_register",
             PREDICT_ERR_OPTIONS);
  if (strcmp(runtime, "onnx") != 0 && strcmp(runtime, "ggml") != 0)
    REG_FAIL("%s: predict_register handles onnx|ggml runtimes, got '%s'",
             PREDICT_ERR_OPTIONS, runtime);
  if (!uri)
    REG_FAIL("%s: %s models need a weights_uri", PREDICT_ERR_OPTIONS, runtime);

  char content_hash[PREDICT_HEX_BUFSIZE];
  if (predict0_hash_file(uri, content_hash, &emsg)) {
    sqlite3_result_error(context, emsg, -1);
    sqlite3_free(emsg);
    goto reg_cleanup;
  }

  /* Derive the io_spec by reading the model when the caller didn't give one. */
  if (!io_spec) {
#ifdef SQLITE_PREDICT_ONNX
    if (strcmp(runtime, "onnx") == 0) {
      if (predict0_onnx_introspect(db, uri, &io_spec, &emsg) != SQLITE_OK) {
        sqlite3_result_error(context, emsg, -1);
        sqlite3_free(emsg);
        goto reg_cleanup;
      }
      /* introspection can't know the SQL target column of an in-context
       * model; splice in a top-level `target` when the caller gave one. */
      if (target) {
        sqlite3_stmt *js = NULL;
        if (sqlite3_prepare_v2(db, "SELECT json_set(?1,'$.target',?2)", -1, &js,
                               NULL) == SQLITE_OK) {
          sqlite3_bind_text(js, 1, io_spec, -1, SQLITE_STATIC);
          sqlite3_bind_text(js, 2, target, -1, SQLITE_STATIC);
          if (sqlite3_step(js) == SQLITE_ROW) {
            char *merged =
                sqlite3_mprintf("%s", (const char *)sqlite3_column_text(js, 0));
            sqlite3_free(io_spec);
            io_spec = merged;
          }
          sqlite3_finalize(js);
        }
      }
    } else {
      REG_FAIL("%s: %s models need an explicit io_spec", PREDICT_ERR_IO_SPEC,
               runtime);
    }
#else
    REG_FAIL("%s: io_spec is required in this build; the onnx build"
             " (loadable-onnx) derives it from the model",
             PREDICT_ERR_IO_SPEC);
#endif
  }

  sqlite3_stmt *ins = NULL;
  if (sqlite3_prepare_v2(
          db,
          "INSERT INTO _predict_models"
          " (model_id, kind, runtime, weights_uri, io_spec, content_hash,"
          "  license) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
          -1, &ins, NULL) != SQLITE_OK)
    REG_FAIL("%s: cannot prepare registry insert", PREDICT_ERR_RESOURCE);
  sqlite3_bind_text(ins, 1, model_id, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 2, kind, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 3, runtime, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 4, uri, -1, SQLITE_STATIC);
  if (io_spec)
    sqlite3_bind_text(ins, 5, io_spec, -1, SQLITE_STATIC);
  else
    sqlite3_bind_null(ins, 5);
  sqlite3_bind_text(ins, 6, content_hash, -1, SQLITE_STATIC);
  sqlite3_bind_text(ins, 7, license, -1, SQLITE_STATIC);
  int irc = sqlite3_step(ins);
  sqlite3_finalize(ins);
  if (irc == SQLITE_CONSTRAINT)
    REG_FAIL("%s: model '%s' is already registered (or fails the weight-source"
             " rule)", PREDICT_ERR_MODEL_EXISTS, model_id);
  if (irc != SQLITE_DONE)
    REG_FAIL("%s: registry insert failed: %s", PREDICT_ERR_RESOURCE,
             sqlite3_errmsg(db));
  sqlite3_result_text(context, content_hash, -1, SQLITE_TRANSIENT);

#undef REG_FAIL
reg_cleanup:
  sqlite3_free(runtime);
  sqlite3_free(kind);
  sqlite3_free(license);
  sqlite3_free(uri);
  sqlite3_free(io_spec);
  sqlite3_free(target);
}

#pragma endregion

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_predict_init(sqlite3 *db, char **pzErrMsg,
                             const sqlite3_api_routines *pApi) {
  UNUSED_PARAMETER(pzErrMsg);
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#else
  UNUSED_PARAMETER(pApi);
#endif
  int rc = SQLITE_OK;
  const int flags = SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC;
  /* predict_version tags its result with the JSON subtype. SQLite 3.45+
   * refuses sqlite3_result_subtype() unless the function opted in with
   * this flag (older SQLite ignores the unknown flag and never enforced
   * the check). Scoped to the one function that sets a subtype. */
#ifdef SQLITE_RESULT_SUBTYPE
  const int subtype_flags = flags | SQLITE_RESULT_SUBTYPE;
#else
  const int subtype_flags = flags;
#endif

  rc = sqlite3_create_function_v2(db, "predict_version", 0, subtype_flags,
                                  NULL, predict_version_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function_v2(db, "predict_debug", 0, flags, NULL,
                                  predict_debug_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function_v2(db, "predict_ulid", 1, flags, NULL,
                                  predict_ulid_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;
  /* predict_register mutates the registry, so it is not INNOCUOUS and not
   * DETERMINISTIC (its content_hash depends on a file read). */
  rc = sqlite3_create_function_v2(db, "predict_register", 2, SQLITE_UTF8,
                                  NULL, predict_register_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = predict0_forecast_init(db);
  if (rc != SQLITE_OK)
    return rc;
  rc = predict0_tabular_init(db);
  if (rc != SQLITE_OK)
    return rc;
  rc = predict0_distill_init(db);
  return rc;
}
