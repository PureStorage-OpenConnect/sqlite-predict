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
      "{\"extension\":\"%s\",\"spec\":\"%s\",\"runtimes\":[\"stat\"],"
      "\"models\":[]}",
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

int predict0_parse_timestamp(const char *s, i64 *out_ms) {
  int y, mo, d, h = 0, mi = 0, sec = 0;
  int n = 0;
  if (!s)
    return 1;
  if (sscanf(s, "%4d-%2d-%2d%n", &y, &mo, &d, &n) != 3)
    return 1;
  if (s[n] == 'T' || s[n] == ' ') {
    int n2 = 0;
    if (sscanf(s + n + 1, "%2d:%2d%n", &h, &mi, &n2) != 2)
      return 1;
    n += 1 + n2;
    if (s[n] == ':') {
      int n3 = 0;
      if (sscanf(s + n + 1, "%2d%n", &sec, &n3) != 1)
        return 1;
      n += 1 + n3;
      if (s[n] == '.') { /* skip fractional seconds */
        n++;
        while (s[n] >= '0' && s[n] <= '9')
          n++;
      }
    }
  }
  if (s[n] == 'Z')
    n++;
  if (s[n] != '\0')
    return 1;
  if (mo < 1 || mo > 12 || d < 1 || d > days_in_month(y, mo) || h > 23 ||
      mi > 59 || sec > 60)
    return 1;

  /* days since 1970-01-01, civil-from-days style */
  i64 days = 0;
  for (int yy = 1970; yy < y; yy++)
    days += ((yy % 4 == 0 && yy % 100 != 0) || yy % 400 == 0) ? 366 : 365;
  for (int mm = 1; mm < mo; mm++)
    days += days_in_month(y, mm);
  days += d - 1;
  *out_ms = ((days * 24 + h) * 60 + mi) * (i64)60 * 1000 + (i64)sec * 1000;
  return 0;
}

void predict0_format_timestamp(i64 ms, char *buf, usize bufsize) {
  i64 secs = ms / 1000;
  i64 days = secs / 86400;
  int rem = (int)(secs % 86400);
  int y = 1970;
  for (;;) {
    int len = ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 366 : 365;
    if (days < len)
      break;
    days -= len;
    y++;
  }
  int mo = 1;
  while (days >= days_in_month(y, mo)) {
    days -= days_in_month(y, mo);
    mo++;
  }
  snprintf(buf, bufsize, "%04d-%02d-%02dT%02d:%02d:%02dZ", y, mo,
           (int)days + 1, rem / 3600, (rem / 60) % 60, rem % 60);
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
    *errmsg = sqlite3_mprintf("%s: options parsing unavailable (%s)",
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
    /* a null value is equivalent to omitting the key (lets recorded
     * receipt params round-trip as options) */
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

void predict0_ulid_min(i64 ms, char *buf) {
  u8 zeros[10] = {0};
  ulid_encode(ms, zeros, buf);
}

void predict0_ulid_new(i64 ms, char *buf) {
  u8 rnd[10];
  sqlite3_randomness(sizeof(rnd), rnd);
  ulid_encode(ms, rnd, buf);
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
  char buf[27];
  predict0_ulid_min(ms, buf);
  sqlite3_result_text(context, buf, 26, SQLITE_TRANSIENT);
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

  rc = sqlite3_create_function_v2(db, "predict_version", 0, flags, NULL,
                                  predict_version_fn, NULL, NULL, NULL);
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

  rc = predict0_receipts_init(db);
  if (rc != SQLITE_OK)
    return rc;
  rc = predict0_forecast_init(db);
  return rc;
}
