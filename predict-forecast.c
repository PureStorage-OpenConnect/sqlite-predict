/* forecast(query, horizon, options) — eponymous table-valued function.
 * RFC 0005 §4.2.1. Vtab shape follows SQLite's ext/misc/series.c. */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define FORECAST_MAX_HORIZON 1000
#define FORECAST_DEFAULT_CONTEXT_LIMIT 4096
#define FORECAST_MIN_HISTORY 8
#define FORECAST_MAX_TOTAL_ROWS (1 << 21)
#define FORECAST_MAX_GROUP_COLS 8

/* column order must match the CREATE TABLE below */
#define FC_COL_SERIES_KEY 0
#define FC_COL_STEP 1
#define FC_COL_TS 2
#define FC_COL_FORECAST 3
#define FC_COL_LOWER 4
#define FC_COL_UPPER 5
#define FC_COL_STATUS 6
#define FC_COL_RECEIPT 7
#define FC_COL_QUERY 8
#define FC_COL_HORIZON 9
#define FC_COL_OPTIONS 10

#pragma region models

/* Detect a seasonal period: autocorrelation of the OLS-detrended series
 * (a trend inflates raw ACF at every lag and buries the seasonal peak),
 * accepting only local ACF peaks above threshold. Returns 0 when nothing
 * convincing is found. */
static int detect_period(const f64 *y, int n) {
  if (n < 3 * 2)
    return 0;
  int max_lag = n / 2;
  if (max_lag > 366)
    max_lag = 366;

  f64 *d = sqlite3_malloc(sizeof(f64) * n);
  if (!d)
    return 0;
  f64 st = 0, sy = 0, stt = 0, sty = 0;
  for (int i = 0; i < n; i++) {
    st += i;
    sy += y[i];
    stt += (f64)i * i;
    sty += (f64)i * y[i];
  }
  f64 denom = (f64)n * stt - st * st;
  f64 b = denom != 0 ? ((f64)n * sty - st * sy) / denom : 0;
  f64 a = (sy - b * st) / n;
  for (int i = 0; i < n; i++)
    d[i] = y[i] - (a + b * i);

  f64 c0 = 0;
  for (int i = 0; i < n; i++)
    c0 += d[i] * d[i];
  if (c0 <= 0) {
    sqlite3_free(d);
    return 0;
  }

  f64 *acf = sqlite3_malloc(sizeof(f64) * (max_lag + 2));
  if (!acf) {
    sqlite3_free(d);
    return 0;
  }
  for (int lag = 1; lag <= max_lag + 1 && lag < n; lag++) {
    f64 c = 0;
    for (int i = lag; i < n; i++)
      c += d[i] * d[i - lag];
    acf[lag] = c / c0;
  }

  int best = 0;
  f64 best_acf = 0.3; /* threshold: below this, treat as non-seasonal */
  for (int lag = 2; lag <= max_lag && lag + 1 < n; lag++) {
    int is_peak = acf[lag] > acf[lag - 1] && acf[lag] >= acf[lag + 1];
    if (is_peak && acf[lag] > best_acf) {
      best_acf = acf[lag];
      best = lag;
    }
  }
  sqlite3_free(d);
  sqlite3_free(acf);
  return best;
}

/* Seasonal naive with drift. Baseline floor: repeat last season (or last
 * value) plus the global drift. sigma_1 from one-step naive residuals. */
static void model_seasonal_naive(const f64 *y, int n, int horizon, f64 *fc,
                                 f64 *sigma) {
  int p = detect_period(y, n);
  f64 drift = n > 1 ? (y[n - 1] - y[0]) / (n - 1) : 0;

  f64 ss = 0;
  int m = 0;
  for (int i = 1; i < n; i++) {
    f64 r = y[i] - y[i - 1];
    ss += r * r;
    m++;
  }
  f64 sigma1 = m > 0 ? sqrt(ss / m) : 0;

  for (int h = 1; h <= horizon; h++) {
    f64 base;
    i64 gap; /* steps between the base observation and the target */
    if (p > 0 && n >= p) {
      int base_idx = n - p + ((h - 1) % p);
      base = y[base_idx];
      gap = (n - 1 + h) - base_idx;
    } else {
      base = y[n - 1];
      gap = h;
    }
    fc[h - 1] = base + drift * (f64)gap;
    sigma[h - 1] = sigma1 * sqrt((f64)h);
  }
}

/* Classic Theta(0,2) with additive seasonal adjustment: average of the
 * linear-regression line extrapolation and SES (grid-searched alpha) on
 * the theta=2 line. The strongest of the simple statistical methods. */
static void model_theta(const f64 *y, int n, int horizon, f64 *fc,
                        f64 *sigma) {
  int p = detect_period(y, n);
  f64 *yd = sqlite3_malloc(sizeof(f64) * n);
  f64 *seas = NULL;
  if (!yd) {
    model_seasonal_naive(y, n, horizon, fc, sigma);
    return;
  }
  memcpy(yd, y, sizeof(f64) * n);

  /* additive seasonal indices from period-phase means of the DETRENDED
   * series — computing them on raw values leaks the trend slope into
   * the indices, one phase-width of drift per index */
  if (p > 1 && n >= 2 * p) {
    seas = sqlite3_malloc(sizeof(f64) * p);
    if (seas) {
      f64 st0 = 0, sy0 = 0, stt0 = 0, sty0 = 0;
      for (int i = 0; i < n; i++) {
        st0 += i;
        sy0 += y[i];
        stt0 += (f64)i * i;
        sty0 += (f64)i * y[i];
      }
      f64 den0 = (f64)n * stt0 - st0 * st0;
      f64 b0 = den0 != 0 ? ((f64)n * sty0 - st0 * sy0) / den0 : 0;
      f64 a0 = (sy0 - b0 * st0) / n;
      for (int k = 0; k < p; k++) {
        f64 s = 0;
        int c = 0;
        for (int i = k; i < n; i += p) {
          s += y[i] - (a0 + b0 * i);
          c++;
        }
        seas[k] = c ? s / c : 0;
      }
      for (int i = 0; i < n; i++)
        yd[i] -= seas[i % p];
    }
  }

  /* OLS line a + b*t over the deseasonalized series */
  f64 st = 0, sy = 0, stt = 0, sty = 0;
  for (int i = 0; i < n; i++) {
    st += i;
    sy += yd[i];
    stt += (f64)i * i;
    sty += i * yd[i];
  }
  f64 denom = n * stt - st * st;
  f64 b = denom != 0 ? (n * sty - st * sy) / denom : 0;
  f64 a = (sy - b * st) / n;

  /* SES over the theta=2 line, alpha by grid search on one-step SSE */
  f64 best_alpha = 0.5, best_sse = -1, best_level = yd[n - 1];
  for (f64 alpha = 0.05; alpha < 0.96; alpha += 0.05) {
    f64 level = 2 * yd[0] - (a);
    f64 sse = 0;
    for (int i = 1; i < n; i++) {
      f64 z = 2 * yd[i] - (a + b * i);
      f64 e = z - level;
      sse += e * e;
      level += alpha * e;
    }
    if (best_sse < 0 || sse < best_sse) {
      best_sse = sse;
      best_alpha = alpha;
      best_level = level;
    }
  }
  UNUSED_PARAMETER(best_alpha);
  f64 sigma1 = n > 1 ? sqrt(best_sse / (n - 1)) / 2 : 0;

  for (int h = 1; h <= horizon; h++) {
    f64 line = a + b * (n - 1 + h);
    f64 theta_fc = 0.5 * line + 0.5 * best_level;
    if (seas && p > 0)
      theta_fc += seas[(n + h - 1) % p];
    fc[h - 1] = theta_fc;
    sigma[h - 1] = sigma1 * sqrt((f64)h);
  }

  sqlite3_free(yd);
  sqlite3_free(seas);
}

typedef void (*predict0_ts_model)(const f64 *, int, int, f64 *, f64 *);

typedef struct {
  const char *id;
  predict0_ts_model run;
} ts_model_entry;

static const ts_model_entry TS_MODELS[] = {
    {"stub-seasonal-naive", model_seasonal_naive},
    {"theta-classic", model_theta},
};

static predict0_ts_model resolve_ts_model(const char *name) {
  if (!name || strcmp(name, "default-ts") == 0)
    return model_theta;
  for (usize i = 0; i < countof(TS_MODELS); i++) {
    if (strcmp(TS_MODELS[i].id, name) == 0)
      return TS_MODELS[i].run;
  }
  return NULL;
}

static const char *resolve_ts_model_id(const char *name) {
  if (!name || strcmp(name, "default-ts") == 0)
    return "theta-classic";
  for (usize i = 0; i < countof(TS_MODELS); i++) {
    if (strcmp(TS_MODELS[i].id, name) == 0)
      return TS_MODELS[i].id;
  }
  return NULL;
}

#pragma endregion

#pragma region options

typedef struct {
  char *time_col;
  char *value_col;
  char *group_cols[FORECAST_MAX_GROUP_COLS];
  int n_group_cols;
  f64 confidence;
  int context_limit;
  char *model;
  int receipt;
} ForecastOpts;

static void forecast_opts_free(ForecastOpts *o) {
  sqlite3_free(o->time_col);
  sqlite3_free(o->value_col);
  sqlite3_free(o->model);
  for (int i = 0; i < o->n_group_cols; i++)
    sqlite3_free(o->group_cols[i]);
}

static char *dup_text(sqlite3_value *v) {
  const char *t = (const char *)sqlite3_value_text(v);
  return t ? sqlite3_mprintf("%s", t) : NULL;
}

static int forecast_opt_cb(void *ctx, const char *key, sqlite3_value *value,
                           char **errmsg) {
  ForecastOpts *o = ctx;
  int vtype = sqlite3_value_type(value);

  if (strcmp(key, "time_col") == 0 || strcmp(key, "value_col") == 0 ||
      strcmp(key, "model") == 0) {
    if (vtype != SQLITE_TEXT)
      goto wrong_type;
    char **dst = strcmp(key, "time_col") == 0  ? &o->time_col
                 : strcmp(key, "value_col") == 0 ? &o->value_col
                                                 : &o->model;
    *dst = dup_text(value);
    return 0;
  }
  if (strcmp(key, "confidence_level") == 0 ||
      strcmp(key, "anomaly_prob_threshold") == 0) {
    if (vtype != SQLITE_FLOAT && vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->confidence = sqlite3_value_double(value);
    if (o->confidence <= 0 || o->confidence >= 1) {
      *errmsg = sqlite3_mprintf(
          "%s: %s must be in (0,1)",
          strcmp(key, "confidence_level") == 0 ? PREDICT_ERR_OPTIONS
                                               : PREDICT_ERR_THRESHOLD,
          key);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "context_limit") == 0) {
    if (vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->context_limit = sqlite3_value_int(value);
    if (o->context_limit < 1) {
      *errmsg = sqlite3_mprintf("%s: context_limit must be >= 1",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "receipt") == 0) {
    if (vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->receipt = sqlite3_value_int(value) != 0;
    return 0;
  }
  if (strcmp(key, "group_cols") == 0) {
    /* JSON array of names, or a bare string as a one-element array */
    const char *t = (const char *)sqlite3_value_text(value);
    if (!t)
      goto wrong_type;
    if (t[0] != '[') {
      if (vtype != SQLITE_TEXT)
        goto wrong_type;
      o->group_cols[0] = sqlite3_mprintf("%s", t);
      o->n_group_cols = 1;
      return 0;
    }
    /* crude but sufficient: strip [ ] " and split on , — column names
     * with commas or quotes are not supported (and not sane) */
    const char *pch = t + 1;
    while (*pch && *pch != ']') {
      while (*pch == ' ' || *pch == '"' || *pch == ',')
        pch++;
      if (*pch == ']' || !*pch)
        break;
      const char *end = pch;
      while (*end && *end != '"' && *end != ',' && *end != ']')
        end++;
      if (o->n_group_cols >= FORECAST_MAX_GROUP_COLS) {
        *errmsg = sqlite3_mprintf("%s: too many group_cols (max %d)",
                                  PREDICT_ERR_OPTIONS, FORECAST_MAX_GROUP_COLS);
        return 1;
      }
      o->group_cols[o->n_group_cols++] =
          sqlite3_mprintf("%.*s", (int)(end - pch), pch);
      pch = end;
    }
    if (o->n_group_cols == 0) {
      *errmsg = sqlite3_mprintf("%s: group_cols must not be empty",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }

wrong_type:
  *errmsg = sqlite3_mprintf("%s: wrong type for option '%s'",
                            PREDICT_ERR_OPTIONS, key);
  return 1;
}

static const char *const FORECAST_OPTION_KEYS[] = {
    "time_col", "value_col", "group_cols", "confidence_level",
    "context_limit", "model", "receipt", NULL};

/* detect_anomalies shares the struct; anomaly_prob_threshold lands in
 * .confidence (same (0,1) validation) with a 0.99 default. */
static const char *const ANOMALY_OPTION_KEYS[] = {
    "time_col", "value_col", "group_cols", "anomaly_prob_threshold",
    "context_limit", "model", "receipt", NULL};

#pragma endregion

#pragma region vtab

typedef struct {
  char *series_key;
  int step;
  char ts[24];
  f64 forecast, lower, upper;
  const char *status; /* static strings only */
  int has_values;     /* 0 for status-only rows */
  char receipt_id[27];
} ForecastRow;

static int forecast_row_cmp(const void *a, const void *b) {
  const ForecastRow *x = *(ForecastRow *const *)a;
  const ForecastRow *y = *(ForecastRow *const *)b;
  int c = strcmp(x->series_key ? x->series_key : "",
                 y->series_key ? y->series_key : "");
  if (c)
    return c;
  return x->step - y->step;
}

typedef struct forecast_vtab {
  sqlite3_vtab base;
  sqlite3 *db;
} forecast_vtab;

typedef struct forecast_cursor {
  sqlite3_vtab_cursor base;
  ForecastRow *rows;
  int n_rows;
  int i;
} forecast_cursor;

/* one input series during collection */
typedef struct {
  char *key;
  i64 *ts;
  f64 *val;
  int n, cap;
  int non_numeric;
  int truncated;
} SeriesBuf;

static int fc_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  forecast_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(series_key TEXT, step INTEGER,"
          " forecast_timestamp TEXT, forecast REAL, lower_bound REAL,"
          " upper_bound REAL, status TEXT, receipt_id TEXT,"
          " query HIDDEN, horizon HIDDEN, options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int fc_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int fc_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_query = 0, seen_horizon = 0, argv = 1;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    if (c->iColumn == FC_COL_QUERY) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_query = 1;
    } else if (c->iColumn == FC_COL_HORIZON) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 2;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_horizon = 1;
    } else if (c->iColumn == FC_COL_OPTIONS) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 3;
      pIdx->aConstraintUsage[i].omit = 1;
    }
  }
  UNUSED_PARAMETER(argv);
  if (!seen_query || !seen_horizon) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: forecast(query, horizon) requires both arguments",
        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int fc_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  forecast_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void fc_rows_free(forecast_cursor *c) {
  for (int i = 0; i < c->n_rows; i++)
    sqlite3_free(c->rows[i].series_key);
  sqlite3_free(c->rows);
  c->rows = NULL;
  c->n_rows = 0;
  c->i = 0;
}

static int fc_close(sqlite3_vtab_cursor *pCur) {
  forecast_cursor *c = (forecast_cursor *)pCur;
  fc_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int fc_error(forecast_cursor *cur, const char *code, const char *fmt,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s%s", code, fmt, detail ? detail : "");
  return SQLITE_ERROR;
}

static SeriesBuf *series_find(SeriesBuf **all, int *n_series, int *cap,
                              const char *key) {
  for (int i = 0; i < *n_series; i++) {
    if (strcmp((*all)[i].key, key) == 0)
      return &(*all)[i];
  }
  if (*n_series == *cap) {
    int nc = *cap ? *cap * 2 : 8;
    SeriesBuf *grown = sqlite3_realloc(*all, sizeof(SeriesBuf) * nc);
    if (!grown)
      return NULL;
    *all = grown;
    *cap = nc;
  }
  SeriesBuf *s = &(*all)[(*n_series)++];
  memset(s, 0, sizeof(*s));
  s->key = sqlite3_mprintf("%s", key);
  return s;
}

/* sort series rows chronologically (insertion sort: input is usually
 * already ordered, making this near-linear) */
static void series_sort(SeriesBuf *s) {
  for (int i = 1; i < s->n; i++) {
    i64 t = s->ts[i];
    f64 v = s->val[i];
    int j = i - 1;
    while (j >= 0 && s->ts[j] > t) {
      s->ts[j + 1] = s->ts[j];
      s->val[j + 1] = s->val[j];
      j--;
    }
    s->ts[j + 1] = t;
    s->val[j + 1] = v;
  }
}

static int fc_filter(sqlite3_vtab_cursor *pCur, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  forecast_cursor *cur = (forecast_cursor *)pCur;
  forecast_vtab *vtab = (forecast_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  fc_rows_free(cur);

  if (argc < 2)
    return fc_error(cur, PREDICT_ERR_SCHEMA,
                    "forecast(query, horizon) requires both arguments", NULL);

  const char *query = (const char *)sqlite3_value_text(argv[0]);
  if (!query)
    return fc_error(cur, PREDICT_ERR_SCHEMA, "query must be text", NULL);

  if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER)
    return fc_error(cur, PREDICT_ERR_HORIZON, "horizon must be an integer",
                    NULL);
  int horizon = sqlite3_value_int(argv[1]);
  if (horizon < 1 || horizon > FORECAST_MAX_HORIZON)
    return fc_error(cur, PREDICT_ERR_HORIZON,
                    "horizon out of range 1..1000", NULL);

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.95;
  opts.context_limit = FORECAST_DEFAULT_CONTEXT_LIMIT;
  opts.receipt = 1;

  const char *options_json =
      argc >= 3 ? (const char *)sqlite3_value_text(argv[2]) : NULL;
  char *errmsg = NULL;
  if (predict0_options_parse(db, options_json, FORECAST_OPTION_KEYS,
                             forecast_opt_cb, &opts, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }

  predict0_ts_model model = resolve_ts_model(opts.model);
  if (!model) {
    int rc = fc_error(cur, PREDICT_ERR_MODEL_NOT_FOUND, "no such model: ",
                      opts.model);
    forecast_opts_free(&opts);
    return rc;
  }

  /* prepare the inner query */
  sqlite3_stmt *stmt = NULL;
  const char *tail = NULL;
  int rc = sqlite3_prepare_v2(db, query, -1, &stmt, &tail);
  if (rc != SQLITE_OK || !stmt) {
    if (stmt)
      sqlite3_finalize(stmt);
    rc = fc_error(cur, PREDICT_ERR_SCHEMA, "query does not parse: ",
                  sqlite3_errmsg(db));
    forecast_opts_free(&opts);
    return rc;
  }
  while (tail && (*tail == ' ' || *tail == '\n' || *tail == ';' ||
                  *tail == '\t'))
    tail++;
  if (tail && *tail != '\0') {
    sqlite3_finalize(stmt);
    forecast_opts_free(&opts);
    return fc_error(cur, PREDICT_ERR_SCHEMA,
                    "query must be a single statement", NULL);
  }
  if (!sqlite3_stmt_readonly(stmt)) {
    sqlite3_finalize(stmt);
    forecast_opts_free(&opts);
    return fc_error(cur, PREDICT_ERR_QUERY_NOT_READONLY,
                    "query must be a read-only SELECT", NULL);
  }

  int ncol = sqlite3_column_count(stmt);
  if (ncol < 2) {
    sqlite3_finalize(stmt);
    forecast_opts_free(&opts);
    return fc_error(cur, PREDICT_ERR_SCHEMA,
                    "query must yield at least a time and a value column",
                    NULL);
  }

  /* resolve named columns */
  int time_idx = -1, value_idx = -1;
  int group_idx[FORECAST_MAX_GROUP_COLS];
  for (int g = 0; g < opts.n_group_cols; g++)
    group_idx[g] = -1;
  for (int i = 0; i < ncol; i++) {
    const char *name = sqlite3_column_name(stmt, i);
    if (opts.time_col && name && strcmp(name, opts.time_col) == 0)
      time_idx = i;
    if (opts.value_col && name && strcmp(name, opts.value_col) == 0)
      value_idx = i;
    for (int g = 0; g < opts.n_group_cols; g++) {
      if (name && strcmp(name, opts.group_cols[g]) == 0)
        group_idx[g] = i;
    }
  }
  if (opts.time_col && time_idx < 0) {
    sqlite3_finalize(stmt);
    int rc2 = fc_error(cur, PREDICT_ERR_SCHEMA, "no such time_col: ",
                       opts.time_col);
    forecast_opts_free(&opts);
    return rc2;
  }
  if (opts.value_col && value_idx < 0) {
    sqlite3_finalize(stmt);
    int rc2 = fc_error(cur, PREDICT_ERR_SCHEMA, "no such value_col: ",
                       opts.value_col);
    forecast_opts_free(&opts);
    return rc2;
  }
  for (int g = 0; g < opts.n_group_cols; g++) {
    if (group_idx[g] < 0) {
      sqlite3_finalize(stmt);
      int rc2 = fc_error(cur, PREDICT_ERR_SCHEMA, "no such group col: ",
                         opts.group_cols[g]);
      forecast_opts_free(&opts);
      return rc2;
    }
  }

  /* collect rows */
  SeriesBuf *series = NULL;
  int n_series = 0, series_cap = 0;
  i64 total_rows = 0;
  int inference_done = (time_idx >= 0 && value_idx >= 0);
  char keybuf[512];

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (!inference_done) {
      /* infer from the first row: time = named or first parseable/int
       * column; value = named or first numeric column that isn't time */
      for (int i = 0; time_idx < 0 && i < ncol; i++) {
        int t = sqlite3_column_type(stmt, i);
        if (t == SQLITE_INTEGER) {
          time_idx = i;
        } else if (t == SQLITE_TEXT) {
          i64 ms;
          if (predict0_parse_timestamp(
                  (const char *)sqlite3_column_text(stmt, i), &ms) == 0)
            time_idx = i;
        }
      }
      for (int i = 0; value_idx < 0 && i < ncol; i++) {
        if (i == time_idx)
          continue;
        int in_groups = 0;
        for (int g = 0; g < opts.n_group_cols; g++)
          if (group_idx[g] == i)
            in_groups = 1;
        if (in_groups)
          continue;
        int t = sqlite3_column_type(stmt, i);
        if (t == SQLITE_INTEGER || t == SQLITE_FLOAT)
          value_idx = i;
      }
      if (time_idx < 0 || value_idx < 0) {
        sqlite3_finalize(stmt);
        for (int i = 0; i < n_series; i++) {
          sqlite3_free(series[i].key);
          sqlite3_free(series[i].ts);
          sqlite3_free(series[i].val);
        }
        sqlite3_free(series);
        forecast_opts_free(&opts);
        return fc_error(cur, PREDICT_ERR_SCHEMA,
                        "could not infer time/value columns", NULL);
      }
      inference_done = 1;
    }

    if (++total_rows > FORECAST_MAX_TOTAL_ROWS) {
      sqlite3_finalize(stmt);
      for (int i = 0; i < n_series; i++) {
        sqlite3_free(series[i].key);
        sqlite3_free(series[i].ts);
        sqlite3_free(series[i].val);
      }
      sqlite3_free(series);
      forecast_opts_free(&opts);
      return fc_error(cur, PREDICT_ERR_RESOURCE, "too many input rows", NULL);
    }

    /* series key: group values joined with 0x1F */
    keybuf[0] = '\0';
    usize kl = 0;
    for (int g = 0; g < opts.n_group_cols; g++) {
      const char *gv = (const char *)sqlite3_column_text(stmt, group_idx[g]);
      int wrote = snprintf(keybuf + kl, sizeof(keybuf) - kl, "%s%s",
                           g ? "\x1f" : "", gv ? gv : "");
      if (wrote > 0)
        kl += (usize)wrote;
      if (kl >= sizeof(keybuf) - 1)
        break;
    }

    SeriesBuf *s = series_find(&series, &n_series, &series_cap, keybuf);
    if (!s) {
      rc = SQLITE_NOMEM;
      break;
    }
    if (s->non_numeric)
      continue;

    i64 ms = 0;
    int tt = sqlite3_column_type(stmt, time_idx);
    if (tt == SQLITE_INTEGER) {
      ms = sqlite3_column_int64(stmt, time_idx) * 1000;
    } else if (predict0_parse_timestamp(
                   (const char *)sqlite3_column_text(stmt, time_idx), &ms)) {
      s->non_numeric = 1; /* unparseable time in this series */
      continue;
    }
    int vt = sqlite3_column_type(stmt, value_idx);
    if (vt != SQLITE_INTEGER && vt != SQLITE_FLOAT) {
      s->non_numeric = 1;
      continue;
    }

    if (s->n == s->cap) {
      int nc = s->cap ? s->cap * 2 : 64;
      i64 *nts = sqlite3_realloc(s->ts, sizeof(i64) * nc);
      f64 *nval = sqlite3_realloc(s->val, sizeof(f64) * nc);
      if (!nts || !nval) {
        sqlite3_free(nts);
        rc = SQLITE_NOMEM;
        break;
      }
      s->ts = nts;
      s->val = nval;
      s->cap = nc;
    }
    s->ts[s->n] = ms;
    s->val[s->n] = sqlite3_column_double(stmt, value_idx);
    s->n++;
  }
  /* capture resolved column names for the receipt before finalize */
  char *resolved_time = NULL, *resolved_value = NULL;
  if (time_idx >= 0)
    resolved_time = sqlite3_mprintf("%s", sqlite3_column_name(stmt, time_idx));
  if (value_idx >= 0)
    resolved_value =
        sqlite3_mprintf("%s", sqlite3_column_name(stmt, value_idx));

  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE && rc != SQLITE_ROW && rc != SQLITE_OK) {
    sqlite3_free(resolved_time);
    sqlite3_free(resolved_value);
    for (int i = 0; i < n_series; i++) {
      sqlite3_free(series[i].key);
      sqlite3_free(series[i].ts);
      sqlite3_free(series[i].val);
    }
    sqlite3_free(series);
    forecast_opts_free(&opts);
    return fc_error(cur, PREDICT_ERR_RESOURCE, "query execution failed",
                    NULL);
  }

  /* forecast every series into cursor rows */
  f64 z = predict0_norm_quantile(0.5 + opts.confidence / 2);
  int max_rows = 0;
  for (int i = 0; i < n_series; i++)
    max_rows += series[i].non_numeric || series[i].n < FORECAST_MIN_HISTORY
                    ? 1
                    : horizon;
  if (n_series == 0)
    max_rows = 0;

  cur->rows = sqlite3_malloc(sizeof(ForecastRow) * (max_rows ? max_rows : 1));
  if (!cur->rows)
    rc = SQLITE_NOMEM;
  cur->n_rows = 0;

  f64 *fc = sqlite3_malloc(sizeof(f64) * horizon);
  f64 *sg = sqlite3_malloc(sizeof(f64) * horizon);

  for (int i = 0; cur->rows && fc && sg && i < n_series; i++) {
    SeriesBuf *s = &series[i];
    const char *status = NULL;
    if (s->non_numeric)
      status = "non_numeric";
    else if (s->n < FORECAST_MIN_HISTORY)
      status = "insufficient_history";

    if (status) {
      ForecastRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->status = status;
      continue;
    }

    series_sort(s);
    int truncated = 0;
    f64 *y = s->val;
    i64 *ts = s->ts;
    int n = s->n;
    if (n > opts.context_limit) {
      y += n - opts.context_limit;
      ts += n - opts.context_limit;
      n = opts.context_limit;
      truncated = 1;
    }

    /* median step from sorted timestamps */
    i64 step_ms = 3600000;
    if (n > 1) {
      i64 span = ts[n - 1] - ts[0];
      step_ms = span > 0 ? span / (n - 1) : 3600000;
    }

    model(y, n, horizon, fc, sg);
    for (int h = 1; h <= horizon; h++) {
      ForecastRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->step = h;
      predict0_format_timestamp(ts[n - 1] + step_ms * h, r->ts,
                                sizeof(r->ts));
      r->forecast = fc[h - 1];
      r->lower = fc[h - 1] - z * sg[h - 1];
      r->upper = fc[h - 1] + z * sg[h - 1];
      r->status = truncated ? "truncated" : "ok";
      r->has_values = 1;
    }
  }

  sqlite3_free(fc);
  sqlite3_free(sg);
  for (int i = 0; i < n_series; i++) {
    sqlite3_free(series[i].key);
    sqlite3_free(series[i].ts);
    sqlite3_free(series[i].val);
  }
  sqlite3_free(series);

  if (rc == SQLITE_NOMEM) {
    sqlite3_free(resolved_time);
    sqlite3_free(resolved_value);
    forecast_opts_free(&opts);
    return SQLITE_NOMEM;
  }

  /* receipt: RFC §4.1.2 — digest BEFORE insert; params canonical via
   * json_object with alphabetical keys, resolved column names recorded */
  if (opts.receipt) {
    char *errmsg = NULL;
    const char *model_id = resolve_ts_model_id(opts.model);
    if (predict0_receipts_ensure(db, &errmsg)) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = errmsg;
      goto receipt_fail;
    }
    char *model_hash = predict0_registry_model_hash(db, model_id);
    if (!model_hash) {
      vtab->base.zErrMsg = sqlite3_mprintf(
          "%s: %s not in _predict_models", PREDICT_ERR_MODEL_NOT_FOUND,
          model_id);
      goto receipt_fail;
    }
    char digest[65];
    if (predict0_logical_digest(db, digest, &errmsg)) {
      sqlite3_free(model_hash);
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = errmsg;
      goto receipt_fail;
    }

    /* result hash over rows sorted by (series_key, step) */
    ForecastRow **order =
        sqlite3_malloc(sizeof(ForecastRow *) * (cur->n_rows ? cur->n_rows : 1));
    if (!order) {
      sqlite3_free(model_hash);
      goto receipt_fail;
    }
    for (int i = 0; i < cur->n_rows; i++)
      order[i] = &cur->rows[i];
    qsort(order, (usize)cur->n_rows, sizeof(ForecastRow *),
          forecast_row_cmp);
    predict0_hasher h;
    predict0_hash_init(&h);
    for (int i = 0; i < cur->n_rows; i++) {
      ForecastRow *r = order[i];
      predict0_hash_text(&h, r->series_key);
      if (r->has_values) {
        predict0_hash_int(&h, r->step);
        predict0_hash_text(&h, r->ts);
        predict0_hash_real(&h, r->forecast);
        predict0_hash_real(&h, r->lower);
        predict0_hash_real(&h, r->upper);
      } else {
        for (int k = 0; k < 5; k++)
          predict0_hash_null(&h);
      }
      predict0_hash_row_end(&h);
    }
    sqlite3_free(order);
    char result_hash[65];
    predict0_hash_hex(&h, result_hash);

    /* canonical params via json_object (keys alphabetical) */
    char group_json[600] = "";
    if (opts.n_group_cols) {
      usize gl = 0;
      gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl, "[");
      for (int g = 0; g < opts.n_group_cols && gl < sizeof(group_json) - 4;
           g++)
        gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl,
                              "%s\"%s\"", g ? "," : "", opts.group_cols[g]);
      snprintf(group_json + gl, sizeof(group_json) - gl, "]");
    }
    sqlite3_stmt *pj = NULL;
    char *params = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_object('confidence_level', ?1, 'context_limit', ?2,"
            " 'group_cols', CASE WHEN ?3 = '' THEN NULL ELSE json(?3) END,"
            " 'horizon', ?4, 'model', ?5, 'receipt', 1,"
            " 'time_col', ?6, 'value_col', ?7)",
            -1, &pj, NULL) == SQLITE_OK) {
      sqlite3_bind_double(pj, 1, opts.confidence);
      sqlite3_bind_int(pj, 2, opts.context_limit);
      sqlite3_bind_text(pj, 3, group_json, -1, SQLITE_STATIC);
      sqlite3_bind_int(pj, 4, horizon);
      sqlite3_bind_text(pj, 5, model_id, -1, SQLITE_STATIC);
      if (resolved_time)
        sqlite3_bind_text(pj, 6, resolved_time, -1, SQLITE_STATIC);
      if (resolved_value)
        sqlite3_bind_text(pj, 7, resolved_value, -1, SQLITE_STATIC);
      if (sqlite3_step(pj) == SQLITE_ROW)
        params = sqlite3_mprintf(
            "%s", (const char *)sqlite3_column_text(pj, 0));
      sqlite3_finalize(pj);
    }
    if (!params) {
      sqlite3_free(model_hash);
      vtab->base.zErrMsg = sqlite3_mprintf(
          "%s: could not canonicalize params", PREDICT_ERR_RESOURCE);
      goto receipt_fail;
    }

    char receipt_id[27];
    int irc = predict0_receipt_insert(db, "forecast", model_id, model_hash,
                                      "logical-digest", digest, params, query,
                                      result_hash, receipt_id, &errmsg);
    sqlite3_free(model_hash);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = errmsg;
      goto receipt_fail;
    }
    for (int i = 0; i < cur->n_rows; i++)
      memcpy(cur->rows[i].receipt_id, receipt_id, sizeof(receipt_id));
  }

  sqlite3_free(resolved_time);
  sqlite3_free(resolved_value);
  forecast_opts_free(&opts);
  cur->i = 0;
  return SQLITE_OK;

receipt_fail:
  sqlite3_free(resolved_time);
  sqlite3_free(resolved_value);
  forecast_opts_free(&opts);
  fc_rows_free(cur);
  return SQLITE_ERROR;
}

static int fc_next(sqlite3_vtab_cursor *pCur) {
  ((forecast_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int fc_eof(sqlite3_vtab_cursor *pCur) {
  forecast_cursor *c = (forecast_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int fc_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  forecast_cursor *c = (forecast_cursor *)pCur;
  ForecastRow *r = &c->rows[c->i];
  switch (col) {
  case FC_COL_SERIES_KEY:
    sqlite3_result_text(ctx, r->series_key ? r->series_key : "", -1,
                        SQLITE_TRANSIENT);
    break;
  case FC_COL_STEP:
    if (r->has_values)
      sqlite3_result_int(ctx, r->step);
    break;
  case FC_COL_TS:
    if (r->has_values)
      sqlite3_result_text(ctx, r->ts, -1, SQLITE_TRANSIENT);
    break;
  case FC_COL_FORECAST:
    if (r->has_values)
      sqlite3_result_double(ctx, r->forecast);
    break;
  case FC_COL_LOWER:
    if (r->has_values)
      sqlite3_result_double(ctx, r->lower);
    break;
  case FC_COL_UPPER:
    if (r->has_values)
      sqlite3_result_double(ctx, r->upper);
    break;
  case FC_COL_STATUS:
    sqlite3_result_text(ctx, r->status, -1, SQLITE_STATIC);
    break;
  case FC_COL_RECEIPT:
    if (r->receipt_id[0])
      sqlite3_result_text(ctx, r->receipt_id, -1, SQLITE_TRANSIENT);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int fc_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((forecast_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module forecastModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL, /* eponymous only */
    /* xConnect    */ fc_connect,
    /* xBestIndex  */ fc_best_index,
    /* xDisconnect */ fc_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ fc_open,
    /* xClose      */ fc_close,
    /* xFilter     */ fc_filter,
    /* xNext       */ fc_next,
    /* xEof        */ fc_eof,
    /* xColumn     */ fc_column,
    /* xRowid      */ fc_rowid,
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

#pragma region detect_anomalies

/* detect_anomalies(query, options) — RFC §4.2.2. Scoring is causal:
 * every point is scored against a one-step-ahead prediction built from
 * PRIOR points only (residual sigma via Welford, updated after each
 * point is scored), so the anomaly cannot hide inside its own fit. */

#define AN_COL_SERIES_KEY 0
#define AN_COL_TS 1
#define AN_COL_VALUE 2
#define AN_COL_FORECAST 3
#define AN_COL_LOWER 4
#define AN_COL_UPPER 5
#define AN_COL_IS 6
#define AN_COL_PROB 7
#define AN_COL_STATUS 8
#define AN_COL_RECEIPT 9
#define AN_COL_QUERY 10
#define AN_COL_OPTIONS 11

typedef struct {
  char *series_key;
  char ts[24];
  f64 value, fc, lo, hi, prob;
  int is_anom;
  int has_pred; /* 0 during warmup */
  int has_values;
  const char *status;
  char receipt_id[27];
} AnomRow;

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} anom_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  AnomRow *rows;
  int n_rows;
  int i;
} anom_cursor;

static int anom_row_cmp(const void *a, const void *b) {
  const AnomRow *x = *(AnomRow *const *)a;
  const AnomRow *y = *(AnomRow *const *)b;
  int c = strcmp(x->series_key ? x->series_key : "",
                 y->series_key ? y->series_key : "");
  if (c)
    return c;
  return strcmp(x->ts, y->ts);
}

/* One-step-ahead scorer. Fills pred/sig/valid arrays (index 0 never
 * valid: nothing precedes it). theta_mode swaps the naive base for an
 * SES level over a running seasonal adjustment. */
static void score_online(const f64 *y, int n, int p, int theta_mode,
                         f64 *pred, f64 *sig, u8 *valid) {
  f64 m = 0, M2 = 0; /* Welford over past residuals */
  int c = 0;
  f64 L = y[0]; /* SES level (theta mode) */
  const f64 alpha = 0.3;
  f64 *phase_sum = NULL;
  int *phase_cnt = NULL;
  f64 run_sum = y[0];
  int run_cnt = 1;
  if (theta_mode && p > 1) {
    phase_sum = sqlite3_malloc(sizeof(f64) * p);
    phase_cnt = sqlite3_malloc(sizeof(int) * p);
    if (phase_sum && phase_cnt) {
      memset(phase_sum, 0, sizeof(f64) * p);
      memset(phase_cnt, 0, sizeof(int) * p);
      phase_sum[0] = y[0];
      phase_cnt[0] = 1;
    } else {
      sqlite3_free(phase_sum);
      sqlite3_free(phase_cnt);
      phase_sum = NULL;
      phase_cnt = NULL;
    }
  }

  valid[0] = 0;
  pred[0] = y[0];
  sig[0] = 0;
  for (int t = 1; t < n; t++) {
    f64 pr;
    if (theta_mode) {
      f64 seas = 0;
      if (phase_sum && phase_cnt[t % p] > 0)
        seas = phase_sum[t % p] / phase_cnt[t % p] - run_sum / run_cnt;
      pr = L + seas;
      /* update after prediction */
      f64 deseason = y[t] - seas;
      L += alpha * (deseason - L);
      if (phase_sum) {
        phase_sum[t % p] += y[t];
        phase_cnt[t % p]++;
      }
      run_sum += y[t];
      run_cnt++;
    } else {
      f64 drift = (y[t - 1] - y[0]) / (t > 1 ? (t - 1) : 1);
      if (p > 0 && t >= p)
        pr = y[t - p] + drift * p;
      else
        pr = y[t - 1] + drift;
    }
    pred[t] = pr;
    f64 s = c >= 4 ? sqrt(M2 / c) : 0;
    sig[t] = s;
    valid[t] = (u8)(s > 0 && t >= FORECAST_MIN_HISTORY);

    f64 r = y[t] - pr;
    c++;
    f64 d = r - m;
    m += d / c;
    M2 += d * (r - m);
  }
  sqlite3_free(phase_sum);
  sqlite3_free(phase_cnt);
}

static int an_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  anom_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(series_key TEXT, ts TEXT, value REAL,"
          " forecast REAL, lower_bound REAL, upper_bound REAL,"
          " is_anomaly INTEGER, anomaly_probability REAL, status TEXT,"
          " receipt_id TEXT, query HIDDEN, options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int an_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int an_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_query = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    if (c->iColumn == AN_COL_QUERY) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_query = 1;
    } else if (c->iColumn == AN_COL_OPTIONS) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 2;
      pIdx->aConstraintUsage[i].omit = 1;
    }
  }
  if (!seen_query) {
    pVtab->zErrMsg =
        sqlite3_mprintf("%s: detect_anomalies(query) requires a query",
                        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int an_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  anom_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void an_rows_free(anom_cursor *c) {
  for (int i = 0; i < c->n_rows; i++)
    sqlite3_free(c->rows[i].series_key);
  sqlite3_free(c->rows);
  c->rows = NULL;
  c->n_rows = 0;
  c->i = 0;
}

static int an_close(sqlite3_vtab_cursor *pCur) {
  anom_cursor *c = (anom_cursor *)pCur;
  an_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int an_error(anom_cursor *cur, const char *code, const char *msg,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s%s", code, msg, detail ? detail : "");
  return SQLITE_ERROR;
}

static int an_filter(sqlite3_vtab_cursor *pCur, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  anom_cursor *cur = (anom_cursor *)pCur;
  anom_vtab *vtab = (anom_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  an_rows_free(cur);

  if (argc < 1)
    return an_error(cur, PREDICT_ERR_SCHEMA, "query required", NULL);
  const char *query = (const char *)sqlite3_value_text(argv[0]);
  if (!query)
    return an_error(cur, PREDICT_ERR_SCHEMA, "query must be text", NULL);

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.99; /* anomaly_prob_threshold default */
  opts.context_limit = FORECAST_DEFAULT_CONTEXT_LIMIT;
  opts.receipt = 1;

  const char *options_json =
      argc >= 2 ? (const char *)sqlite3_value_text(argv[1]) : NULL;
  char *errmsg = NULL;
  if (predict0_options_parse(db, options_json, ANOMALY_OPTION_KEYS,
                             forecast_opt_cb, &opts, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }
  int theta_mode = 1; /* default-ts and theta-classic */
  const char *model_id = resolve_ts_model_id(opts.model);
  if (!model_id) {
    int rc = an_error(cur, PREDICT_ERR_MODEL_NOT_FOUND, "no such model: ",
                      opts.model);
    forecast_opts_free(&opts);
    return rc;
  }
  if (strcmp(model_id, "stub-seasonal-naive") == 0)
    theta_mode = 0;

  /* --- lean collection (same validation path as forecast) --- */
  sqlite3_stmt *stmt = NULL;
  const char *tail = NULL;
  int rc = sqlite3_prepare_v2(db, query, -1, &stmt, &tail);
  if (rc != SQLITE_OK || !stmt) {
    if (stmt)
      sqlite3_finalize(stmt);
    rc = an_error(cur, PREDICT_ERR_SCHEMA, "query does not parse: ",
                  sqlite3_errmsg(db));
    forecast_opts_free(&opts);
    return rc;
  }
  while (tail && (*tail == ' ' || *tail == '\n' || *tail == ';' ||
                  *tail == '\t'))
    tail++;
  if ((tail && *tail != '\0') || !sqlite3_stmt_readonly(stmt)) {
    int readonly = sqlite3_stmt_readonly(stmt);
    sqlite3_finalize(stmt);
    forecast_opts_free(&opts);
    return readonly
               ? an_error(cur, PREDICT_ERR_SCHEMA,
                          "query must be a single statement", NULL)
               : an_error(cur, PREDICT_ERR_QUERY_NOT_READONLY,
                          "query must be a read-only SELECT", NULL);
  }
  int ncol = sqlite3_column_count(stmt);
  if (ncol < 2) {
    sqlite3_finalize(stmt);
    forecast_opts_free(&opts);
    return an_error(cur, PREDICT_ERR_SCHEMA,
                    "query must yield at least a time and a value column",
                    NULL);
  }
  int time_idx = -1, value_idx = -1;
  int group_idx[FORECAST_MAX_GROUP_COLS];
  for (int g = 0; g < opts.n_group_cols; g++)
    group_idx[g] = -1;
  for (int i = 0; i < ncol; i++) {
    const char *name = sqlite3_column_name(stmt, i);
    if (opts.time_col && name && strcmp(name, opts.time_col) == 0)
      time_idx = i;
    if (opts.value_col && name && strcmp(name, opts.value_col) == 0)
      value_idx = i;
    for (int g = 0; g < opts.n_group_cols; g++)
      if (name && strcmp(name, opts.group_cols[g]) == 0)
        group_idx[g] = i;
  }
  if ((opts.time_col && time_idx < 0) || (opts.value_col && value_idx < 0)) {
    sqlite3_finalize(stmt);
    int r = an_error(cur, PREDICT_ERR_SCHEMA, "no such named column", NULL);
    forecast_opts_free(&opts);
    return r;
  }
  for (int g = 0; g < opts.n_group_cols; g++) {
    if (group_idx[g] < 0) {
      sqlite3_finalize(stmt);
      int r = an_error(cur, PREDICT_ERR_SCHEMA, "no such group col: ",
                       opts.group_cols[g]);
      forecast_opts_free(&opts);
      return r;
    }
  }

  SeriesBuf *series = NULL;
  int n_series = 0, series_cap = 0;
  i64 total_rows = 0;
  int inference_done = (time_idx >= 0 && value_idx >= 0);
  char keybuf[512];

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (!inference_done) {
      for (int i = 0; time_idx < 0 && i < ncol; i++) {
        int t = sqlite3_column_type(stmt, i);
        if (t == SQLITE_INTEGER) {
          time_idx = i;
        } else if (t == SQLITE_TEXT) {
          i64 ms;
          if (predict0_parse_timestamp(
                  (const char *)sqlite3_column_text(stmt, i), &ms) == 0)
            time_idx = i;
        }
      }
      for (int i = 0; value_idx < 0 && i < ncol; i++) {
        if (i == time_idx)
          continue;
        int in_groups = 0;
        for (int g = 0; g < opts.n_group_cols; g++)
          if (group_idx[g] == i)
            in_groups = 1;
        if (in_groups)
          continue;
        int t = sqlite3_column_type(stmt, i);
        if (t == SQLITE_INTEGER || t == SQLITE_FLOAT)
          value_idx = i;
      }
      if (time_idx < 0 || value_idx < 0)
        break;
      inference_done = 1;
    }
    if (++total_rows > FORECAST_MAX_TOTAL_ROWS)
      break;

    keybuf[0] = '\0';
    usize kl = 0;
    for (int g = 0; g < opts.n_group_cols; g++) {
      const char *gv = (const char *)sqlite3_column_text(stmt, group_idx[g]);
      int wrote = snprintf(keybuf + kl, sizeof(keybuf) - kl, "%s%s",
                           g ? "\x1f" : "", gv ? gv : "");
      if (wrote > 0)
        kl += (usize)wrote;
      if (kl >= sizeof(keybuf) - 1)
        break;
    }
    SeriesBuf *s = series_find(&series, &n_series, &series_cap, keybuf);
    if (!s) {
      rc = SQLITE_NOMEM;
      break;
    }
    if (s->non_numeric)
      continue;
    i64 ms = 0;
    int tt = sqlite3_column_type(stmt, time_idx);
    if (tt == SQLITE_INTEGER) {
      ms = sqlite3_column_int64(stmt, time_idx) * 1000;
    } else if (predict0_parse_timestamp(
                   (const char *)sqlite3_column_text(stmt, time_idx), &ms)) {
      s->non_numeric = 1;
      continue;
    }
    int vt = sqlite3_column_type(stmt, value_idx);
    if (vt != SQLITE_INTEGER && vt != SQLITE_FLOAT) {
      s->non_numeric = 1;
      continue;
    }
    if (s->n == s->cap) {
      int nc = s->cap ? s->cap * 2 : 64;
      i64 *nts = sqlite3_realloc(s->ts, sizeof(i64) * nc);
      f64 *nval = sqlite3_realloc(s->val, sizeof(f64) * nc);
      if (!nts || !nval) {
        sqlite3_free(nts);
        rc = SQLITE_NOMEM;
        break;
      }
      s->ts = nts;
      s->val = nval;
      s->cap = nc;
    }
    s->ts[s->n] = ms;
    s->val[s->n] = sqlite3_column_double(stmt, value_idx);
    s->n++;
  }
  int schema_fail = !inference_done && total_rows > 0;
  int too_many = total_rows > FORECAST_MAX_TOTAL_ROWS;
  char *resolved_time = NULL, *resolved_value = NULL;
  if (time_idx >= 0)
    resolved_time = sqlite3_mprintf("%s", sqlite3_column_name(stmt, time_idx));
  if (value_idx >= 0)
    resolved_value =
        sqlite3_mprintf("%s", sqlite3_column_name(stmt, value_idx));
  sqlite3_finalize(stmt);

#define AN_FREE_SERIES()                                                      \
  do {                                                                        \
    for (int i = 0; i < n_series; i++) {                                      \
      sqlite3_free(series[i].key);                                            \
      sqlite3_free(series[i].ts);                                             \
      sqlite3_free(series[i].val);                                            \
    }                                                                         \
    sqlite3_free(series);                                                     \
    sqlite3_free(resolved_time);                                              \
    sqlite3_free(resolved_value);                                             \
    forecast_opts_free(&opts);                                                \
  } while (0)

  if (schema_fail) {
    AN_FREE_SERIES();
    return an_error(cur, PREDICT_ERR_SCHEMA,
                    "could not infer time/value columns", NULL);
  }
  if (too_many) {
    AN_FREE_SERIES();
    return an_error(cur, PREDICT_ERR_RESOURCE, "too many input rows", NULL);
  }
  if (rc == SQLITE_NOMEM) {
    AN_FREE_SERIES();
    return SQLITE_NOMEM;
  }

  /* score every series */
  f64 z_thr = predict0_norm_quantile(0.5 + opts.confidence / 2);
  int max_rows = 0;
  for (int i = 0; i < n_series; i++) {
    SeriesBuf *s = &series[i];
    int usable = !s->non_numeric && s->n >= FORECAST_MIN_HISTORY;
    max_rows += usable ? (s->n > opts.context_limit ? opts.context_limit
                                                    : s->n)
                       : 1;
  }
  cur->rows = sqlite3_malloc(sizeof(AnomRow) * (max_rows ? max_rows : 1));
  if (!cur->rows) {
    AN_FREE_SERIES();
    return SQLITE_NOMEM;
  }
  cur->n_rows = 0;

  for (int i = 0; i < n_series; i++) {
    SeriesBuf *s = &series[i];
    const char *status = NULL;
    if (s->non_numeric)
      status = "non_numeric";
    else if (s->n < FORECAST_MIN_HISTORY)
      status = "insufficient_history";
    if (status) {
      AnomRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->status = status;
      continue;
    }
    series_sort(s);
    int truncated = 0;
    f64 *y = s->val;
    i64 *ts = s->ts;
    int n = s->n;
    if (n > opts.context_limit) {
      y += n - opts.context_limit;
      ts += n - opts.context_limit;
      n = opts.context_limit;
      truncated = 1;
    }
    int p = detect_period(y, n);
    f64 *pred = sqlite3_malloc(sizeof(f64) * n);
    f64 *sig = sqlite3_malloc(sizeof(f64) * n);
    u8 *valid = sqlite3_malloc(n);
    if (!pred || !sig || !valid) {
      sqlite3_free(pred);
      sqlite3_free(sig);
      sqlite3_free(valid);
      break;
    }
    score_online(y, n, p, theta_mode, pred, sig, valid);
    for (int t = 0; t < n; t++) {
      AnomRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      predict0_format_timestamp(ts[t], r->ts, sizeof(r->ts));
      r->value = y[t];
      r->has_values = 1;
      r->status = truncated ? "truncated" : "ok";
      if (valid[t]) {
        r->has_pred = 1;
        r->fc = pred[t];
        r->lo = pred[t] - z_thr * sig[t];
        r->hi = pred[t] + z_thr * sig[t];
        f64 z = (y[t] - pred[t]) / sig[t];
        r->prob = erf(fabs(z) / 1.4142135623730951);
        r->is_anom = r->prob > opts.confidence;
      }
    }
    sqlite3_free(pred);
    sqlite3_free(sig);
    sqlite3_free(valid);
  }

  /* receipt */
  if (opts.receipt) {
    char *rerr = NULL;
    if (predict0_receipts_ensure(db, &rerr)) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = rerr;
      AN_FREE_SERIES();
      an_rows_free(cur);
      return SQLITE_ERROR;
    }
    char *model_hash = predict0_registry_model_hash(db, model_id);
    char digest[65];
    if (!model_hash || predict0_logical_digest(db, digest, &rerr)) {
      sqlite3_free(model_hash);
      if (rerr) {
        sqlite3_free(vtab->base.zErrMsg);
        vtab->base.zErrMsg = rerr;
      } else {
        an_error(cur, PREDICT_ERR_MODEL_NOT_FOUND, model_id, NULL);
      }
      AN_FREE_SERIES();
      an_rows_free(cur);
      return SQLITE_ERROR;
    }

    AnomRow **order =
        sqlite3_malloc(sizeof(AnomRow *) * (cur->n_rows ? cur->n_rows : 1));
    if (!order) {
      sqlite3_free(model_hash);
      AN_FREE_SERIES();
      an_rows_free(cur);
      return SQLITE_NOMEM;
    }
    for (int i = 0; i < cur->n_rows; i++)
      order[i] = &cur->rows[i];
    qsort(order, (usize)cur->n_rows, sizeof(AnomRow *), anom_row_cmp);
    predict0_hasher h;
    predict0_hash_init(&h);
    for (int i = 0; i < cur->n_rows; i++) {
      AnomRow *r = order[i];
      predict0_hash_text(&h, r->series_key);
      if (r->has_values) {
        predict0_hash_text(&h, r->ts);
        predict0_hash_real(&h, r->value);
      } else {
        predict0_hash_null(&h);
        predict0_hash_null(&h);
      }
      if (r->has_pred) {
        predict0_hash_real(&h, r->fc);
        predict0_hash_real(&h, r->lo);
        predict0_hash_real(&h, r->hi);
      } else {
        predict0_hash_null(&h);
        predict0_hash_null(&h);
        predict0_hash_null(&h);
      }
      if (r->has_values) {
        predict0_hash_int(&h, r->is_anom);
        predict0_hash_real(&h, r->prob);
      } else {
        predict0_hash_null(&h);
        predict0_hash_null(&h);
      }
      predict0_hash_row_end(&h);
    }
    sqlite3_free(order);
    char result_hash[65];
    predict0_hash_hex(&h, result_hash);

    char group_json[600] = "";
    if (opts.n_group_cols) {
      usize gl = 0;
      gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl, "[");
      for (int g = 0; g < opts.n_group_cols && gl < sizeof(group_json) - 4;
           g++)
        gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl,
                              "%s\"%s\"", g ? "," : "", opts.group_cols[g]);
      snprintf(group_json + gl, sizeof(group_json) - gl, "]");
    }
    sqlite3_stmt *pj = NULL;
    char *params = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_object('anomaly_prob_threshold', ?1,"
            " 'context_limit', ?2,"
            " 'group_cols', CASE WHEN ?3 = '' THEN NULL ELSE json(?3) END,"
            " 'model', ?4, 'receipt', 1, 'time_col', ?5, 'value_col', ?6)",
            -1, &pj, NULL) == SQLITE_OK) {
      sqlite3_bind_double(pj, 1, opts.confidence);
      sqlite3_bind_int(pj, 2, opts.context_limit);
      sqlite3_bind_text(pj, 3, group_json, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 4, model_id, -1, SQLITE_STATIC);
      if (resolved_time)
        sqlite3_bind_text(pj, 5, resolved_time, -1, SQLITE_STATIC);
      if (resolved_value)
        sqlite3_bind_text(pj, 6, resolved_value, -1, SQLITE_STATIC);
      if (sqlite3_step(pj) == SQLITE_ROW)
        params = sqlite3_mprintf(
            "%s", (const char *)sqlite3_column_text(pj, 0));
      sqlite3_finalize(pj);
    }
    if (!params) {
      sqlite3_free(model_hash);
      AN_FREE_SERIES();
      an_rows_free(cur);
      return an_error(cur, PREDICT_ERR_RESOURCE,
                      "could not canonicalize params", NULL);
    }
    char receipt_id[27];
    int irc = predict0_receipt_insert(db, "detect_anomalies", model_id,
                                      model_hash, "logical-digest", digest,
                                      params, query, result_hash, receipt_id,
                                      &rerr);
    sqlite3_free(model_hash);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = rerr;
      AN_FREE_SERIES();
      an_rows_free(cur);
      return SQLITE_ERROR;
    }
    for (int i = 0; i < cur->n_rows; i++)
      memcpy(cur->rows[i].receipt_id, receipt_id, sizeof(receipt_id));
  }

  AN_FREE_SERIES();
#undef AN_FREE_SERIES
  cur->i = 0;
  return SQLITE_OK;
}

static int an_next(sqlite3_vtab_cursor *pCur) {
  ((anom_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int an_eof(sqlite3_vtab_cursor *pCur) {
  anom_cursor *c = (anom_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int an_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  anom_cursor *c = (anom_cursor *)pCur;
  AnomRow *r = &c->rows[c->i];
  switch (col) {
  case AN_COL_SERIES_KEY:
    sqlite3_result_text(ctx, r->series_key ? r->series_key : "", -1,
                        SQLITE_TRANSIENT);
    break;
  case AN_COL_TS:
    if (r->has_values)
      sqlite3_result_text(ctx, r->ts, -1, SQLITE_TRANSIENT);
    break;
  case AN_COL_VALUE:
    if (r->has_values)
      sqlite3_result_double(ctx, r->value);
    break;
  case AN_COL_FORECAST:
    if (r->has_pred)
      sqlite3_result_double(ctx, r->fc);
    break;
  case AN_COL_LOWER:
    if (r->has_pred)
      sqlite3_result_double(ctx, r->lo);
    break;
  case AN_COL_UPPER:
    if (r->has_pred)
      sqlite3_result_double(ctx, r->hi);
    break;
  case AN_COL_IS:
    if (r->has_values)
      sqlite3_result_int(ctx, r->is_anom);
    break;
  case AN_COL_PROB:
    if (r->has_values)
      sqlite3_result_double(ctx, r->prob);
    break;
  case AN_COL_STATUS:
    sqlite3_result_text(ctx, r->status, -1, SQLITE_STATIC);
    break;
  case AN_COL_RECEIPT:
    if (r->receipt_id[0])
      sqlite3_result_text(ctx, r->receipt_id, -1, SQLITE_TRANSIENT);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int an_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((anom_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module anomModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ an_connect,
    /* xBestIndex  */ an_best_index,
    /* xDisconnect */ an_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ an_open,
    /* xClose      */ an_close,
    /* xFilter     */ an_filter,
    /* xNext       */ an_next,
    /* xEof        */ an_eof,
    /* xColumn     */ an_column,
    /* xRowid      */ an_rowid,
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

int predict0_forecast_init(sqlite3 *db) {
  int rc = sqlite3_create_module(db, "forecast", &forecastModule, NULL);
  if (rc != SQLITE_OK)
    return rc;
  return sqlite3_create_module(db, "detect_anomalies", &anomModule, NULL);
}
