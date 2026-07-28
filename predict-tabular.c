/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* predict(train_query, apply_query, options) — RFC 0005 §4.2.5.
 * v0 backing: knn5-incontext (kind 'tabular-stat'), the measured honest
 * baseline from the benchmark campaign: z-scored 5-NN with per-column
 * categorical coding. Foundation-model teachers arrive via distill_predict()
 * and the opt-in ONNX build; the benchmarks showed raw in-context FMs
 * are teachers, not serving paths (30s+/call on CPU). */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define TAB_MAX_TRAIN 100000 /* knn5-incontext declared context capacity */
#define TAB_MAX_FEAT PREDICT_MAX_FEAT
#define TAB_K 5

#define PR_COL_REF 0
#define PR_COL_PRED 1
#define PR_COL_CONF 2
#define PR_COL_STATUS 3
#define PR_COL_TRAINQ 4
#define PR_COL_APPLYQ 5
#define PR_COL_OPTIONS 6

typedef struct {
  int type; /* SQLITE_INTEGER/FLOAT/TEXT/NULL */
  i64 i;
  f64 f;
  char *t;
} RefVal;

typedef struct {
  RefVal ref;
  char *prediction; /* NULL when status row */
  f64 confidence;
  int has_conf;
  const char *status;
} PredRow;

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} pred_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  PredRow *rows;
  int n_rows;
  int i;
} pred_cursor;

typedef struct {
  char *target;
  char *task; /* 'classify' | 'regress' */
  char *model;
  char *device;         /* onnx: 'cpu'|'coreml'|'cuda'|'tensorrt' */
  char *precision;      /* onnx: 'fp32'|'fp16'|'int8' */
  char *accept_license; /* onnx: SPDX the caller accepts */
} PredOpts;

static void pred_opts_free(PredOpts *o) {
  sqlite3_free(o->target);
  sqlite3_free(o->task);
  sqlite3_free(o->model);
  sqlite3_free(o->device);
  sqlite3_free(o->precision);
  sqlite3_free(o->accept_license);
}

static int pred_opt_cb(void *ctx, const char *key, sqlite3_value *value,
                       char **errmsg) {
  PredOpts *o = ctx;
  if (sqlite3_value_type(value) != SQLITE_TEXT) {
    *errmsg = sqlite3_mprintf("%s: wrong type for option '%s'",
                              PREDICT_ERR_OPTIONS, key);
    return 1;
  }
  /* free-before-assign: a duplicate JSON key invokes this twice for the
   * same option, and the second assignment must not leak the first */
  if (strcmp(key, "target") == 0) {
    sqlite3_free(o->target);
    o->target = sqlite3_mprintf(
        "%s", (const char *)sqlite3_value_text(value));
  } else if (strcmp(key, "task") == 0) {
    sqlite3_free(o->task);
    o->task = sqlite3_mprintf(
        "%s", (const char *)sqlite3_value_text(value));
  } else if (strcmp(key, "model") == 0) {
    sqlite3_free(o->model);
    o->model = sqlite3_mprintf(
        "%s", (const char *)sqlite3_value_text(value));
  } else if (strcmp(key, "device") == 0) {
    sqlite3_free(o->device);
    o->device = sqlite3_mprintf(
        "%s", (const char *)sqlite3_value_text(value));
  } else if (strcmp(key, "precision") == 0) {
    sqlite3_free(o->precision);
    o->precision = sqlite3_mprintf(
        "%s", (const char *)sqlite3_value_text(value));
  } else if (strcmp(key, "accept_license") == 0) {
    sqlite3_free(o->accept_license);
    o->accept_license = sqlite3_mprintf(
        "%s", (const char *)sqlite3_value_text(value));
  }
  return 0;
}

static const char *const PRED_OPTION_KEYS[] = {
    "target", "task", "model", "device", "precision", "accept_license", NULL};

/* per-feature categorical vocabulary (linear; fine at spike scale) */
typedef struct {
  char **words;
  int n, cap;
} Vocab;

static f64 vocab_code(Vocab *v, const char *w) {
  for (int i = 0; i < v->n; i++)
    if (strcmp(v->words[i], w) == 0)
      return (f64)i;
  if (v->n == v->cap) {
    int nc = v->cap ? v->cap * 2 : 8;
    char **g = sqlite3_realloc(v->words, sizeof(char *) * nc);
    if (!g)
      return -1;
    v->words = g;
    v->cap = nc;
  }
  v->words[v->n] = sqlite3_mprintf("%s", w);
  return (f64)v->n++;
}

static int pr_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  pred_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(row_ref, prediction TEXT, confidence REAL,"
          " status TEXT, train_query HIDDEN,"
          " apply_query HIDDEN, options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int pr_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int pr_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_train = 0, seen_apply = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    int argv = c->iColumn == PR_COL_TRAINQ   ? 1
               : c->iColumn == PR_COL_APPLYQ ? 2
               : c->iColumn == PR_COL_OPTIONS ? 3
                                              : 0;
    if (!argv)
      continue;
    if (!c->usable)
      return SQLITE_CONSTRAINT;
    pIdx->aConstraintUsage[i].argvIndex = argv;
    pIdx->aConstraintUsage[i].omit = 1;
    if (argv == 1)
      seen_train = 1;
    if (argv == 2)
      seen_apply = 1;
  }
  if (!seen_train || !seen_apply) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: predict(train_query, apply_query) requires both arguments",
        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int pr_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  pred_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void pr_rows_free(pred_cursor *c) {
  for (int i = 0; i < c->n_rows; i++) {
    sqlite3_free(c->rows[i].ref.t);
    sqlite3_free(c->rows[i].prediction);
  }
  sqlite3_free(c->rows);
  c->rows = NULL;
  c->n_rows = 0;
  c->i = 0;
}

static int pr_close(sqlite3_vtab_cursor *pCur) {
  pred_cursor *c = (pred_cursor *)pCur;
  pr_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int pr_error(pred_cursor *cur, const char *code, const char *msg,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s%s", code, msg, detail ? detail : "");
  return SQLITE_ERROR;
}

/* prepare + validate an inner query (read-only, single statement) */
static sqlite3_stmt *pr_prepare(pred_cursor *cur, sqlite3 *db,
                                const char *sql, int *err) {
  sqlite3_stmt *stmt = NULL;
  char *emsg = NULL;
  *err = 0;
  if (predict0_prepare_ro(db, sql, "query", &stmt, &emsg) != SQLITE_OK) {
    sqlite3_free(cur->base.pVtab->zErrMsg);
    cur->base.pVtab->zErrMsg = emsg; /* already "CODE: detail" formed */
    *err = SQLITE_ERROR;
    return NULL;
  }
  return stmt;
}

/* Adopt a runtime backend's neutral result array into the cursor's PredRow
 * rows, transferring string ownership so the neutral array frees cleanly. */
static int pr_adopt(pred_cursor *cur, predict0_result *res, int n) {
  if (n > 0) {
    cur->rows = sqlite3_malloc(sizeof(PredRow) * n);
    if (!cur->rows) {
      predict0_results_free(res, n);
      return SQLITE_NOMEM;
    }
    memset(cur->rows, 0, sizeof(PredRow) * n);
  }
  for (int i = 0; i < n; i++) {
    PredRow *o = &cur->rows[i];
    predict0_result *r = &res[i];
    o->ref.type = r->ref_type;
    o->ref.i = r->ref_i;
    o->ref.f = r->ref_f;
    o->ref.t = r->ref_t; /* transfer */
    r->ref_t = NULL;
    o->prediction = r->prediction; /* transfer */
    r->prediction = NULL;
    o->confidence = r->confidence;
    o->has_conf = r->has_conf;
    o->status = r->status;
    cur->n_rows++;
  }
  predict0_results_free(res, n);
  cur->i = 0;
  return SQLITE_OK;
}

static void backend_opts_from(predict0_backend_opts *b, PredOpts *o) {
  b->device = o->device;
  b->precision = o->precision;
  b->accept_license = o->accept_license;
}

/* Dispatch to the native tree student (zero-dependency core) and adopt. */
static int run_student(pred_cursor *cur, sqlite3 *db, const char *model_id,
                       const char *apply_sql, const predict0_model_row *m,
                       PredOpts *opts) {
  predict0_backend_opts bopts;
  backend_opts_from(&bopts, opts);
  predict0_result *res = NULL;
  int n = 0;
  char *emsg = NULL;
  int rc = predict0_tree_run(db, model_id, apply_sql, m, &bopts, &res, &n,
                             &emsg);
  if (rc != SQLITE_OK) {
    sqlite3_free(cur->base.pVtab->zErrMsg);
    cur->base.pVtab->zErrMsg = emsg;
    return SQLITE_ERROR;
  }
  return pr_adopt(cur, res, n);
}

#ifdef SQLITE_PREDICT_ONNX
/* Dispatch to the onnx backend and adopt its results into the cursor. */
static int run_onnx(pred_cursor *cur, sqlite3 *db, const char *model_id,
                    const char *train_sql, const char *apply_sql,
                    const predict0_model_row *m, PredOpts *opts) {
  predict0_backend_opts bopts;
  backend_opts_from(&bopts, opts);
  predict0_result *res = NULL;
  int n = 0;
  char *emsg = NULL;
  int rc = predict0_onnx_predict(db, model_id, train_sql, apply_sql, m, &bopts,
                                 &res, &n, &emsg);
  if (rc != SQLITE_OK) {
    sqlite3_free(cur->base.pVtab->zErrMsg);
    cur->base.pVtab->zErrMsg = emsg; /* already PREDICT_ERR_*-prefixed */
    return SQLITE_ERROR;
  }
  return pr_adopt(cur, res, n);
}
#endif

static int pr_filter(sqlite3_vtab_cursor *pCur, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  pred_cursor *cur = (pred_cursor *)pCur;
  pred_vtab *vtab = (pred_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  pr_rows_free(cur);

  if (argc < 2)
    return pr_error(cur, PREDICT_ERR_SCHEMA,
                    "train_query and apply_query required", NULL);
  const char *train_sql =
      sqlite3_value_type(argv[0]) == SQLITE_NULL
          ? NULL
          : (const char *)sqlite3_value_text(argv[0]);
  const char *apply_sql = (const char *)sqlite3_value_text(argv[1]);

  PredOpts opts;
  memset(&opts, 0, sizeof(opts));
  char *errmsg = NULL;
  const char *options_json =
      argc >= 3 ? (const char *)sqlite3_value_text(argv[2]) : NULL;
  if (predict0_options_parse(db, options_json, PRED_OPTION_KEYS, pred_opt_cb,
                             &opts, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    pred_opts_free(&opts);
    return SQLITE_ERROR;
  }

  int rc = SQLITE_ERROR;
  const char *model_id = opts.model && strcmp(opts.model, "default-tabular")
                             ? opts.model
                             : "knn5-incontext";

  /* Non-default model: dispatch through the registry to a runtime backend.
   * The default knn path below never consults the registry, so it runs on
   * read-only databases. */
  if (strcmp(model_id, "knn5-incontext") != 0) {
    predict0_model_row m;
    int look = predict0_registry_lookup(db, model_id, &m);
    if (look == 1) {
      rc = pr_error(cur, PREDICT_ERR_MODEL_NOT_FOUND, "no such model: ",
                    model_id);
      pred_opts_free(&opts);
      return rc;
    }
    if (look == 2) {
      rc = pr_error(cur, PREDICT_ERR_MODEL_HASH,
                    "weights do not match content_hash: ", model_id);
      pred_opts_free(&opts);
      return rc;
    }
    if (look != 0) {
      rc = pr_error(cur, PREDICT_ERR_RESOURCE,
                    "model registry unavailable (needs a writable db)", NULL);
      pred_opts_free(&opts);
      return rc;
    }
    if (m.runtime && strcmp(m.runtime, "tree") == 0) {
      if (train_sql) {
        /* a student carries its training in its blob; a train_query
         * here would be silently ignored, so reject it loudly */
        predict0_model_row_free(&m);
        rc = pr_error(cur, PREDICT_ERR_OPTIONS,
                      "a distilled student takes no train_query; pass NULL: ",
                      model_id);
        pred_opts_free(&opts);
        return rc;
      }
      rc = run_student(cur, db, model_id, apply_sql, &m, &opts);
    } else if (m.runtime && strcmp(m.runtime, "onnx") == 0) {
#ifdef SQLITE_PREDICT_ONNX
      rc = run_onnx(cur, db, model_id, train_sql, apply_sql, &m, &opts);
#else
      rc = pr_error(cur, PREDICT_ERR_RUNTIME_UNAVAILABLE,
                    "onnx runtime is not in this build: ", model_id);
#endif
    } else {
      rc = pr_error(cur, PREDICT_ERR_RUNTIME_UNAVAILABLE,
                    "predict() cannot serve models with runtime: ",
                    m.runtime ? m.runtime : "unknown");
    }
    predict0_model_row_free(&m);
    pred_opts_free(&opts);
    return rc;
  }

  /* default knn5-incontext path: the onnx-only options make no sense here */
  if (opts.device || opts.precision || opts.accept_license) {
    rc = pr_error(cur, PREDICT_ERR_OPTIONS,
                  "device/precision/accept_license require an onnx model",
                  NULL);
    pred_opts_free(&opts);
    return rc;
  }
  if (!train_sql) {
    rc = pr_error(cur, PREDICT_ERR_SCHEMA,
                  "train_query is required for in-context models", NULL);
    pred_opts_free(&opts);
    return rc;
  }
  int classify;
  if (!opts.task || strcmp(opts.task, "classify") == 0)
    classify = 1;
  else if (strcmp(opts.task, "regress") == 0)
    classify = 0;
  else {
    rc = pr_error(cur, PREDICT_ERR_TASK, "task must be classify|regress: ",
                  opts.task);
    pred_opts_free(&opts);
    return rc;
  }
  if (!opts.target) {
    rc = pr_error(cur, PREDICT_ERR_TARGET, "target option is required",
                  NULL);
    pred_opts_free(&opts);
    return rc;
  }

  /* ---- train pass ---- */
  int err = 0;
  sqlite3_stmt *ts = pr_prepare(cur, db, train_sql, &err);
  if (!ts) {
    pred_opts_free(&opts);
    return err;
  }
  int tn = sqlite3_column_count(ts);
  int target_idx = -1;
  char *feat_names[TAB_MAX_FEAT];
  int feat_idx[TAB_MAX_FEAT];
  int nfeat = 0;
  for (int i = 0; i < tn; i++) {
    const char *nm = sqlite3_column_name(ts, i);
    if (nm && strcmp(nm, opts.target) == 0) {
      target_idx = i;
    } else {
      if (nfeat == TAB_MAX_FEAT) {
        sqlite3_finalize(ts);
        for (int f = 0; f < nfeat; f++)
          sqlite3_free(feat_names[f]);
        rc = pr_error(cur, PREDICT_ERR_SCHEMA,
                      "too many feature columns (max 64)", NULL);
        pred_opts_free(&opts);
        return rc;
      }
      feat_names[nfeat] = sqlite3_mprintf("%s", nm ? nm : "");
      feat_idx[nfeat] = i;
      nfeat++;
    }
  }
  if (target_idx < 0 || nfeat == 0) {
    sqlite3_finalize(ts);
    for (int i = 0; i < nfeat; i++)
      sqlite3_free(feat_names[i]);
    rc = pr_error(cur, target_idx < 0 ? PREDICT_ERR_TARGET
                                      : PREDICT_ERR_SCHEMA,
                  target_idx < 0 ? "no such target column: "
                                 : "train_query needs feature columns",
                  target_idx < 0 ? opts.target : NULL);
    pred_opts_free(&opts);
    return rc;
  }

  Vocab vocab[TAB_MAX_FEAT];
  memset(vocab, 0, sizeof(vocab));
  int is_text[TAB_MAX_FEAT];
  for (int i = 0; i < nfeat; i++)
    is_text[i] = -1; /* undecided */

  f64 *X = NULL;
  char **ylab = NULL;
  f64 *yval = NULL;
  int ntr = 0, cap = 0;
  int bad_target = 0;

  while ((rc = sqlite3_step(ts)) == SQLITE_ROW) {
    if (ntr >= TAB_MAX_TRAIN) {
      sqlite3_finalize(ts);
      rc = pr_error(cur, PREDICT_ERR_CONTEXT_TOO_LARGE,
                    "train set exceeds knn5-incontext capacity", NULL);
      goto fail_train;
    }
    if (ntr == cap) {
      cap = cap ? cap * 2 : 256;
      f64 *gX = sqlite3_realloc(X, sizeof(f64) * cap * nfeat);
      char **gl = sqlite3_realloc(ylab, sizeof(char *) * cap);
      f64 *gv = sqlite3_realloc(yval, sizeof(f64) * cap);
      if (!gX || !gl || !gv) {
        sqlite3_finalize(ts);
        rc = SQLITE_NOMEM;
        goto fail_train;
      }
      X = gX;
      ylab = gl;
      yval = gv;
    }
    for (int i = 0; i < nfeat; i++) {
      int ct = sqlite3_column_type(ts, feat_idx[i]);
      if (is_text[i] < 0 && ct != SQLITE_NULL)
        is_text[i] = ct == SQLITE_TEXT;
      f64 v = 0;
      if (ct == SQLITE_NULL) {
        v = 0;
      } else if (is_text[i] == 1) {
        v = vocab_code(&vocab[i],
                       (const char *)sqlite3_column_text(ts, feat_idx[i]));
      } else {
        v = sqlite3_column_double(ts, feat_idx[i]);
      }
      X[ntr * nfeat + i] = v;
    }
    int tt = sqlite3_column_type(ts, target_idx);
    if (classify) {
      ylab[ntr] = sqlite3_mprintf(
          "%s", tt == SQLITE_NULL
                    ? ""
                    : (const char *)sqlite3_column_text(ts, target_idx));
    } else {
      ylab[ntr] = NULL;
      if (tt != SQLITE_INTEGER && tt != SQLITE_FLOAT)
        bad_target = 1;
      yval[ntr] = sqlite3_column_double(ts, target_idx);
    }
    ntr++;
  }
  sqlite3_finalize(ts);
  if (rc != SQLITE_DONE) {
    rc = pr_error(cur, PREDICT_ERR_RESOURCE, "train query failed", NULL);
    goto fail_train;
  }
  if (bad_target) {
    rc = pr_error(cur, PREDICT_ERR_TARGET,
                  "regress target must be numeric", NULL);
    goto fail_train;
  }
  if (ntr < TAB_K) {
    rc = pr_error(cur, PREDICT_ERR_SCHEMA,
                  "train set smaller than k=5", NULL);
    goto fail_train;
  }

  /* z-score params */
  f64 mu[TAB_MAX_FEAT], sd[TAB_MAX_FEAT];
  for (int i = 0; i < nfeat; i++) {
    f64 s = 0;
    for (int r = 0; r < ntr; r++)
      s += X[r * nfeat + i];
    mu[i] = s / ntr;
    f64 v = 0;
    for (int r = 0; r < ntr; r++) {
      f64 d = X[r * nfeat + i] - mu[i];
      v += d * d;
    }
    sd[i] = sqrt(v / ntr);
    if (sd[i] == 0)
      sd[i] = 1;
    for (int r = 0; r < ntr; r++)
      X[r * nfeat + i] = (X[r * nfeat + i] - mu[i]) / sd[i];
  }

  /* ---- apply pass ---- */
  sqlite3_stmt *as = pr_prepare(cur, db, apply_sql, &err);
  if (!as) {
    rc = err;
    goto fail_train;
  }
  int an = sqlite3_column_count(as);
  if (an < 2) {
    sqlite3_finalize(as);
    rc = pr_error(cur, PREDICT_ERR_SCHEMA,
                  "apply_query needs a row_ref column plus features", NULL);
    goto fail_train;
  }
  /* map apply cols 1..an-1 to train features by name (exact set) */
  int amap[TAB_MAX_FEAT];
  int nmap = 0;
  for (int i = 1; i < an; i++) {
    const char *nm = sqlite3_column_name(as, i);
    int found = -1;
    for (int f = 0; f < nfeat; f++)
      if (nm && strcmp(nm, feat_names[f]) == 0)
        found = f;
    if (found < 0 || nmap >= TAB_MAX_FEAT) {
      /* copy the name before finalize invalidates it */
      char namebuf[128];
      snprintf(namebuf, sizeof(namebuf), "%s", nm ? nm : "?");
      sqlite3_finalize(as);
      rc = pr_error(cur, PREDICT_ERR_SCHEMA,
                    "apply feature not in train_query: ", namebuf);
      goto fail_train;
    }
    amap[nmap++] = found;
  }
  if (nmap != nfeat) {
    sqlite3_finalize(as);
    rc = pr_error(cur, PREDICT_ERR_SCHEMA,
                  "apply features must match train features exactly", NULL);
    goto fail_train;
  }

  int rcap = 0;
  while ((rc = sqlite3_step(as)) == SQLITE_ROW) {
    if (cur->n_rows == rcap) {
      rcap = rcap ? rcap * 2 : 256;
      PredRow *g = sqlite3_realloc(cur->rows, sizeof(PredRow) * rcap);
      if (!g) {
        sqlite3_finalize(as);
        rc = SQLITE_NOMEM;
        goto fail_train;
      }
      cur->rows = g;
    }
    PredRow *out = &cur->rows[cur->n_rows];
    memset(out, 0, sizeof(*out));
    out->ref.type = sqlite3_column_type(as, 0);
    if (out->ref.type == SQLITE_INTEGER)
      out->ref.i = sqlite3_column_int64(as, 0);
    else if (out->ref.type == SQLITE_FLOAT)
      out->ref.f = sqlite3_column_double(as, 0);
    else if (out->ref.type != SQLITE_NULL)
      out->ref.t =
          sqlite3_mprintf("%s", (const char *)sqlite3_column_text(as, 0));

    f64 q[TAB_MAX_FEAT];
    int row_bad = 0;
    for (int i = 1; i < an; i++) {
      int f = amap[i - 1];
      int ct = sqlite3_column_type(as, i);
      f64 v = 0;
      if (ct == SQLITE_NULL) {
        row_bad = 1;
      } else if (is_text[f] == 1) {
        if (ct != SQLITE_TEXT)
          row_bad = 1;
        else
          v = vocab_code(&vocab[f],
                         (const char *)sqlite3_column_text(as, i));
      } else {
        if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT)
          row_bad = 1;
        else
          v = sqlite3_column_double(as, i);
      }
      q[f] = (v - mu[f]) / sd[f];
    }
    if (row_bad) {
      out->status = "non_numeric";
      cur->n_rows++;
      continue;
    }

    /* 5 nearest by squared euclid; strict < keeps earliest on ties */
    int best[TAB_K];
    f64 bestd[TAB_K];
    for (int k = 0; k < TAB_K; k++) {
      best[k] = -1;
      bestd[k] = 1e300;
    }
    for (int r = 0; r < ntr; r++) {
      f64 d = 0;
      for (int f = 0; f < nfeat; f++) {
        f64 dd = X[r * nfeat + f] - q[f];
        d += dd * dd;
      }
      for (int k = 0; k < TAB_K; k++) {
        if (d < bestd[k]) {
          for (int m = TAB_K - 1; m > k; m--) {
            bestd[m] = bestd[m - 1];
            best[m] = best[m - 1];
          }
          bestd[k] = d;
          best[k] = r;
          break;
        }
      }
    }

    if (classify) {
      /* majority among the K labels; ties -> lexicographically smallest */
      const char *win = NULL;
      int win_count = 0;
      for (int k = 0; k < TAB_K; k++) {
        const char *cand = ylab[best[k]];
        int count = 0;
        for (int m = 0; m < TAB_K; m++)
          if (strcmp(ylab[best[m]], cand) == 0)
            count++;
        if (count > win_count ||
            (count == win_count && (!win || strcmp(cand, win) < 0))) {
          win = cand;
          win_count = count;
        }
      }
      out->prediction = sqlite3_mprintf("%s", win);
      out->confidence = (f64)win_count / TAB_K;
      out->has_conf = 1;
    } else {
      f64 s = 0;
      for (int k = 0; k < TAB_K; k++)
        s += yval[best[k]];
      out->prediction = sqlite3_mprintf("%.17g", s / TAB_K);
    }
    out->status = "ok";
    cur->n_rows++;
  }
  sqlite3_finalize(as);
  if (rc != SQLITE_DONE) {
    rc = pr_error(cur, PREDICT_ERR_RESOURCE, "apply query failed", NULL);
    goto fail_train;
  }

  rc = SQLITE_OK;
  cur->i = 0;
  goto cleanup;

fail_train:
  pr_rows_free(cur);
cleanup:
  for (int i = 0; i < nfeat; i++) {
    sqlite3_free(feat_names[i]);
    for (int w = 0; w < vocab[i].n; w++)
      sqlite3_free(vocab[i].words[w]);
    sqlite3_free(vocab[i].words);
  }
  for (int i = 0; i < ntr; i++)
    sqlite3_free(ylab ? ylab[i] : NULL);
  sqlite3_free(X);
  sqlite3_free(ylab);
  sqlite3_free(yval);
  pred_opts_free(&opts);
  return rc;
}

static int pr_next(sqlite3_vtab_cursor *pCur) {
  ((pred_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int pr_eof(sqlite3_vtab_cursor *pCur) {
  pred_cursor *c = (pred_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int pr_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  pred_cursor *c = (pred_cursor *)pCur;
  PredRow *r = &c->rows[c->i];
  switch (col) {
  case PR_COL_REF:
    switch (r->ref.type) {
    case SQLITE_INTEGER:
      sqlite3_result_int64(ctx, r->ref.i);
      break;
    case SQLITE_FLOAT:
      sqlite3_result_double(ctx, r->ref.f);
      break;
    case SQLITE_NULL:
      break;
    default:
      sqlite3_result_text(ctx, r->ref.t ? r->ref.t : "", -1,
                          SQLITE_TRANSIENT);
      break;
    }
    break;
  case PR_COL_PRED:
    if (r->prediction)
      sqlite3_result_text(ctx, r->prediction, -1, SQLITE_TRANSIENT);
    break;
  case PR_COL_CONF:
    if (r->has_conf)
      sqlite3_result_double(ctx, r->confidence);
    break;
  case PR_COL_STATUS:
    sqlite3_result_text(ctx, r->status, -1, SQLITE_STATIC);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int pr_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((pred_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module predictModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ pr_connect,
    /* xBestIndex  */ pr_best_index,
    /* xDisconnect */ pr_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ pr_open,
    /* xClose      */ pr_close,
    /* xFilter     */ pr_filter,
    /* xNext       */ pr_next,
    /* xEof        */ pr_eof,
    /* xColumn     */ pr_column,
    /* xRowid      */ pr_rowid,
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

int predict0_tabular_init(sqlite3 *db) {
  return sqlite3_create_module(db, "predict", &predictModule, NULL);
}
