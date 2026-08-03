/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* predict(train_query, apply_query, options).
 * v0 backing: knn5-incontext (kind 'tabular-stat'), the strongest
 * measured baseline from the benchmark campaign: z-scored 5-NN with per-column
 * categorical coding. Foundation-model teachers arrive via distill_predict()
 * and the opt-in ONNX build; the benchmarks showed raw in-context FMs
 * are teachers, not serving paths (30s+/call on CPU). */
#include "predict-internal.h"
#include "predict-train.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define TAB_MAX_TRAIN 100000 /* knn5-incontext declared context capacity */
#define TAB_MAX_FEAT PREDICT_MAX_FEAT
#define TAB_K 5

/* Compile-time guarantee (the codebase builds -std=c99, so no _Static_assert):
 * every fixed feature buffer here is sized to TAB_MAX_FEAT, and predict() fills
 * one with a loaded student's features. The student loader caps a student's
 * nfeat at TREE_MAX_FEAT (predict-student.h), so TAB_MAX_FEAT MUST be at least
 * that or a loaded student would overflow these buffers. Both currently alias
 * PREDICT_MAX_FEAT; if they ever diverge this array size goes negative and the
 * build fails here, instead of overflowing at runtime. */
typedef char predict0_assert_tab_buf_holds_max_student
    [(TREE_MAX_FEAT <= TAB_MAX_FEAT) ? 1 : -1];

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
        "%s: predict_batch(train_query, apply_query) requires both arguments",
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
    else if (out->ref.type != SQLITE_NULL) {
      out->ref.t =
          sqlite3_mprintf("%s", (const char *)sqlite3_column_text(as, 0));
      if (!out->ref.t) {
        sqlite3_finalize(as);
        rc = SQLITE_NOMEM;
        goto fail_train;
      }
    }

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

/* ======================================================================
 * fit() aggregate + predict() scalar: the guessable tabular pair.
 * fit([name,] f1, ..., fN, label [, options]) trains a native student over the
 * rows of the statement (features are numeric and positional, the label is the
 * last positional argument, no target option). An optional leading TEXT argument
 * names and registers the model, mirroring predict(model, ...); the same name may
 * instead be given as {"register":"my-id"}, but supplying it both ways is an
 * error. With a name fit() returns the registered id text; with neither it
 * returns the serialized model blob. The optional trailing options is a TEXT JSON
 * object; once a feature and a label precede it (argc - has_name >= 3) a trailing
 * '{...}' is read as options, so a class label at that arity must not itself be a
 * JSON-object string. At the lowest arity (a single feature and the label, as in
 * fit(x, '{}')) the trailing object is the label.
 * predict(model, f1, ..., fN [, options]) serves that student per row, features
 * positional, deserialized once and cached on the model argument.
 * ====================================================================== */

/* ---- fit() aggregate ---- */

typedef struct {
  char *kind; /* 'gbt' (default) | 'tree' */
  char *task; /* 'classify' (default) | 'regress' */
  char *reg;  /* register name, or NULL to return the blob */
} FitOpts;

static void fit_opts_free(FitOpts *o) {
  sqlite3_free(o->kind);
  sqlite3_free(o->task);
  sqlite3_free(o->reg);
}

static int fit_opt_cb(void *ctx, const char *key, sqlite3_value *value,
                      char **errmsg) {
  FitOpts *o = ctx;
  if (sqlite3_value_type(value) != SQLITE_TEXT) {
    *errmsg = sqlite3_mprintf("%s: wrong type for option '%s'",
                              PREDICT_ERR_OPTIONS, key);
    return 1;
  }
  char **slot = (strcmp(key, "kind") == 0 || strcmp(key, "student_kind") == 0)
                    ? &o->kind
                : strcmp(key, "task") == 0     ? &o->task
                : strcmp(key, "register") == 0 ? &o->reg
                                               : NULL;
  if (slot) {
    char *copy = sqlite3_mprintf("%s", (const char *)sqlite3_value_text(value));
    if (!copy) {
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      return 1;
    }
    sqlite3_free(*slot); /* free-before-assign: duplicate JSON key */
    *slot = copy;
  }
  return 0;
}

static const char *const FIT_OPTION_KEYS[] = {"kind", "student_kind", "task",
                                              "register", NULL};

/* Per-aggregate accumulator. Lives in sqlite3_aggregate_context (SQLite owns
 * the block); fit_ctx_free releases the pointers it holds. */
typedef struct {
  int configured;
  int has_opts;
  int has_name;   /* a leading TEXT model-name argument was present */
  char *opts_raw; /* trailing options text of the first row (owned), or NULL */
  int nfeat;
  int classify;
  char *kind;   /* borrowed by predict0_train_student; owned here */
  char *reg_id; /* register name, or NULL */
  f32 *X;       /* row-major [cap, nfeat] */
  char **ylab;  /* [cap] classify labels */
  f64 *yval;    /* [cap] regress values */
  int n, cap;
  int err;      /* SQLITE_ code, reported in fit_final */
  char *errmsg;
} FitCtx;

static void fit_ctx_free(FitCtx *c) {
  if (!c)
    return;
  sqlite3_free(c->X);
  if (c->ylab) {
    for (int i = 0; i < c->n; i++)
      sqlite3_free(c->ylab[i]);
    sqlite3_free(c->ylab);
  }
  sqlite3_free(c->yval);
  sqlite3_free(c->kind);
  sqlite3_free(c->reg_id);
  sqlite3_free(c->opts_raw);
  sqlite3_free(c->errmsg);
  memset(c, 0, sizeof(*c));
}

static void fit_step(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  FitCtx *c = sqlite3_aggregate_context(ctx, sizeof(FitCtx));
  if (!c) {
    /* NULL means the context allocation failed; SQLite prescribes signalling
     * NOMEM here rather than deferring to a misleading no-rows message. */
    sqlite3_result_error_nomem(ctx);
    return;
  }
  if (c->err)
    return;
  /* An optional leading model name: a TEXT first argument. Features are numeric,
   * so a text argv[0] is unambiguously the register name — the guessable mirror of
   * predict(model, ...). Like the options object it must be constant across the
   * group, and it is reconciled against {"register":...} below (supplying the name
   * both ways is an error, not a silent precedence). Detected before the trailing
   * options so the options arity can discount it. */
  int has_name = (argc >= 1 && sqlite3_value_type(argv[0]) == SQLITE_TEXT);
  const char *name_txt = NULL;
  if (has_name) {
    name_txt = (const char *)sqlite3_value_text(argv[0]);
    if (!name_txt) { /* TEXT value but NULL text pointer means an allocation failed */
      c->err = SQLITE_NOMEM;
      return;
    }
    /* The name flows through mprintf("%s") and strcmp below, so an embedded NUL
     * would silently truncate it to a different id than the caller passed. Reject
     * it rather than register/return the wrong name. */
    if (memchr(name_txt, '\0', (size_t)sqlite3_value_bytes(argv[0]))) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: model name must not contain a NUL byte",
                                  PREDICT_ERR_OPTIONS);
      return;
    }
  }

  /* This row's trailing options: the last argument when it is TEXT beginning with
   * '{'. It is options only when a feature and a label still precede it after the
   * optional name (argc - has_name >= 3); with fewer arguments the trailing '{...}'
   * is the label, exactly as the name-less form treats it, so both forms stay
   * consistent. Presence and content must be constant across the group, or the
   * feature/label split and the model config would depend on the order SQLite
   * happens to visit rows in. Consequence: a trailing '{...}' with a feature and
   * label present is ALWAYS the options object, so a class label must not be a
   * JSON-object string (it would be consumed as options). A malformed options
   * object fails loudly at predict0_options_parse; the only quiet case is a label
   * that is itself a well-formed, valid-key options object, which is why the label
   * contract is documented rather than guessed. */
  int has_opts = 0;
  const char *opts_txt = NULL;
  if (argc - has_name >= 3 && sqlite3_value_type(argv[argc - 1]) == SQLITE_TEXT) {
    const char *t = (const char *)sqlite3_value_text(argv[argc - 1]);
    if (t && t[0] == '{') {
      has_opts = 1;
      opts_txt = t;
    }
  }

  if (!c->configured) {
    int nfeat = argc - has_name - 1 - has_opts;
    if (nfeat < 1) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: fit([name,] feature, ..., label [, options])"
                                  " needs at least one feature and a label",
                                  PREDICT_ERR_SCHEMA);
      return;
    }
    if (nfeat > TAB_MAX_FEAT) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: too many feature columns (max %d)",
                                  PREDICT_ERR_SCHEMA, TAB_MAX_FEAT);
      return;
    }
    FitOpts o;
    memset(&o, 0, sizeof(o));
    if (has_opts) {
      char *emsg = NULL;
      if (predict0_options_parse(sqlite3_context_db_handle(ctx), opts_txt,
                                 FIT_OPTION_KEYS, fit_opt_cb, &o, &emsg)) {
        c->err = SQLITE_ERROR;
        c->errmsg = emsg;
        fit_opts_free(&o);
        return;
      }
    }
    /* The model name is given at most once: a leading argument OR {"register":...},
     * never both — a conflicting pair is a mistake, not a precedence to resolve. */
    if (has_name && o.reg) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf(
          "%s: model name given twice (leading argument and \"register\" option)",
          PREDICT_ERR_OPTIONS);
      fit_opts_free(&o);
      return;
    }
    int classify = 1;
    if (o.task) {
      if (strcmp(o.task, "regress") == 0)
        classify = 0;
      else if (strcmp(o.task, "classify") != 0) {
        c->err = SQLITE_ERROR;
        c->errmsg = sqlite3_mprintf("%s: task must be classify|regress: %s",
                                    PREDICT_ERR_TASK, o.task);
        fit_opts_free(&o);
        return;
      }
    }
    c->nfeat = nfeat;
    c->classify = classify;
    c->kind = o.kind; /* transfer ownership */
    o.kind = NULL;
    if (has_name) {
      c->reg_id = sqlite3_mprintf("%s", name_txt);
      if (!c->reg_id) {
        c->err = SQLITE_NOMEM;
        fit_opts_free(&o);
        return;
      }
    } else {
      c->reg_id = o.reg; /* transfer ownership */
      o.reg = NULL;
    }
    fit_opts_free(&o);
    c->has_opts = has_opts;
    c->has_name = has_name;
    if (has_opts) {
      c->opts_raw = sqlite3_mprintf("%s", opts_txt);
      if (!c->opts_raw) {
        c->err = SQLITE_NOMEM;
        return;
      }
    }
    c->configured = 1;
  } else if (has_opts != c->has_opts || has_name != c->has_name ||
             (has_opts && (!c->opts_raw || strcmp(opts_txt, c->opts_raw) != 0)) ||
             (has_name && (!c->reg_id || strcmp(name_txt, c->reg_id) != 0))) {
    c->err = SQLITE_ERROR;
    c->errmsg = sqlite3_mprintf(
        "%s: fit options and model name must be constant within a group",
        PREDICT_ERR_OPTIONS);
    return;
  }

  if (c->n == c->cap) {
    /* Bound the training set: fit() buffers the whole matrix in memory, and an
     * uncapped c->cap*2 (int) goes negative past INT_MAX/2, making the byte size
     * a huge size_t (UB). Cap at TAB_MAX_TRAIN and fail loudly instead. */
    if (c->n >= TAB_MAX_TRAIN) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: fit training set exceeds %d rows",
                                  PREDICT_ERR_CONTEXT_TOO_LARGE, TAB_MAX_TRAIN);
      return;
    }
    int nc = c->cap ? c->cap * 2 : 256;
    if (nc > TAB_MAX_TRAIN)
      nc = TAB_MAX_TRAIN;
    f32 *gx = sqlite3_realloc(c->X, sizeof(f32) * (size_t)nc * c->nfeat);
    if (!gx) {
      c->err = SQLITE_NOMEM;
      return;
    }
    c->X = gx;
    if (c->classify) {
      char **g = sqlite3_realloc(c->ylab, sizeof(char *) * nc);
      if (!g) {
        c->err = SQLITE_NOMEM;
        return;
      }
      c->ylab = g;
    } else {
      f64 *g = sqlite3_realloc(c->yval, sizeof(f64) * nc);
      if (!g) {
        c->err = SQLITE_NOMEM;
        return;
      }
      c->yval = g;
    }
    c->cap = nc;
  }
  f32 *row = &c->X[(size_t)c->n * c->nfeat];
  for (int i = 0; i < c->nfeat; i++) {
    /* features start after the optional leading name (argv[0]) */
    int t = sqlite3_value_type(argv[has_name + i]);
    if (t != SQLITE_INTEGER && t != SQLITE_FLOAT) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: fit feature %d must be numeric",
                                  PREDICT_ERR_SCHEMA, i + 1);
      return;
    }
    f64 fv = sqlite3_value_double(argv[has_name + i]);
    if (!isfinite(fv)) { /* 1e999 -> +Inf as SQLITE_FLOAT; NaN poisons splits */
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: fit feature %d must be finite",
                                  PREDICT_ERR_SCHEMA, i + 1);
      return;
    }
    row[i] = (f32)fv;
  }
  sqlite3_value *lv = argv[has_name + c->nfeat];
  if (c->classify) {
    /* A NULL class label is not a real class; fail loudly rather than train a
     * silent empty-string class. A NULL text under memory pressure is NOMEM. */
    if (sqlite3_value_type(lv) == SQLITE_NULL) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: classify label must not be NULL",
                                  PREDICT_ERR_TARGET);
      return;
    }
    const char *s = (const char *)sqlite3_value_text(lv);
    if (!s) {
      c->err = SQLITE_NOMEM;
      return;
    }
    c->ylab[c->n] = sqlite3_mprintf("%s", s);
    if (!c->ylab[c->n]) {
      c->err = SQLITE_NOMEM;
      return;
    }
  } else {
    int t = sqlite3_value_type(lv);
    if (t != SQLITE_INTEGER && t != SQLITE_FLOAT) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: regress label must be numeric",
                                  PREDICT_ERR_TARGET);
      return;
    }
    f64 lval = sqlite3_value_double(lv);
    if (!isfinite(lval)) {
      c->err = SQLITE_ERROR;
      c->errmsg = sqlite3_mprintf("%s: regress label must be finite",
                                  PREDICT_ERR_TARGET);
      return;
    }
    c->yval[c->n] = lval;
  }
  c->n++;
}

static void fit_final(sqlite3_context *ctx) {
  FitCtx *c = sqlite3_aggregate_context(ctx, 0);
  if (!c) {
    sqlite3_result_error(
        ctx, PREDICT_ERR_SCHEMA ": fit received no training rows", -1);
    return;
  }
  /* A first-row config error (bad task, bad arity, options parse failure) sets
   * c->err but returns before c->configured, so surface the real diagnostic
   * before the generic no-rows message, or it would mask every such failure. */
  if (c->err) {
    if (c->err == SQLITE_NOMEM && !c->errmsg)
      sqlite3_result_error_nomem(ctx);
    else
      sqlite3_result_error(ctx, c->errmsg ? c->errmsg : "fit failed", -1);
    fit_ctx_free(c);
    return;
  }
  if (!c->configured) {
    sqlite3_result_error(
        ctx, PREDICT_ERR_SCHEMA ": fit received no training rows", -1);
    fit_ctx_free(c);
    return;
  }
  char **fn = sqlite3_malloc(sizeof(char *) * c->nfeat);
  if (!fn) {
    sqlite3_result_error_nomem(ctx);
    fit_ctx_free(c);
    return;
  }
  memset(fn, 0, sizeof(char *) * c->nfeat);
  int ok = 1;
  for (int i = 0; i < c->nfeat && ok; i++)
    if (!(fn[i] = sqlite3_mprintf("f%d", i)))
      ok = 0;
  if (!ok) {
    for (int i = 0; i < c->nfeat; i++)
      sqlite3_free(fn[i]);
    sqlite3_free(fn);
    sqlite3_result_error_nomem(ctx);
    fit_ctx_free(c);
    return;
  }
  void *blob = NULL;
  int blob_len = 0;
  char *emsg = NULL;
  int rc = predict0_train_student(
      sqlite3_context_db_handle(ctx), c->X, c->n, c->nfeat, fn, c->classify,
      c->classify ? c->ylab : NULL, c->classify ? NULL : c->yval, c->kind,
      c->reg_id, &blob, &blob_len, &emsg);
  for (int i = 0; i < c->nfeat; i++)
    sqlite3_free(fn[i]);
  sqlite3_free(fn);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(ctx, emsg ? emsg : "fit training failed", -1);
    sqlite3_free(emsg);
    fit_ctx_free(c);
    return;
  }
  if (c->reg_id)
    sqlite3_result_text(ctx, c->reg_id, -1, SQLITE_TRANSIENT);
  else
    sqlite3_result_blob(ctx, blob, blob_len, SQLITE_TRANSIENT);
  sqlite3_free(blob);
  fit_ctx_free(c);
}

/* ---- predict() scalar ---- */

typedef struct {
  int proba;
} PredScalarOpts;

static int predict_scalar_opt_cb(void *ctx, const char *key,
                                 sqlite3_value *value, char **errmsg) {
  PredScalarOpts *o = ctx;
  if (strcmp(key, "proba") == 0) {
    int t = sqlite3_value_type(value);
    if (t != SQLITE_INTEGER && t != SQLITE_FLOAT) {
      *errmsg = sqlite3_mprintf("%s: wrong type for option '%s'",
                                PREDICT_ERR_OPTIONS, key);
      return 1;
    }
    o->proba = sqlite3_value_int(value) != 0;
  }
  return 0;
}

static const char *const PRED_SCALAR_OPTION_KEYS[] = {"proba", NULL};

/* auxdata destructor with the exact void(*)(void*) type SQLite invokes it
 * through. Casting predict0_student_free (which takes predict0_loaded_student*)
 * to that pointer type and calling through it is undefined behavior that
 * -fsanitize=function flags on Linux. */
static void predict_scalar_free_aux(void *p) {
  predict0_student_free((predict0_loaded_student *)p);
}

static void predict_scalar(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  if (argc < 2) {
    sqlite3_result_error(ctx,
                         PREDICT_ERR_SCHEMA ": predict(model, feature, ...)"
                                            " needs a model and >= 1 feature",
                         -1);
    return;
  }
  int has_opts = 0;
  if (sqlite3_value_type(argv[argc - 1]) == SQLITE_TEXT) {
    const char *t = (const char *)sqlite3_value_text(argv[argc - 1]);
    if (t && t[0] == '{')
      has_opts = 1;
  }
  int nfeat_in = argc - 1 - has_opts;
  if (nfeat_in < 1) {
    sqlite3_result_error(
        ctx, PREDICT_ERR_SCHEMA ": predict needs >= 1 feature after the model",
        -1);
    return;
  }
  PredScalarOpts po;
  memset(&po, 0, sizeof(po));
  if (has_opts) {
    char *emsg = NULL;
    if (predict0_options_parse(sqlite3_context_db_handle(ctx),
                               (const char *)sqlite3_value_text(argv[argc - 1]),
                               PRED_SCALAR_OPTION_KEYS, predict_scalar_opt_cb,
                               &po, &emsg)) {
      sqlite3_result_error(
          ctx, emsg ? emsg : PREDICT_ERR_OPTIONS ": cannot parse options", -1);
      sqlite3_free(emsg);
      return;
    }
  }

  predict0_loaded_student *st = sqlite3_get_auxdata(ctx, 0);
  if (!st) {
    int mt = sqlite3_value_type(argv[0]);
    const void *blob = NULL;
    int blen = 0;
    predict0_model_row mrow;
    int owned = 0;
    char *emsg = NULL;
    if (mt == SQLITE_BLOB) {
      blob = sqlite3_value_blob(argv[0]);
      blen = sqlite3_value_bytes(argv[0]);
    } else if (mt == SQLITE_TEXT) {
      const char *id = (const char *)sqlite3_value_text(argv[0]);
      int look =
          predict0_registry_lookup(sqlite3_context_db_handle(ctx), id, &mrow);
      if (look == 1) {
        char *m = sqlite3_mprintf("%s: no such model: %s",
                                  PREDICT_ERR_MODEL_NOT_FOUND, id);
        sqlite3_result_error(ctx, m, -1);
        sqlite3_free(m);
        return;
      }
      if (look == 2) {
        char *m = sqlite3_mprintf("%s: weights do not match content_hash: %s",
                                  PREDICT_ERR_MODEL_HASH, id);
        sqlite3_result_error(ctx, m, -1);
        sqlite3_free(m);
        return;
      }
      if (look != 0) {
        sqlite3_result_error(
            ctx, PREDICT_ERR_RESOURCE ": model registry unavailable", -1);
        return;
      }
      owned = 1;
      if (!mrow.runtime || strcmp(mrow.runtime, "tree") != 0) {
        char *m = sqlite3_mprintf(
            "%s: predict() serves native students; use predict_batch() for"
            " runtime '%s'",
            PREDICT_ERR_RUNTIME_UNAVAILABLE, mrow.runtime ? mrow.runtime : "?");
        predict0_model_row_free(&mrow);
        sqlite3_result_error(ctx, m, -1);
        sqlite3_free(m);
        return;
      }
      if (!mrow.weights) {
        predict0_model_row_free(&mrow);
        sqlite3_result_error(
            ctx, PREDICT_ERR_SCHEMA ": model has no inline weights", -1);
        return;
      }
      blob = mrow.weights;
      blen = mrow.weights_len;
    } else {
      sqlite3_result_error(ctx,
                           PREDICT_ERR_SCHEMA ": predict model must be a"
                                              " registered id (text) or a"
                                              " fit() blob",
                           -1);
      return;
    }
    int rc = predict0_student_load(blob, blen, &st, &emsg);
    if (owned)
      predict0_model_row_free(&mrow);
    if (rc != SQLITE_OK) {
      sqlite3_result_error(ctx, emsg ? emsg : "cannot load model", -1);
      sqlite3_free(emsg);
      return;
    }
    sqlite3_set_auxdata(ctx, 0, st, predict_scalar_free_aux);
    /* SQLite may invoke the destructor immediately (before set_auxdata even
     * returns) if it cannot store the value under memory pressure, so st must
     * not be used afterward. Re-fetch it and bail if it was discarded. */
    st = sqlite3_get_auxdata(ctx, 0);
    if (!st) {
      sqlite3_result_error_nomem(ctx);
      return;
    }
  }

  /* need <= TAB_MAX_FEAT holds by construction, so filling x[TAB_MAX_FEAT] below
   * cannot overflow: the loader caps a student's nfeat at TREE_MAX_FEAT and the
   * compile-time assert (predict0_assert_tab_buf_holds_max_student) guarantees
   * TREE_MAX_FEAT <= TAB_MAX_FEAT. A feature-count mismatch is still reported. */
  int need = predict0_student_nfeat(st);
  if (nfeat_in != need) {
    char *m = sqlite3_mprintf(
        "%s: predict got %d features but the model expects %d",
        PREDICT_ERR_SCHEMA, nfeat_in, need);
    sqlite3_result_error(ctx, m, -1);
    sqlite3_free(m);
    return;
  }
  f32 x[TAB_MAX_FEAT];
  for (int i = 0; i < nfeat_in; i++) {
    int t = sqlite3_value_type(argv[1 + i]);
    if (t != SQLITE_INTEGER && t != SQLITE_FLOAT) {
      sqlite3_result_error(
          ctx, PREDICT_ERR_SCHEMA ": predict features must be numeric", -1);
      return;
    }
    f64 fv = sqlite3_value_double(argv[1 + i]);
    if (!isfinite(fv)) { /* symmetry with fit(): a non-finite feature can't be
                            trained on, so it must not be scored on either */
      sqlite3_result_error(
          ctx, PREDICT_ERR_SCHEMA ": predict features must be finite", -1);
      return;
    }
    x[i] = (f32)fv;
  }
  char *pred = NULL;
  f64 conf = 0;
  int hc = 0;
  char *emsg = NULL;
  if (predict0_student_predict(st, x, &pred, &conf, &hc, &emsg) != SQLITE_OK) {
    sqlite3_result_error(ctx, emsg ? emsg : "predict failed", -1);
    sqlite3_free(emsg);
    return;
  }
  if (po.proba) {
    sqlite3_str *j = sqlite3_str_new(NULL);
    sqlite3_str_appendall(j, "{\"prediction\":");
    predict0_json_str(j, pred);
    if (hc)
      sqlite3_str_appendf(j, ",\"confidence\":%.17g", conf);
    sqlite3_str_appendchar(j, 1, '}');
    char *js = sqlite3_str_finish(j);
    if (js)
      sqlite3_result_text(ctx, js, -1, sqlite3_free);
    else
      sqlite3_result_error_nomem(ctx);
  } else {
    sqlite3_result_text(ctx, pred, -1, SQLITE_TRANSIENT);
  }
  sqlite3_free(pred);
}

int predict0_tabular_init(sqlite3 *db) {
  /* predict_batch: the former predict() TVF, now the batched escape hatch
   * (knn5-incontext, onnx, and registered-student batch serving over an apply
   * query). The guessable per-row scalar predict() is registered below. It is
   * NOT marked deterministic: a registered id resolves against _predict_models,
   * so the result depends on external state, not on the SQL arguments alone. */
  int rc = sqlite3_create_module(db, "predict_batch", &predictModule, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function(db, "predict", -1, SQLITE_UTF8, NULL,
                               predict_scalar, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;
  return sqlite3_create_function(db, "fit", -1, SQLITE_UTF8, NULL, NULL,
                                 fit_step, fit_final);
}
