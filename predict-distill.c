/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* distill_predict() / distill_forecast(): the distillation recipes.
 *
 * Fits a native student via the trainers in predict-train.c. distill_predict() fits a student
 * on a training signal (the target column by default, or a named teacher
 * model's predictions), evaluates it on a held-out fraction, serializes it via
 * predict-student.c, and registers it in _predict_models. The student then
 * serves through predict0_tree_run with no onnxruntime. The blob format,
 * deserializers, and inference runtime all live in predict-student.c. */
#include "predict-internal.h"
#include "predict-student.h"
#include "predict-train.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif


/* ---- the distill_predict() operation ---- */

typedef struct {
  char *target, *task, *student_id, *teacher, *student_kind;
  char *proba;   /* JSON array of soft-target probability column names */
  char *classes; /* JSON array of class labels, same order as `proba` */
} DistOpts;

static void dist_opts_free(DistOpts *o) {
  sqlite3_free(o->target);
  sqlite3_free(o->task);
  sqlite3_free(o->student_id);
  sqlite3_free(o->teacher);
  sqlite3_free(o->student_kind);
  sqlite3_free(o->proba);
  sqlite3_free(o->classes);
}

/* Return the class index for `s`, interning it into *labels (growing *cap and
 * *nclass on first sight). Returns -1 and sets *rc = SQLITE_NOMEM on OOM; a
 * valid index is always >= 0. */

static int dist_opt_cb(void *ctx, const char *key, sqlite3_value *value,
                       char **errmsg) {
  DistOpts *o = ctx;
  if (sqlite3_value_type(value) != SQLITE_TEXT) {
    *errmsg = sqlite3_mprintf("%s: wrong type for option '%s'",
                              PREDICT_ERR_OPTIONS, key);
    return 1;
  }
  char **slot = strcmp(key, "target") == 0          ? &o->target
                : strcmp(key, "task") == 0          ? &o->task
                : strcmp(key, "student_id") == 0    ? &o->student_id
                : strcmp(key, "teacher") == 0       ? &o->teacher
                : strcmp(key, "student_kind") == 0  ? &o->student_kind
                : strcmp(key, "proba") == 0         ? &o->proba
                : strcmp(key, "classes") == 0       ? &o->classes
                                                    : NULL;
  if (slot) {
    sqlite3_free(*slot); /* free-before-assign: duplicate JSON key */
    *slot = sqlite3_mprintf("%s", (const char *)sqlite3_value_text(value));
  }
  return 0;
}

static const char *const DIST_OPTION_KEYS[] = {
    "target", "task",  "student_id", "teacher", "student_kind",
    "proba",  "classes", NULL};

static int name_index(char *const *arr, int n, const char *s) {
  for (int i = 0; i < n; i++)
    if (strcmp(arr[i], s) == 0)
      return i;
  return -1;
}

typedef struct {
  char *model_id;
  char content_hash[PREDICT_HEX_BUFSIZE];
  int train_rows;
  double metric;
} DistResult;

static void append_ident(sqlite3_str *s, const char *nm) {
  sqlite3_str_appendchar(s, 1, '"');
  for (const char *c = nm; *c; c++) {
    if (*c == '"')
      sqlite3_str_appendchar(s, 1, '"');
    sqlite3_str_appendchar(s, 1, *c);
  }
  sqlite3_str_appendchar(s, 1, '"');
}

/* Hash a serialized student blob and register it under student_id
 * (kind='student'; runtime='tree' also covers MLP/forest blobs, "native
 * in-core runtime" as opposed to onnx). hash_out receives the
 * content_hash hex. Returns SQLITE_OK, or SQLITE_ERROR with *errmsg set
 * (STUDENT_EXISTS on an id collision). */

/* The training pipeline. Fills *res on success. */
static int distill_train(sqlite3 *db, const char *tq, DistOpts *o,
                         DistResult *res, char **errmsg) {
  int rc = SQLITE_OK;
  int classify = !o->task || strcmp(o->task, "classify") == 0;
  memset(res, 0, sizeof(*res));

  /* ---- 1. introspect train_query columns ---- */
  sqlite3_stmt *iq = NULL;
  if (predict0_prepare_ro(db, tq, "train_query", &iq, errmsg) != SQLITE_OK)
    return SQLITE_ERROR;
  int ncol = sqlite3_column_count(iq);
  int target_col = -1;
  char *feat_names[TREE_MAX_FEAT];
  int feat_col[TREE_MAX_FEAT];
  int nfeat = 0;
  for (int i = 0; i < ncol; i++) {
    const char *nm = sqlite3_column_name(iq, i);
    if (nm && o->target && strcmp(nm, o->target) == 0) {
      target_col = i;
    } else if (nfeat < TREE_MAX_FEAT) {
      feat_names[nfeat] = sqlite3_mprintf("%s", nm ? nm : "");
      feat_col[nfeat] = i;
      nfeat++;
    } else {
      sqlite3_finalize(iq);
      for (int f = 0; f < nfeat; f++)
        sqlite3_free(feat_names[f]);
      *errmsg = sqlite3_mprintf("%s: too many feature columns (max %d)",
                                PREDICT_ERR_SCHEMA, TREE_MAX_FEAT);
      return SQLITE_ERROR;
    }
  }
  sqlite3_finalize(iq);
  if (target_col < 0 || nfeat == 0) {
    for (int f = 0; f < nfeat; f++)
      sqlite3_free(feat_names[f]);
    *errmsg = sqlite3_mprintf(
        target_col < 0 ? "%s: no such target column: %s"
                       : "%s: train_query needs feature columns%s",
        target_col < 0 ? PREDICT_ERR_TARGET : PREDICT_ERR_SCHEMA,
        target_col < 0 ? (o->target ? o->target : "(none)") : "");
    return SQLITE_ERROR;
  }

  /* quoted feature list for the teacher's apply query */
  sqlite3_str *fl = sqlite3_str_new(db);
  for (int f = 0; f < nfeat; f++) {
    if (f)
      sqlite3_str_appendchar(fl, 1, ',');
    append_ident(fl, feat_names[f]);
  }
  char *feat_list = sqlite3_str_finish(fl);

  /* all further allocations flow to a single cleanup */
  f32 *X = NULL;
  char **y_true_c = NULL; /* classify: true labels */
  f64 *y_true_r = NULL;   /* regress: true values */
  i32 *y_teach = NULL;    /* classify: teacher class index */
  f32 *y_teach_r = NULL;  /* regress: teacher value */
  char **labels = NULL;   /* teacher class vocabulary */
  int nclass = 0, nlab_cap = 0;
  int n = 0, cap = 0;
  Tree tree;
  memset(&tree, 0, sizeof(tree));
  Forest forest;
  memset(&forest, 0, sizeof(forest));
  MLP mlp;
  memset(&mlp, 0, sizeof(mlp));
  void *blob = NULL;
  int blob_len = 0;
  int *idx = NULL;
  char *read_sql = NULL, *apply_sql = NULL, *teacher_sql = NULL;
  sqlite3_stmt *rq = NULL, *tqs = NULL;
  char **proba_names = NULL, **soft_labels = NULL; /* soft distillation */
  int nproba = 0, nsoft = 0, *proba_col = NULL;
  f32 *soft_P = NULL; /* [n, nproba] teacher class probabilities */
  sqlite3_stmt *sps = NULL;
  int soft = o->proba != NULL;

  const char *teacher = o->teacher; /* NULL => train directly on the target */
  const char *task = classify ? "classify" : "regress";

  /* ---- soft-label distillation setup: teacher class probabilities live in
   * named columns (`proba`), one per class (`classes`), which the gbt student
   * matches instead of a hard label. Exclude those columns from the features
   * and capture their indices. ---- */
  if (soft) {
    if (!classify) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: 'proba' (soft distillation) is"
                                " classification only",
                                PREDICT_ERR_OPTIONS);
      goto done;
    }
    if (o->student_kind && strcmp(o->student_kind, "gbt") != 0 &&
        strcmp(o->student_kind, "mlp") != 0) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: soft distillation requires student_kind"
                                " 'gbt' or 'mlp'",
                                PREDICT_ERR_OPTIONS);
      goto done;
    }
    if (!o->classes) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: 'proba' requires 'classes'",
                                PREDICT_ERR_OPTIONS);
      goto done;
    }
    rc = predict0_json_str_array(db, o->proba, NULL, &proba_names, &nproba,
                                 PREDICT0_MAX_CLASS, errmsg);
    if (rc != SQLITE_OK)
      goto done;
    rc = predict0_json_str_array(db, o->classes, NULL, &soft_labels, &nsoft,
                                 PREDICT0_MAX_CLASS, errmsg);
    if (rc != SQLITE_OK)
      goto done;
    if (nproba < 2 || nproba != nsoft) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: 'proba' and 'classes' must be equal-length arrays of >= 2",
          PREDICT_ERR_OPTIONS);
      goto done;
    }
    /* Defensive invariant: predict0_json_str_array already caps proba/classes
     * at PREDICT0_MAX_CLASS during parsing; kept in case that changes. */
    if (nproba > PREDICT0_MAX_CLASS) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: too many classes (%d); the maximum is %d",
                                PREDICT_ERR_SCHEMA, nproba, PREDICT0_MAX_CLASS);
      goto done;
    }
    proba_col = sqlite3_malloc(sizeof(int) * nproba);
    if (!proba_col) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    for (int k = 0; k < nproba; k++)
      proba_col[k] = -1;
    int w = 0;
    for (int r = 0; r < nfeat; r++) {
      int k = name_index(proba_names, nproba, feat_names[r]);
      if (k >= 0) {
        proba_col[k] = feat_col[r]; /* query column index of this class prob */
        sqlite3_free(feat_names[r]);
      } else {
        feat_names[w] = feat_names[r];
        feat_col[w] = feat_col[r];
        w++;
      }
    }
    nfeat = w;
    for (int k = 0; k < nproba; k++)
      if (proba_col[k] < 0) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf("%s: proba column '%s' not in train_query",
                                  PREDICT_ERR_SCHEMA, proba_names[k]);
        goto done;
      }
    if (nfeat == 0) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: no feature columns left after excluding"
                                " proba columns",
                                PREDICT_ERR_SCHEMA);
      goto done;
    }
  }

  /* ---- 2. read features + true labels (row_number keeps a stable order) ---- */
  read_sql = sqlite3_mprintf(
      "SELECT (row_number() OVER ()) AS _rid, * FROM (%s) ORDER BY _rid", tq);
  if (!read_sql) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  if (sqlite3_prepare_v2(db, read_sql, -1, &rq, NULL) != SQLITE_OK) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: could not read train_query: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
    goto done;
  }
  int sr;
  while ((sr = sqlite3_step(rq)) == SQLITE_ROW) {
    if (n == cap) {
      cap = cap ? cap * 2 : 256;
      f32 *gx = sqlite3_realloc(X, sizeof(f32) * (size_t)cap * nfeat);
      if (!gx) {
        rc = SQLITE_NOMEM;
        goto done;
      }
      X = gx;
      if (classify) {
        char **g = sqlite3_realloc(y_true_c, sizeof(char *) * cap);
        i32 *gt = sqlite3_realloc(y_teach, sizeof(i32) * cap);
        if (!g || !gt) {
          sqlite3_free(g);
          sqlite3_free(gt);
          rc = SQLITE_NOMEM;
          goto done;
        }
        y_true_c = g;
        y_teach = gt;
      } else {
        f64 *g = sqlite3_realloc(y_true_r, sizeof(f64) * cap);
        f32 *gt = sqlite3_realloc(y_teach_r, sizeof(f32) * cap);
        if (!g || !gt) {
          sqlite3_free(g);
          sqlite3_free(gt);
          rc = SQLITE_NOMEM;
          goto done;
        }
        y_true_r = g;
        y_teach_r = gt;
      }
    }
    f32 *row = &X[(size_t)n * nfeat];
    for (int f = 0; f < nfeat; f++) {
      int c = feat_col[f] + 1; /* +1 for the _rid prefix column */
      int ct = sqlite3_column_type(rq, c);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf("%s: train feature '%s' is not numeric",
                                  PREDICT_ERR_SCHEMA, feat_names[f]);
        goto done;
      }
      row[f] = (f32)sqlite3_column_double(rq, c);
    }
    if (classify) {
      const char *lab =
          (const char *)sqlite3_column_text(rq, target_col + 1);
      y_true_c[n] = sqlite3_mprintf("%s", lab ? lab : "");
    } else {
      int ct = sqlite3_column_type(rq, target_col + 1);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf("%s: regress target must be numeric",
                                  PREDICT_ERR_TARGET);
        goto done;
      }
      y_true_r[n] = sqlite3_column_double(rq, target_col + 1);
    }
    n++;
  }
  /* A terminal step code other than DONE is a read failure, not end-of-data:
   * surface it rather than train on the partial rows collected so far. Capture
   * the message before finalize clears it. */
  if (rc == SQLITE_OK && sr != SQLITE_DONE) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: could not read train_query: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
  }
  sqlite3_finalize(rq);
  rq = NULL;
  if (rc != SQLITE_OK)
    goto done;
  if (n < DISTILL_MIN_ROWS) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: need at least %d train rows, got %d",
                              PREDICT_ERR_SCHEMA, DISTILL_MIN_ROWS, n);
    goto done;
  }

  /* soft distillation: read the probability columns in the same row order and
   * normalize each row to a distribution (defensive: clamp negatives, rescale;
   * a degenerate row falls back to uniform). */
  if (soft) {
    soft_P = sqlite3_malloc(sizeof(f32) * (size_t)n * nproba);
    if (!soft_P) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    if (sqlite3_prepare_v2(db, read_sql, -1, &sps, NULL) != SQLITE_OK) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: could not re-read proba columns: %s",
                                PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
      goto done;
    }
    int si = 0;
    while (si < n && sqlite3_step(sps) == SQLITE_ROW) {
      f64 sum = 0;
      for (int k = 0; k < nproba; k++) {
        f64 v = sqlite3_column_double(sps, proba_col[k] + 1); /* +1 for _rid */
        if (v < 0)
          v = 0;
        soft_P[(size_t)si * nproba + k] = (f32)v;
        sum += v;
      }
      if (sum > 1e-12)
        for (int k = 0; k < nproba; k++)
          soft_P[(size_t)si * nproba + k] /= (f32)sum;
      else
        for (int k = 0; k < nproba; k++)
          soft_P[(size_t)si * nproba + k] = 1.f / nproba;
      si++;
    }
    sqlite3_finalize(sps);
    sps = NULL;
    if (si != n) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: proba re-read yielded %d of %d rows",
                                PREDICT_ERR_RESOURCE, si, n);
      goto done;
    }
  }

  /* ---- 3. training signal per row ----
   * Soft distillation: the class vocabulary is the given `classes`, and the
   * per-row soft targets are already in soft_P; there is no teacher to run.
   * Otherwise, with no teacher (the default) the student trains directly on
   * the target column: it already holds the labels, or a strong teacher's
   * precomputed predictions (e.g. an offline TabFM run) that you want
   * compressed into a native student that runs anywhere. A named teacher is a
   * registered predict() model, re-run over the same rows (aligned by _rid) to
   * relabel them. */
  if (soft) {
    labels = soft_labels; /* transfer ownership; matched to soft_P columns */
    soft_labels = NULL;
    nclass = nproba;
  } else if (!teacher) {
    for (int i = 0; i < n; i++) {
      if (classify) {
        int cl = predict0_intern_label(&labels, &nclass, &nlab_cap,
                                       y_true_c[i], PREDICT0_MAX_CLASS, &rc,
                                       errmsg);
        if (cl < 0)
          goto done;
        y_teach[i] = cl;
      } else {
        y_teach_r[i] = (f32)y_true_r[i];
      }
    }
  } else {
    apply_sql = sqlite3_mprintf(
        "SELECT (row_number() OVER ()) AS _rid, %s FROM (%s) ORDER BY _rid",
        feat_list, tq);
    teacher_sql = sqlite3_mprintf(
        "SELECT prediction FROM predict_batch(%Q, %Q, json_object('target',%Q,'task',"
        "%Q,'model',%Q))",
        tq, apply_sql, o->target, task, teacher);
    if (!apply_sql || !teacher_sql) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    if (sqlite3_prepare_v2(db, teacher_sql, -1, &tqs, NULL) != SQLITE_OK) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: teacher query for '%s' does not"
                                " prepare: %s",
                                PREDICT_ERR_SCHEMA, teacher,
                                sqlite3_errmsg(db));
      goto done;
    }
    int ti = 0;
    int tstep;
    while ((tstep = sqlite3_step(tqs)) == SQLITE_ROW && ti < n) {
      if (classify) {
        const char *pred = (const char *)sqlite3_column_text(tqs, 0);
        int cl = predict0_intern_label(&labels, &nclass, &nlab_cap,
                                       pred ? pred : "", PREDICT0_MAX_CLASS, &rc,
                                       errmsg);
        if (cl < 0)
          goto done;
        y_teach[ti] = cl;
      } else {
        y_teach_r[ti] = (f32)sqlite3_column_double(tqs, 0);
      }
      ti++;
    }
    /* Require exactly n teacher labels: DONE with ti == n. If the loop stopped
     * because ti hit n while the query still had rows (tstep == SQLITE_ROW), the
     * teacher over-produced; fail rather than silently drop the extra. */
    int tdone = tstep == SQLITE_DONE && ti == n;
    char *terr = tdone ? NULL : sqlite3_mprintf("%s", sqlite3_errmsg(db));
    sqlite3_finalize(tqs);
    tqs = NULL;
    if (!tdone) {
      int too_many = tstep == SQLITE_ROW;
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: teacher produced %s labels for %d train rows (%s)",
          PREDICT_ERR_SCHEMA, too_many ? "too many" : "too few", n,
          terr ? terr : (too_many ? "extra rows" : "short read"));
      sqlite3_free(terr);
      goto done;
    }
    sqlite3_free(terr);
  }
  if (classify && nclass < 2) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: %s produced a single class; nothing to distill", PREDICT_ERR_SCHEMA,
        teacher ? "teacher" : "target column");
    goto done;
  }
  /* Defensive invariant: predict0_intern_label and the soft-label nproba check
   * already cap this before any allocation; kept so the bound holds even if
   * those paths change. */
  if (classify && nclass > PREDICT0_MAX_CLASS) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: too many classes (%d); the maximum is %d",
                              PREDICT_ERR_SCHEMA, nclass, PREDICT0_MAX_CLASS);
    goto done;
  }

  /* ---- 4. fit the student on the fit split (teacher targets) ---- */
  int n_hold = n / 5;
  if (n_hold < 1)
    n_hold = 1;
  int n_fit = n - n_hold;
  int is_mlp_student = o->student_kind && strcmp(o->student_kind, "mlp") == 0;
  int is_gbt = !is_mlp_student &&
               (soft || (o->student_kind && strcmp(o->student_kind, "gbt") == 0));
  int correct = 0;
  f64 sse = 0;

  if (is_mlp_student) {
    if (!classify) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: mlp student is classification only",
                                PREDICT_ERR_OPTIONS);
      goto done;
    }
    rc = predict0_train_mlp(X, n_fit, nfeat, nclass, y_teach, soft ? soft_P : NULL,
                   0 /* classify */, MLP_HIDDEN, MLP_EPOCHS, MLP_LR,
                   0 /* no skip */, &mlp, errmsg);
    if (rc != SQLITE_OK)
      goto done;
    mlp.feat_names = sqlite3_malloc(sizeof(char *) * nfeat);
    if (!mlp.feat_names) {
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
    for (int f = 0; f < nfeat; f++)
      mlp.feat_names[f] = feat_names[f];
    mlp.labels = labels; /* transfer */
    nfeat = 0;
    labels = NULL;

    f32 *hb = sqlite3_malloc(sizeof(f32) * mlp.nhid);
    f32 *ob = sqlite3_malloc(sizeof(f32) * mlp.nout);
    if (!hb || !ob) {
      sqlite3_free(hb);
      sqlite3_free(ob);
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
    for (int i = n_fit; i < n; i++) {
      char *pred = NULL;
      f64 conf = 0;
      int hc = 0;
      if (predict0_mlp_predict_row(&mlp, &X[(size_t)i * mlp.nfeat], hb, ob, &pred, &conf,
                          &hc) != SQLITE_OK) {
        sqlite3_free(hb);
        sqlite3_free(ob);
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
      if (strcmp(pred, y_true_c[i]) == 0)
        correct++;
      sqlite3_free(pred);
    }
    sqlite3_free(hb);
    sqlite3_free(ob);
    res->metric = (f64)correct / n_hold;
    rc = predict0_mlp_serialize(&mlp, &blob, &blob_len);
    if (rc != SQLITE_OK) {
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
  } else if (is_gbt) {
    rc = predict0_train_gbt(X, n_fit, nfeat, classify ? 0 : 1, nclass, y_teach,
                   y_teach_r, soft ? soft_P : NULL, &forest, errmsg);
    if (rc != SQLITE_OK)
      goto done;
    forest.feat_names = sqlite3_malloc(sizeof(char *) * nfeat);
    if (!forest.feat_names) {
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
    for (int f = 0; f < nfeat; f++)
      forest.feat_names[f] = feat_names[f];
    forest.labels = labels; /* transfer */
    nfeat = 0;
    labels = NULL;

    f64 *scbuf = classify ? sqlite3_malloc(sizeof(f64) * forest.n_score) : NULL;
    if (classify && !scbuf) {
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
    for (int i = n_fit; i < n; i++) {
      char *pred = NULL;
      f64 conf = 0;
      int hc = 0;
      if (predict0_forest_predict_row(&forest, &X[(size_t)i * forest.nfeat], scbuf,
                             &pred, &conf, &hc) != SQLITE_OK) {
        sqlite3_free(scbuf);
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
      if (classify) {
        if (strcmp(pred, y_true_c[i]) == 0)
          correct++;
      } else {
        f64 d = strtod(pred, NULL) - y_true_r[i];
        sse += d * d;
      }
      sqlite3_free(pred);
    }
    sqlite3_free(scbuf);
    res->metric = classify ? (f64)correct / n_hold : sqrt(sse / n_hold);
    rc = predict0_forest_serialize(&forest, &blob, &blob_len);
    if (rc != SQLITE_OK) {
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
  } else {
    idx = sqlite3_malloc(sizeof(int) * n_fit);
    if (!idx) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    for (int i = 0; i < n_fit; i++)
      idx[i] = i;

    Builder b;
    memset(&b, 0, sizeof(b));
    b.X = X;
    b.nfeat = nfeat;
    b.yc = y_teach;
    b.yr = y_teach_r;
    b.nclass = nclass;
    b.task = classify ? 0 : 1;
    int root = predict0_bld_build(&b, idx, n_fit, 0);
    if (root < 0) {
      sqlite3_free(b.nodes);
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
    /* root is index 0 (first node allocated). Move the feature names into a
     * heap array the tree owns (the local is a stack array). */
    tree.feat_names = sqlite3_malloc(sizeof(char *) * nfeat);
    if (!tree.feat_names) {
      sqlite3_free(b.nodes);
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
    for (int f = 0; f < nfeat; f++)
      tree.feat_names[f] = feat_names[f];
    tree.task = b.task;
    tree.nfeat = nfeat;
    tree.nclass = nclass;
    tree.labels = labels;
    tree.n_nodes = b.n;
    tree.nodes = b.nodes;
    nfeat = 0; /* feature-name strings now owned by the tree */
    labels = NULL;

    for (int i = n_fit; i < n; i++) {
      int leaf = predict0_tree_walk(&tree, &X[(size_t)i * tree.nfeat]);
      if (leaf < 0)
        continue;
      const TreeNode *ln = &tree.nodes[leaf];
      if (classify) {
        if (strcmp(tree.labels[ln->klass], y_true_c[i]) == 0)
          correct++;
      } else {
        f64 d = (f64)ln->value - y_true_r[i];
        sse += d * d;
      }
    }
    res->metric = classify ? (f64)correct / n_hold : sqrt(sse / n_hold);
    rc = predict0_tree_serialize(&tree, &blob, &blob_len);
    if (rc != SQLITE_OK) {
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
  }
  rc = predict0_register_student(db, o->student_id, blob, blob_len, res->content_hash,
                        errmsg);
  if (rc != SQLITE_OK)
    goto done;

  res->model_id = sqlite3_mprintf("%s", o->student_id);
  res->train_rows = n;

  rc = SQLITE_OK;

done:
  sqlite3_free(feat_list);
  sqlite3_free(read_sql);
  sqlite3_free(apply_sql);
  sqlite3_free(teacher_sql);
  if (rq)
    sqlite3_finalize(rq);
  if (tqs)
    sqlite3_finalize(tqs);
  sqlite3_free(X);
  if (y_true_c)
    for (int i = 0; i < n; i++)
      sqlite3_free(y_true_c[i]);
  sqlite3_free(y_true_c);
  sqlite3_free(y_true_r);
  sqlite3_free(y_teach);
  sqlite3_free(y_teach_r);
  for (int f = 0; f < nfeat; f++) /* only if not transferred to the tree */
    sqlite3_free(feat_names[f]);
  if (labels)
    for (int i = 0; i < nclass; i++)
      sqlite3_free(labels[i]);
  sqlite3_free(labels);
  if (proba_names)
    for (int k = 0; k < nproba; k++)
      sqlite3_free(proba_names[k]);
  sqlite3_free(proba_names);
  if (soft_labels) /* NULL once transferred to `labels` */
    for (int k = 0; k < nsoft; k++)
      sqlite3_free(soft_labels[k]);
  sqlite3_free(soft_labels);
  sqlite3_free(proba_col);
  sqlite3_free(soft_P);
  if (sps)
    sqlite3_finalize(sps);
  sqlite3_free(idx);
  sqlite3_free(blob);
  predict0_tree_free(&tree);
  predict0_forest_free(&forest);
  predict0_mlp_free(&mlp);
  return rc;
}

/* ---- distill vtab ---- */

#define DL_MODEL 0
#define DL_HASH 1
#define DL_ROWS 2
#define DL_METRIC 3
#define DL_TRAINQ 4
#define DL_OPTIONS 5

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} dl_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  DistResult res;
  int done;
} dl_cursor;

static int dl_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  dl_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(model_id TEXT, content_hash TEXT, train_rows INTEGER,"
          " holdout_metric REAL, train_query HIDDEN,"
          " options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int dl_disconnect(sqlite3_vtab *p) {
  sqlite3_free(p);
  return SQLITE_OK;
}

static int dl_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_train = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    int argv = c->iColumn == DL_TRAINQ ? 1 : c->iColumn == DL_OPTIONS ? 2 : 0;
    if (!argv)
      continue;
    if (!c->usable)
      return SQLITE_CONSTRAINT;
    pIdx->aConstraintUsage[i].argvIndex = argv;
    pIdx->aConstraintUsage[i].omit = 1;
    if (argv == 1)
      seen_train = 1;
  }
  if (!seen_train) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: distill_predict(train_query, options) requires a train_query",
        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1e6;
  return SQLITE_OK;
}

static int dl_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  dl_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static int dl_close(sqlite3_vtab_cursor *pCur) {
  dl_cursor *c = (dl_cursor *)pCur;
  sqlite3_free(c->res.model_id);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int dl_filter(sqlite3_vtab_cursor *pCur, int idxNum, const char *idxStr,
                     int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  dl_cursor *cur = (dl_cursor *)pCur;
  dl_vtab *vtab = (dl_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  sqlite3_free(cur->res.model_id);
  memset(&cur->res, 0, sizeof(cur->res));
  cur->done = 0;

  const char *tq =
      argc >= 1 ? (const char *)sqlite3_value_text(argv[0]) : NULL;
  const char *options =
      argc >= 2 ? (const char *)sqlite3_value_text(argv[1]) : NULL;
  if (!tq) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg =
        sqlite3_mprintf("%s: train_query is required", PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }

  DistOpts o;
  memset(&o, 0, sizeof(o));
  char *emsg = NULL;
  if (predict0_options_parse(db, options, DIST_OPTION_KEYS, dist_opt_cb, &o,
                             &emsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = emsg;
    dist_opts_free(&o);
    return SQLITE_ERROR;
  }

#define DL_FAIL(...)                                                           \
  do {                                                                        \
    sqlite3_free(vtab->base.zErrMsg);                                         \
    vtab->base.zErrMsg = sqlite3_mprintf(__VA_ARGS__);                        \
    dist_opts_free(&o);                                                       \
    return SQLITE_ERROR;                                                      \
  } while (0)

  if (!o.target)
    DL_FAIL("%s: distill_predict requires a target option", PREDICT_ERR_TARGET);
  if (!o.student_id)
    DL_FAIL("%s: distill_predict requires a student_id option",
            PREDICT_ERR_OPTIONS);
  if (o.task && strcmp(o.task, "classify") != 0 &&
      strcmp(o.task, "regress") != 0)
    DL_FAIL("%s: task must be classify|regress: %s", PREDICT_ERR_TASK, o.task);
  if (o.student_kind && strcmp(o.student_kind, "tree") != 0 &&
      strcmp(o.student_kind, "gbt") != 0 &&
      strcmp(o.student_kind, "mlp") != 0)
    DL_FAIL("%s: student_kind '%s' is not available; use 'tree', 'gbt', or"
            " 'mlp'",
            PREDICT_ERR_OPTIONS, o.student_kind);

  char *ensure_err = NULL;
  if (predict0_registry_ensure(db, &ensure_err) != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = ensure_err;
    dist_opts_free(&o);
    return SQLITE_ERROR;
  }
  char *existing = predict0_registry_model_hash(db, o.student_id);
  if (existing) {
    sqlite3_free(existing);
    DL_FAIL("%s: student '%s' already exists", PREDICT_ERR_STUDENT_EXISTS,
            o.student_id);
  }

#undef DL_FAIL
  int rc = distill_train(db, tq, &o, &cur->res, &emsg);
  dist_opts_free(&o);
  if (rc != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = emsg;
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int dl_next(sqlite3_vtab_cursor *pCur) {
  ((dl_cursor *)pCur)->done = 1;
  return SQLITE_OK;
}

static int dl_eof(sqlite3_vtab_cursor *pCur) {
  return ((dl_cursor *)pCur)->done;
}

static int dl_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx, int col) {
  dl_cursor *c = (dl_cursor *)pCur;
  switch (col) {
  case DL_MODEL:
    sqlite3_result_text(ctx, c->res.model_id, -1, SQLITE_TRANSIENT);
    break;
  case DL_HASH:
    sqlite3_result_text(ctx, c->res.content_hash, -1, SQLITE_TRANSIENT);
    break;
  case DL_ROWS:
    sqlite3_result_int(ctx, c->res.train_rows);
    break;
  case DL_METRIC:
    sqlite3_result_double(ctx, c->res.metric);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int dl_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  UNUSED_PARAMETER(pCur);
  *pRowid = 1;
  return SQLITE_OK;
}

static sqlite3_module distillModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ dl_connect,
    /* xBestIndex  */ dl_best_index,
    /* xDisconnect */ dl_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ dl_open,
    /* xClose      */ dl_close,
    /* xFilter     */ dl_filter,
    /* xNext       */ dl_next,
    /* xEof        */ dl_eof,
    /* xColumn     */ dl_column,
    /* xRowid      */ dl_rowid,
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

/* ---- distill_forecast: train a native forecast student (PSFCST) ----
 *
 * The train_query returns context + horizon columns per row: the first
 * `context` columns are a raw history window, the next `horizon` are the
 * teacher's forecast for that window (computed offline, e.g. by a foundation
 * model). Each row is instance-normalized by its own window before a multi-
 * output regression MLP is fit to reproduce the teacher. The student serves
 * through forecast() with no teacher and no onnxruntime. */

#define FCST_HIDDEN 256 /* forecast student residual width (benchmark-chosen:
                         * 256 is the m4_hourly optimum, 128 and 512 both worse) */
#define FCST_EPOCHS 1500 /* forecast regression trains longer at a gentler LR */
#define FCST_LR 0.005f

typedef struct {
  char *model_id;
  char content_hash[PREDICT_HEX_BUFSIZE];
  int train_rows;
  f64 metric; /* holdout RMSE in normalized space */
} FcstResult;

#define FCST_TEACHER_MAX_WIN 4000 /* teacher= mode: window budget per call */

/* Train + register a forecast student from a prepared, instance-normalized
 * training matrix X [n,L], Y [n,H*Q]. Does not own X/Y. */
static int fdistill_fit(sqlite3 *db, const f32 *X, const f32 *Y, int n, int L,
                        int H, int Q, const f32 *levels, const char *student_id,
                        int nhid, int epochs, f32 lr, FcstResult *res,
                        char **errmsg) {
  int nout = H * Q, rc = SQLITE_OK, blob_len = 0;
  void *blob = NULL;
  MLP mlp;
  memset(&mlp, 0, sizeof(mlp));
  if (n < 2) {
    *errmsg = sqlite3_mprintf("%s: need at least 2 training windows, got %d",
                              PREDICT_ERR_SCHEMA, n);
    return SQLITE_ERROR;
  }
  int n_hold = n / 5;
  if (n_hold < 1)
    n_hold = 1;
  int n_fit = n - n_hold;
  rc = predict0_train_mlp(X, n_fit, L, nout, NULL, Y, 1 /* regress */, nhid, epochs, lr,
                 1 /* linear skip */, &mlp, errmsg);
  if (rc != SQLITE_OK)
    goto done;
  { /* holdout RMSE (normalized space) as the reported metric */
    f32 *hid = sqlite3_malloc(sizeof(f32) * (nhid > 0 ? nhid : 1));
    f32 *out = sqlite3_malloc(sizeof(f32) * nout);
    if (!hid || !out) {
      sqlite3_free(hid);
      sqlite3_free(out);
      rc = SQLITE_NOMEM;
      goto done;
    }
    f64 se = 0;
    int m = 0;
    for (int r = n_fit; r < n; r++) {
      predict0_mlp_forward(&mlp, &X[(size_t)r * L], hid, out);
      for (int k = 0; k < nout; k++) {
        f64 e = (f64)out[k] - Y[(size_t)r * nout + k];
        se += e * e;
        m++;
      }
    }
    res->metric = m ? sqrt(se / m) : 0;
    sqlite3_free(hid);
    sqlite3_free(out);
  }
  ForecastStudent fs = {
      .mlp = mlp, .horizon = H, .nquant = Q, .levels = (f32 *)levels};
  rc = predict0_fcst_serialize(&fs, &blob, &blob_len);
  if (rc != SQLITE_OK) {
    *errmsg =
        sqlite3_mprintf("%s: student serialize failed", PREDICT_ERR_RESOURCE);
    goto done;
  }
  rc = predict0_register_student(db, student_id, blob, blob_len, res->content_hash,
                        errmsg);
  if (rc != SQLITE_OK)
    goto done;
  res->model_id = sqlite3_mprintf("%s", student_id);
  res->train_rows = n;
  rc = SQLITE_OK;
done:
  sqlite3_free(blob);
  predict0_mlp_free(&mlp);
  return rc;
}

/* Provided-labels mode: each train_query row is context + H*Q teacher-quantile
 * columns; instance-normalize and fit. */
static int fdistill_train(sqlite3 *db, const char *tq, int L, int H, int Q,
                          const f32 *levels, const char *student_id, int nhid,
                          int epochs, f32 lr, FcstResult *res,
                          char **errmsg) {
  int nout = H * Q, rc = SQLITE_OK, ncol = L + nout, cap = 0, n = 0;
  f32 *X = NULL, *Y = NULL;
  f64 *wrow = NULL;
  sqlite3_stmt *q = NULL;
  if (predict0_prepare_ro(db, tq, "train_query", &q, errmsg) != SQLITE_OK)
    return SQLITE_ERROR;
  if (sqlite3_column_count(q) != ncol) {
    *errmsg = sqlite3_mprintf(
        "%s: train_query must return context + horizon*nquant = %d columns,"
        " got %d",
        PREDICT_ERR_SCHEMA, ncol, sqlite3_column_count(q));
    rc = SQLITE_ERROR;
    goto done;
  }
  wrow = sqlite3_malloc(sizeof(f64) * ncol);
  if (!wrow) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  int sr;
  while ((sr = sqlite3_step(q)) == SQLITE_ROW) {
    for (int c = 0; c < ncol; c++) {
      /* sqlite3_column_double coerces NULL/text to 0.0, which would silently
       * skew the normalized windows; require a real, finite number instead. */
      int ct = sqlite3_column_type(q, c);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        *errmsg = sqlite3_mprintf(
            "%s: train_query cell at column %d is not numeric",
            PREDICT_ERR_SCHEMA, c);
        rc = SQLITE_ERROR;
        goto done;
      }
      wrow[c] = sqlite3_column_double(q, c);
      if (!isfinite(wrow[c])) {
        *errmsg = sqlite3_mprintf(
            "%s: train_query cell at column %d is not finite",
            PREDICT_ERR_SCHEMA, c);
        rc = SQLITE_ERROR;
        goto done;
      }
    }
    f64 mu = 0;
    for (int i = 0; i < L; i++)
      mu += wrow[i];
    mu /= L;
    f64 v = 0;
    for (int i = 0; i < L; i++) {
      f64 d = wrow[i] - mu;
      v += d * d;
    }
    f64 sd = sqrt(v / L);
    if (sd < 1e-9)
      sd = 1e-9;
    if (n == cap) {
      int nc = cap ? cap * 2 : 256;
      f32 *nX = sqlite3_realloc64(X, sizeof(f32) * (size_t)nc * L);
      f32 *nY = sqlite3_realloc64(Y, sizeof(f32) * (size_t)nc * nout);
      if (nX)
        X = nX;
      if (nY)
        Y = nY;
      if (!nX || !nY) {
        rc = SQLITE_NOMEM;
        goto done;
      }
      cap = nc;
    }
    for (int i = 0; i < L; i++)
      X[(size_t)n * L + i] = (f32)((wrow[i] - mu) / sd);
    for (int k = 0; k < nout; k++)
      Y[(size_t)n * nout + k] = (f32)((wrow[L + k] - mu) / sd);
    n++;
  }
  /* A terminal step code other than DONE is a read failure, not end-of-data. */
  if (rc == SQLITE_OK && sr != SQLITE_DONE) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: could not read teacher rows: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
  }
  if (rc == SQLITE_OK)
    rc = fdistill_fit(db, X, Y, n, L, H, Q, levels, student_id, nhid, epochs,
                      lr, res, errmsg);
done:
  if (q)
    sqlite3_finalize(q);
  sqlite3_free(wrow);
  sqlite3_free(X);
  sqlite3_free(Y);
  return rc;
}

#ifdef SQLITE_PREDICT_ONNX
/* Label one series' sliding windows into X/Y with the onnx teacher's
 * quantile fan, so the trainer holds at most one series in memory.
 * Grows *X and *Y (realloc'd, *n windows of *cap capacity); the first
 * flush sets *Q and *levels from the teacher. Returns SQLITE_OK or an
 * error with *errmsg set. */
static int fcst_teacher_flush(sqlite3 *db, const predict0_model_row *trow,
                              const predict0_backend_opts *bopts,
                              const f64 *series, int sn, int L, int H,
                              f32 **X, f32 **Y, int *n, int *cap, int *Q,
                              f32 **levels, char **errmsg) {
  int stride = H >= 8 ? H / 4 : 1; /* dense enough for window diversity */
  for (int t = L; t <= sn && *n < FCST_TEACHER_MAX_WIN; t += stride) {
    const f64 *win = series + (t - L);
    f64 wm = 0;
    for (int i = 0; i < L; i++)
      wm += win[i];
    wm /= L;
    f64 wv = 0;
    for (int i = 0; i < L; i++) {
      f64 d = win[i] - wm;
      wv += d * d;
    }
    f64 ws = sqrt(wv / L);
    if (ws < 1e-6 * (fabs(wm) + 1))
      continue; /* degenerate window: instance-norm blows up */
    f64 *fan = NULL;
    f32 *lv = NULL;
    int fq = 0;
    int rc = predict0_onnx_forecast_fan(db, trow, bopts, win, L, H, &fan, &lv,
                                        &fq, errmsg);
    if (rc != SQLITE_OK)
      return rc;
    if (*Q == 0) {
      *Q = fq;
      *levels = lv;
    } else
      sqlite3_free(lv);
    int nout = H * *Q;
    if (*n == *cap) {
      int nc = *cap ? *cap * 2 : 256;
      f32 *nX = sqlite3_realloc64(*X, sizeof(f32) * (size_t)nc * L);
      f32 *nY = sqlite3_realloc64(*Y, sizeof(f32) * (size_t)nc * nout);
      if (nX)
        *X = nX;
      if (nY)
        *Y = nY;
      if (!nX || !nY) {
        sqlite3_free(fan);
        return SQLITE_NOMEM;
      }
      *cap = nc;
    }
    for (int i = 0; i < L; i++)
      (*X)[(size_t)*n * L + i] = (f32)((win[i] - wm) / ws);
    for (int k = 0; k < nout; k++)
      (*Y)[(size_t)*n * nout + k] = (f32)((fan[k] - wm) / ws);
    sqlite3_free(fan);
    (*n)++;
  }
  return SQLITE_OK;
}

/* teacher= mode: the train_query returns (value) rows for one series, or
 * (series_key, value) rows for several (in series-then-time order). The onnx
 * teacher labels sliding context windows in-DB; instance-normalize and fit.
 * The student's quantile levels come from the teacher's fan. */
static int fdistill_train_teacher(sqlite3 *db, const char *tq,
                                  const char *teacher, int L, int H,
                                  const char *student_id, int nhid, int epochs,
                                  f32 lr, FcstResult *res,
                                  char **errmsg) {
  int rc = SQLITE_OK, Q = 0, cap = 0, n = 0;
  f32 *X = NULL, *Y = NULL, *levels = NULL;
  f64 *series = NULL;
  int sn = 0, scap = 0;
  char *curkey = NULL;
  sqlite3_stmt *q = NULL;
  predict0_model_row trow;
  memset(&trow, 0, sizeof(trow));
  int have_trow = 0;
  predict0_backend_opts bopts = {NULL, NULL, NULL};

  if (predict0_registry_lookup(db, teacher, &trow) != 0 || !trow.runtime ||
      strcmp(trow.runtime, "onnx") != 0) {
    if (trow.runtime || trow.io_spec)
      predict0_model_row_free(&trow);
    *errmsg = sqlite3_mprintf("%s: teacher '%s' is not a registered onnx"
                              " forecast model",
                              PREDICT_ERR_MODEL_NOT_FOUND, teacher);
    return SQLITE_ERROR;
  }
  have_trow = 1;
  if (predict0_prepare_ro(db, tq, "train_query", &q, errmsg) != SQLITE_OK) {
    rc = SQLITE_ERROR;
    goto done;
  }
  int qcol = sqlite3_column_count(q), has_key = qcol >= 2;

  /* A single-key series never trips the n < FCST_TEACHER_MAX_WIN flush guard, so
   * bound the raw rows kept per series. The window budget consumes at most this
   * many rows; beyond it a series yields no further windows and would only grow
   * memory. Computed in 64-bit and clamped so the bound itself cannot overflow. */
  int win_stride = H >= 8 ? H / 4 : 1;
  sqlite3_int64 budget_rows =
      (sqlite3_int64)L + (sqlite3_int64)FCST_TEACHER_MAX_WIN * win_stride + H;
  int max_series = budget_rows > (1 << 23) ? (1 << 23) : (int)budget_rows;

  int sr;
  while ((sr = sqlite3_step(q)) == SQLITE_ROW && n < FCST_TEACHER_MAX_WIN) {
    const char *key = has_key ? (const char *)sqlite3_column_text(q, 0) : "";
    if (!key)
      key = "";
    if (has_key && (!curkey || strcmp(key, curkey) != 0)) {
      if (curkey) {
        rc = fcst_teacher_flush(db, &trow, &bopts, series, sn, L, H, &X, &Y,
                                &n, &cap, &Q, &levels, errmsg);
        if (rc != SQLITE_OK)
          goto done;
        sn = 0;
      }
      sqlite3_free(curkey);
      curkey = sqlite3_mprintf("%s", key);
    } else if (!curkey) {
      curkey = sqlite3_mprintf("%s", key);
    }
    if (sn < max_series) {
      if (sn == scap) {
        int nc = scap ? scap * 2 : 512;
        if (nc > max_series)
          nc = max_series;
        /* 64-bit sizing so the byte product can't narrow to int and overrun */
        f64 *ns = sqlite3_realloc64(series, sizeof(f64) * (size_t)nc);
        if (!ns) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        series = ns;
        scap = nc;
      }
      /* sqlite3_column_double coerces NULL/text to 0.0 and lets NaN/Inf through,
       * which would train on corrupted rows; require a real, finite number. */
      int ct = sqlite3_column_type(q, qcol - 1);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        *errmsg = sqlite3_mprintf("%s: teacher series value is not numeric",
                                  PREDICT_ERR_SCHEMA);
        rc = SQLITE_ERROR;
        goto done;
      }
      f64 v = sqlite3_column_double(q, qcol - 1);
      if (!isfinite(v)) {
        *errmsg = sqlite3_mprintf("%s: teacher series value is not finite",
                                  PREDICT_ERR_SCHEMA);
        rc = SQLITE_ERROR;
        goto done;
      }
      series[sn++] = v;
    }
  }
  /* sr == ROW means the FCST_TEACHER_MAX_WIN cap stopped us (not an error); any
   * terminal code other than ROW or DONE is a read failure, not end-of-data. */
  if (rc == SQLITE_OK && sr != SQLITE_ROW && sr != SQLITE_DONE) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: could not read teacher rows: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
  }
  if (rc == SQLITE_OK && sn >= L) {
    rc = fcst_teacher_flush(db, &trow, &bopts, series, sn, L, H, &X, &Y, &n,
                            &cap, &Q, &levels, errmsg);
    if (rc != SQLITE_OK)
      goto done;
  }

  if (rc == SQLITE_OK)
    rc = fdistill_fit(db, X, Y, n, L, H, Q, levels, student_id, nhid, epochs,
                      lr, res, errmsg);
done:
  if (q)
    sqlite3_finalize(q);
  if (have_trow)
    predict0_model_row_free(&trow);
  sqlite3_free(curkey);
  sqlite3_free(series);
  sqlite3_free(levels);
  sqlite3_free(X);
  sqlite3_free(Y);
  return rc;
}
#endif /* SQLITE_PREDICT_ONNX */

/* distill_forecast option keys. Values are extracted with json_extract
 * below; this closed list exists so an unknown or misspelled key fails
 * loudly (the same contract as every other operation) instead of being
 * silently ignored. */
static const char *const FCST_DIST_OPTION_KEYS[] = {
    "context", "horizon", "student_id", "hidden",
    "epochs",  "lr",      "teacher",    "quantiles", NULL};

static int fdist_opt_validate_cb(void *ctx, const char *key,
                                 sqlite3_value *value, char **errmsg) {
  UNUSED_PARAMETER(ctx);
  UNUSED_PARAMETER(key);
  UNUSED_PARAMETER(value);
  UNUSED_PARAMETER(errmsg);
  return 0; /* keys validated by the parser; values read via json_extract */
}

/* ---- distill_forecast vtab ---- */

#define FD_MODEL 0
#define FD_HASH 1
#define FD_ROWS 2
#define FD_METRIC 3
#define FD_TRAINQ 4
#define FD_OPTIONS 5

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} fd_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  FcstResult res;
  int done;
} fd_cursor;

static int fd_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  fd_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(model_id TEXT, content_hash TEXT, train_rows INTEGER,"
          " train_rmse REAL, train_query HIDDEN,"
          " options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int fd_disconnect(sqlite3_vtab *p) {
  sqlite3_free(p);
  return SQLITE_OK;
}

static int fd_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_train = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    int argv = c->iColumn == FD_TRAINQ ? 1 : c->iColumn == FD_OPTIONS ? 2 : 0;
    if (!argv)
      continue;
    if (!c->usable)
      return SQLITE_CONSTRAINT;
    pIdx->aConstraintUsage[i].argvIndex = argv;
    pIdx->aConstraintUsage[i].omit = 1;
    if (argv == 1)
      seen_train = 1;
  }
  if (!seen_train) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: distill_forecast(train_query, options) requires a train_query",
        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1e6;
  return SQLITE_OK;
}

static int fd_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  fd_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static int fd_close(sqlite3_vtab_cursor *pCur) {
  fd_cursor *c = (fd_cursor *)pCur;
  sqlite3_free(c->res.model_id);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int fd_filter(sqlite3_vtab_cursor *pCur, int idxNum, const char *idxStr,
                     int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  fd_cursor *cur = (fd_cursor *)pCur;
  fd_vtab *vtab = (fd_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  sqlite3_free(cur->res.model_id);
  memset(&cur->res, 0, sizeof(cur->res));
  cur->done = 0;

  const char *tq = argc >= 1 ? (const char *)sqlite3_value_text(argv[0]) : NULL;
  const char *options =
      argc >= 2 ? (const char *)sqlite3_value_text(argv[1]) : NULL;

#define FD_FAIL(...)                                                           \
  do {                                                                         \
    sqlite3_free(vtab->base.zErrMsg);                                          \
    vtab->base.zErrMsg = sqlite3_mprintf(__VA_ARGS__);                         \
    sqlite3_free(student_id);                                                  \
    sqlite3_free(levels);                                                      \
    sqlite3_free(teacher);                                                     \
    return SQLITE_ERROR;                                                       \
  } while (0)

  char *student_id = NULL, *teacher = NULL;
  f32 *levels = NULL;
  int L = 0, H = 0, Q = 1, nhid = FCST_HIDDEN, epochs = FCST_EPOCHS;
  f32 lr = FCST_LR;
  if (!tq)
    FD_FAIL("%s: train_query is required", PREDICT_ERR_SCHEMA);
  {
    /* reject unknown option keys before extracting the known ones */
    char *operr = NULL;
    if (predict0_options_parse(db, options, FCST_DIST_OPTION_KEYS,
                               fdist_opt_validate_cb, NULL, &operr)) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = operr;
      sqlite3_free(student_id);
      sqlite3_free(levels);
      sqlite3_free(teacher);
      return SQLITE_ERROR;
    }
    sqlite3_stmt *op = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_extract(?1,'$.context'), json_extract(?1,'$.horizon'),"
            " json_extract(?1,'$.student_id'), json_extract(?1,'$.hidden'),"
            " json_extract(?1,'$.epochs'),"
            " json_extract(?1,'$.lr'), json_extract(?1,'$.teacher')",
            -1, &op, NULL) == SQLITE_OK) {
      sqlite3_bind_text(op, 1, options, -1, SQLITE_STATIC);
      if (sqlite3_step(op) == SQLITE_ROW) {
        L = sqlite3_column_int(op, 0);
        H = sqlite3_column_int(op, 1);
        const char *sid = (const char *)sqlite3_column_text(op, 2);
        if (sid)
          student_id = sqlite3_mprintf("%s", sid);
        if (sqlite3_column_type(op, 3) != SQLITE_NULL)
          nhid = sqlite3_column_int(op, 3);
        if (sqlite3_column_type(op, 4) != SQLITE_NULL)
          epochs = sqlite3_column_int(op, 4);
        if (sqlite3_column_type(op, 5) != SQLITE_NULL)
          lr = (f32)sqlite3_column_double(op, 5);
        const char *tch = (const char *)sqlite3_column_text(op, 6);
        if (tch)
          teacher = sqlite3_mprintf("%s", tch);
      }
      sqlite3_finalize(op);
    }
  }
  if (L <= 0 || H <= 0 || !student_id)
    FD_FAIL("%s: distill_forecast requires context, horizon, and student_id",
            PREDICT_ERR_OPTIONS);
  if (L > FCST_MAX_CONTEXT)
    FD_FAIL("%s: context %d exceeds the maximum %d", PREDICT_ERR_OPTIONS, L,
            FCST_MAX_CONTEXT);
  if (nhid < 0 || nhid > 2048)
    FD_FAIL("%s: hidden must be in 0..2048 (0 = pure linear student)",
            PREDICT_ERR_OPTIONS);
  if (epochs <= 0 || epochs > 100000)
    FD_FAIL("%s: epochs must be in 1..100000", PREDICT_ERR_OPTIONS);
  if (!(lr > 0.0f) || lr > 10.0f)
    FD_FAIL("%s: lr must be in (0,10]", PREDICT_ERR_OPTIONS);

  { /* optional quantile fan (a JSON array of levels); default = point student */
    sqlite3_stmt *qs = NULL;
    int cap = 0, cnt = 0;
    if (sqlite3_prepare_v2(db, "SELECT value FROM json_each(?1,'$.quantiles')",
                           -1, &qs, NULL) == SQLITE_OK) {
      sqlite3_bind_text(qs, 1, options, -1, SQLITE_STATIC);
      int sr;
      while ((sr = sqlite3_step(qs)) == SQLITE_ROW) {
        if (cnt == cap) {
          int nc = cap ? cap * 2 : 8;
          f32 *nl = sqlite3_realloc64(levels, sizeof(f32) * (size_t)nc);
          if (!nl) {
            sqlite3_finalize(qs);
            FD_FAIL("%s: out of memory", PREDICT_ERR_RESOURCE);
          }
          levels = nl;
          cap = nc;
        }
        levels[cnt++] = (f32)sqlite3_column_double(qs, 0);
      }
      /* A terminal step code other than DONE is a read failure. Build the
       * message from db before finalize clears it, then take FD_FAIL's cleanup
       * path (free student_id/levels/teacher, return). */
      if (sr != SQLITE_DONE) {
        sqlite3_free(vtab->base.zErrMsg);
        vtab->base.zErrMsg = sqlite3_mprintf(
            "%s: could not read quantile levels: %s", PREDICT_ERR_OPTIONS,
            sqlite3_errmsg(db));
        sqlite3_finalize(qs);
        sqlite3_free(student_id);
        sqlite3_free(levels);
        sqlite3_free(teacher);
        return SQLITE_ERROR;
      }
      sqlite3_finalize(qs);
    } else {
      /* Prepare failure is a real error, not "no quantiles": an absent
       * $.quantiles key prepares fine and yields zero rows (the median-only
       * path below). Fail loudly rather than silently train the wrong model. */
      FD_FAIL("%s: could not parse quantiles: %s", PREDICT_ERR_OPTIONS,
              sqlite3_errmsg(db));
    }
    if (cnt > 0)
      Q = cnt;
    else { /* no quantiles key: point student (median only) */
      levels = sqlite3_malloc(sizeof(f32));
      if (!levels)
        FD_FAIL("%s: out of memory", PREDICT_ERR_RESOURCE);
      levels[0] = 0.5f;
      Q = 1;
    }
  }
  if (Q > FCST_MAX_QUANT)
    FD_FAIL("%s: at most %d quantile levels", PREDICT_ERR_OPTIONS,
            FCST_MAX_QUANT);
  for (int i = 1; i < Q; i++) /* insertion sort ascending (Q is small) */
    for (int j = i; j > 0 && levels[j] < levels[j - 1]; j--) {
      f32 t = levels[j];
      levels[j] = levels[j - 1];
      levels[j - 1] = t;
    }
  for (int i = 0; i < Q; i++)
    if (!(levels[i] > 0.0f && levels[i] < 1.0f) ||
        (i > 0 && levels[i] == levels[i - 1]))
      FD_FAIL("%s: quantile levels must be distinct values in (0,1)",
              PREDICT_ERR_OPTIONS);

  char *ensure_err = NULL;
  if (predict0_registry_ensure(db, &ensure_err) != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = ensure_err;
    sqlite3_free(student_id);
    sqlite3_free(levels);
    sqlite3_free(teacher);
    return SQLITE_ERROR;
  }
  char *existing = predict0_registry_model_hash(db, student_id);
  if (existing) {
    sqlite3_free(existing);
    FD_FAIL("%s: student '%s' already exists", PREDICT_ERR_STUDENT_EXISTS,
            student_id);
  }
#undef FD_FAIL

  char *emsg = NULL;
  int rc;
  if (teacher) { /* in-DB labeling: an onnx teacher labels the series' windows */
#ifdef SQLITE_PREDICT_ONNX
    rc = fdistill_train_teacher(db, tq, teacher, L, H, student_id, nhid, epochs,
                                lr, &cur->res, &emsg);
#else
    rc = SQLITE_ERROR;
    emsg = sqlite3_mprintf("%s: onnx runtime is not in this build: teacher",
                           PREDICT_ERR_RUNTIME_UNAVAILABLE);
#endif
  } else {
    rc = fdistill_train(db, tq, L, H, Q, levels, student_id, nhid, epochs, lr,
                        &cur->res, &emsg);
  }
  sqlite3_free(student_id);
  sqlite3_free(levels);
  sqlite3_free(teacher);
  if (rc != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = emsg;
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int fd_next(sqlite3_vtab_cursor *pCur) {
  ((fd_cursor *)pCur)->done = 1;
  return SQLITE_OK;
}

static int fd_eof(sqlite3_vtab_cursor *pCur) {
  return ((fd_cursor *)pCur)->done;
}

static int fd_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx, int col) {
  fd_cursor *c = (fd_cursor *)pCur;
  switch (col) {
  case FD_MODEL:
    sqlite3_result_text(ctx, c->res.model_id, -1, SQLITE_TRANSIENT);
    break;
  case FD_HASH:
    sqlite3_result_text(ctx, c->res.content_hash, -1, SQLITE_TRANSIENT);
    break;
  case FD_ROWS:
    sqlite3_result_int(ctx, c->res.train_rows);
    break;
  case FD_METRIC:
    sqlite3_result_double(ctx, c->res.metric);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int fd_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  UNUSED_PARAMETER(pCur);
  *pRowid = 1;
  return SQLITE_OK;
}

static sqlite3_module forecastDistillModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ fd_connect,
    /* xBestIndex  */ fd_best_index,
    /* xDisconnect */ fd_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ fd_open,
    /* xClose      */ fd_close,
    /* xFilter     */ fd_filter,
    /* xNext       */ fd_next,
    /* xEof        */ fd_eof,
    /* xColumn     */ fd_column,
    /* xRowid      */ fd_rowid,
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

int predict0_distill_init(sqlite3 *db) {
  int rc = sqlite3_create_module(db, "distill_predict", &distillModule, NULL);
  if (rc != SQLITE_OK)
    return rc;
  return sqlite3_create_module(db, "distill_forecast", &forecastDistillModule,
                               NULL);
}
