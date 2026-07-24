/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* distill() — RFC 0005 §4.2.5 — and the native tree student it produces.
 *
 * distill fits a native student (a CART tree or a gradient-boosted forest) on
 * a training signal, evaluates it on a held-out fraction, and stores it as an
 * inline-BLOB student in _predict_models. By default the signal is the target
 * column itself -- your labels, or a strong teacher's predictions computed
 * offline and stored in a column (the way to compress a 33 s TabFM run into a
 * microsecond student). Pass teacher='<model-id>' to instead relabel the rows
 * with a registered predict() model first (e.g. compress the in-context knn5
 * into a standalone tree). Either way the student runs in the zero-dependency
 * core with no onnxruntime.
 *
 * This file is compiled into the core build. The student blob format is
 * implementation-defined (RFC §4.2.5); it is little-endian and rigorously
 * bounds-checked on read, because the registry is writable by any SQL caller
 * (RFC §6.2). */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define TREE_MAX_DEPTH 8
#define TREE_MIN_SPLIT 5
#define TREE_MAX_FEAT 64
#define DISTILL_MIN_ROWS 8

/* ---- tree representation ---- */

typedef struct {
  i32 feature; /* -1 = leaf */
  f32 threshold;
  i32 left, right;
  f32 value; /* leaf: regression value */
  i32 klass; /* leaf classify: class index; -1 internal */
  f32 conf;  /* leaf classify: confidence (fraction) */
} TreeNode;

typedef struct {
  int task; /* 0 classify, 1 regress */
  int nfeat;
  char **feat_names;
  int nclass;
  char **labels;
  int n_nodes;
  TreeNode *nodes;
} Tree;

static void tree_free(Tree *t) {
  if (!t)
    return;
  for (int i = 0; i < t->nfeat; i++)
    sqlite3_free(t->feat_names[i]);
  sqlite3_free(t->feat_names);
  for (int i = 0; i < t->nclass; i++)
    sqlite3_free(t->labels[i]);
  sqlite3_free(t->labels);
  sqlite3_free(t->nodes);
  memset(t, 0, sizeof(*t));
}

/* ---- little-endian serialization ---- */

static const char TREE_MAGIC[8] = {'P', 'S', 'T', 'R', 'E', 'E', '0', '1'};

static void put_u32(u8 **p, u32 v) {
  (*p)[0] = (u8)v;
  (*p)[1] = (u8)(v >> 8);
  (*p)[2] = (u8)(v >> 16);
  (*p)[3] = (u8)(v >> 24);
  *p += 4;
}
static void put_f32(u8 **p, f32 v) {
  u32 u;
  memcpy(&u, &v, 4);
  put_u32(p, u);
}
static void put_str(u8 **p, const char *s) {
  u32 n = (u32)strlen(s);
  put_u32(p, n);
  memcpy(*p, s, n);
  *p += n;
}

/* Bounds-checked reader over [buf, end). Sets *err on any overrun. */
typedef struct {
  const u8 *p, *end;
  int err;
} Reader;

static u32 rd_u32(Reader *r) {
  if (r->err || r->p + 4 > r->end) {
    r->err = 1;
    return 0;
  }
  u32 v = (u32)r->p[0] | ((u32)r->p[1] << 8) | ((u32)r->p[2] << 16) |
          ((u32)r->p[3] << 24);
  r->p += 4;
  return v;
}
static f32 rd_f32(Reader *r) {
  u32 u = rd_u32(r);
  f32 v;
  memcpy(&v, &u, 4);
  return v;
}
static char *rd_str(Reader *r) {
  u32 n = rd_u32(r);
  if (r->err || n > (u32)(r->end - r->p)) {
    r->err = 1;
    return NULL;
  }
  char *s = sqlite3_malloc((int)n + 1);
  if (!s) {
    r->err = 1;
    return NULL;
  }
  memcpy(s, r->p, n);
  s[n] = '\0';
  r->p += n;
  return s;
}

static int tree_serialize(const Tree *t, void **blob_out, int *len_out) {
  size_t sz = sizeof(TREE_MAGIC) + 4 * 4;
  for (int i = 0; i < t->nfeat; i++)
    sz += 4 + strlen(t->feat_names[i]);
  for (int i = 0; i < t->nclass; i++)
    sz += 4 + strlen(t->labels[i]);
  sz += (size_t)t->n_nodes * (4 + 4 + 4 + 4 + 4 + 4 + 4);

  u8 *buf = sqlite3_malloc((int)sz);
  if (!buf)
    return SQLITE_NOMEM;
  u8 *p = buf;
  memcpy(p, TREE_MAGIC, sizeof(TREE_MAGIC));
  p += sizeof(TREE_MAGIC);
  put_u32(&p, (u32)t->task);
  put_u32(&p, (u32)t->nfeat);
  put_u32(&p, (u32)t->nclass);
  put_u32(&p, (u32)t->n_nodes);
  for (int i = 0; i < t->nfeat; i++)
    put_str(&p, t->feat_names[i]);
  for (int i = 0; i < t->nclass; i++)
    put_str(&p, t->labels[i]);
  for (int i = 0; i < t->n_nodes; i++) {
    const TreeNode *n = &t->nodes[i];
    put_u32(&p, (u32)n->feature);
    put_f32(&p, n->threshold);
    put_u32(&p, (u32)n->left);
    put_u32(&p, (u32)n->right);
    put_f32(&p, n->value);
    put_u32(&p, (u32)n->klass);
    put_f32(&p, n->conf);
  }
  *blob_out = buf;
  *len_out = (int)sz;
  return SQLITE_OK;
}

/* Deserialize + validate. Every field is range-checked so a hand-crafted
 * blob cannot drive an out-of-bounds read or a bad tree traversal. */
static int tree_deserialize(const void *blob, int len, Tree *t, char **errmsg) {
  memset(t, 0, sizeof(*t));
  Reader r = {.p = blob, .end = (const u8 *)blob + len, .err = 0};
  if (!blob || len < (int)sizeof(TREE_MAGIC) ||
      memcmp(blob, TREE_MAGIC, sizeof(TREE_MAGIC)) != 0) {
    *errmsg = sqlite3_mprintf("%s: not a tree student blob", PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  r.p += sizeof(TREE_MAGIC);
  t->task = (int)rd_u32(&r);
  t->nfeat = (int)rd_u32(&r);
  t->nclass = (int)rd_u32(&r);
  t->n_nodes = (int)rd_u32(&r);
  if (r.err || t->task < 0 || t->task > 1 || t->nfeat <= 0 ||
      t->nfeat > TREE_MAX_FEAT || t->nclass < 0 || t->n_nodes <= 0 ||
      t->n_nodes > (1 << 24))
    goto bad;

  t->feat_names = sqlite3_malloc(sizeof(char *) * t->nfeat);
  if (!t->feat_names)
    goto oom;
  memset(t->feat_names, 0, sizeof(char *) * t->nfeat);
  for (int i = 0; i < t->nfeat; i++) {
    t->feat_names[i] = rd_str(&r);
    if (r.err)
      goto bad;
  }
  if (t->nclass > 0) {
    t->labels = sqlite3_malloc(sizeof(char *) * t->nclass);
    if (!t->labels)
      goto oom;
    memset(t->labels, 0, sizeof(char *) * t->nclass);
    for (int i = 0; i < t->nclass; i++) {
      t->labels[i] = rd_str(&r);
      if (r.err)
        goto bad;
    }
  }
  t->nodes = sqlite3_malloc(sizeof(TreeNode) * t->n_nodes);
  if (!t->nodes)
    goto oom;
  for (int i = 0; i < t->n_nodes; i++) {
    TreeNode *n = &t->nodes[i];
    n->feature = (i32)rd_u32(&r);
    n->threshold = rd_f32(&r);
    n->left = (i32)rd_u32(&r);
    n->right = (i32)rd_u32(&r);
    n->value = rd_f32(&r);
    n->klass = (i32)rd_u32(&r);
    n->conf = rd_f32(&r);
    if (r.err)
      goto bad;
    /* structural validity: internal children in range, leaf class in range */
    if (n->feature >= 0) {
      if (n->feature >= t->nfeat || n->left < 0 || n->left >= t->n_nodes ||
          n->right < 0 || n->right >= t->n_nodes)
        goto bad;
    } else if (t->task == 0 && (n->klass < 0 || n->klass >= t->nclass)) {
      goto bad;
    }
  }
  return SQLITE_OK;

oom:
  tree_free(t);
  *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  return SQLITE_NOMEM;
bad:
  tree_free(t);
  *errmsg = sqlite3_mprintf("%s: corrupt tree student blob", PREDICT_ERR_SCHEMA);
  return SQLITE_ERROR;
}

/* Traverse the tree for one feature vector. Guards against a cycle by
 * bounding the hop count at n_nodes (a validated tree is acyclic, but the
 * bound is cheap insurance). Returns the leaf node index, or -1 on trouble. */
static int tree_walk(const Tree *t, const f32 *x) {
  int node = 0;
  for (int hops = 0; hops <= t->n_nodes; hops++) {
    const TreeNode *n = &t->nodes[node];
    if (n->feature < 0)
      return node; /* leaf */
    node = x[n->feature] < n->threshold ? n->left : n->right;
  }
  return -1;
}

/* ---- the tree runtime (execute a student over apply rows) ---- */

/* ---- gradient-boosted tree student (student_kind='gbt') ---- */

/* A single depth-8 CART underfits complex boundaries; an additive ensemble of
 * shallow regression trees (gradient boosting) closes most of the gap and is
 * deterministic by construction — no bootstrap or feature-sampling randomness,
 * so the student stays reproducible. Classification uses softmax with one
 * score function per class; regression uses squared loss. */

#define GBT_ROUNDS 200
#define GBT_DEPTH 3
#define GBT_MIN_SPLIT 5
#define GBT_LR 0.1f     /* shrinkage: many small steps generalize better than few big ones */
#define GBT_LAMBDA 1.0f /* L2 leaf regularization (XGBoost reg_lambda default) */

static const char GBT_MAGIC[8] = {'P', 'S', 'G', 'B', 'T', '0', '1', '\0'};

typedef struct {
  int task; /* 0 classify, 1 regress */
  int nfeat;
  char **feat_names;
  int nclass;
  char **labels;
  int n_score; /* score functions: nclass (classify) or 1 (regress) */
  int n_rounds;
  f32 lr;
  f32 *init;       /* [n_score] */
  int n_trees;     /* n_rounds * n_score */
  int *tree_off;   /* [n_trees+1] offsets into nodes (per-tree local roots at 0) */
  TreeNode *nodes; /* flat pool; tree j uses [tree_off[j], tree_off[j+1]) */
} Forest;

static void forest_free(Forest *f) {
  if (!f)
    return;
  for (int i = 0; i < f->nfeat; i++)
    sqlite3_free(f->feat_names[i]);
  sqlite3_free(f->feat_names);
  for (int i = 0; i < f->nclass; i++)
    sqlite3_free(f->labels[i]);
  sqlite3_free(f->labels);
  sqlite3_free(f->init);
  sqlite3_free(f->tree_off);
  sqlite3_free(f->nodes);
  memset(f, 0, sizeof(*f));
}

/* Walk one weak learner (regression tree) whose nodes start at `nd` with
 * 0-based left/right, returning its leaf value. `guard` bounds the hops. */
static f32 reg_tree_value(const TreeNode *nd, int guard, const f32 *x) {
  int node = 0;
  for (int h = 0; h <= guard; h++) {
    if (nd[node].feature < 0)
      return nd[node].value;
    node = x[nd[node].feature] < nd[node].threshold ? nd[node].left
                                                     : nd[node].right;
  }
  return 0.f;
}

static f32 forest_tree_value(const Forest *f, int j, const f32 *x) {
  int base = f->tree_off[j], guard = f->tree_off[j + 1] - base;
  return reg_tree_value(&f->nodes[base], guard, x);
}

/* Predict one row: sets *pred (sqlite3_malloc'd) and, for classify, *conf. */
static int forest_predict_row(const Forest *f, const f32 *x, f64 *scbuf,
                              char **pred, f64 *conf, int *has_conf) {
  if (f->task == 1) {
    f64 s = f->init[0];
    for (int j = 0; j < f->n_trees; j++)
      s += f->lr * forest_tree_value(f, j, x);
    *has_conf = 0;
    *pred = sqlite3_mprintf("%.17g", s);
    return *pred ? SQLITE_OK : SQLITE_NOMEM;
  }
  for (int c = 0; c < f->n_score; c++)
    scbuf[c] = f->init[c];
  for (int r = 0; r < f->n_rounds; r++)
    for (int s = 0; s < f->n_score; s++)
      scbuf[s] += f->lr * forest_tree_value(f, r * f->n_score + s, x);
  f64 mx = scbuf[0];
  int arg = 0;
  for (int c = 1; c < f->n_score; c++)
    if (scbuf[c] > mx) {
      mx = scbuf[c];
      arg = c;
    }
  f64 sum = 0;
  for (int c = 0; c < f->n_score; c++)
    sum += exp(scbuf[c] - mx);
  *conf = 1.0 / sum; /* softmax at the argmax */
  *has_conf = 1;
  *pred = sqlite3_mprintf("%s", f->labels[arg]);
  return *pred ? SQLITE_OK : SQLITE_NOMEM;
}

static int forest_serialize(const Forest *f, void **blob_out, int *len_out) {
  size_t sz = sizeof(GBT_MAGIC) + 4 * 5 + 4 /*lr*/ + 4 * (size_t)f->n_score;
  for (int i = 0; i < f->nfeat; i++)
    sz += 4 + strlen(f->feat_names[i]);
  for (int i = 0; i < f->nclass; i++)
    sz += 4 + strlen(f->labels[i]);
  sz += 4; /* n_trees */
  for (int j = 0; j < f->n_trees; j++)
    sz += 4 + (size_t)(f->tree_off[j + 1] - f->tree_off[j]) * (4 + 4 + 4 + 4 + 4);

  u8 *buf = sqlite3_malloc((int)sz);
  if (!buf)
    return SQLITE_NOMEM;
  u8 *p = buf;
  memcpy(p, GBT_MAGIC, sizeof(GBT_MAGIC));
  p += sizeof(GBT_MAGIC);
  put_u32(&p, (u32)f->task);
  put_u32(&p, (u32)f->nfeat);
  put_u32(&p, (u32)f->nclass);
  put_u32(&p, (u32)f->n_score);
  put_u32(&p, (u32)f->n_rounds);
  put_f32(&p, f->lr);
  for (int i = 0; i < f->n_score; i++)
    put_f32(&p, f->init[i]);
  for (int i = 0; i < f->nfeat; i++)
    put_str(&p, f->feat_names[i]);
  for (int i = 0; i < f->nclass; i++)
    put_str(&p, f->labels[i]);
  put_u32(&p, (u32)f->n_trees);
  for (int j = 0; j < f->n_trees; j++) {
    int m = f->tree_off[j + 1] - f->tree_off[j];
    put_u32(&p, (u32)m);
    for (int k = 0; k < m; k++) {
      const TreeNode *n = &f->nodes[f->tree_off[j] + k];
      put_u32(&p, (u32)n->feature);
      put_f32(&p, n->threshold);
      put_u32(&p, (u32)n->left);
      put_u32(&p, (u32)n->right);
      put_f32(&p, n->value);
    }
  }
  *blob_out = buf;
  *len_out = (int)sz;
  return SQLITE_OK;
}

static int forest_deserialize(const void *blob, int len, Forest *f,
                              char **errmsg) {
  memset(f, 0, sizeof(*f));
  Reader r = {.p = blob, .end = (const u8 *)blob + len, .err = 0};
  if (!blob || len < (int)sizeof(GBT_MAGIC) ||
      memcmp(blob, GBT_MAGIC, sizeof(GBT_MAGIC)) != 0) {
    *errmsg = sqlite3_mprintf("%s: not a gbt student blob", PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  r.p += sizeof(GBT_MAGIC);
  f->task = (int)rd_u32(&r);
  f->nfeat = (int)rd_u32(&r);
  f->nclass = (int)rd_u32(&r);
  f->n_score = (int)rd_u32(&r);
  f->n_rounds = (int)rd_u32(&r);
  f->lr = rd_f32(&r);
  int want_score = f->task == 0 ? f->nclass : 1;
  if (r.err || f->task < 0 || f->task > 1 || f->nfeat <= 0 ||
      f->nfeat > TREE_MAX_FEAT || f->nclass < 0 || f->n_rounds <= 0 ||
      f->n_rounds > (1 << 20) || f->n_score != want_score || f->n_score <= 0)
    goto bad;

  f->init = sqlite3_malloc(sizeof(f32) * f->n_score);
  if (!f->init)
    goto oom;
  for (int i = 0; i < f->n_score; i++)
    f->init[i] = rd_f32(&r);
  f->feat_names = sqlite3_malloc(sizeof(char *) * f->nfeat);
  if (!f->feat_names)
    goto oom;
  memset(f->feat_names, 0, sizeof(char *) * f->nfeat);
  for (int i = 0; i < f->nfeat; i++)
    if (!(f->feat_names[i] = rd_str(&r)))
      goto bad;
  if (f->nclass > 0) {
    f->labels = sqlite3_malloc(sizeof(char *) * f->nclass);
    if (!f->labels)
      goto oom;
    memset(f->labels, 0, sizeof(char *) * f->nclass);
    for (int i = 0; i < f->nclass; i++)
      if (!(f->labels[i] = rd_str(&r)))
        goto bad;
  }
  f->n_trees = (int)rd_u32(&r);
  if (r.err || f->n_trees != f->n_rounds * f->n_score)
    goto bad;
  f->tree_off = sqlite3_malloc(sizeof(int) * (f->n_trees + 1));
  if (!f->tree_off)
    goto oom;
  f->tree_off[0] = 0;
  int pool_cap = 0;
  for (int j = 0; j < f->n_trees; j++) {
    int m = (int)rd_u32(&r);
    if (r.err || m <= 0 || m > (1 << 20))
      goto bad;
    int need = f->tree_off[j] + m;
    if (need > pool_cap) {
      pool_cap = need > pool_cap * 2 ? need : pool_cap * 2;
      TreeNode *g = sqlite3_realloc(f->nodes, sizeof(TreeNode) * pool_cap);
      if (!g)
        goto oom;
      f->nodes = g;
    }
    for (int k = 0; k < m; k++) {
      TreeNode *n = &f->nodes[f->tree_off[j] + k];
      memset(n, 0, sizeof(*n));
      n->feature = (i32)rd_u32(&r);
      n->threshold = rd_f32(&r);
      n->left = (i32)rd_u32(&r);
      n->right = (i32)rd_u32(&r);
      n->value = rd_f32(&r);
      n->klass = -1;
      if (r.err)
        goto bad;
      if (n->feature >= 0 &&
          (n->feature >= f->nfeat || n->left < 0 || n->left >= m ||
           n->right < 0 || n->right >= m))
        goto bad; /* internal children stay within this tree */
    }
    f->tree_off[j + 1] = need;
  }
  return SQLITE_OK;

oom:
  forest_free(f);
  *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  return SQLITE_NOMEM;
bad:
  forest_free(f);
  *errmsg = sqlite3_mprintf("%s: corrupt gbt student blob", PREDICT_ERR_SCHEMA);
  return SQLITE_ERROR;
}

int predict0_tree_run(sqlite3 *db, const char *model_id, const char *apply_sql,
                      const predict0_model_row *model,
                      const predict0_backend_opts *opts,
                      predict0_result **out_rows, int *out_n,
                      char receipt_id_out[PREDICT_ULID_BUFSIZE], char **errmsg) {
  *out_rows = NULL;
  *out_n = 0;
  *errmsg = NULL;

  /* onnx-only options are meaningless for a native tree */
  if (opts->device || opts->precision || opts->accept_license) {
    *errmsg = sqlite3_mprintf(
        "%s: device/precision/accept_license do not apply to a tree student",
        PREDICT_ERR_OPTIONS);
    return SQLITE_ERROR;
  }
  if (!model->weights || model->weights_len <= 0) {
    *errmsg = sqlite3_mprintf("%s: tree student has no weights",
                              PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }

  /* one runtime for both native students: a single CART (PSTREE) or a
   * gradient-boosted forest (PSGBT), told apart by the blob magic */
  int is_forest = model->weights_len >= (int)sizeof(GBT_MAGIC) &&
                  memcmp(model->weights, GBT_MAGIC, sizeof(GBT_MAGIC)) == 0;
  Tree tree;
  Forest forest;
  memset(&tree, 0, sizeof(tree));
  memset(&forest, 0, sizeof(forest));
  int rc = is_forest ? forest_deserialize(model->weights, model->weights_len,
                                          &forest, errmsg)
                     : tree_deserialize(model->weights, model->weights_len,
                                        &tree, errmsg);
  if (rc != SQLITE_OK)
    return rc;
  int nfeat = is_forest ? forest.nfeat : tree.nfeat;
  char **feat_names = is_forest ? forest.feat_names : tree.feat_names;
  int classify = (is_forest ? forest.task : tree.task) == 0;
  f64 *scbuf = NULL;
  if (is_forest && classify) {
    scbuf = sqlite3_malloc(sizeof(f64) * forest.n_score);
    if (!scbuf) {
      forest_free(&forest);
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      return SQLITE_NOMEM;
    }
  }

  predict0_result *rows = NULL;
  int nrows = 0, rcap = 0;
  int *amap = NULL; /* apply feature col -> student feature slot */
  sqlite3_stmt *as = NULL;

  if (sqlite3_prepare_v2(db, apply_sql, -1, &as, NULL) != SQLITE_OK || !as) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: apply query does not parse: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
    goto done;
  }
  if (!sqlite3_stmt_readonly(as)) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: apply query must be a read-only SELECT",
                              PREDICT_ERR_QUERY_NOT_READONLY);
    goto done;
  }
  int an = sqlite3_column_count(as);
  if (an - 1 != nfeat) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: apply features (%d) must match the student's (%d)",
        PREDICT_ERR_SCHEMA, an - 1, nfeat);
    goto done;
  }
  amap = sqlite3_malloc(sizeof(int) * (an - 1));
  if (!amap) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }
  for (int i = 1; i < an; i++) {
    const char *nm = sqlite3_column_name(as, i);
    int found = -1;
    for (int f = 0; f < nfeat; f++)
      if (nm && strcmp(nm, feat_names[f]) == 0)
        found = f;
    if (found < 0) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: apply column '%s' is not a student feature", PREDICT_ERR_SCHEMA,
          nm ? nm : "?");
      goto done;
    }
    amap[i - 1] = found;
  }

  int step;
  while ((step = sqlite3_step(as)) == SQLITE_ROW) {
    if (nrows == rcap) {
      rcap = rcap ? rcap * 2 : 256;
      predict0_result *g =
          sqlite3_realloc(rows, sizeof(predict0_result) * rcap);
      if (!g) {
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
      rows = g;
    }
    predict0_result *out = &rows[nrows];
    memset(out, 0, sizeof(*out));
    out->ref_type = sqlite3_column_type(as, 0);
    if (out->ref_type == SQLITE_INTEGER)
      out->ref_i = sqlite3_column_int64(as, 0);
    else if (out->ref_type == SQLITE_FLOAT)
      out->ref_f = sqlite3_column_double(as, 0);
    else if (out->ref_type != SQLITE_NULL)
      out->ref_t = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(as, 0));

    f32 x[TREE_MAX_FEAT];
    int bad = 0;
    for (int i = 1; i < an; i++) {
      int ct = sqlite3_column_type(as, i);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        bad = 1;
        break;
      }
      x[amap[i - 1]] = (f32)sqlite3_column_double(as, i);
    }
    if (bad) {
      out->status = "non_numeric";
      nrows++;
      continue;
    }
    if (is_forest) {
      rc = forest_predict_row(&forest, x, scbuf, &out->prediction,
                              &out->confidence, &out->has_conf);
      if (rc != SQLITE_OK) {
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
    } else {
      int leaf = tree_walk(&tree, x);
      if (leaf < 0) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf("%s: malformed tree traversal",
                                  PREDICT_ERR_SCHEMA);
        goto done;
      }
      const TreeNode *ln = &tree.nodes[leaf];
      if (classify) {
        out->prediction = sqlite3_mprintf("%s", tree.labels[ln->klass]);
        out->confidence = ln->conf;
        out->has_conf = 1;
      } else {
        out->prediction = sqlite3_mprintf("%.17g", (f64)ln->value);
      }
      if (!out->prediction) {
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
    }
    out->status = "ok";
    nrows++;
  }
  if (step != SQLITE_DONE) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: apply query failed: %s",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
    goto done;
  }

  if (opts->receipt) {
    predict0_hasher h;
    predict0_hash_init(&h);
    for (int i = 0; i < nrows; i++) {
      predict0_result *r = &rows[i];
      switch (r->ref_type) {
      case SQLITE_INTEGER:
        predict0_hash_int(&h, r->ref_i);
        break;
      case SQLITE_FLOAT:
        predict0_hash_real(&h, r->ref_f);
        break;
      case SQLITE_NULL:
        predict0_hash_null(&h);
        break;
      default:
        predict0_hash_text(&h, r->ref_t);
        break;
      }
      if (r->prediction)
        predict0_hash_text(&h, r->prediction);
      else
        predict0_hash_null(&h);
      if (r->has_conf)
        predict0_hash_real(&h, r->confidence);
      else
        predict0_hash_null(&h);
      predict0_hash_row_end(&h);
    }
    char result_hash[PREDICT_HEX_BUFSIZE];
    predict0_hash_hex(&h, result_hash);

    char *params = NULL, *input_json = NULL;
    sqlite3_stmt *pj = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_object('model', ?1, 'receipt', 1),"
            " json_object('apply', ?2)",
            -1, &pj, NULL) == SQLITE_OK) {
      sqlite3_bind_text(pj, 1, model_id, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 2, apply_sql, -1, SQLITE_STATIC);
      if (sqlite3_step(pj) == SQLITE_ROW) {
        params = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 0));
        input_json =
            sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 1));
      }
      sqlite3_finalize(pj);
    }
    if (!params || !input_json) {
      sqlite3_free(params);
      sqlite3_free(input_json);
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: could not canonicalize params",
                                PREDICT_ERR_RESOURCE);
      goto done;
    }
    char *rerr = NULL;
    rc = predict0_emit_receipt(db, "predict", model_id, params, input_json,
                               result_hash, receipt_id_out, &rerr);
    sqlite3_free(params);
    sqlite3_free(input_json);
    if (rc != SQLITE_OK) {
      *errmsg = rerr ? rerr
                     : sqlite3_mprintf("%s: receipt write failed",
                                       PREDICT_ERR_RESOURCE);
      goto done;
    }
  }
  rc = SQLITE_OK;

done:
  if (as)
    sqlite3_finalize(as);
  sqlite3_free(amap);
  sqlite3_free(scbuf);
  if (is_forest)
    forest_free(&forest);
  else
    tree_free(&tree);
  if (rc == SQLITE_OK) {
    *out_rows = rows;
    *out_n = nrows;
  } else {
    predict0_results_free(rows, nrows);
  }
  return rc;
}

/* ---- CART training ---- */

typedef struct {
  TreeNode *nodes;
  int n, cap;
  const f32 *X; /* [nrow, nfeat] */
  int nfeat;
  const i32 *yc; /* classify targets (teacher class index) */
  const f32 *yr; /* regress targets */
  int nclass;
  int task;
  int max_depth; /* 0 => TREE_MAX_DEPTH; shallow for GBT weak learners */
  int min_split; /* 0 => TREE_MIN_SPLIT */
  const f32 *hess; /* GBT only: per-row Hessian; NULL => mean-value leaves */
  f32 lambda;      /* GBT only: L2 leaf regularization */
} Builder;

static int bld_new_node(Builder *b) {
  if (b->n == b->cap) {
    int nc = b->cap ? b->cap * 2 : 64;
    TreeNode *g = sqlite3_realloc(b->nodes, sizeof(TreeNode) * nc);
    if (!g)
      return -1;
    b->nodes = g;
    b->cap = nc;
  }
  memset(&b->nodes[b->n], 0, sizeof(TreeNode));
  return b->n++;
}

/* Fill node `ni` as a leaf over rows idx[0..n). */
static void bld_leaf(Builder *b, int ni, const int *idx, int n) {
  TreeNode *nd = &b->nodes[ni];
  nd->feature = -1;
  nd->left = nd->right = -1;
  if (b->task == 0) {
    int *cnt = sqlite3_malloc(sizeof(int) * b->nclass);
    int best = 0, bestc = -1;
    if (cnt) {
      memset(cnt, 0, sizeof(int) * b->nclass);
      for (int i = 0; i < n; i++)
        cnt[b->yc[idx[i]]]++;
      for (int c = 0; c < b->nclass; c++)
        if (cnt[c] > bestc) {
          bestc = cnt[c];
          best = c;
        }
      sqlite3_free(cnt);
    }
    nd->klass = best;
    nd->conf = n ? (f32)bestc / (f32)n : 0.f;
  } else if (b->hess) {
    /* Newton leaf: the tree fits the gradient (b->yr), but the leaf value is
     * the second-order step sum(grad) / (sum(hess) + lambda) -- what lifts a
     * gradient booster to XGBoost-quality on non-squared losses. */
    f64 g = 0, h = 0;
    for (int i = 0; i < n; i++) {
      g += b->yr[idx[i]];
      h += b->hess[idx[i]];
    }
    nd->value = (f32)(g / (h + b->lambda));
    nd->klass = -1;
  } else {
    f64 s = 0;
    for (int i = 0; i < n; i++)
      s += b->yr[idx[i]];
    nd->value = n ? (f32)(s / n) : 0.f;
    nd->klass = -1;
  }
}

/* Best (feature, threshold) split minimizing impurity, or feature<0 if none
 * improves. cmp buffer of (value, target) sorted per feature. */
typedef struct {
  f32 v;
  i32 yc;
  f32 yr;
} VY;
static int vy_cmp(const void *a, const void *b) {
  f32 x = ((const VY *)a)->v, y = ((const VY *)b)->v;
  return x < y ? -1 : x > y ? 1 : 0;
}

static int bld_best_split(Builder *b, const int *idx, int n, int *feat_out,
                          f32 *thr_out) {
  *feat_out = -1;
  VY *buf = sqlite3_malloc(sizeof(VY) * n);
  if (!buf)
    return SQLITE_NOMEM;
  f64 best_score = 0; /* impurity decrease; want > 0 */

  for (int f = 0; f < b->nfeat; f++) {
    for (int i = 0; i < n; i++) {
      buf[i].v = b->X[(size_t)idx[i] * b->nfeat + f];
      if (b->task == 0)
        buf[i].yc = b->yc[idx[i]];
      else
        buf[i].yr = b->yr[idx[i]];
    }
    qsort(buf, n, sizeof(VY), vy_cmp);
    if (buf[0].v == buf[n - 1].v)
      continue; /* constant feature */

    if (b->task == 0) {
      int *ltot = sqlite3_malloc(sizeof(int) * b->nclass * 2);
      if (!ltot) {
        sqlite3_free(buf);
        return SQLITE_NOMEM;
      }
      int *rtot = ltot + b->nclass;
      memset(ltot, 0, sizeof(int) * b->nclass * 2);
      for (int i = 0; i < n; i++)
        rtot[buf[i].yc]++;
      int nl = 0;
      for (int i = 0; i < n - 1; i++) {
        ltot[buf[i].yc]++;
        rtot[buf[i].yc]--;
        nl++;
        if (buf[i].v == buf[i + 1].v)
          continue; /* can't split between equal values */
        int nr = n - nl;
        f64 gl = 1, gr = 1;
        for (int c = 0; c < b->nclass; c++) {
          f64 pl = (f64)ltot[c] / nl, pr = (f64)rtot[c] / nr;
          gl -= pl * pl;
          gr -= pr * pr;
        }
        f64 score = -((f64)nl * gl + (f64)nr * gr) / n; /* maximize */
        if (*feat_out < 0 || score > best_score) {
          best_score = score;
          *feat_out = f;
          *thr_out = (buf[i].v + buf[i + 1].v) / 2.f;
        }
      }
      sqlite3_free(ltot);
    } else {
      f64 tot = 0, totsq = 0;
      for (int i = 0; i < n; i++) {
        tot += buf[i].yr;
        totsq += (f64)buf[i].yr * buf[i].yr;
      }
      f64 lsum = 0, lsq = 0;
      int nl = 0;
      for (int i = 0; i < n - 1; i++) {
        lsum += buf[i].yr;
        lsq += (f64)buf[i].yr * buf[i].yr;
        nl++;
        if (buf[i].v == buf[i + 1].v)
          continue;
        int nr = n - nl;
        f64 rsum = tot - lsum, rsq = totsq - lsq;
        f64 lvar = lsq - lsum * lsum / nl;   /* SSE left */
        f64 rvar = rsq - rsum * rsum / nr;   /* SSE right */
        f64 score = -(lvar + rvar);          /* maximize (minimize SSE) */
        if (*feat_out < 0 || score > best_score) {
          best_score = score;
          *feat_out = f;
          *thr_out = (buf[i].v + buf[i + 1].v) / 2.f;
        }
      }
    }
  }
  sqlite3_free(buf);
  return SQLITE_OK;
}

/* Recursively build; returns the node index, or -1 on OOM. */
static int bld_build(Builder *b, int *idx, int n, int depth) {
  int ni = bld_new_node(b);
  if (ni < 0)
    return -1;

  int pure = 1;
  if (b->task == 0) {
    for (int i = 1; i < n; i++)
      if (b->yc[idx[i]] != b->yc[idx[0]]) {
        pure = 0;
        break;
      }
  } else {
    pure = 0; /* regression leaves split on impurity, not purity */
  }
  int maxd = b->max_depth ? b->max_depth : TREE_MAX_DEPTH;
  int mins = b->min_split ? b->min_split : TREE_MIN_SPLIT;
  if (depth >= maxd || n < mins || pure) {
    bld_leaf(b, ni, idx, n);
    return ni;
  }
  int feat;
  f32 thr;
  if (bld_best_split(b, idx, n, &feat, &thr) != SQLITE_OK)
    return -1;
  if (feat < 0) {
    bld_leaf(b, ni, idx, n);
    return ni;
  }
  /* partition idx: rows with X[.,feat] < thr to the front */
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    if (b->X[(size_t)idx[lo] * b->nfeat + feat] < thr) {
      lo++;
    } else {
      int tmp = idx[lo];
      idx[lo] = idx[hi];
      idx[hi] = tmp;
      hi--;
    }
  }
  int nl = lo;
  if (nl == 0 || nl == n) { /* degenerate split; make a leaf */
    bld_leaf(b, ni, idx, n);
    return ni;
  }
  int L = bld_build(b, idx, nl, depth + 1);
  if (L < 0)
    return -1;
  int R = bld_build(b, idx + nl, n - nl, depth + 1);
  if (R < 0)
    return -1;
  /* node index ni is stable across the reallocs above; set it now */
  b->nodes[ni].feature = feat;
  b->nodes[ni].threshold = thr;
  b->nodes[ni].left = L;
  b->nodes[ni].right = R;
  b->nodes[ni].klass = -1;
  return ni;
}


/* Train a GBT on the teacher targets. Fills the numeric parts of *fo; the
 * caller sets feat_names and labels (as for the single-tree path). For
 * classification, `soft` (when
 * non-NULL) is a row-major [n, nclass] matrix of teacher class probabilities
 * that replaces the hard one-hot label: the student then matches the teacher's
 * whole distribution (soft-label distillation), transferring the calibrated
 * probabilities a hard argmax throws away. NULL `soft` keeps the hard-label
 * path. Regression ignores `soft`. */
static int train_gbt(const f32 *X, int n, int nfeat, int task, int nclass,
                     const i32 *yc, const f32 *yr, const f32 *soft, Forest *fo,
                     char **errmsg) {
  memset(fo, 0, sizeof(*fo));
  int rounds = GBT_ROUNDS, nscore = task == 0 ? nclass : 1;
  fo->task = task;
  fo->nfeat = nfeat;
  fo->nclass = nclass;
  fo->n_score = nscore;
  fo->n_rounds = rounds;
  fo->lr = GBT_LR;

  int rc = SQLITE_OK;
  fo->init = sqlite3_malloc(sizeof(f32) * nscore);
  fo->tree_off = sqlite3_malloc(sizeof(int) * (rounds * nscore + 1));
  f64 *F = sqlite3_malloc(sizeof(f64) * (size_t)n * nscore);
  f64 *p = task == 0 ? sqlite3_malloc(sizeof(f64) * (size_t)n * nscore) : NULL;
  int *idx = sqlite3_malloc(sizeof(int) * n); /* scratch for bld_build */
  f32 *grad = sqlite3_malloc(sizeof(f32) * n);
  f32 *hess = task == 0 ? sqlite3_malloc(sizeof(f32) * n) : NULL;
  if (!fo->init || !fo->tree_off || !F || !idx || !grad ||
      (task == 0 && (!p || !hess))) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }

  if (task == 0) {
    for (int c = 0; c < nscore; c++) {
      f64 pri;
      if (soft) { /* mean teacher probability for the class */
        f64 sum = 0;
        for (int i = 0; i < n; i++)
          sum += soft[(size_t)i * nscore + c];
        pri = sum / n;
        if (pri < 1e-6)
          pri = 1e-6;
      } else {
        int cnt = 0;
        for (int i = 0; i < n; i++)
          cnt += yc[i] == c;
        pri = (cnt + 1.0) / (n + nscore); /* smoothed */
      }
      fo->init[c] = (f32)log(pri);
    }
    for (int i = 0; i < n; i++)
      for (int c = 0; c < nscore; c++)
        F[(size_t)i * nscore + c] = fo->init[c];
  } else {
    f64 m = 0;
    for (int i = 0; i < n; i++)
      m += yr[i];
    m /= n;
    fo->init[0] = (f32)m;
    for (int i = 0; i < n; i++)
      F[i] = m;
  }

  fo->tree_off[0] = 0;
  int nt = 0, pool_cap = 0;
  for (int r = 0; r < rounds; r++) {
    if (task == 0) { /* round-start softmax for every row */
      for (int i = 0; i < n; i++) {
        f64 *Fi = &F[(size_t)i * nscore], mx = Fi[0];
        for (int c = 1; c < nscore; c++)
          if (Fi[c] > mx)
            mx = Fi[c];
        f64 sum = 0;
        for (int c = 0; c < nscore; c++)
          sum += (p[(size_t)i * nscore + c] = exp(Fi[c] - mx));
        for (int c = 0; c < nscore; c++)
          p[(size_t)i * nscore + c] /= sum;
      }
    }
    for (int s = 0; s < nscore; s++) {
      if (task == 0)
        for (int i = 0; i < n; i++) {
          f64 pi = p[(size_t)i * nscore + s];
          f32 tgt =
              soft ? soft[(size_t)i * nscore + s] : (yc[i] == s ? 1.f : 0.f);
          grad[i] = tgt - (f32)pi;
          hess[i] = (f32)(pi * (1.0 - pi)); /* softmax curvature */
        }
      else
        for (int i = 0; i < n; i++)
          grad[i] = (f32)(yr[i] - F[i]);

      Builder b;
      memset(&b, 0, sizeof(b));
      b.X = X;
      b.nfeat = nfeat;
      b.yr = grad;
      b.task = 1;
      b.max_depth = GBT_DEPTH;
      b.min_split = GBT_MIN_SPLIT;
      b.hess = hess; /* NULL for regression => mean leaves (Newton for MSE) */
      b.lambda = GBT_LAMBDA;
      for (int i = 0; i < n; i++)
        idx[i] = i;
      int root = bld_build(&b, idx, n, 0);
      if (root < 0) {
        sqlite3_free(b.nodes);
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
      for (int i = 0; i < n; i++)
        F[(size_t)i * nscore + s] +=
            fo->lr * reg_tree_value(b.nodes, b.n, &X[(size_t)i * nfeat]);

      int need = fo->tree_off[nt] + b.n;
      if (need > pool_cap) {
        pool_cap = need > pool_cap * 2 ? need : pool_cap * 2;
        TreeNode *g = sqlite3_realloc(fo->nodes, sizeof(TreeNode) * pool_cap);
        if (!g) {
          sqlite3_free(b.nodes);
          rc = SQLITE_NOMEM;
          *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
          goto done;
        }
        fo->nodes = g;
      }
      memcpy(&fo->nodes[fo->tree_off[nt]], b.nodes, sizeof(TreeNode) * b.n);
      fo->tree_off[nt + 1] = need;
      nt++;
      sqlite3_free(b.nodes);
    }
  }
  fo->n_trees = nt;

done:
  sqlite3_free(F);
  sqlite3_free(p);
  sqlite3_free(idx);
  sqlite3_free(grad);
  sqlite3_free(hess);
  if (rc != SQLITE_OK)
    forest_free(fo);
  return rc;
}

/* ---- the distill() operation ---- */

typedef struct {
  char *target, *task, *student_id, *teacher, *student_kind;
  char *proba;   /* JSON array of soft-target probability column names */
  char *classes; /* JSON array of class labels, same order as `proba` */
  int receipt;
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
static int intern_label(char ***labels, int *nclass, int *cap, const char *s,
                        int *rc) {
  for (int k = 0; k < *nclass; k++)
    if (strcmp((*labels)[k], s) == 0)
      return k;
  if (*nclass == *cap) {
    int nc = *cap ? *cap * 2 : 8;
    char **g = sqlite3_realloc(*labels, sizeof(char *) * nc);
    if (!g) {
      *rc = SQLITE_NOMEM;
      return -1;
    }
    *labels = g;
    *cap = nc;
  }
  (*labels)[*nclass] = sqlite3_mprintf("%s", s);
  if (!(*labels)[*nclass]) {
    *rc = SQLITE_NOMEM;
    return -1;
  }
  return (*nclass)++;
}

static int dist_opt_cb(void *ctx, const char *key, sqlite3_value *value,
                       char **errmsg) {
  DistOpts *o = ctx;
  if (strcmp(key, "receipt") == 0) {
    if (sqlite3_value_type(value) != SQLITE_INTEGER) {
      *errmsg = sqlite3_mprintf("%s: wrong type for option 'receipt'",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    o->receipt = sqlite3_value_int(value) != 0;
    return 0;
  }
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
    "target",       "task",    "student_id", "teacher", "student_kind",
    "proba",        "classes", "receipt",    NULL};

/* Parse a JSON array of strings into a heap array of sqlite3_mprintf'd
 * strings. Returns SQLITE_OK and sets out and n (caller frees each element and
 * the array). On a non-array or parse error, returns SQLITE_ERROR with errmsg
 * set. */
static int parse_str_array(sqlite3 *db, const char *json, char ***out, int *n,
                           char **errmsg) {
  *out = NULL;
  *n = 0;
  int cap = 0;
  sqlite3_stmt *st = NULL;
  if (sqlite3_prepare_v2(db, "SELECT value FROM json_each(?) WHERE"
                             " json_type(?) = 'array'",
                         -1, &st, NULL) != SQLITE_OK) {
    *errmsg = sqlite3_mprintf("%s: cannot parse option array", PREDICT_ERR_OPTIONS);
    return SQLITE_ERROR;
  }
  sqlite3_bind_text(st, 1, json, -1, SQLITE_STATIC);
  sqlite3_bind_text(st, 2, json, -1, SQLITE_STATIC);
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
    (*out)[*n] = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(st, 0));
    if (!(*out)[*n]) {
      rc = SQLITE_NOMEM;
      break;
    }
    (*n)++;
  }
  sqlite3_finalize(st);
  return rc;
}

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
  char receipt_id[PREDICT_ULID_BUFSIZE];
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

/* The training pipeline. Fills *res on success. */
static int distill_train(sqlite3 *db, const char *tq, DistOpts *o,
                         DistResult *res, char **errmsg) {
  int rc = SQLITE_OK;
  int classify = !o->task || strcmp(o->task, "classify") == 0;
  memset(res, 0, sizeof(*res));

  /* ---- 1. introspect train_query columns ---- */
  sqlite3_stmt *iq = NULL;
  if (sqlite3_prepare_v2(db, tq, -1, &iq, NULL) != SQLITE_OK || !iq) {
    *errmsg = sqlite3_mprintf("%s: train_query does not parse: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
    return SQLITE_ERROR;
  }
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
  void *blob = NULL;
  int blob_len = 0;
  int *idx = NULL;
  char *read_sql = NULL, *apply_sql = NULL, *teacher_sql = NULL, *params = NULL;
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
    if (o->student_kind && strcmp(o->student_kind, "gbt") != 0) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: soft distillation requires"
                                " student_kind 'gbt'",
                                PREDICT_ERR_OPTIONS);
      goto done;
    }
    if (!o->classes) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: 'proba' requires 'classes'",
                                PREDICT_ERR_OPTIONS);
      goto done;
    }
    if (parse_str_array(db, o->proba, &proba_names, &nproba, errmsg) !=
            SQLITE_OK ||
        parse_str_array(db, o->classes, &soft_labels, &nsoft, errmsg) !=
            SQLITE_OK) {
      rc = SQLITE_ERROR;
      goto done;
    }
    if (nproba < 2 || nproba != nsoft) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: 'proba' and 'classes' must be equal-length arrays of >= 2",
          PREDICT_ERR_OPTIONS);
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
  while (sqlite3_step(rq) == SQLITE_ROW) {
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
  sqlite3_finalize(rq);
  rq = NULL;
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
        int cl = intern_label(&labels, &nclass, &nlab_cap, y_true_c[i], &rc);
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
        "SELECT prediction FROM predict(%Q, %Q, json_object('target',%Q,'task',"
        "%Q,'model',%Q,'receipt',0))",
        tq, apply_sql, o->target, task, teacher);
    if (!apply_sql || !teacher_sql) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    if (sqlite3_prepare_v2(db, teacher_sql, -1, &tqs, NULL) != SQLITE_OK) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: teacher '%s' failed: %s",
                                PREDICT_ERR_MODEL_NOT_FOUND, teacher,
                                sqlite3_errmsg(db));
      goto done;
    }
    int ti = 0;
    int tstep;
    while ((tstep = sqlite3_step(tqs)) == SQLITE_ROW && ti < n) {
      if (classify) {
        const char *pred = (const char *)sqlite3_column_text(tqs, 0);
        int cl =
            intern_label(&labels, &nclass, &nlab_cap, pred ? pred : "", &rc);
        if (cl < 0)
          goto done;
        y_teach[ti] = cl;
      } else {
        y_teach_r[ti] = (f32)sqlite3_column_double(tqs, 0);
      }
      ti++;
    }
    int tdone = tstep == SQLITE_DONE || ti == n;
    char *terr = tdone ? NULL : sqlite3_mprintf("%s", sqlite3_errmsg(db));
    sqlite3_finalize(tqs);
    tqs = NULL;
    if (!tdone || ti != n) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: teacher produced %d labels for %d rows (%s)",
          PREDICT_ERR_RESOURCE, ti, n, terr ? terr : "short read");
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

  /* ---- 4. fit the student on the fit split (teacher targets) ---- */
  int n_hold = n / 5;
  if (n_hold < 1)
    n_hold = 1;
  int n_fit = n - n_hold;
  int is_gbt = soft || (o->student_kind && strcmp(o->student_kind, "gbt") == 0);
  int correct = 0;
  f64 sse = 0;

  if (is_gbt) {
    rc = train_gbt(X, n_fit, nfeat, classify ? 0 : 1, nclass, y_teach,
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
      if (forest_predict_row(&forest, &X[(size_t)i * forest.nfeat], scbuf,
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
    rc = forest_serialize(&forest, &blob, &blob_len);
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
    int root = bld_build(&b, idx, n_fit, 0);
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
      int leaf = tree_walk(&tree, &X[(size_t)i * tree.nfeat]);
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
    rc = tree_serialize(&tree, &blob, &blob_len);
    if (rc != SQLITE_OK) {
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto done;
    }
  }
  {
    predict0_hasher h;
    predict0_hash_init(&h);
    sha256_update(&h.sha, (const u8 *)blob, (usize)blob_len);
    predict0_hash_hex(&h, res->content_hash);
  }
  {
    sqlite3_stmt *ins = NULL;
    if (sqlite3_prepare_v2(
            db,
            "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
            " io_spec, content_hash, license) VALUES (?1,'student','tree',?2,"
            " NULL,?3,'unspecified')",
            -1, &ins, NULL) != SQLITE_OK) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: cannot prepare student insert",
                                PREDICT_ERR_RESOURCE);
      goto done;
    }
    sqlite3_bind_text(ins, 1, o->student_id, -1, SQLITE_STATIC);
    sqlite3_bind_blob(ins, 2, blob, blob_len, SQLITE_STATIC);
    sqlite3_bind_text(ins, 3, res->content_hash, -1, SQLITE_STATIC);
    int irc = sqlite3_step(ins);
    sqlite3_finalize(ins);
    if (irc == SQLITE_CONSTRAINT) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: student '%s' already exists",
                                PREDICT_ERR_STUDENT_EXISTS, o->student_id);
      goto done;
    }
    if (irc != SQLITE_DONE) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: student insert failed: %s",
                                PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
      goto done;
    }
  }

  res->model_id = sqlite3_mprintf("%s", o->student_id);
  res->train_rows = n;

  /* ---- 7. receipt (anchors the training data state) ---- */
  if (o->receipt) {
    predict0_hasher rh;
    predict0_hash_init(&rh);
    predict0_hash_text(&rh, res->model_id);
    predict0_hash_text(&rh, res->content_hash);
    predict0_hash_int(&rh, res->train_rows);
    predict0_hash_real(&rh, res->metric);
    predict0_hash_row_end(&rh);
    char result_hash[PREDICT_HEX_BUFSIZE];
    predict0_hash_hex(&rh, result_hash);

    sqlite3_stmt *pj = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_object('target',?1,'task',?2,'teacher',?3,"
            "'student_kind',?4,'proba',json(?5),'classes',json(?6),"
            "'receipt',1)",
            -1, &pj, NULL) == SQLITE_OK) {
      sqlite3_bind_text(pj, 1, o->target, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 2, task, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 3, teacher, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 4,
                        o->student_kind ? o->student_kind
                                        : (soft ? "gbt" : "tree"),
                        -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 5, o->proba, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 6, o->classes, -1, SQLITE_STATIC);
      if (sqlite3_step(pj) == SQLITE_ROW)
        params = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 0));
      sqlite3_finalize(pj);
    }
    char *rerr = NULL;
    rc = predict0_emit_receipt(db, "distill", res->model_id,
                               params ? params : "{}", tq, result_hash,
                               res->receipt_id, &rerr);
    if (rc != SQLITE_OK) {
      *errmsg = rerr ? rerr
                     : sqlite3_mprintf("%s: receipt write failed",
                                       PREDICT_ERR_RESOURCE);
      goto done;
    }
  }
  rc = SQLITE_OK;

done:
  sqlite3_free(feat_list);
  sqlite3_free(read_sql);
  sqlite3_free(apply_sql);
  sqlite3_free(teacher_sql);
  sqlite3_free(params);
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
  tree_free(&tree);
  forest_free(&forest);
  return rc;
}

/* ---- distill vtab ---- */

#define DL_MODEL 0
#define DL_HASH 1
#define DL_ROWS 2
#define DL_METRIC 3
#define DL_RECEIPT 4
#define DL_TRAINQ 5
#define DL_OPTIONS 6

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
          " holdout_metric REAL, receipt_id TEXT, train_query HIDDEN,"
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
        "%s: distill(train_query, options) requires a train_query",
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
  o.receipt = 1;
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
    DL_FAIL("%s: distill requires a target option", PREDICT_ERR_TARGET);
  if (!o.student_id)
    DL_FAIL("%s: distill requires a student_id option", PREDICT_ERR_OPTIONS);
  if (o.task && strcmp(o.task, "classify") != 0 &&
      strcmp(o.task, "regress") != 0)
    DL_FAIL("%s: task must be classify|regress: %s", PREDICT_ERR_TASK, o.task);
  if (o.student_kind && strcmp(o.student_kind, "tree") != 0 &&
      strcmp(o.student_kind, "gbt") != 0)
    DL_FAIL("%s: student_kind '%s' is not available; use 'tree' or 'gbt'",
            PREDICT_ERR_OPTIONS, o.student_kind);

  char *ensure_err = NULL;
  if (predict0_receipts_ensure(db, &ensure_err) != SQLITE_OK) {
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
  case DL_RECEIPT:
    if (c->res.receipt_id[0])
      sqlite3_result_text(ctx, c->res.receipt_id, -1, SQLITE_TRANSIENT);
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

int predict0_distill_init(sqlite3 *db) {
  return sqlite3_create_module(db, "distill", &distillModule, NULL);
}
