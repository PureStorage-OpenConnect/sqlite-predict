/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* Native student format + serving runtime (RFC 0005 §4.1.6). Reads and
 * executes the inline student blobs distill_predict() produces: a single tree
 * (PSTREE01), a gradient-boosted forest (PSGBT01), or a one-hidden-layer MLP
 * (PSMLP01), told apart by the 8-byte magic. Every field is bounds-checked on
 * read, because the registry is writable by any SQL caller (RFC §6.2). This
 * file is compiled into the core build and links no training code. */
#include "predict-internal.h"
#include "predict-student.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

/* ---- decision tree (PSTREE01) ---- */

void tree_free(Tree *t) {
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

int tree_serialize(const Tree *t, void **blob_out, int *len_out) {
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
int tree_walk(const Tree *t, const f32 *x) {
  int node = 0;
  for (int hops = 0; hops <= t->n_nodes; hops++) {
    const TreeNode *n = &t->nodes[node];
    if (n->feature < 0)
      return node; /* leaf */
    node = x[n->feature] < n->threshold ? n->left : n->right;
  }
  return -1;
}

/* ---- gradient-boosted forest (PSGBT01) ---- */

static const char GBT_MAGIC[8] = {'P', 'S', 'G', 'B', 'T', '0', '1', '\0'};

void forest_free(Forest *f) {
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
f32 reg_tree_value(const TreeNode *nd, int guard, const f32 *x) {
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
int forest_predict_row(const Forest *f, const f32 *x, f64 *scbuf, char **pred,
                       f64 *conf, int *has_conf) {
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

int forest_serialize(const Forest *f, void **blob_out, int *len_out) {
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

/* ---- MLP (PSMLP01): one hidden layer (tanh), softmax head ---- */

static const char MLP_MAGIC[8] = {'P', 'S', 'M', 'L', 'P', '0', '1', '\0'};

void mlp_free(MLP *m) {
  if (m->feat_names)
    for (int i = 0; i < m->nfeat; i++)
      sqlite3_free(m->feat_names[i]);
  sqlite3_free(m->feat_names);
  if (m->labels)
    for (int i = 0; i < m->nclass; i++)
      sqlite3_free(m->labels[i]);
  sqlite3_free(m->labels);
  sqlite3_free(m->mean);
  sqlite3_free(m->sd);
  sqlite3_free(m->W1);
  sqlite3_free(m->b1);
  sqlite3_free(m->W2);
  sqlite3_free(m->b2);
  memset(m, 0, sizeof(*m));
}

/* forward on RAW features x; writes hidden[nhid] and out[nout] logits. */
void mlp_forward(const MLP *m, const f32 *x, f32 *hid, f32 *out) {
  for (int j = 0; j < m->nhid; j++) {
    f64 s = m->b1[j];
    for (int i = 0; i < m->nfeat; i++)
      s += (f64)m->W1[j * m->nfeat + i] * ((x[i] - m->mean[i]) / m->sd[i]);
    hid[j] = (f32)tanh(s);
  }
  for (int k = 0; k < m->nout; k++) {
    f64 s = m->b2[k];
    for (int j = 0; j < m->nhid; j++)
      s += (f64)m->W2[k * m->nhid + j] * hid[j];
    out[k] = (f32)s;
  }
}

int mlp_predict_row(const MLP *m, const f32 *x, f32 *hid, f32 *out, char **pred,
                    f64 *conf, int *has_conf) {
  mlp_forward(m, x, hid, out);
  if (m->task == 0) {
    f64 mx = out[0];
    int best = 0;
    for (int k = 1; k < m->nout; k++)
      if (out[k] > mx) {
        mx = out[k];
        best = k;
      }
    f64 sum = 0;
    for (int k = 0; k < m->nout; k++)
      sum += exp((f64)out[k] - mx);
    *pred = sqlite3_mprintf("%s", m->labels[best]);
    *conf = sum > 0 ? 1.0 / sum : 0; /* softmax prob of the argmax class */
    *has_conf = 1;
  } else {
    *pred = sqlite3_mprintf("%.17g", (f64)out[0]);
    *has_conf = 0;
  }
  return *pred ? SQLITE_OK : SQLITE_NOMEM;
}

int mlp_serialize(const MLP *m, void **blob_out, int *len_out) {
  size_t sz = sizeof(MLP_MAGIC) + 4 * 5;
  sz += 4 * (size_t)m->nfeat * 2;
  sz += 4 * ((size_t)m->nhid * m->nfeat + m->nhid);
  sz += 4 * ((size_t)m->nout * m->nhid + m->nout);
  for (int i = 0; i < m->nfeat; i++)
    sz += 4 + strlen(m->feat_names[i]);
  for (int i = 0; i < m->nclass; i++)
    sz += 4 + strlen(m->labels[i]);
  u8 *buf = sqlite3_malloc((int)sz);
  if (!buf)
    return SQLITE_NOMEM;
  u8 *p = buf;
  memcpy(p, MLP_MAGIC, sizeof(MLP_MAGIC));
  p += sizeof(MLP_MAGIC);
  put_u32(&p, (u32)m->task);
  put_u32(&p, (u32)m->nfeat);
  put_u32(&p, (u32)m->nhid);
  put_u32(&p, (u32)m->nout);
  put_u32(&p, (u32)m->nclass);
  for (int i = 0; i < m->nfeat; i++)
    put_f32(&p, m->mean[i]);
  for (int i = 0; i < m->nfeat; i++)
    put_f32(&p, m->sd[i]);
  for (int i = 0; i < m->nhid * m->nfeat; i++)
    put_f32(&p, m->W1[i]);
  for (int i = 0; i < m->nhid; i++)
    put_f32(&p, m->b1[i]);
  for (int i = 0; i < m->nout * m->nhid; i++)
    put_f32(&p, m->W2[i]);
  for (int i = 0; i < m->nout; i++)
    put_f32(&p, m->b2[i]);
  for (int i = 0; i < m->nfeat; i++)
    put_str(&p, m->feat_names[i]);
  for (int i = 0; i < m->nclass; i++)
    put_str(&p, m->labels[i]);
  *blob_out = buf;
  *len_out = (int)sz;
  return SQLITE_OK;
}

static int mlp_deserialize(const void *blob, int len, MLP *m, char **errmsg) {
  memset(m, 0, sizeof(*m));
  Reader r = {.p = blob, .end = (const u8 *)blob + len, .err = 0};
  if (!blob || len < (int)sizeof(MLP_MAGIC) ||
      memcmp(blob, MLP_MAGIC, sizeof(MLP_MAGIC)) != 0) {
    *errmsg = sqlite3_mprintf("%s: not an mlp student blob", PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  r.p += sizeof(MLP_MAGIC);
  m->task = (int)rd_u32(&r);
  m->nfeat = (int)rd_u32(&r);
  m->nhid = (int)rd_u32(&r);
  m->nout = (int)rd_u32(&r);
  m->nclass = (int)rd_u32(&r);
  if (r.err || m->task < 0 || m->task > 1 || m->nfeat <= 0 ||
      m->nfeat > TREE_MAX_FEAT || m->nhid <= 0 || m->nhid > 2048 ||
      m->nout <= 0 || m->nout > 2048 || m->nclass < 0 || m->nclass > 2048 ||
      (m->task == 0 && m->nout != m->nclass) || (m->task == 1 && m->nout != 1))
    goto bad;
  m->mean = sqlite3_malloc(sizeof(f32) * m->nfeat);
  m->sd = sqlite3_malloc(sizeof(f32) * m->nfeat);
  m->W1 = sqlite3_malloc(sizeof(f32) * (size_t)m->nhid * m->nfeat);
  m->b1 = sqlite3_malloc(sizeof(f32) * m->nhid);
  m->W2 = sqlite3_malloc(sizeof(f32) * (size_t)m->nout * m->nhid);
  m->b2 = sqlite3_malloc(sizeof(f32) * m->nout);
  if (!m->mean || !m->sd || !m->W1 || !m->b1 || !m->W2 || !m->b2)
    goto oom;
  for (int i = 0; i < m->nfeat; i++)
    m->mean[i] = rd_f32(&r);
  for (int i = 0; i < m->nfeat; i++)
    m->sd[i] = rd_f32(&r);
  for (int i = 0; i < m->nhid * m->nfeat; i++)
    m->W1[i] = rd_f32(&r);
  for (int i = 0; i < m->nhid; i++)
    m->b1[i] = rd_f32(&r);
  for (int i = 0; i < m->nout * m->nhid; i++)
    m->W2[i] = rd_f32(&r);
  for (int i = 0; i < m->nout; i++)
    m->b2[i] = rd_f32(&r);
  if (r.err)
    goto bad;
  m->feat_names = sqlite3_malloc(sizeof(char *) * m->nfeat);
  if (!m->feat_names)
    goto oom;
  memset(m->feat_names, 0, sizeof(char *) * m->nfeat);
  for (int i = 0; i < m->nfeat; i++) {
    m->feat_names[i] = rd_str(&r);
    if (r.err)
      goto bad;
  }
  if (m->nclass > 0) {
    m->labels = sqlite3_malloc(sizeof(char *) * m->nclass);
    if (!m->labels)
      goto oom;
    memset(m->labels, 0, sizeof(char *) * m->nclass);
    for (int i = 0; i < m->nclass; i++) {
      m->labels[i] = rd_str(&r);
      if (r.err)
        goto bad;
    }
  }
  return SQLITE_OK;
bad:
  mlp_free(m);
  *errmsg = sqlite3_mprintf("%s: malformed mlp student blob", PREDICT_ERR_SCHEMA);
  return SQLITE_ERROR;
oom:
  mlp_free(m);
  *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  return SQLITE_NOMEM;
}

/* ---- Forecast student (PSFCST01): multi-output regression MLP ---- */

static const char FCST_MAGIC[8] = {'P', 'S', 'F', 'C', 'S', 'T', '0', '1'};

void fcst_student_free(ForecastStudent *fs) {
  sqlite3_free(fs->levels);
  fs->levels = NULL;
  fs->horizon = fs->nquant = 0;
  mlp_free(&fs->mlp); /* memsets the embedded MLP */
}

int fcst_serialize(const ForecastStudent *fs, void **blob_out, int *len_out) {
  const MLP *m = &fs->mlp;
  size_t sz = sizeof(FCST_MAGIC) + 4 * 4; /* nfeat, nhid, horizon, nquant */
  sz += 4 * (size_t)fs->nquant;           /* quantile levels */
  sz += 4 * (size_t)m->nfeat * 2;         /* mean, sd */
  sz += 4 * ((size_t)m->nhid * m->nfeat + m->nhid);
  sz += 4 * ((size_t)m->nout * m->nhid + m->nout);
  u8 *buf = sqlite3_malloc((int)sz);
  if (!buf)
    return SQLITE_NOMEM;
  u8 *p = buf;
  memcpy(p, FCST_MAGIC, sizeof(FCST_MAGIC));
  p += sizeof(FCST_MAGIC);
  put_u32(&p, (u32)m->nfeat);
  put_u32(&p, (u32)m->nhid);
  put_u32(&p, (u32)fs->horizon);
  put_u32(&p, (u32)fs->nquant);
  for (int i = 0; i < fs->nquant; i++)
    put_f32(&p, fs->levels[i]);
  for (int i = 0; i < m->nfeat; i++)
    put_f32(&p, m->mean[i]);
  for (int i = 0; i < m->nfeat; i++)
    put_f32(&p, m->sd[i]);
  for (int i = 0; i < m->nhid * m->nfeat; i++)
    put_f32(&p, m->W1[i]);
  for (int i = 0; i < m->nhid; i++)
    put_f32(&p, m->b1[i]);
  for (int i = 0; i < m->nout * m->nhid; i++)
    put_f32(&p, m->W2[i]);
  for (int i = 0; i < m->nout; i++)
    put_f32(&p, m->b2[i]);
  *blob_out = buf;
  *len_out = (int)sz;
  return SQLITE_OK;
}

int fcst_deserialize(const void *blob, int len, ForecastStudent *fs,
                     char **errmsg) {
  memset(fs, 0, sizeof(*fs));
  MLP *m = &fs->mlp;
  m->task = 1; /* regression head; no classes or feature names */
  Reader r = {.p = blob, .end = (const u8 *)blob + len, .err = 0};
  if (!blob || len < (int)sizeof(FCST_MAGIC) ||
      memcmp(blob, FCST_MAGIC, sizeof(FCST_MAGIC)) != 0) {
    *errmsg =
        sqlite3_mprintf("%s: not a forecast student blob", PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  r.p += sizeof(FCST_MAGIC);
  m->nfeat = (int)rd_u32(&r);
  m->nhid = (int)rd_u32(&r);
  fs->horizon = (int)rd_u32(&r);
  fs->nquant = (int)rd_u32(&r);
  if (r.err || m->nfeat <= 0 || m->nfeat > FCST_MAX_CONTEXT || m->nhid <= 0 ||
      m->nhid > 2048 || fs->horizon <= 0 || fs->horizon > 2048 ||
      fs->nquant <= 0 || fs->nquant > FCST_MAX_QUANT)
    goto bad;
  m->nout = fs->horizon * fs->nquant;
  if (m->nout > 8192) /* bound the largest allocation (W2 = nout*nhid) */
    goto bad;
  fs->levels = sqlite3_malloc(sizeof(f32) * fs->nquant);
  m->mean = sqlite3_malloc(sizeof(f32) * m->nfeat);
  m->sd = sqlite3_malloc(sizeof(f32) * m->nfeat);
  m->W1 = sqlite3_malloc(sizeof(f32) * (size_t)m->nhid * m->nfeat);
  m->b1 = sqlite3_malloc(sizeof(f32) * m->nhid);
  m->W2 = sqlite3_malloc(sizeof(f32) * (size_t)m->nout * m->nhid);
  m->b2 = sqlite3_malloc(sizeof(f32) * m->nout);
  if (!fs->levels || !m->mean || !m->sd || !m->W1 || !m->b1 || !m->W2 || !m->b2)
    goto oom;
  for (int i = 0; i < fs->nquant; i++)
    fs->levels[i] = rd_f32(&r);
  /* levels MUST be strictly ascending within (0,1) for the serving-time
   * quantile interpolation (the registry is caller-writable). */
  for (int i = 0; i < fs->nquant; i++)
    if (!(fs->levels[i] > 0.0f && fs->levels[i] < 1.0f) ||
        (i > 0 && fs->levels[i] <= fs->levels[i - 1]))
      goto bad;
  for (int i = 0; i < m->nfeat; i++)
    m->mean[i] = rd_f32(&r);
  for (int i = 0; i < m->nfeat; i++)
    m->sd[i] = rd_f32(&r);
  for (int i = 0; i < m->nhid * m->nfeat; i++)
    m->W1[i] = rd_f32(&r);
  for (int i = 0; i < m->nhid; i++)
    m->b1[i] = rd_f32(&r);
  for (int i = 0; i < m->nout * m->nhid; i++)
    m->W2[i] = rd_f32(&r);
  for (int i = 0; i < m->nout; i++)
    m->b2[i] = rd_f32(&r);
  if (r.err)
    goto bad;
  return SQLITE_OK;
bad:
  fcst_student_free(fs);
  *errmsg =
      sqlite3_mprintf("%s: malformed forecast student blob", PREDICT_ERR_SCHEMA);
  return SQLITE_ERROR;
oom:
  fcst_student_free(fs);
  *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  return SQLITE_NOMEM;
}

/* ---- the serving runtime: execute a student over apply rows ---- */

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

  /* one runtime for the native students: a single CART (PSTREE), a
   * gradient-boosted forest (PSGBT), or a neural net (PSMLP), told apart by
   * the blob magic */
  int is_forest = model->weights_len >= (int)sizeof(GBT_MAGIC) &&
                  memcmp(model->weights, GBT_MAGIC, sizeof(GBT_MAGIC)) == 0;
  int is_mlp = model->weights_len >= (int)sizeof(MLP_MAGIC) &&
               memcmp(model->weights, MLP_MAGIC, sizeof(MLP_MAGIC)) == 0;
  Tree tree;
  Forest forest;
  MLP mlp;
  memset(&tree, 0, sizeof(tree));
  memset(&forest, 0, sizeof(forest));
  memset(&mlp, 0, sizeof(mlp));
  int rc = is_mlp ? mlp_deserialize(model->weights, model->weights_len, &mlp,
                                    errmsg)
           : is_forest ? forest_deserialize(model->weights, model->weights_len,
                                            &forest, errmsg)
                       : tree_deserialize(model->weights, model->weights_len,
                                          &tree, errmsg);
  if (rc != SQLITE_OK)
    return rc;
  int nfeat = is_mlp ? mlp.nfeat : is_forest ? forest.nfeat : tree.nfeat;
  char **feat_names =
      is_mlp ? mlp.feat_names : is_forest ? forest.feat_names : tree.feat_names;
  int classify = (is_mlp ? mlp.task : is_forest ? forest.task : tree.task) == 0;
  f64 *scbuf = NULL;
  f32 *mhid = NULL, *mout = NULL; /* mlp per-row scratch */
  if (is_forest && classify) {
    scbuf = sqlite3_malloc(sizeof(f64) * forest.n_score);
    if (!scbuf) {
      forest_free(&forest);
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      return SQLITE_NOMEM;
    }
  }
  if (is_mlp) {
    mhid = sqlite3_malloc(sizeof(f32) * mlp.nhid);
    mout = sqlite3_malloc(sizeof(f32) * mlp.nout);
    if (!mhid || !mout) {
      sqlite3_free(mhid);
      sqlite3_free(mout);
      mlp_free(&mlp);
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
    if (is_mlp) {
      rc = mlp_predict_row(&mlp, x, mhid, mout, &out->prediction,
                           &out->confidence, &out->has_conf);
      if (rc != SQLITE_OK) {
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
    } else if (is_forest) {
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
  sqlite3_free(mhid);
  sqlite3_free(mout);
  if (is_mlp)
    mlp_free(&mlp);
  else if (is_forest)
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
