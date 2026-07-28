/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* distill_predict() -- RFC 0005 §4.2.6 -- and the native-student TRAINERS.
 *
 * The training half of distillation: the CART / gradient-boosted / MLP
 * trainers and the distill_predict() table-valued function. distill_predict() fits a student
 * on a training signal (the target column by default, or a named teacher
 * model's predictions), evaluates it on a held-out fraction, serializes it via
 * predict-student.c, and registers it in _predict_models. The student then
 * serves through predict0_tree_run with no onnxruntime. The blob format,
 * deserializers, and inference runtime all live in predict-student.c. */
#include "predict-internal.h"
#include "predict-student.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define TREE_MAX_DEPTH 8
#define TREE_MIN_SPLIT 5
#define DISTILL_MIN_ROWS 8

#define GBT_ROUNDS 200
#define GBT_DEPTH 3
#define GBT_MIN_SPLIT 5
#define GBT_LR 0.1f     /* shrinkage: many small steps generalize better than few big ones */
#define GBT_LAMBDA 1.0f /* L2 leaf regularization (XGBoost reg_lambda default) */

/* ---- MLP student trainer (deterministic full-batch Adam) ----
 * A smooth learner for boundaries an axis-aligned tree ensemble cannot render.
 * Seeded init and no shuffle, so the blob and its predictions replay exactly.
 * The blob format and inference runtime live in predict-student.c. */
#define MLP_HIDDEN 48
#define MLP_EPOCHS 400
#define MLP_LR 0.02
#define MLP_L2 1e-4
#define MLP_BETA1 0.9
#define MLP_BETA2 0.999
/* FCST_RES_SCALE (the linear-skip hidden-path scale) is in predict-student.h,
 * shared with the serving forward. */

/* deterministic xorshift32 -> f32 in [-1, 1) */
static f32 mlp_rng(u32 *s) {
  u32 x = *s ? *s : 1;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return (f32)((f64)x / 2147483648.0 - 1.0);
}

/* One Adam step over a parameter array (with L2), then zero its gradient. */
static void mlp_adam(f32 *p, f64 *g, f64 *mm, f64 *vv, int sz, f64 lr,
                     f64 scale, f64 bc1, f64 bc2) {
  for (int i = 0; i < sz; i++) {
    f64 gg = g[i] * scale + MLP_L2 * p[i];
    mm[i] = MLP_BETA1 * mm[i] + (1 - MLP_BETA1) * gg;
    vv[i] = MLP_BETA2 * vv[i] + (1 - MLP_BETA2) * gg * gg;
    p[i] -= (f32)(lr * (mm[i] / bc1) / (sqrt(vv[i] / bc2) + 1e-8));
    g[i] = 0;
  }
}

/* Train an MLP with a configurable width and task.
 * task 0 (classify): softmax head over `nout` classes; the target is `soft`
 * (a row-major [n, nout] teacher-probability matrix) when non-NULL, else the
 * hard class yc. task 1 (regress): linear head, MSE against the [n, nout]
 * target matrix `soft` (yc unused) -- this is how the multi-output forecast
 * student is fit. `use_skip` adds a direct linear map Wskip*x to the output (a
 * DLinear/TiDE skip: the linear part carries seasonal-naive + trend, the hidden
 * path a scaled nonlinear correction); with nhid=0 the model is purely linear.
 * Deterministic full-batch Adam; the caller sets any feat_names/labels. */
static int train_mlp(const f32 *X, int n, int nfeat, int nout, const i32 *yc,
                     const f32 *soft, int task, int nhid, int epochs, f32 lr,
                     int use_skip, MLP *m, char **errmsg) {
  memset(m, 0, sizeof(*m));
  int nW1 = nhid * nfeat, nW2 = nout * nhid, nWs = use_skip ? nout * nfeat : 0;
  f64 rs = use_skip ? FCST_RES_SCALE : 1.0; /* hidden-path scale (see #define) */
  m->task = task;
  m->nfeat = nfeat;
  m->nhid = nhid;
  m->nout = nout;
  m->nclass = task == 0 ? nout : 0;

  int rc = SQLITE_OK;
  m->mean = sqlite3_malloc(sizeof(f32) * nfeat);
  m->sd = sqlite3_malloc(sizeof(f32) * nfeat);
  m->b2 = sqlite3_malloc(sizeof(f32) * nout);
  /* hidden-layer blocks exist only when nhid>0; skip block only when use_skip */
  if (nhid > 0) {
    m->W1 = sqlite3_malloc(sizeof(f32) * nW1);
    m->b1 = sqlite3_malloc(sizeof(f32) * nhid);
    m->W2 = sqlite3_malloc(sizeof(f32) * nW2);
  }
  if (use_skip)
    m->Wskip = sqlite3_malloc(sizeof(f32) * nWs);
  f64 *mW1 = nhid ? sqlite3_malloc(sizeof(f64) * nW1) : NULL,
      *vW1 = nhid ? sqlite3_malloc(sizeof(f64) * nW1) : NULL,
      *gW1 = nhid ? sqlite3_malloc(sizeof(f64) * nW1) : NULL;
  f64 *mW2 = nhid ? sqlite3_malloc(sizeof(f64) * nW2) : NULL,
      *vW2 = nhid ? sqlite3_malloc(sizeof(f64) * nW2) : NULL,
      *gW2 = nhid ? sqlite3_malloc(sizeof(f64) * nW2) : NULL;
  f64 *mb1 = nhid ? sqlite3_malloc(sizeof(f64) * nhid) : NULL,
      *vb1 = nhid ? sqlite3_malloc(sizeof(f64) * nhid) : NULL,
      *gb1 = nhid ? sqlite3_malloc(sizeof(f64) * nhid) : NULL;
  f64 *mWs = use_skip ? sqlite3_malloc(sizeof(f64) * nWs) : NULL,
      *vWs = use_skip ? sqlite3_malloc(sizeof(f64) * nWs) : NULL,
      *gWs = use_skip ? sqlite3_malloc(sizeof(f64) * nWs) : NULL;
  f64 *mb2 = sqlite3_malloc(sizeof(f64) * nout),
      *vb2 = sqlite3_malloc(sizeof(f64) * nout),
      *gb2 = sqlite3_malloc(sizeof(f64) * nout);
  f32 *hid = nhid ? sqlite3_malloc(sizeof(f32) * nhid) : NULL,
      *out = sqlite3_malloc(sizeof(f32) * nout),
      *xs = sqlite3_malloc(sizeof(f32) * nfeat);
  f64 *dout = sqlite3_malloc(sizeof(f64) * nout);
  int hid_ok = nhid == 0 || (m->W1 && m->b1 && m->W2 && mW1 && vW1 && gW1 &&
                             mW2 && vW2 && gW2 && mb1 && vb1 && gb1 && hid);
  int skip_ok = !use_skip || (m->Wskip && mWs && vWs && gWs);
  if (!m->mean || !m->sd || !m->b2 || !mb2 || !vb2 || !gb2 || !out || !xs ||
      !dout || !hid_ok || !skip_ok) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }
  for (int i = 0; i < nW1; i++)
    mW1[i] = vW1[i] = gW1[i] = 0;
  for (int i = 0; i < nW2; i++)
    mW2[i] = vW2[i] = gW2[i] = 0;
  for (int i = 0; i < nhid; i++)
    mb1[i] = vb1[i] = gb1[i] = 0;
  for (int i = 0; i < nWs; i++)
    mWs[i] = vWs[i] = gWs[i] = 0;
  for (int i = 0; i < nout; i++)
    mb2[i] = vb2[i] = gb2[i] = 0;

  for (int i = 0; i < nfeat; i++) { /* standardize features */
    f64 mu = 0;
    for (int r = 0; r < n; r++)
      mu += X[(size_t)r * nfeat + i];
    mu /= n;
    f64 var = 0;
    for (int r = 0; r < n; r++) {
      f64 d = X[(size_t)r * nfeat + i] - mu;
      var += d * d;
    }
    var /= n;
    f64 s = sqrt(var);
    m->mean[i] = (f32)mu;
    m->sd[i] = (f32)(s > 1e-6 ? s : 1e-6);
  }

  u32 seed = 0x243f6a88u; /* Xavier-uniform init, deterministic */
  f32 a1 = (f32)sqrt(6.0 / (nfeat + (nhid ? nhid : 1)));
  for (int i = 0; i < nW1; i++)
    m->W1[i] = a1 * mlp_rng(&seed);
  for (int j = 0; j < nhid; j++)
    m->b1[j] = 0;
  f32 a2 = (f32)sqrt(6.0 / ((nhid ? nhid : 1) + nout));
  for (int i = 0; i < nW2; i++)
    m->W2[i] = a2 * mlp_rng(&seed);
  f32 as = (f32)sqrt(6.0 / (nfeat + nout));
  for (int i = 0; i < nWs; i++)
    m->Wskip[i] = as * mlp_rng(&seed);
  for (int k = 0; k < nout; k++)
    m->b2[k] = 0;

  f64 b1p = 1, b2p = 1;
  for (int ep = 0; ep < epochs; ep++) {
    b1p *= MLP_BETA1;
    b2p *= MLP_BETA2;
    for (int row = 0; row < n; row++) {
      const f32 *x = &X[(size_t)row * nfeat];
      for (int i = 0; i < nfeat; i++)
        xs[i] = (x[i] - m->mean[i]) / m->sd[i];
      for (int j = 0; j < nhid; j++) {
        f64 s = m->b1[j];
        for (int i = 0; i < nfeat; i++)
          s += (f64)m->W1[j * nfeat + i] * xs[i];
        hid[j] = (f32)tanh(s);
      }
      for (int k = 0; k < nout; k++) {
        f64 s = m->b2[k];
        for (int j = 0; j < nhid; j++)
          s += rs * (f64)m->W2[k * nhid + j] * hid[j];
        if (use_skip)
          for (int i = 0; i < nfeat; i++)
            s += (f64)m->Wskip[k * nfeat + i] * xs[i];
        out[k] = (f32)s;
      }
      if (task == 0) { /* softmax cross-entropy: dL/dlogit = softmax - target */
        f64 mx = out[0];
        for (int k = 1; k < nout; k++)
          if (out[k] > mx)
            mx = out[k];
        f64 sm = 0;
        for (int k = 0; k < nout; k++) {
          dout[k] = exp((f64)out[k] - mx);
          sm += dout[k];
        }
        for (int k = 0; k < nout; k++) {
          f64 tgt = soft ? soft[(size_t)row * nout + k] : (yc[row] == k ? 1.0 : 0.0);
          dout[k] = dout[k] / sm - tgt;
        }
      } else { /* regression MSE on the linear head: dL/dout = out - target */
        for (int k = 0; k < nout; k++)
          dout[k] = (f64)out[k] - soft[(size_t)row * nout + k];
      }
      for (int k = 0; k < nout; k++) {
        gb2[k] += dout[k];
        for (int j = 0; j < nhid; j++)
          gW2[k * nhid + j] += dout[k] * rs * hid[j];
        if (use_skip)
          for (int i = 0; i < nfeat; i++)
            gWs[k * nfeat + i] += dout[k] * xs[i];
      }
      for (int j = 0; j < nhid; j++) {
        f64 dh = 0;
        for (int k = 0; k < nout; k++)
          dh += dout[k] * rs * m->W2[k * nhid + j];
        dh *= 1.0 - (f64)hid[j] * hid[j]; /* tanh' */
        gb1[j] += dh;
        for (int i = 0; i < nfeat; i++)
          gW1[j * nfeat + i] += dh * xs[i];
      }
    }
    f64 scale = 1.0 / n, bc1 = 1 - b1p, bc2 = 1 - b2p;
    if (nhid > 0) {
      mlp_adam(m->W1, gW1, mW1, vW1, nW1, lr, scale, bc1, bc2);
      mlp_adam(m->b1, gb1, mb1, vb1, nhid, lr, scale, bc1, bc2);
      mlp_adam(m->W2, gW2, mW2, vW2, nW2, lr, scale, bc1, bc2);
    }
    if (use_skip)
      mlp_adam(m->Wskip, gWs, mWs, vWs, nWs, lr, scale, bc1, bc2);
    mlp_adam(m->b2, gb2, mb2, vb2, nout, lr, scale, bc1, bc2);
  }

done:
  sqlite3_free(mWs);
  sqlite3_free(vWs);
  sqlite3_free(gWs);
  sqlite3_free(mW1);
  sqlite3_free(vW1);
  sqlite3_free(gW1);
  sqlite3_free(mW2);
  sqlite3_free(vW2);
  sqlite3_free(gW2);
  sqlite3_free(mb1);
  sqlite3_free(vb1);
  sqlite3_free(gb1);
  sqlite3_free(mb2);
  sqlite3_free(vb2);
  sqlite3_free(gb2);
  sqlite3_free(hid);
  sqlite3_free(out);
  sqlite3_free(xs);
  sqlite3_free(dout);
  if (rc != SQLITE_OK)
    predict0_mlp_free(m);
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
            fo->lr * predict0_reg_tree_value(b.nodes, b.n, &X[(size_t)i * nfeat]);

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
    predict0_forest_free(fo);
  return rc;
}

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
static int register_student(sqlite3 *db, const char *student_id,
                            const void *blob, int blob_len,
                            char hash_out[PREDICT_HEX_BUFSIZE],
                            char **errmsg) {
  predict0_hasher h;
  predict0_hash_init(&h);
  sha256_update(&h.sha, (const u8 *)blob, (usize)blob_len);
  predict0_hash_hex(&h, hash_out);

  sqlite3_stmt *ins = NULL;
  if (sqlite3_prepare_v2(
          db,
          "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
          " io_spec, content_hash, license) VALUES (?1,'student','tree',?2,"
          " NULL,?3,'unspecified')",
          -1, &ins, NULL) != SQLITE_OK) {
    *errmsg = sqlite3_mprintf("%s: cannot prepare student insert",
                              PREDICT_ERR_RESOURCE);
    return SQLITE_ERROR;
  }
  sqlite3_bind_text(ins, 1, student_id, -1, SQLITE_STATIC);
  sqlite3_bind_blob(ins, 2, blob, blob_len, SQLITE_STATIC);
  sqlite3_bind_text(ins, 3, hash_out, -1, SQLITE_STATIC);
  int irc = sqlite3_step(ins);
  sqlite3_finalize(ins);
  if (irc == SQLITE_CONSTRAINT) {
    *errmsg = sqlite3_mprintf("%s: student '%s' already exists",
                              PREDICT_ERR_STUDENT_EXISTS, student_id);
    return SQLITE_ERROR;
  }
  if (irc != SQLITE_DONE) {
    *errmsg = sqlite3_mprintf("%s: student insert failed: %s",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

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
    if (predict0_json_str_array(db, o->proba, NULL, &proba_names,
                                &nproba, errmsg) !=
            SQLITE_OK ||
        predict0_json_str_array(db, o->classes, NULL, &soft_labels,
                                &nsoft, errmsg) !=
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
    rc = train_mlp(X, n_fit, nfeat, nclass, y_teach, soft ? soft_P : NULL,
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
  rc = register_student(db, o->student_id, blob, blob_len, res->content_hash,
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

/* ---- distill_forecast: train a native forecast student (PSFCST, §4.1.3) ----
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
  rc = train_mlp(X, n_fit, L, nout, NULL, Y, 1 /* regress */, nhid, epochs, lr,
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
  rc = register_student(db, student_id, blob, blob_len, res->content_hash,
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
  while (sqlite3_step(q) == SQLITE_ROW) {
    for (int c = 0; c < ncol; c++)
      wrow[c] = sqlite3_column_double(q, c);
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
      f32 *nX = sqlite3_realloc(X, (int)(sizeof(f32) * (size_t)nc * L));
      f32 *nY = sqlite3_realloc(Y, (int)(sizeof(f32) * (size_t)nc * nout));
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
      f32 *nX = sqlite3_realloc(*X, (int)(sizeof(f32) * (size_t)nc * L));
      f32 *nY = sqlite3_realloc(*Y, (int)(sizeof(f32) * (size_t)nc * nout));
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


  while (sqlite3_step(q) == SQLITE_ROW && n < FCST_TEACHER_MAX_WIN) {
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
    if (sn == scap) {
      int nc = scap ? scap * 2 : 512;
      f64 *ns = sqlite3_realloc(series, (int)(sizeof(f64) * nc));
      if (!ns) {
        rc = SQLITE_NOMEM;
        goto done;
      }
      series = ns;
      scap = nc;
    }
    series[sn++] = sqlite3_column_double(q, qcol - 1);
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
      while (sqlite3_step(qs) == SQLITE_ROW) {
        if (cnt == cap) {
          int nc = cap ? cap * 2 : 8;
          f32 *nl = sqlite3_realloc(levels, (int)(sizeof(f32) * nc));
          if (!nl) {
            sqlite3_finalize(qs);
            FD_FAIL("%s: out of memory", PREDICT_ERR_RESOURCE);
          }
          levels = nl;
          cap = nc;
        }
        levels[cnt++] = (f32)sqlite3_column_double(qs, 0);
      }
      sqlite3_finalize(qs);
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
