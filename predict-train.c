/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* Native-student TRAINERS: the CART / gradient-boosted / MLP fitters, shared
 * by fit() (predict-tabular.c) and the distill recipes (predict-distill.c).
 * Blob format, deserializers, and inference live in predict-student.c. */
#include "predict-internal.h"
#include "predict-student.h"
#include "predict-train.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define TREE_MAX_DEPTH 8
#define TREE_MIN_SPLIT 5
#define GBT_ROUNDS 200
#define GBT_DEPTH 3
#define GBT_MIN_SPLIT 5
#define GBT_LR 0.1f     /* shrinkage: many small steps generalize better */
#define GBT_LAMBDA 1.0f /* L2 leaf regularization (XGBoost reg_lambda default) */
#define MLP_L2 1e-4
#define MLP_BETA1 0.9
#define MLP_BETA2 0.999

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

/* All n elements finite? A NULL array with n == 0 is vacuously finite (the
 * hidden/skip blocks are absent when nhid == 0 / !use_skip). */
static int arr_finite(const f32 *a, int n) {
  for (int i = 0; i < n; i++)
    if (!isfinite(a[i]))
      return 0;
  return 1;
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
int predict0_train_mlp(const f32 *X, int n, int nfeat, int nout, const i32 *yc,
                     const f32 *soft, int task, int nhid, int epochs, f32 lr,
                     int use_skip, MLP *m, char **errmsg) {
  memset(m, 0, sizeof(*m));
  /* Target contract: task 1 (regress) needs the soft matrix; task 0 (classify)
   * needs soft or the hard labels yc. Fail loudly at this cross-TU entry rather
   * than dereference a NULL target in the training loop. */
  if ((task == 1 && !soft) || (task == 0 && !soft && !yc)) {
    *errmsg = sqlite3_mprintf("%s: mlp training target missing",
                              PREDICT_ERR_TARGET);
    return SQLITE_ERROR;
  }
  /* Shape contract: non-positive dimensions would make the weight sizes below
   * (nhid*nfeat, nout*nhid, ...) zero or negative and the training loop read out
   * of bounds; reject them loudly rather than train on a degenerate shape. */
  if (n <= 0 || nfeat <= 0 || nout <= 0 || nhid < 0 || epochs <= 0) {
    *errmsg = sqlite3_mprintf(
        "%s: mlp shape invalid (n=%d nfeat=%d nout=%d nhid=%d epochs=%d)",
        PREDICT_ERR_SCHEMA, n, nfeat, nout, nhid, epochs);
    return SQLITE_ERROR;
  }
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

  /* Divergence guard on the WRITE path: Adam (tanh head, no leaf regularization)
   * can push weights to NaN/Inf on a bad lr or degenerate data. Reject at fit
   * time rather than serialize a blob the finite-checked loader (rd_f32) would
   * later refuse. mean/sd are input-derived, finite when the inputs are; gbt and
   * tree students are finite by construction (bounded, lambda-regularized leaves
   * / means of finite data), so only the mlp needs this. */
  if (rc == SQLITE_OK &&
      (!arr_finite(m->W1, nW1) || !arr_finite(m->b1, nhid) ||
       !arr_finite(m->W2, nW2) || !arr_finite(m->b2, nout) ||
       !arr_finite(m->Wskip, nWs))) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: training diverged to non-finite weights; reduce epochs or lr",
        PREDICT_ERR_SCHEMA);
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

/* Fill node `ni` as a leaf over rows idx[0..n). Returns SQLITE_NOMEM (never a
 * bogus leaf) if the class-count buffer cannot be allocated. */
static int bld_leaf(Builder *b, int ni, const int *idx, int n) {
  TreeNode *nd = &b->nodes[ni];
  nd->feature = -1;
  nd->left = nd->right = -1;
  if (b->task == 0) {
    int *cnt = sqlite3_malloc(sizeof(int) * b->nclass);
    if (!cnt)
      return SQLITE_NOMEM;
    memset(cnt, 0, sizeof(int) * b->nclass);
    int best = 0, bestc = -1;
    for (int i = 0; i < n; i++)
      cnt[b->yc[idx[i]]]++;
    for (int c = 0; c < b->nclass; c++)
      if (cnt[c] > bestc) {
        bestc = cnt[c];
        best = c;
      }
    sqlite3_free(cnt);
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
  return SQLITE_OK;
}

/* Best (feature, threshold) split minimizing impurity, or feature<0 if none
 * improves. cmp buffer of (value, target) sorted per feature. */
typedef struct {
  f32 v;
  i32 row; /* original row index: a stable tiebreak for a total order */
  i32 yc;
  f32 yr;
} VY;
/* Total order on (v, row). Sorting only by v leaves equal-value rows in a
 * libc-defined order, and the regression branch then sums f64-of-f32 in that
 * order, so rounding — and near-tied split selection — would differ across
 * platforms, breaking content_hash reproducibility. The row tiebreak fixes the
 * order; the target union fields are never read here (yc/yr are only valid for
 * one task, so comparing them would be undefined). */
static int vy_cmp(const void *a, const void *b) {
  const VY *x = (const VY *)a, *y = (const VY *)b;
  if (x->v != y->v)
    return x->v < y->v ? -1 : 1;
  return x->row < y->row ? -1 : x->row > y->row ? 1 : 0;
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
      buf[i].row = idx[i];
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
int predict0_bld_build(Builder *b, int *idx, int n, int depth) {
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
    if (bld_leaf(b, ni, idx, n) != SQLITE_OK)
      return -1;
    return ni;
  }
  int feat;
  f32 thr;
  if (bld_best_split(b, idx, n, &feat, &thr) != SQLITE_OK)
    return -1;
  if (feat < 0) {
    if (bld_leaf(b, ni, idx, n) != SQLITE_OK)
      return -1;
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
    if (bld_leaf(b, ni, idx, n) != SQLITE_OK)
      return -1;
    return ni;
  }
  int L = predict0_bld_build(b, idx, nl, depth + 1);
  if (L < 0)
    return -1;
  int R = predict0_bld_build(b, idx + nl, n - nl, depth + 1);
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
int predict0_train_gbt(const f32 *X, int n, int nfeat, int task, int nclass,
                     const i32 *yc, const f32 *yr, const f32 *soft, Forest *fo,
                     char **errmsg) {
  memset(fo, 0, sizeof(*fo));
  /* Shape/target contract at this cross-TU entry (also called from the distill
   * recipes): non-positive dims make sum/n and m/n below NaN and size arrays to
   * zero/negative, and a missing target is dereferenced (yc/yr/soft). The fit
   * path validates these upstream in predict0_train_student; guard here too so
   * the boundary is safe on its own, mirroring predict0_train_mlp. */
  if (n <= 0 || nfeat <= 0 || task < 0 || task > 1 ||
      (task == 0 && (nclass < 2 || nclass > PREDICT0_MAX_CLASS))) {
    *errmsg = sqlite3_mprintf(
        "%s: gbt shape invalid (n=%d nfeat=%d task=%d nclass=%d)",
        PREDICT_ERR_SCHEMA, n, nfeat, task, nclass);
    return SQLITE_ERROR;
  }
  if ((task == 0 && !soft && !yc) || (task == 1 && !yr)) {
    *errmsg = sqlite3_mprintf("%s: gbt training target missing",
                              PREDICT_ERR_TARGET);
    return SQLITE_ERROR;
  }
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
  /* These scale with the row count, so their size_t byte product can exceed
   * INT_MAX; use sqlite3_malloc64 to avoid narrowing the request to int and
   * later overrunning F/p. */
  f64 *F = sqlite3_malloc64(sizeof(f64) * (size_t)n * nscore);
  f64 *p =
      task == 0 ? sqlite3_malloc64(sizeof(f64) * (size_t)n * nscore) : NULL;
  int *idx =
      sqlite3_malloc64(sizeof(int) * (size_t)n); /* scratch for bld_build */
  f32 *grad = sqlite3_malloc64(sizeof(f32) * (size_t)n);
  f32 *hess = task == 0 ? sqlite3_malloc64(sizeof(f32) * (size_t)n) : NULL;
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
      int root = predict0_bld_build(&b, idx, n, 0);
      if (root < 0) {
        sqlite3_free(b.nodes);
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
      for (int i = 0; i < n; i++)
        F[(size_t)i * nscore + s] +=
            (f64)fo->lr *
            predict0_reg_tree_value(b.nodes, b.n, &X[(size_t)i * nfeat]);

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

int predict0_intern_label(char ***labels, int *nclass, int *cap, const char *s,
                        int max, int *rc, char **errmsg) {
  for (int k = 0; k < *nclass; k++)
    if (strcmp((*labels)[k], s) == 0)
      return k;
  /* New distinct label. Enforce the cap here, before growing the array or
   * allocating the string, so an oversized vocabulary fails loud instead of
   * hitting SQLITE_NOMEM. max <= 0 means unbounded. */
  if (max > 0 && *nclass >= max) {
    *rc = SQLITE_ERROR;
    if (errmsg)
      *errmsg = sqlite3_mprintf("%s: too many classes (%d); the maximum is %d",
                                PREDICT_ERR_SCHEMA, *nclass + 1, max);
    return -1;
  }
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

int predict0_register_student(sqlite3 *db, const char *student_id,
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
    *errmsg = sqlite3_mprintf("%s: cannot prepare student insert: %s",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
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

/* Shared tabular training core (declared in predict-train.h). The fit()
 * aggregate calls this with an in-memory matrix; distill_predict() keeps its own
 * pipeline (teacher relabeling, soft labels, holdout metric) for now. */
int predict0_train_student(sqlite3 *db, const f32 *X, int n, int nfeat,
                           char *const *feat_names, int classify,
                           char *const *ylab, const f64 *yval, const char *kind,
                           const char *register_id, void **blob_out,
                           int *blob_len_out, char **errmsg) {
  *blob_out = NULL;
  *blob_len_out = 0;
  *errmsg = NULL; /* required out-param; own its initial state, don't rely on the
                     caller pre-zeroing it for the done: NOMEM check */
  if (n < DISTILL_MIN_ROWS) {
    *errmsg = sqlite3_mprintf("%s: need at least %d train rows, got %d",
                              PREDICT_ERR_SCHEMA, DISTILL_MIN_ROWS, n);
    return SQLITE_ERROR;
  }
  int is_gbt = !kind || strcmp(kind, "gbt") == 0;
  int is_tree = kind && strcmp(kind, "tree") == 0;
  if (!is_gbt && !is_tree) {
    *errmsg = sqlite3_mprintf("%s: fit kind must be 'gbt' or 'tree' (mlp,"
                              " soft-label, and teacher paths via"
                              " distill_predict): %s",
                              PREDICT_ERR_OPTIONS, kind);
    return SQLITE_ERROR;
  }

  int rc = SQLITE_OK;
  char **labels = NULL; /* class vocabulary (classify) */
  int nclass = 0, lcap = 0;
  i32 *y_teach = NULL;   /* classify: class indices */
  f32 *y_teach_r = NULL; /* regress: values */
  int *idx = NULL;
  Tree tree;
  memset(&tree, 0, sizeof(tree));
  Forest forest;
  memset(&forest, 0, sizeof(forest));
  void *blob = NULL;
  int blob_len = 0;

  if (classify) {
    y_teach = sqlite3_malloc(sizeof(i32) * n);
    if (!y_teach) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    for (int i = 0; i < n; i++) {
      int k = predict0_intern_label(&labels, &nclass, &lcap,
                                    ylab[i] ? ylab[i] : "", PREDICT0_MAX_CLASS,
                                    &rc, errmsg);
      if (k < 0)
        goto done;
      y_teach[i] = k;
    }
    if (nclass < 2) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: classify needs >= 2 distinct labels (got %d); pass"
          " '{\"task\":\"regress\"}' for a numeric target",
          PREDICT_ERR_TARGET, nclass);
      goto done;
    }
    /* Defensive invariant: predict0_intern_label already caps this during
     * interning; kept so the bound holds even if that path changes. */
    if (nclass > PREDICT0_MAX_CLASS) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: too many classes (%d); the maximum is %d",
                                PREDICT_ERR_SCHEMA, nclass, PREDICT0_MAX_CLASS);
      goto done;
    }
  } else {
    y_teach_r = sqlite3_malloc(sizeof(f32) * n);
    if (!y_teach_r) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    for (int i = 0; i < n; i++)
      y_teach_r[i] = (f32)yval[i];
  }

  if (is_gbt) {
    rc = predict0_train_gbt(X, n, nfeat, classify ? 0 : 1, nclass, y_teach, y_teach_r,
                   NULL, &forest, errmsg);
    if (rc != SQLITE_OK)
      goto done;
    forest.feat_names = sqlite3_malloc(sizeof(char *) * nfeat);
    if (!forest.feat_names) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    memset(forest.feat_names, 0, sizeof(char *) * nfeat);
    for (int f = 0; f < nfeat; f++)
      if (!(forest.feat_names[f] = sqlite3_mprintf("%s", feat_names[f]))) {
        rc = SQLITE_NOMEM;
        goto done;
      }
    forest.labels = labels; /* transfer ownership */
    labels = NULL;
    rc = predict0_forest_serialize(&forest, &blob, &blob_len);
    if (rc != SQLITE_OK)
      goto done;
  } else { /* single tree */
    idx = sqlite3_malloc(sizeof(int) * n);
    if (!idx) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    for (int i = 0; i < n; i++)
      idx[i] = i;
    Builder b;
    memset(&b, 0, sizeof(b));
    b.X = X;
    b.nfeat = nfeat;
    b.yc = y_teach;
    b.yr = y_teach_r;
    b.nclass = nclass;
    b.task = classify ? 0 : 1;
    int root = predict0_bld_build(&b, idx, n, 0);
    if (root < 0) {
      sqlite3_free(b.nodes);
      rc = SQLITE_NOMEM;
      goto done;
    }
    tree.feat_names = sqlite3_malloc(sizeof(char *) * nfeat);
    if (!tree.feat_names) {
      sqlite3_free(b.nodes);
      rc = SQLITE_NOMEM;
      goto done;
    }
    memset(tree.feat_names, 0, sizeof(char *) * nfeat);
    tree.nfeat = nfeat; /* set first so predict0_tree_free() releases partial
                           feat_names copies if an allocation below fails */
    for (int f = 0; f < nfeat; f++)
      if (!(tree.feat_names[f] = sqlite3_mprintf("%s", feat_names[f]))) {
        sqlite3_free(b.nodes);
        rc = SQLITE_NOMEM;
        goto done;
      }
    tree.task = b.task;
    tree.nclass = nclass;
    tree.labels = labels; /* transfer ownership */
    labels = NULL;
    tree.n_nodes = b.n;
    tree.nodes = b.nodes;
    rc = predict0_tree_serialize(&tree, &blob, &blob_len);
    if (rc != SQLITE_OK)
      goto done;
  }

  if (register_id) {
    char hash[PREDICT_HEX_BUFSIZE];
    rc = predict0_registry_ensure(db, errmsg); /* create _predict_models if new */
    if (rc != SQLITE_OK)
      goto done;
    rc = predict0_register_student(db, register_id, blob, blob_len, hash, errmsg);
    if (rc != SQLITE_OK)
      goto done;
  }
  *blob_out = blob;
  blob = NULL;
  *blob_len_out = blob_len;
  rc = SQLITE_OK;

done:
  if (rc == SQLITE_NOMEM && !*errmsg)
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  sqlite3_free(blob);
  sqlite3_free(idx);
  sqlite3_free(y_teach);
  sqlite3_free(y_teach_r);
  if (labels) {
    for (int k = 0; k < nclass; k++)
      sqlite3_free(labels[k]);
    sqlite3_free(labels);
  }
  predict0_forest_free(&forest); /* frees transferred feat_names/labels */
  predict0_tree_free(&tree);
  return rc;
}
