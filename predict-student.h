/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* Native student model format and serving runtime (RFC 0005 §4.1.3).
 *
 * The serving/format half of the distillation feature: the blob layouts
 * (PSTREE01 / PSGBT01 / PSMLP01), their bounds-checked deserializers, the
 * decision-tree / gradient-boosted-forest / MLP inference runtimes, and
 * predict0_tree_run (the runtime='tree' serving entry point, declared in
 * predict-internal.h). It carries no training code. The trainers and the
 * distill_predict() vtab live in predict-distill.c and write blobs in exactly this
 * format; this file reads and executes them. */
#ifndef PREDICT_STUDENT_H
#define PREDICT_STUDENT_H

#include "predict-internal.h"

#define TREE_MAX_FEAT PREDICT_MAX_FEAT /* alias; see predict-internal.h */
#define FCST_MAX_CONTEXT 4096 /* forecast student: max context length (nfeat) */
#define FCST_MAX_QUANT 64 /* forecast student: max quantile levels */
/* Forecast student with a linear skip: the hidden path is a small correction on
 * the dominant linear map, so its output is scaled by this. Shared by the
 * trainer (train_mlp) and the serving forward (predict0_mlp_forward). */
#define FCST_RES_SCALE 0.1

/* A decision-tree node (also the weak learner of a gbt forest). */
typedef struct {
  i32 feature; /* -1 = leaf */
  f32 threshold;
  i32 left, right;
  f32 value; /* leaf: regression value */
  i32 klass; /* leaf classify: class index; -1 internal */
  f32 conf;  /* leaf classify: confidence (fraction) */
} TreeNode;

/* A single decision-tree student (PSTREE01). */
typedef struct {
  int task; /* 0 classify, 1 regress */
  int nfeat;
  char **feat_names;
  int nclass;
  char **labels;
  int n_nodes;
  TreeNode *nodes;
} Tree;

/* A gradient-boosted forest student (PSGBT01). */
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

/* A softmax/regression MLP student (PSMLP01 classify, PSFCST forecast).
 * `Wskip` is an optional direct linear map from the standardized input to the
 * output (a DLinear/TiDE-style skip): out = Wskip*x + W2*tanh(W1*x). It is used
 * only by the forecast student, where the linear skip carries seasonal-naive +
 * trend and the hidden path adds a nonlinear correction; nhid=0 makes it a pure
 * linear model. NULL for the classification MLP (unchanged). */
typedef struct {
  int task, nfeat, nhid, nout, nclass;
  char **feat_names; /* [nfeat] */
  char **labels;     /* [nclass] (classification) */
  f32 *mean, *sd;    /* [nfeat] feature standardization */
  f32 *W1, *b1;      /* [nhid*nfeat], [nhid] (absent when nhid=0) */
  f32 *W2, *b2;      /* [nout*nhid], [nout] */
  f32 *Wskip;        /* [nout*nfeat] linear skip, or NULL */
} MLP;

/* Serialize a trained student to its inline blob (caller frees *blob_out with
 * sqlite3_free). Used by distill_predict(); the matching deserializers are internal to
 * the runtime and picked by blob magic. */
int predict0_tree_serialize(const Tree *t, void **blob_out, int *len_out);
int predict0_forest_serialize(const Forest *f, void **blob_out, int *len_out);
int predict0_mlp_serialize(const MLP *m, void **blob_out, int *len_out);

/* Forecast student (PSFCST01): a multi-output regression MLP whose outputs are
 * `nquant` quantiles per horizon step (nout = horizon * nquant), distilled from
 * a teacher's quantile fan. A point student is nquant=1 with level {0.5}. The
 * forecast() serving path instance-normalizes a context window of length nfeat,
 * applies the net, de-normalizes, and reads the point + interval off the fan. */
typedef struct {
  MLP mlp;     /* task=1 regression net; nout = horizon * nquant */
  int horizon; /* H forecast steps */
  int nquant;  /* Q quantile levels (1 = point student) */
  f32 *levels; /* [nquant] ascending; {0.5} for a point student */
} ForecastStudent;

int predict0_fcst_serialize(const ForecastStudent *fs, void **blob_out, int *len_out);
int predict0_fcst_deserialize(const void *blob, int len, ForecastStudent *fs,
                     char **errmsg);
void predict0_fcst_student_free(ForecastStudent *fs);

/* Free a student's owned memory. */
void predict0_tree_free(Tree *t);
void predict0_forest_free(Forest *f);
void predict0_mlp_free(MLP *m);

/* Inference, shared by predict0_tree_run and distill_predict()'s holdout evaluation. */
int predict0_tree_walk(const Tree *t, const f32 *x);
f32 predict0_reg_tree_value(const TreeNode *nd, int guard, const f32 *x);
int predict0_forest_predict_row(const Forest *f, const f32 *x, f64 *scbuf, char **pred,
                       f64 *conf, int *has_conf);
int predict0_mlp_predict_row(const MLP *m, const f32 *x, f32 *hid, f32 *out, char **pred,
                    f64 *conf, int *has_conf);
/* Raw forward pass: writes hid[nhid] and out[nout]. Used by forecast()
 * serving, which supplies an instance-normalized window as x. */
void predict0_mlp_forward(const MLP *m, const f32 *x, f32 *hid, f32 *out);

#endif /* PREDICT_STUDENT_H */
