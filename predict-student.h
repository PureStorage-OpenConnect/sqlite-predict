/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* Native student model format and serving runtime (RFC 0005 §4.1.6).
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

#define TREE_MAX_FEAT 64 /* apply-time feature cap; bounds a student's nfeat */
#define FCST_MAX_CONTEXT 4096 /* forecast student: max context length (nfeat) */

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

/* A one-hidden-layer softmax MLP student (PSMLP01). */
typedef struct {
  int task, nfeat, nhid, nout, nclass;
  char **feat_names; /* [nfeat] */
  char **labels;     /* [nclass] (classification) */
  f32 *mean, *sd;    /* [nfeat] feature standardization */
  f32 *W1, *b1;      /* [nhid*nfeat], [nhid] */
  f32 *W2, *b2;      /* [nout*nhid], [nout] */
} MLP;

/* Serialize a trained student to its inline blob (caller frees *blob_out with
 * sqlite3_free). Used by distill_predict(); the matching deserializers are internal to
 * the runtime and picked by blob magic. */
int tree_serialize(const Tree *t, void **blob_out, int *len_out);
int forest_serialize(const Forest *f, void **blob_out, int *len_out);
int mlp_serialize(const MLP *m, void **blob_out, int *len_out);

/* Forecast student (PSFCST01): a multi-output regression MLP (nout = horizon)
 * reusing the MLP struct with task=1, no class labels or feature names. The
 * forecast() serving path instance-normalizes a context window of length nfeat,
 * applies the net, and de-normalizes the nout outputs. */
int fcst_serialize(const MLP *m, void **blob_out, int *len_out);
int fcst_deserialize(const void *blob, int len, MLP *m, char **errmsg);

/* Free a student's owned memory. */
void tree_free(Tree *t);
void forest_free(Forest *f);
void mlp_free(MLP *m);

/* Inference, shared by predict0_tree_run and distill_predict()'s holdout evaluation. */
int tree_walk(const Tree *t, const f32 *x);
f32 reg_tree_value(const TreeNode *nd, int guard, const f32 *x);
int forest_predict_row(const Forest *f, const f32 *x, f64 *scbuf, char **pred,
                       f64 *conf, int *has_conf);
int mlp_predict_row(const MLP *m, const f32 *x, f32 *hid, f32 *out, char **pred,
                    f64 *conf, int *has_conf);
/* Raw forward pass: writes hid[nhid] and out[nout]. Used by forecast()
 * serving, which supplies an instance-normalized window as x. */
void mlp_forward(const MLP *m, const f32 *x, f32 *hid, f32 *out);

#endif /* PREDICT_STUDENT_H */
