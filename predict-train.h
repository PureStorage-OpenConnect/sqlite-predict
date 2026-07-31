/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
#ifndef PREDICT_TRAIN_H
#define PREDICT_TRAIN_H

#include "predict-internal.h"
#include "predict-student.h"

/* Shared training defaults (the core here + the distill recipes). */
#define DISTILL_MIN_ROWS 8
#define MLP_HIDDEN 48
#define MLP_EPOCHS 400
#define MLP_LR 0.02

/* Tree builder. The distill recipe constructs one in place and calls predict0_bld_build. */
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

int predict0_bld_build(Builder *b, int *idx, int n, int depth);
int predict0_train_gbt(const f32 *X, int n, int nfeat, int task, int nclass,
              const i32 *yc, const f32 *yr, const f32 *soft, Forest *fo,
              char **errmsg);
int predict0_train_mlp(const f32 *X, int n, int nfeat, int nout, const i32 *yc,
              const f32 *soft, int task, int nhid, int epochs, f32 lr,
              int use_skip, MLP *m, char **errmsg);
/* Intern a label into the vocabulary, returning its class index (>=0), or -1
 * with *rc set on failure. max caps the distinct-class count: a new label is
 * rejected (SQLITE_ERROR + *errmsg) before the array grows or the string is
 * allocated once *nclass would exceed max; max <= 0 means unbounded. */
int predict0_intern_label(char ***labels, int *nclass, int *cap, const char *s,
                          int max, int *rc, char **errmsg);
int predict0_register_student(sqlite3 *db, const char *student_id, const void *blob,
                     int blob_len, char hash_out[PREDICT_HEX_BUFSIZE],
                     char **errmsg);

/* Train a native tree/gbt student from an in-memory feature matrix. X is
 * row-major [n, nfeat]. For classify, ylab holds n labels (yval NULL); for
 * regress, yval holds n values (ylab NULL). feat_names[nfeat] name the columns
 * (placeholders are fine; the scalar predict() maps features by position).
 * kind is "gbt" (default) or "tree". On SQLITE_OK, blob_out and blob_len_out
 * receive the serialized student (caller frees the blob with sqlite3_free);
 * if register_id is non-NULL the student is also inserted into _predict_models.
 * Returns an SQLITE_ code with *errmsg set (PREDICT_ERR_* lead) on failure. */
int predict0_train_student(sqlite3 *db, const f32 *X, int n, int nfeat,
                           char *const *feat_names, int classify,
                           char *const *ylab, const f64 *yval, const char *kind,
                           const char *register_id, void **blob_out,
                           int *blob_len_out, char **errmsg);

#endif /* PREDICT_TRAIN_H */
