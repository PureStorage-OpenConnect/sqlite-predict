/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
#ifndef PREDICT_INTERNAL_H
#define PREDICT_INTERNAL_H

#include "sqlite-predict.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int8_t i8;
typedef uint8_t u8;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;
typedef float f32;
typedef double f64;
typedef size_t usize;

#define UNUSED_PARAMETER(X) (void)(X)
#define countof(x) (sizeof(x) / sizeof((x)[0]))

/* Wire-format contracts. These are duplicated across the key builder,
 * the result hasher, and the RFC §4.1.3 spec; they MUST agree, so they
 * live in exactly one place. */
#define PREDICT_FS 0x1f /* field separator: series keys + result hash */
#define PREDICT_RS 0x1e /* row separator: result hash */

/* Buffer sizes encoding fixed-width invariants. */
#define PREDICT_TS_BUFSIZE 24   /* "YYYY-MM-DDTHH:MM:SSZ" (20) + slack */
#define PREDICT_ULID_BUFSIZE 27 /* 26 Crockford base32 chars + NUL */
#define PREDICT_HEX_BUFSIZE 65  /* 64 lowercase SHA-256 hex chars + NUL */

/* Representable timestamp domain: proleptic Gregorian years 0001..9999
 * (the ISO 8601 range), in epoch milliseconds. Every value handed to
 * predict0_format_timestamp is clamped here, so formatting is total and
 * buffer-safe for any i64. */
#define PREDICT_MS_MIN (-62135596800000LL) /* 0001-01-01T00:00:00Z */
#define PREDICT_MS_MAX (253402300799999LL) /* 9999-12-31T23:59:59.999Z */

#define PREDICT_MS_PER_HOUR 3600000LL
/* Integers at or above this are already epoch ms; below, epoch seconds.
 * 1e11 s is year 5138; 1e11 ms is year 1973 — no realistic overlap. */
#define PREDICT_EPOCH_MS_THRESHOLD 100000000000LL

/* Closed call-error set, RFC 0005 §4.3. Raised as "PREDICT_ERR_<NAME>: detail". */
#define PREDICT_ERR_OPTIONS "PREDICT_ERR_OPTIONS"
#define PREDICT_ERR_QUERY_NOT_READONLY "PREDICT_ERR_QUERY_NOT_READONLY"
#define PREDICT_ERR_SCHEMA "PREDICT_ERR_SCHEMA"
#define PREDICT_ERR_HORIZON "PREDICT_ERR_HORIZON"
#define PREDICT_ERR_THRESHOLD "PREDICT_ERR_THRESHOLD"
#define PREDICT_ERR_CONTEXT_TOO_LARGE "PREDICT_ERR_CONTEXT_TOO_LARGE"
#define PREDICT_ERR_TARGET "PREDICT_ERR_TARGET"
#define PREDICT_ERR_TASK "PREDICT_ERR_TASK"
#define PREDICT_ERR_LICENSE "PREDICT_ERR_LICENSE"
#define PREDICT_ERR_MODEL_NOT_FOUND "PREDICT_ERR_MODEL_NOT_FOUND"
#define PREDICT_ERR_MODEL_EXISTS "PREDICT_ERR_MODEL_EXISTS"
#define PREDICT_ERR_MODEL_HASH "PREDICT_ERR_MODEL_HASH"
#define PREDICT_ERR_RUNTIME_UNAVAILABLE "PREDICT_ERR_RUNTIME_UNAVAILABLE"
#define PREDICT_ERR_IO_SPEC "PREDICT_ERR_IO_SPEC"
#define PREDICT_ERR_INFERENCE "PREDICT_ERR_INFERENCE"
#define PREDICT_ERR_STUDENT_EXISTS "PREDICT_ERR_STUDENT_EXISTS"
#define PREDICT_ERR_RESOURCE "PREDICT_ERR_RESOURCE"
#define PREDICT_ERR_RECEIPT_NOT_FOUND "PREDICT_ERR_RECEIPT_NOT_FOUND"
#define PREDICT_ERR_ANCHOR_UNAVAILABLE "PREDICT_ERR_ANCHOR_UNAVAILABLE"
#define PREDICT_ERR_REPLAY_MISMATCH "PREDICT_ERR_REPLAY_MISMATCH"

int predict0_forecast_init(sqlite3 *db);
int predict0_receipts_init(sqlite3 *db);
int predict0_tabular_init(sqlite3 *db);
int predict0_distill_init(sqlite3 *db);

/* ---- shared helpers (sqlite-predict.c) ---- */

/* Parse an ISO-8601 UTC timestamp ("YYYY-MM-DD[ T]HH:MM[:SS][Z]") or a
 * bare date to epoch milliseconds. Returns 0 on success. */
int predict0_parse_timestamp(const char *s, i64 *out_ms);

/* Format epoch ms back to "YYYY-MM-DDTHH:MM:SSZ" into buf[21+]. */
void predict0_format_timestamp(i64 ms, char *buf, usize bufsize);

/* Inverse standard-normal CDF (Acklam's rational approximation).
 * p in (0,1). Used for prediction-interval z values. */
f64 predict0_norm_quantile(f64 p);

/* Options parsing: the trailing JSON options argument. keys is a
 * NULL-terminated array of allowed key names; each parsed key/value is
 * delivered to the callback (value as sqlite3_value*). Returns 0 on
 * success; on failure sets *errmsg (sqlite3_malloc'd, starts with a
 * PREDICT_ERR_* code). NULL/absent json means no options: success. */
typedef int (*predict0_option_cb)(void *ctx, const char *key,
                                  sqlite3_value *value, char **errmsg);
int predict0_options_parse(sqlite3 *db, const char *json,
                           const char *const *keys, predict0_option_cb cb,
                           void *ctx, char **errmsg);

/* Smallest ULID for the given epoch ms (random component zeroed),
 * Crockford base32, 26 chars + NUL into buf[27]. */
void predict0_ulid_min(i64 ms, char *buf);

/* Fresh ULID for the given epoch ms with sqlite3_randomness entropy. */
void predict0_ulid_new(i64 ms, char *buf);

/* ---- receipts (predict-receipts.c) ---- */

#include "sha256.h"

/* RFC §4.1.3 canonical row hashing: type-tagged fields, 0x1F between
 * fields, 0x1E after each row. */
typedef struct {
  sha256_ctx sha;
} predict0_hasher;

void predict0_hash_init(predict0_hasher *h);
void predict0_hash_null(predict0_hasher *h);
void predict0_hash_int(predict0_hasher *h, i64 v);
void predict0_hash_real(predict0_hasher *h, f64 v);
void predict0_hash_text(predict0_hasher *h, const char *s);
void predict0_hash_row_end(predict0_hasher *h);
void predict0_hash_hex(predict0_hasher *h, char hex[PREDICT_HEX_BUFSIZE]);

/* Idempotent DDL for _predict_models/_predict_receipts + bundled rows. */
int predict0_receipts_ensure(sqlite3 *db, char **errmsg);

/* content_hash of a registered model; sqlite3_malloc'd. NULL = absent. */
char *predict0_registry_model_hash(sqlite3 *db, const char *model_id);

/* A registry row, as the dispatcher and runtime backends see it. All
 * strings are sqlite3_malloc'd; weights is a malloc'd copy of the inline
 * BLOB (NULL when the model is URI-referenced or has no local weights). */
typedef struct {
  char *runtime;      /* 'onnx' | 'ggml' | 'tree' | 'remote' | 'bundled' */
  char *kind;         /* 'tabular-fm' | 'student' | ... */
  char *weights_uri;  /* external path to weights; NULL if inline/none */
  char *io_spec;      /* JSON tensor mapping; NULL if not set */
  char *license;      /* SPDX id */
  char *content_hash; /* hex sha-256 pinning the exact weights */
  void *weights;      /* inline weight bytes; NULL if URI/none */
  int weights_len;
} predict0_model_row;

/* Look up a model. Returns 0 and fills *out on hit, 1 if absent, or an
 * SQLITE_ error. Free *out with predict0_model_row_free on a hit. */
int predict0_registry_lookup(sqlite3 *db, const char *model_id,
                             predict0_model_row *out);
void predict0_model_row_free(predict0_model_row *m);

/* SHA-256 of a file's bytes, streamed. hex into out[65]. Returns 0 on
 * success; on failure sets *errmsg (sqlite3_malloc'd, PREDICT_ERR_* lead). */
int predict0_hash_file(const char *path, char out[PREDICT_HEX_BUFSIZE],
                       char **errmsg);

/* ---- runtime backends (predict-onnx.c, opt-in build) ---- */

/* Backend-relevant options, parsed from the predict() JSON. Borrowed
 * pointers into the caller's parsed options; valid for the call only. */
typedef struct {
  const char *device;         /* 'cpu'|'coreml'|'cuda'|'tensorrt'; NULL=cpu */
  const char *precision;      /* 'fp32'|'fp16'|'int8'; NULL=fp32 */
  const char *accept_license; /* SPDX the caller accepts; NULL=none */
  int receipt;                /* emit a receipt? */
} predict0_backend_opts;

/* One prediction a runtime backend hands back, in apply-query order. */
typedef struct {
  int ref_type; /* SQLITE_INTEGER/FLOAT/TEXT/NULL of the row_ref */
  i64 ref_i;
  f64 ref_f;
  char *ref_t;      /* sqlite3_malloc'd, for a TEXT ref */
  char *prediction; /* sqlite3_malloc'd; NULL on status/error rows */
  f64 confidence;
  int has_conf;
  const char *status; /* static string ('ok','non_numeric',...) */
} predict0_result;

void predict0_results_free(predict0_result *rows, int n);

/* Execute a native tree student (runtime='tree') from its inline-BLOB
 * weights, over the rows of apply_sql. Fills rows/n (apply order; free with
 * predict0_results_free) and, when opts->receipt, emits a receipt into
 * receipt_id_out. The tree runtime lives in the zero-dependency core, so a
 * distilled student runs with no onnxruntime. Returns SQLITE_OK or an
 * SQLITE_ code with *errmsg set (PREDICT_ERR_* lead). */
int predict0_tree_run(sqlite3 *db, const char *model_id, const char *apply_sql,
                      const predict0_model_row *model,
                      const predict0_backend_opts *opts,
                      predict0_result **rows, int *n,
                      char receipt_id_out[PREDICT_ULID_BUFSIZE], char **errmsg);

/* Interpolate the quantile at level p over ascending levels lev[Q] with the
 * (ascending) values val[Q], extrapolating the tails (e.g. deciles to a 95%
 * band). Shared by the native forecast student and the onnx forecast backend. */
f64 predict0_quantile_at(const f32 *lev, const f64 *val, int Q, f64 p);

#ifdef SQLITE_PREDICT_ONNX
/* ONNX forecast backend: run a sequence model (a context window -> a quantile
 * fan) and fill fc/lo/hi[horizon] (the point and the interval at `conf`), read
 * off the fan by interpolating over the model's quantile levels. The io_spec
 * (layout 'sequence') names the input/output tensors, the levels, and the patch
 * size the context length is truncated to. Returns SQLITE_OK or an SQLITE_ code
 * with *errmsg set (PREDICT_ERR_* lead). */
int predict0_onnx_forecast(sqlite3 *db, const predict0_model_row *model,
                           const predict0_backend_opts *opts,
                           const f64 *context, int ctx_len, int horizon,
                           f64 conf, f64 *fc, f64 *lo, f64 *hi, char **errmsg);

/* As above but returns the raw quantile fan: *fan_out is malloc'd
 * [horizon*nquant] step-major (fan[k*nquant+q]), *levels_out is malloc'd
 * [nquant] (both freed by the caller). Used by distill_forecast(teacher=...)
 * to label windows with an in-DB onnx teacher. */
int predict0_onnx_forecast_fan(sqlite3 *db, const predict0_model_row *model,
                               const predict0_backend_opts *opts,
                               const f64 *context, int ctx_len, int horizon,
                               f64 **fan_out, f32 **levels_out, int *nquant_out,
                               char **errmsg);

/* ONNX inference for predict(). Dispatches on the model's io_spec layout:
 * 'vector' (a self-contained model mapping features -> prediction, train_sql
 * unused) or 'in_context' (a teacher that ingests the train rows as context
 * each call, so train_sql is required). Reads row_ref + named features from
 * apply_sql, runs batched inference on the requested execution provider, and
 * fills rows/n (apply order; free with predict0_results_free). When
 * opts->receipt, emits a receipt and writes receipt_id_out. Returns
 * SQLITE_OK, or an SQLITE_ code with *errmsg set (PREDICT_ERR_* lead). */
int predict0_onnx_predict(sqlite3 *db, const char *model_id,
                          const char *train_sql, const char *apply_sql,
                          const predict0_model_row *model,
                          const predict0_backend_opts *opts,
                          predict0_result **rows, int *n,
                          char receipt_id_out[PREDICT_ULID_BUFSIZE],
                          char **errmsg);

/* Derive an io_spec by reading a model's input/output tensors, so a caller
 * can register with just a weights path. On success returns SQLITE_OK and a
 * sqlite3_malloc'd JSON io_spec in *io_spec_out (caller frees). Covers the
 * unambiguous shapes (1 input -> vector; 3 inputs named x_train/y_train/
 * x_query -> in_context; a single output whose width picks probs vs value);
 * anything it cannot disambiguate returns an error asking for an explicit
 * io_spec. */
int predict0_onnx_introspect(sqlite3 *db, const char *weights_uri,
                             char **io_spec_out, char **errmsg);
#endif

/* Deterministic logical digest of all user tables (schema + rows,
 * excluding _predict_% and sqlite_%), hex into out[65]. */
int predict0_logical_digest(sqlite3 *db, char out[PREDICT_HEX_BUFSIZE], char **errmsg);

/* Insert one receipt row. anchor/params/input_sql/result_hash owned by
 * caller. receipt_id_out[27] receives the new ULID. */
int predict0_receipt_insert(sqlite3 *db, const char *operation,
                            const char *model_id, const char *model_hash,
                            const char *anchor_kind, const char *anchor,
                            const char *params, const char *input_sql,
                            const char *result_hash, char receipt_id_out[PREDICT_ULID_BUFSIZE],
                            char **errmsg);

/* Shared receipt tail: ensure tables + model hash + digest + insert.
 * Caller supplies the op-specific params and result_hash. */
int predict0_emit_receipt(sqlite3 *db, const char *op, const char *model_id,
                          const char *params, const char *input_sql,
                          const char *result_hash,
                          char receipt_id_out[PREDICT_ULID_BUFSIZE],
                          char **errmsg);

#endif /* PREDICT_INTERNAL_H */
