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

/* Closed call-error set, RFC 0005 §4.3. Raised as "PREDICT_ERR_<NAME>: detail". */
#define PREDICT_ERR_OPTIONS "PREDICT_ERR_OPTIONS"
#define PREDICT_ERR_QUERY_NOT_READONLY "PREDICT_ERR_QUERY_NOT_READONLY"
#define PREDICT_ERR_SCHEMA "PREDICT_ERR_SCHEMA"
#define PREDICT_ERR_HORIZON "PREDICT_ERR_HORIZON"
#define PREDICT_ERR_THRESHOLD "PREDICT_ERR_THRESHOLD"
#define PREDICT_ERR_CONTEXT_TOO_LARGE "PREDICT_ERR_CONTEXT_TOO_LARGE"
#define PREDICT_ERR_TARGET "PREDICT_ERR_TARGET"
#define PREDICT_ERR_TASK "PREDICT_ERR_TASK"
#define PREDICT_ERR_PROBE "PREDICT_ERR_PROBE"
#define PREDICT_ERR_LICENSE "PREDICT_ERR_LICENSE"
#define PREDICT_ERR_MODEL_NOT_FOUND "PREDICT_ERR_MODEL_NOT_FOUND"
#define PREDICT_ERR_MODEL_HASH "PREDICT_ERR_MODEL_HASH"
#define PREDICT_ERR_STUDENT_EXISTS "PREDICT_ERR_STUDENT_EXISTS"
#define PREDICT_ERR_RESOURCE "PREDICT_ERR_RESOURCE"
#define PREDICT_ERR_RECEIPT_NOT_FOUND "PREDICT_ERR_RECEIPT_NOT_FOUND"
#define PREDICT_ERR_ANCHOR_UNAVAILABLE "PREDICT_ERR_ANCHOR_UNAVAILABLE"
#define PREDICT_ERR_REPLAY_MISMATCH "PREDICT_ERR_REPLAY_MISMATCH"

int predict0_forecast_init(sqlite3 *db);
int predict0_receipts_init(sqlite3 *db);

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
void predict0_hash_hex(predict0_hasher *h, char hex[65]);

/* Idempotent DDL for _predict_models/_predict_receipts + bundled rows. */
int predict0_receipts_ensure(sqlite3 *db, char **errmsg);

/* content_hash of a registered model; sqlite3_malloc'd. NULL = absent. */
char *predict0_registry_model_hash(sqlite3 *db, const char *model_id);

/* Deterministic logical digest of all user tables (schema + rows,
 * excluding _predict_% and sqlite_%), hex into out[65]. */
int predict0_logical_digest(sqlite3 *db, char out[65], char **errmsg);

/* Insert one receipt row. anchor/params/input_sql/result_hash owned by
 * caller. receipt_id_out[27] receives the new ULID. */
int predict0_receipt_insert(sqlite3 *db, const char *operation,
                            const char *model_id, const char *model_hash,
                            const char *anchor_kind, const char *anchor,
                            const char *params, const char *input_sql,
                            const char *result_hash, char receipt_id_out[27],
                            char **errmsg);

#endif /* PREDICT_INTERNAL_H */
