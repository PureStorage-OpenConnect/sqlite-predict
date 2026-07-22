#ifndef PREDICT_INTERNAL_H
#define PREDICT_INTERNAL_H

#include "sqlite-predict.h"

#include <assert.h>
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

#endif /* PREDICT_INTERNAL_H */
