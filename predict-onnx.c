/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* ONNX runtime backend for predict() — the opt-in foundation-model
 * serving path (make loadable-onnx, -DSQLITE_PREDICT_ONNX). This file is
 * the only one that links onnxruntime; the core build never sees it.
 *
 * This pass implements the "vector" io_spec layout: a self-contained,
 * pre-trained model that maps a feature vector to a prediction (a
 * distilled student, or any exported tabular classifier/regressor). The
 * "in_context" layout (a teacher such as TabFM that ingests the training
 * rows as context) is the next pass; it is validated against real weights
 * on the gated GPU CI job.
 *
 * Performance shape (see the RFC): the expensive costs are session
 * creation and weight load, so sessions are cached process-global keyed by
 * (weights, device, precision) and reused across calls; query rows are run
 * in batches, not one at a time. Execution-provider selection is explicit
 * and fails loud — a request for a provider this build lacks errors rather
 * than silently dropping to CPU. */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#ifndef SQLITE_PREDICT_ONNX
/* Nothing in this translation unit when the runtime is not compiled in.
 * The dispatcher in predict-tabular.c guards its call the same way. */
#else

#include "onnxruntime_c_api.h"
#ifdef __APPLE__
#include "coreml_provider_factory.h"
#endif

#define ONNX_BATCH 1024 /* query rows per forward pass */

static const OrtApi *g_ort = NULL;
static OrtEnv *g_env = NULL;

/* Process-global session cache. Sessions are expensive to build and safe
 * to Run() concurrently, so we keep them for the process lifetime. The
 * list head is a static global, so the sessions stay reachable at exit
 * (no leak-checker false positives). */
typedef struct onnx_session {
  char *key; /* content_hash|device|precision */
  OrtSession *session;
  struct onnx_session *next;
} onnx_session;
static onnx_session *g_cache = NULL;

static sqlite3_mutex *onnx_mutex(void) {
  return sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1);
}

/* ---- error plumbing ---- */

/* Turn an OrtStatus into a PREDICT_ERR_* message and release it. Returns
 * SQLITE_ERROR so callers can `return onnx_fail(...)`. */
static int onnx_fail(OrtStatus *st, const char *code, const char *ctx,
                     char **errmsg) {
  const char *m = st ? g_ort->GetErrorMessage(st) : "(no status)";
  *errmsg = sqlite3_mprintf("%s: %s: %s", code, ctx, m);
  if (st)
    g_ort->ReleaseStatus(st);
  return SQLITE_ERROR;
}

static int onnx_init(char **errmsg) {
  if (g_env)
    return SQLITE_OK;
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    *errmsg = sqlite3_mprintf("%s: onnxruntime API version mismatch",
                              PREDICT_ERR_RUNTIME_UNAVAILABLE);
    return SQLITE_ERROR;
  }
  OrtStatus *st =
      g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sqlite-predict", &g_env);
  if (st)
    return onnx_fail(st, PREDICT_ERR_RUNTIME_UNAVAILABLE, "CreateEnv", errmsg);
  return SQLITE_OK;
}

/* ---- io_spec parsing ---- */

typedef struct {
  char *input_name;
  char **features;
  int nfeat;
  char *output_name;
  char *output_kind; /* 'probs' | 'logits' | 'label' | 'value' */
  char **labels;
  int nlabels;
} onnx_io;

static void onnx_io_free(onnx_io *io) {
  sqlite3_free(io->input_name);
  for (int i = 0; i < io->nfeat; i++)
    sqlite3_free(io->features[i]);
  sqlite3_free(io->features);
  sqlite3_free(io->output_name);
  sqlite3_free(io->output_kind);
  for (int i = 0; i < io->nlabels; i++)
    sqlite3_free(io->labels[i]);
  sqlite3_free(io->labels);
  memset(io, 0, sizeof(*io));
}

/* scalar json_extract(json, path) -> sqlite3_malloc'd text, NULL if absent */
static char *json_str(sqlite3 *db, const char *json, const char *path) {
  sqlite3_stmt *s = NULL;
  char *out = NULL;
  if (sqlite3_prepare_v2(db, "SELECT json_extract(?1, ?2)", -1, &s, NULL) ==
      SQLITE_OK) {
    sqlite3_bind_text(s, 1, json, -1, SQLITE_STATIC);
    sqlite3_bind_text(s, 2, path, -1, SQLITE_STATIC);
    if (sqlite3_step(s) == SQLITE_ROW &&
        sqlite3_column_type(s, 0) != SQLITE_NULL)
      out = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(s, 0));
  }
  sqlite3_finalize(s);
  return out;
}

/* json array at path -> sqlite3_malloc'd char*[]; *n set. 0 on success. */
static int json_arr(sqlite3 *db, const char *json, const char *path,
                    char ***out, int *n) {
  *out = NULL;
  *n = 0;
  sqlite3_stmt *s = NULL;
  if (sqlite3_prepare_v2(
          db, "SELECT value FROM json_each(?1, ?2)", -1, &s, NULL) != SQLITE_OK)
    return 1;
  sqlite3_bind_text(s, 1, json, -1, SQLITE_STATIC);
  sqlite3_bind_text(s, 2, path, -1, SQLITE_STATIC);
  int cap = 0;
  int rc = 0;
  while (sqlite3_step(s) == SQLITE_ROW) {
    if (*n == cap) {
      cap = cap ? cap * 2 : 8;
      char **g = sqlite3_realloc(*out, sizeof(char *) * cap);
      if (!g) {
        rc = 1;
        break;
      }
      *out = g;
    }
    (*out)[*n] =
        sqlite3_mprintf("%s", (const char *)sqlite3_column_text(s, 0));
    (*n)++;
  }
  sqlite3_finalize(s);
  return rc;
}

static int onnx_io_parse(sqlite3 *db, const char *io_spec, onnx_io *io,
                         char **errmsg) {
  memset(io, 0, sizeof(*io));
  if (!io_spec) {
    *errmsg = sqlite3_mprintf("%s: model has no io_spec", PREDICT_ERR_IO_SPEC);
    return SQLITE_ERROR;
  }
  char *layout = json_str(db, io_spec, "$.layout");
  if (!layout || strcmp(layout, "vector") != 0) {
    *errmsg = sqlite3_mprintf(
        "%s: this build serves the 'vector' io_spec layout; got '%s'",
        PREDICT_ERR_IO_SPEC, layout ? layout : "(none)");
    sqlite3_free(layout);
    return SQLITE_ERROR;
  }
  sqlite3_free(layout);

  io->input_name = json_str(db, io_spec, "$.input");
  io->output_name = json_str(db, io_spec, "$.output.name");
  io->output_kind = json_str(db, io_spec, "$.output.kind");
  json_arr(db, io_spec, "$.features", &io->features, &io->nfeat);
  json_arr(db, io_spec, "$.output.labels", &io->labels, &io->nlabels);

  int ok = io->input_name && io->output_name && io->output_kind &&
           io->nfeat > 0;
  int classify = io->output_kind &&
                 (strcmp(io->output_kind, "probs") == 0 ||
                  strcmp(io->output_kind, "logits") == 0 ||
                  strcmp(io->output_kind, "label") == 0);
  int regress = io->output_kind && strcmp(io->output_kind, "value") == 0;
  if (!ok || (!classify && !regress) || (classify && io->nlabels == 0)) {
    *errmsg = sqlite3_mprintf(
        "%s: io_spec needs input, features[], output.name, output.kind"
        " ('probs'|'logits'|'label'|'value'), and labels[] for classifiers",
        PREDICT_ERR_IO_SPEC);
    onnx_io_free(io);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

/* ---- session cache ---- */

/* Build a session for (model, device, precision). Caller holds the mutex. */
static int onnx_build_session(const predict0_model_row *model,
                              const predict0_backend_opts *opts,
                              OrtSession **out, char **errmsg) {
  int rc = SQLITE_OK;
  OrtSessionOptions *so = NULL;
  *out = NULL;

#define BUILD_CHECK(expr, code, ctx)                                          \
  do {                                                                        \
    OrtStatus *st_ = (expr);                                                  \
    if (st_) {                                                                \
      rc = onnx_fail(st_, code, ctx, errmsg);                                 \
      goto fail;                                                              \
    }                                                                         \
  } while (0)

  BUILD_CHECK(g_ort->CreateSessionOptions(&so), PREDICT_ERR_INFERENCE,
              "CreateSessionOptions");
  /* 0 lets onnxruntime pick a sane thread count. */
  BUILD_CHECK(g_ort->SetIntraOpNumThreads(so, 0), PREDICT_ERR_INFERENCE,
              "SetIntraOpNumThreads");

  const char *device = opts->device ? opts->device : "cpu";
  if (strcmp(device, "cpu") == 0) {
    /* CPU EP is always present; nothing to append. */
  } else if (strcmp(device, "coreml") == 0) {
#ifdef __APPLE__
    BUILD_CHECK(OrtSessionOptionsAppendExecutionProvider_CoreML(so, 0),
                PREDICT_ERR_RUNTIME_UNAVAILABLE, "append CoreML EP");
#else
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: CoreML is only available on Apple platforms",
                              PREDICT_ERR_RUNTIME_UNAVAILABLE);
    goto fail;
#endif
  } else if (strcmp(device, "cuda") == 0 || strcmp(device, "tensorrt") == 0) {
#ifdef SQLITE_PREDICT_ONNX_GPU
    OrtStatus *ep =
        strcmp(device, "cuda") == 0
            ? g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(so, NULL)
            : g_ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(so,
                                                                       NULL);
    BUILD_CHECK(ep, PREDICT_ERR_RUNTIME_UNAVAILABLE, "append GPU EP");
#else
    /* Honest state: the GPU execution providers are validated on the gated
     * GPU CI job and compiled only into the GPU build. This CPU build does
     * not silently fall back to CPU for a cuda/tensorrt request. */
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: device '%s' needs the GPU build (loadable-onnx-gpu); this build"
        " serves cpu and coreml",
        PREDICT_ERR_RUNTIME_UNAVAILABLE, device);
    goto fail;
#endif
  } else {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: unknown device '%s' (cpu|coreml|cuda|tensorrt)",
        PREDICT_ERR_OPTIONS, device);
    goto fail;
  }

  if (model->weights_uri) {
    BUILD_CHECK(g_ort->CreateSession(g_env, model->weights_uri, so, out),
                PREDICT_ERR_INFERENCE, "CreateSession from file");
  } else if (model->weights && model->weights_len > 0) {
    BUILD_CHECK(g_ort->CreateSessionFromArray(
                    g_env, model->weights, (size_t)model->weights_len, so, out),
                PREDICT_ERR_INFERENCE, "CreateSession from blob");
  } else {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: model has no weights to load",
                              PREDICT_ERR_INFERENCE);
    goto fail;
  }

#undef BUILD_CHECK
fail:
  if (so)
    g_ort->ReleaseSessionOptions(so);
  return rc;
}

static int onnx_get_session(const predict0_model_row *model,
                            const predict0_backend_opts *opts,
                            OrtSession **out, char **errmsg) {
  const char *device = opts->device ? opts->device : "cpu";
  const char *precision = opts->precision ? opts->precision : "fp32";
  char *key =
      sqlite3_mprintf("%s|%s|%s", model->content_hash, device, precision);
  if (!key) {
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    return SQLITE_NOMEM;
  }

  sqlite3_mutex *mx = onnx_mutex();
  sqlite3_mutex_enter(mx);
  for (onnx_session *n = g_cache; n; n = n->next) {
    if (strcmp(n->key, key) == 0) {
      *out = n->session;
      sqlite3_mutex_leave(mx);
      sqlite3_free(key);
      return SQLITE_OK;
    }
  }
  OrtSession *sess = NULL;
  int rc = onnx_build_session(model, opts, &sess, errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_mutex_leave(mx);
    sqlite3_free(key);
    return rc;
  }
  onnx_session *node = sqlite3_malloc(sizeof(*node));
  if (!node) {
    g_ort->ReleaseSession(sess);
    sqlite3_mutex_leave(mx);
    sqlite3_free(key);
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    return SQLITE_NOMEM;
  }
  node->key = key;
  node->session = sess;
  node->next = g_cache;
  g_cache = node;
  *out = sess;
  sqlite3_mutex_leave(mx);
  return SQLITE_OK;
}

/* ---- license gate ---- */

/* A permissive license runs freely; anything else requires the caller to
 * name it in accept_license, so a non-commercial model (TabFM) cannot be
 * used by accident. */
static int license_ok(const char *license, const char *accepted) {
  if (!license)
    return 0;
  if (strcmp(license, "MIT") == 0 || strcmp(license, "Apache-2.0") == 0 ||
      strcmp(license, "MIT OR Apache-2.0") == 0 ||
      strcmp(license, "BSD-3-Clause") == 0 || strcmp(license, "CC0-1.0") == 0)
    return 1;
  return accepted && strcmp(accepted, license) == 0;
}

/* ---- one forward pass ---- */

/* Run nbatch feature rows through the session and write predictions back
 * into rows[batch_row[b]]. Owns every ORT handle it creates and releases
 * them on all paths. Returns SQLITE_OK or sets *errmsg. */
static int run_batch(OrtSession *session, OrtMemoryInfo *mem, const onnx_io *io,
                     int classify, const f32 *batch, int nbatch,
                     predict0_result *rows, const int *batch_row,
                     char **errmsg) {
  if (nbatch == 0)
    return SQLITE_OK;
  int rc = SQLITE_OK;
  OrtValue *input = NULL, *output = NULL;
  OrtTensorTypeAndShapeInfo *ti = NULL;

  int64_t shape[2] = {nbatch, io->nfeat};
  OrtStatus *st = g_ort->CreateTensorWithDataAsOrtValue(
      mem, (void *)batch, sizeof(f32) * (size_t)nbatch * io->nfeat, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateTensor", errmsg);
    goto out;
  }
  const char *in_names[1] = {io->input_name};
  const char *out_names[1] = {io->output_name};
  st = g_ort->Run(session, NULL, in_names, (const OrtValue *const *)&input, 1,
                  out_names, 1, &output);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "Run", errmsg);
    goto out;
  }

  size_t ndim = 0;
  int64_t odims[8];
  float *odata = NULL;
  st = g_ort->GetTensorTypeAndShape(output, &ti);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetTensorTypeAndShape", errmsg);
    goto out;
  }
  st = g_ort->GetDimensionsCount(ti, &ndim);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetDimensionsCount", errmsg);
    goto out;
  }
  if (ndim < 1 || ndim > 8) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: output has %zu dims (expected 1-2)",
                              PREDICT_ERR_INFERENCE, ndim);
    goto out;
  }
  st = g_ort->GetDimensions(ti, odims, ndim);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetDimensions", errmsg);
    goto out;
  }
  int width = ndim == 1 ? 1 : (int)odims[ndim - 1];
  st = g_ort->GetTensorMutableData(output, (void **)&odata);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetTensorMutableData", errmsg);
    goto out;
  }
  if (classify && width != io->nlabels) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: output width %d does not match %d labels",
                              PREDICT_ERR_INFERENCE, width, io->nlabels);
    goto out;
  }

  for (int b = 0; b < nbatch; b++) {
    predict0_result *out = &rows[batch_row[b]];
    float *v = &odata[(size_t)b * width];
    if (classify) {
      int arg = 0;
      for (int c = 1; c < width; c++)
        if (v[c] > v[arg])
          arg = c;
      f64 conf;
      if (strcmp(io->output_kind, "logits") == 0) {
        f64 sum = 0, mx = v[arg];
        for (int c = 0; c < width; c++)
          sum += exp((f64)v[c] - mx);
        conf = 1.0 / sum; /* softmax value at the argmax */
      } else {
        conf = (f64)v[arg]; /* already a probability */
      }
      out->prediction = sqlite3_mprintf("%s", io->labels[arg]);
      out->confidence = conf;
      out->has_conf = 1;
    } else {
      out->prediction = sqlite3_mprintf("%.17g", (f64)v[0]);
    }
    if (!out->prediction) {
      rc = SQLITE_NOMEM;
      *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
      goto out;
    }
    out->status = "ok";
  }

out:
  if (ti)
    g_ort->ReleaseTensorTypeAndShapeInfo(ti);
  if (output)
    g_ort->ReleaseValue(output);
  if (input)
    g_ort->ReleaseValue(input);
  return rc;
}

/* ---- the vector inference path ---- */

int predict0_onnx_predict_vector(sqlite3 *db, const char *model_id,
                                 const char *apply_sql,
                                 const predict0_model_row *model,
                                 const predict0_backend_opts *opts,
                                 predict0_result **out_rows, int *out_n,
                                 char receipt_id_out[PREDICT_ULID_BUFSIZE],
                                 char **errmsg) {
  *out_rows = NULL;
  *out_n = 0;
  *errmsg = NULL;

  int rc = onnx_init(errmsg);
  if (rc != SQLITE_OK)
    return rc;

  const char *precision = opts->precision ? opts->precision : "fp32";
  if (strcmp(precision, "fp32") != 0) {
    *errmsg = sqlite3_mprintf(
        "%s: precision '%s' is not in this build (fp32 only; fp16/int8 land"
        " with the GPU path)",
        PREDICT_ERR_RUNTIME_UNAVAILABLE, precision);
    return SQLITE_ERROR;
  }
  if (!license_ok(model->license, opts->accept_license)) {
    *errmsg = sqlite3_mprintf(
        "%s: model license '%s' requires accept_license to match",
        PREDICT_ERR_LICENSE, model->license ? model->license : "(unknown)");
    return SQLITE_ERROR;
  }

  onnx_io io;
  rc = onnx_io_parse(db, model->io_spec, &io, errmsg);
  if (rc != SQLITE_OK)
    return rc;
  int classify = strcmp(io.output_kind, "value") != 0;

  OrtSession *session = NULL;
  rc = onnx_get_session(model, opts, &session, errmsg);
  if (rc != SQLITE_OK) {
    onnx_io_free(&io);
    return rc;
  }

  /* every heap resource is NULL/0 here, so the single cleanup path is safe
   * from any failure below */
  predict0_result *rows = NULL;
  int nrows = 0, rcap = 0;
  f32 *batch = sqlite3_malloc(sizeof(f32) * ONNX_BATCH * io.nfeat);
  int *batch_row = sqlite3_malloc(sizeof(int) * ONNX_BATCH);
  int *amap = NULL;
  OrtMemoryInfo *mem = NULL;
  sqlite3_stmt *as = NULL;

  if (!batch || !batch_row) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }
  {
    OrtStatus *st = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator,
                                               OrtMemTypeDefault, &mem);
    if (st) {
      rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateCpuMemoryInfo", errmsg);
      goto done;
    }
  }

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
  if (an < 2) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: apply_query needs a row_ref column plus features",
        PREDICT_ERR_SCHEMA);
    goto done;
  }
  if (an - 1 != io.nfeat) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: apply features (%d) must match the model's (%d) exactly",
        PREDICT_ERR_SCHEMA, an - 1, io.nfeat);
    goto done;
  }
  /* map each apply feature column to a model feature slot (exact set) */
  amap = sqlite3_malloc(sizeof(int) * (an - 1));
  if (!amap) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }
  for (int i = 1; i < an; i++) {
    const char *nm = sqlite3_column_name(as, i);
    int found = -1;
    for (int f = 0; f < io.nfeat; f++)
      if (nm && strcmp(nm, io.features[f]) == 0)
        found = f;
    if (found < 0) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: apply column '%s' is not a model feature", PREDICT_ERR_SCHEMA,
          nm ? nm : "?");
      goto done;
    }
    amap[i - 1] = found;
  }

  int nbatch = 0;
  for (;;) {
    int step = sqlite3_step(as);
    if (step == SQLITE_ROW) {
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
        out->ref_t =
            sqlite3_mprintf("%s", (const char *)sqlite3_column_text(as, 0));

      f32 *slot = &batch[(size_t)nbatch * io.nfeat];
      int bad = 0;
      for (int i = 1; i < an; i++) {
        int ct = sqlite3_column_type(as, i);
        if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
          bad = 1;
          break;
        }
        slot[amap[i - 1]] = (f32)sqlite3_column_double(as, i);
      }
      if (bad) {
        out->status = "non_numeric"; /* not fed to the model */
        nrows++;
        continue;
      }
      batch_row[nbatch] = nrows;
      nbatch++;
      nrows++;
      if (nbatch < ONNX_BATCH)
        continue;
    } else if (step != SQLITE_DONE) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: apply query failed: %s",
                                PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
      goto done;
    }

    rc = run_batch(session, mem, &io, classify, batch, nbatch, rows, batch_row,
                   errmsg);
    if (rc != SQLITE_OK)
      goto done;
    nbatch = 0;
    if (step == SQLITE_DONE)
      break;
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

    /* params pin the execution provider + precision, so a receipt from a
     * different device is distinguishable and replay reproduces the exact
     * call. params must contain only reproducible predict() options: the
     * determinism annotation is a receipt column added with the GPU path,
     * not a call option. CPU-fp32 is deterministic, so replay is exact. */
    char *params = NULL, *input_json = NULL;
    sqlite3_stmt *pj = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_object('model', ?1, 'device', ?2, 'precision', ?3,"
            " 'receipt', 1),"
            " json_object('apply', ?4)",
            -1, &pj, NULL) == SQLITE_OK) {
      sqlite3_bind_text(pj, 1, model_id, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 2, opts->device ? opts->device : "cpu", -1,
                        SQLITE_STATIC);
      sqlite3_bind_text(pj, 3, precision, -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 4, apply_sql, -1, SQLITE_STATIC);
      if (sqlite3_step(pj) == SQLITE_ROW) {
        params =
            sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 0));
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
  sqlite3_free(batch);
  sqlite3_free(batch_row);
  if (mem)
    g_ort->ReleaseMemoryInfo(mem);
  onnx_io_free(&io);
  if (rc == SQLITE_OK) {
    *out_rows = rows;
    *out_n = nrows;
  } else {
    predict0_results_free(rows, nrows);
  }
  return rc;
}

#endif /* SQLITE_PREDICT_ONNX */
