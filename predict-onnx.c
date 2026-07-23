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

/* ---- io_spec introspection (derive an io_spec from the model) ---- */

/* The last dimension of input/output `idx`, or -1 if dynamic/unknown; sets
 * *elem_type to the tensor element type. */
/* Non-zero if the status is an error; always consumes it. */
static int st_bad(OrtStatus *st) {
  if (st) {
    g_ort->ReleaseStatus(st);
    return 1;
  }
  return 0;
}

static int64_t tensor_last_dim(OrtSession *s, int is_input, size_t idx,
                               ONNXTensorElementDataType *elem_type) {
  OrtTypeInfo *ti = NULL;
  int64_t last = -1;
  *elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (st_bad(is_input ? g_ort->SessionGetInputTypeInfo(s, idx, &ti)
                      : g_ort->SessionGetOutputTypeInfo(s, idx, &ti)))
    return -1;
  const OrtTensorTypeAndShapeInfo *tsi = NULL;
  if (!st_bad(g_ort->CastTypeInfoToTensorInfo(ti, &tsi)) && tsi) {
    size_t ndim = 0;
    if (!st_bad(g_ort->GetTensorElementType(tsi, elem_type)) &&
        !st_bad(g_ort->GetDimensionsCount(tsi, &ndim)) && ndim >= 1 &&
        ndim <= 8) {
      int64_t dims[8];
      if (!st_bad(g_ort->GetDimensions(tsi, dims, ndim)))
        last = dims[ndim - 1];
    }
  }
  g_ort->ReleaseTypeInfo(ti);
  return last;
}

int predict0_onnx_introspect(sqlite3 *db, const char *weights_uri,
                             char **io_spec_out, char **errmsg) {
  *io_spec_out = NULL;
  int rc = onnx_init(errmsg);
  if (rc != SQLITE_OK)
    return rc;

  OrtSessionOptions *so = NULL;
  OrtSession *sess = NULL;
  OrtAllocator *alloc = NULL;
  char *in_names[3] = {NULL, NULL, NULL};
  char *out_name = NULL;
  sqlite3_stmt *bld = NULL;

  OrtStatus *st = g_ort->CreateSessionOptions(&so);
  if (st)
    return onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateSessionOptions", errmsg);
  st = g_ort->CreateSession(g_env, weights_uri, so, &sess);
  g_ort->ReleaseSessionOptions(so);
  if (st)
    return onnx_fail(st, PREDICT_ERR_INFERENCE,
                     "cannot open model for introspection", errmsg);

#define INTRO_FAIL(...)                                                        \
  do {                                                                        \
    *errmsg = sqlite3_mprintf(__VA_ARGS__);                                   \
    rc = SQLITE_ERROR;                                                        \
    goto done;                                                                \
  } while (0)

  size_t n_in = 0, n_out = 0;
  if (st_bad(g_ort->GetAllocatorWithDefaultOptions(&alloc)) ||
      st_bad(g_ort->SessionGetInputCount(sess, &n_in)) ||
      st_bad(g_ort->SessionGetOutputCount(sess, &n_out)))
    INTRO_FAIL("%s: could not read model metadata", PREDICT_ERR_IO_SPEC);

  if (n_out != 1)
    INTRO_FAIL("%s: model has %zu outputs; provide an explicit io_spec with"
               " output.name",
               PREDICT_ERR_IO_SPEC, n_out);
  if (n_in != 1 && n_in != 3)
    INTRO_FAIL("%s: cannot derive io_spec for %zu inputs; provide an explicit"
               " io_spec",
               PREDICT_ERR_IO_SPEC, n_in);

  for (size_t i = 0; i < n_in; i++) {
    char *nm = NULL;
    if (g_ort->SessionGetInputName(sess, i, alloc, &nm) || !nm)
      INTRO_FAIL("%s: could not read input name %zu", PREDICT_ERR_IO_SPEC, i);
    in_names[i] = sqlite3_mprintf("%s", nm);
    alloc->Free(alloc, nm);
  }
  {
    char *nm = NULL;
    if (g_ort->SessionGetOutputName(sess, 0, alloc, &nm) || !nm)
      INTRO_FAIL("%s: could not read output name", PREDICT_ERR_IO_SPEC);
    out_name = sqlite3_mprintf("%s", nm);
    alloc->Free(alloc, nm);
  }

  /* output shape/type -> kind */
  ONNXTensorElementDataType out_type;
  int64_t out_w = tensor_last_dim(sess, 0, 0, &out_type);
  if (out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    INTRO_FAIL("%s: output is not a float tensor; provide an explicit io_spec",
               PREDICT_ERR_IO_SPEC);
  int classify = out_w > 1;   /* [N,K>1] -> probs; [N,1]/[N] -> value */
  int nlabels = classify ? (int)out_w : 0;

  /* feature count from the query input's last dim (may be dynamic = -1) */
  ONNXTensorElementDataType in_type;
  size_t q_idx = 0; /* vector: the only input; in_context: x_query */
  int layout_incontext = (n_in == 3);
  if (layout_incontext) {
    /* need inputs named x_train / y_train / x_query to map them */
    int has_xt = 0, has_yt = 0, has_xq = 0;
    for (int i = 0; i < 3; i++) {
      if (strcmp(in_names[i], "x_train") == 0)
        has_xt = 1;
      else if (strcmp(in_names[i], "y_train") == 0)
        has_yt = 1;
      else if (strcmp(in_names[i], "x_query") == 0) {
        has_xq = 1;
        q_idx = i;
      }
    }
    if (!has_xt || !has_yt || !has_xq)
      INTRO_FAIL("%s: 3-input model must name its inputs x_train/y_train/"
                 "x_query to auto-derive; provide an explicit io_spec",
                 PREDICT_ERR_IO_SPEC);
  }
  int64_t nfeat = tensor_last_dim(sess, 1, q_idx, &in_type);

  /* build the io_spec JSON via json_object (proper escaping of names) */
  char *labels_json = NULL;
  if (classify) {
    sqlite3_str *ls = sqlite3_str_new(db);
    sqlite3_str_appendchar(ls, 1, '[');
    for (int k = 0; k < nlabels; k++)
      sqlite3_str_appendf(ls, "%s\"%d\"", k ? "," : "", k);
    sqlite3_str_appendchar(ls, 1, ']');
    labels_json = sqlite3_str_finish(ls);
  }

  const char *sql =
      layout_incontext
          ? "SELECT json_object('layout','in_context','nfeatures',?1,"
            "'inputs',json_object('x_train','x_train','y_train','y_train',"
            "'x_query','x_query'),'output',"
            "CASE WHEN ?5 THEN json_object('name',?2,'kind','probs','labels',"
            "json(?4)) ELSE json_object('name',?2,'kind','value') END)"
          : "SELECT json_object('layout','vector','nfeatures',?1,'input',?3,"
            "'output',CASE WHEN ?5 THEN json_object('name',?2,'kind','probs',"
            "'labels',json(?4)) ELSE json_object('name',?2,'kind','value')"
            " END)";
  if (sqlite3_prepare_v2(db, sql, -1, &bld, NULL) != SQLITE_OK)
    INTRO_FAIL("%s: could not build io_spec", PREDICT_ERR_IO_SPEC);
  if (nfeat > 0)
    sqlite3_bind_int64(bld, 1, nfeat);
  else
    sqlite3_bind_null(bld, 1); /* dynamic feature dim: validated at run time */
  sqlite3_bind_text(bld, 2, out_name, -1, SQLITE_STATIC);
  sqlite3_bind_text(bld, 3, in_names[0], -1, SQLITE_STATIC);
  if (labels_json)
    sqlite3_bind_text(bld, 4, labels_json, -1, SQLITE_STATIC);
  else
    sqlite3_bind_null(bld, 4);
  sqlite3_bind_int(bld, 5, classify);
  if (sqlite3_step(bld) == SQLITE_ROW)
    *io_spec_out =
        sqlite3_mprintf("%s", (const char *)sqlite3_column_text(bld, 0));
  sqlite3_free(labels_json);
  if (!*io_spec_out)
    INTRO_FAIL("%s: could not serialize io_spec", PREDICT_ERR_IO_SPEC);
  rc = SQLITE_OK;

#undef INTRO_FAIL
done:
  if (bld)
    sqlite3_finalize(bld);
  for (int i = 0; i < 3; i++)
    sqlite3_free(in_names[i]);
  sqlite3_free(out_name);
  if (sess)
    g_ort->ReleaseSession(sess);
  return rc;
}

/* ---- io_spec parsing ---- */

typedef enum { LAYOUT_VECTOR, LAYOUT_INCONTEXT } onnx_layout;

typedef struct {
  onnx_layout layout;
  /* vector: the single feature input. in_context: the x_query input. */
  char *input_name;
  /* in_context only: the training-context inputs and the train y column */
  char *x_train_name;
  char *y_train_name;
  char *target_name;
  char **features; /* feature order (optional); empty => positional mapping */
  int nfeat;       /* count of named features (0 => positional) */
  int nfeatures;   /* expected feature count for validation (0 => unknown) */
  char *output_name;
  char *output_kind; /* 'probs' | 'logits' | 'label' | 'value' */
  char **labels;
  int nlabels;
} onnx_io;

static void onnx_io_free(onnx_io *io) {
  sqlite3_free(io->input_name);
  sqlite3_free(io->x_train_name);
  sqlite3_free(io->y_train_name);
  sqlite3_free(io->target_name);
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
  int is_vector = layout && strcmp(layout, "vector") == 0;
  int is_incontext = layout && strcmp(layout, "in_context") == 0;
  if (!is_vector && !is_incontext) {
    *errmsg = sqlite3_mprintf(
        "%s: io_spec layout must be 'vector' or 'in_context'; got '%s'",
        PREDICT_ERR_IO_SPEC, layout ? layout : "(none)");
    sqlite3_free(layout);
    return SQLITE_ERROR;
  }
  sqlite3_free(layout);
  io->layout = is_incontext ? LAYOUT_INCONTEXT : LAYOUT_VECTOR;

  /* output + feature schema are common to both layouts. features[] is
   * optional: when absent the backend maps apply columns positionally, and
   * nfeatures (if given) is the count to validate against. */
  io->output_name = json_str(db, io_spec, "$.output.name");
  io->output_kind = json_str(db, io_spec, "$.output.kind");
  json_arr(db, io_spec, "$.features", &io->features, &io->nfeat);
  json_arr(db, io_spec, "$.output.labels", &io->labels, &io->nlabels);
  char *nf = json_str(db, io_spec, "$.nfeatures");
  io->nfeatures = nf ? atoi(nf) : io->nfeat; /* names imply their own count */
  sqlite3_free(nf);

  int inputs_ok;
  if (io->layout == LAYOUT_VECTOR) {
    io->input_name = json_str(db, io_spec, "$.input");
    inputs_ok = io->input_name != NULL;
  } else {
    /* the model ingests the training rows as context each call */
    io->input_name = json_str(db, io_spec, "$.inputs.x_query");
    io->x_train_name = json_str(db, io_spec, "$.inputs.x_train");
    io->y_train_name = json_str(db, io_spec, "$.inputs.y_train");
    io->target_name = json_str(db, io_spec, "$.target");
    inputs_ok = io->input_name && io->x_train_name && io->y_train_name &&
                io->target_name;
  }

  int have_out = io->output_name && io->output_kind;
  int classify = io->output_kind &&
                 (strcmp(io->output_kind, "probs") == 0 ||
                  strcmp(io->output_kind, "logits") == 0 ||
                  strcmp(io->output_kind, "label") == 0);
  int regress = io->output_kind && strcmp(io->output_kind, "value") == 0;
  if (!inputs_ok || !have_out || (!classify && !regress) ||
      (classify && io->nlabels == 0)) {
    *errmsg = sqlite3_mprintf(
        "%s: io_spec needs output.name/kind"
        " ('probs'|'logits'|'label'|'value'), labels[] for classifiers, and"
        " the input names for its layout (vector: input; in_context:"
        " inputs.x_train/y_train/x_query + target). features[] is optional"
        " (positional mapping when omitted)",
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
    /* Real provider-options wiring. Appending fails loud if the EP is not in
     * this onnxruntime build (i.e. onnxruntime-gpu is required). Compiled
     * against the CPU headers for the CI compile-check; exercised for real on
     * the gated GPU job. */
    int fp16 = opts->precision && strcmp(opts->precision, "fp16") == 0;
    if (strcmp(device, "cuda") == 0) {
      OrtCUDAProviderOptionsV2 *cu = NULL;
      BUILD_CHECK(g_ort->CreateCUDAProviderOptions(&cu),
                  PREDICT_ERR_RUNTIME_UNAVAILABLE, "CreateCUDAProviderOptions");
      /* CUDA EP runs the model's own dtype; fp16 comes from an fp16 model,
       * not an EP flag. precision is recorded in the receipt regardless. */
      OrtStatus *ap =
          g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(so, cu);
      g_ort->ReleaseCUDAProviderOptions(cu);
      BUILD_CHECK(ap, PREDICT_ERR_RUNTIME_UNAVAILABLE, "append CUDA EP");
    } else {
      OrtTensorRTProviderOptionsV2 *trt = NULL;
      BUILD_CHECK(g_ort->CreateTensorRTProviderOptions(&trt),
                  PREDICT_ERR_RUNTIME_UNAVAILABLE,
                  "CreateTensorRTProviderOptions");
      if (fp16) {
        const char *keys[] = {"trt_fp16_enable"};
        const char *vals[] = {"1"};
        OrtStatus *up =
            g_ort->UpdateTensorRTProviderOptions(trt, keys, vals, 1);
        if (up) {
          g_ort->ReleaseTensorRTProviderOptions(trt);
          rc = onnx_fail(up, PREDICT_ERR_RUNTIME_UNAVAILABLE,
                         "trt_fp16_enable", errmsg);
          goto fail;
        }
      }
      OrtStatus *ap =
          g_ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(so, trt);
      g_ort->ReleaseTensorRTProviderOptions(trt);
      BUILD_CHECK(ap, PREDICT_ERR_RUNTIME_UNAVAILABLE, "append TensorRT EP");
    }
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
  /* 'unspecified' is the default when a caller registers their own model
   * without a license tag: they vouch for it, so it runs. The gate exists to
   * catch redistributed non-commercial weights (e.g. TabFM), which carry an
   * explicit restrictive tag and need accept_license to match. */
  if (strcmp(license, "unspecified") == 0 || strcmp(license, "MIT") == 0 ||
      strcmp(license, "Apache-2.0") == 0 ||
      strcmp(license, "MIT OR Apache-2.0") == 0 ||
      strcmp(license, "BSD-3-Clause") == 0 || strcmp(license, "CC0-1.0") == 0)
    return 1;
  return accepted && strcmp(accepted, license) == 0;
}

/* ---- output decoding + receipts (shared by both layouts) ---- */

/* Decode a [nbatch, width] float output tensor into predictions on
 * rows[batch_row[b]]. Does not take ownership of `output`. */
static int decode_output(OrtValue *output, const onnx_io *io, int classify,
                         predict0_result *rows, const int *batch_row,
                         int nbatch, char **errmsg) {
  int rc = SQLITE_OK;
  OrtTensorTypeAndShapeInfo *ti = NULL;
  size_t ndim = 0;
  int64_t odims[8];
  float *odata = NULL;

  OrtStatus *st = g_ort->GetTensorTypeAndShape(output, &ti);
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
  return rc;
}

/* Vector single-input forward pass: nbatch rows of F features -> preds. */
static int run_batch(OrtSession *session, OrtMemoryInfo *mem, const onnx_io *io,
                     int F, int classify, const f32 *batch, int nbatch,
                     predict0_result *rows, const int *batch_row,
                     char **errmsg) {
  if (nbatch == 0)
    return SQLITE_OK;
  int rc = SQLITE_OK;
  OrtValue *input = NULL, *output = NULL;
  int64_t shape[2] = {nbatch, F};
  OrtStatus *st = g_ort->CreateTensorWithDataAsOrtValue(
      mem, (void *)batch, sizeof(f32) * (size_t)nbatch * F, shape, 2,
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
  rc = decode_output(output, io, classify, rows, batch_row, nbatch, errmsg);

out:
  if (output)
    g_ort->ReleaseValue(output);
  if (input)
    g_ort->ReleaseValue(input);
  return rc;
}

/* Hash results canonically (ref, prediction, confidence per row) and write
 * a receipt. input_json is {"train","apply"} when train_sql is given, else
 * {"apply"}; params pin the model, device, and precision. Shared by the
 * vector and in-context paths. */
static int emit_predict_receipt(sqlite3 *db, const char *model_id,
                                const predict0_backend_opts *opts,
                                predict0_result *rows, int nrows,
                                const char *train_sql, const char *apply_sql,
                                char receipt_id_out[PREDICT_ULID_BUFSIZE],
                                char **errmsg) {
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

  const char *precision = opts->precision ? opts->precision : "fp32";
  const char *device = opts->device ? opts->device : "cpu";
  char *params = NULL, *input_json = NULL;
  sqlite3_stmt *pj = NULL;
  const char *sql =
      train_sql ? "SELECT json_object('model', ?1, 'device', ?2, 'precision',"
                  " ?3, 'receipt', 1), json_object('train', ?4, 'apply', ?5)"
                : "SELECT json_object('model', ?1, 'device', ?2, 'precision',"
                  " ?3, 'receipt', 1), json_object('apply', ?5)";
  if (sqlite3_prepare_v2(db, sql, -1, &pj, NULL) == SQLITE_OK) {
    sqlite3_bind_text(pj, 1, model_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(pj, 2, device, -1, SQLITE_STATIC);
    sqlite3_bind_text(pj, 3, precision, -1, SQLITE_STATIC);
    if (train_sql)
      sqlite3_bind_text(pj, 4, train_sql, -1, SQLITE_STATIC);
    sqlite3_bind_text(pj, 5, apply_sql, -1, SQLITE_STATIC);
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
    *errmsg = sqlite3_mprintf("%s: could not canonicalize params",
                              PREDICT_ERR_RESOURCE);
    return SQLITE_ERROR;
  }
  char *rerr = NULL;
  int rc = predict0_emit_receipt(db, "predict", model_id, params, input_json,
                                 result_hash, receipt_id_out, &rerr);
  sqlite3_free(params);
  sqlite3_free(input_json);
  if (rc != SQLITE_OK)
    *errmsg = rerr ? rerr
                   : sqlite3_mprintf("%s: receipt write failed",
                                     PREDICT_ERR_RESOURCE);
  return rc;
}

/* Map the apply query's feature columns (1..an-1) to model feature slots.
 * With names in io->features, match by name (an exact set). Without names,
 * map positionally (apply column order = model feature order). Sets *F, the
 * tensor feature dimension, and fills amap[0..an-2]. Validates the count
 * against io->nfeatures when it is known. */
static int map_apply_features(const onnx_io *io, sqlite3_stmt *as, int an,
                              int *amap, int *F, char **errmsg) {
  int expected = io->nfeat > 0 ? io->nfeat : io->nfeatures; /* 0 = unknown */
  if (expected > 0 && an - 1 != expected) {
    *errmsg = sqlite3_mprintf(
        "%s: apply features (%d) must match the model's (%d) exactly",
        PREDICT_ERR_SCHEMA, an - 1, expected);
    return SQLITE_ERROR;
  }
  if (io->nfeat > 0) {
    for (int i = 1; i < an; i++) {
      const char *nm = sqlite3_column_name(as, i);
      int found = -1;
      for (int f = 0; f < io->nfeat; f++)
        if (nm && strcmp(nm, io->features[f]) == 0)
          found = f;
      if (found < 0) {
        *errmsg = sqlite3_mprintf(
            "%s: apply column '%s' is not a model feature", PREDICT_ERR_SCHEMA,
            nm ? nm : "?");
        return SQLITE_ERROR;
      }
      amap[i - 1] = found;
    }
    *F = io->nfeat;
  } else {
    for (int i = 1; i < an; i++)
      amap[i - 1] = i - 1; /* positional */
    *F = an - 1;
  }
  return SQLITE_OK;
}

/* ---- the vector inference path ---- */

/* Self-contained model: features -> prediction, train_sql unused. io and
 * session are owned by the router. */
static int onnx_vector(sqlite3 *db, const char *model_id, const char *apply_sql,
                       const predict0_backend_opts *opts, const onnx_io *io,
                       OrtSession *session, predict0_result **out_rows,
                       int *out_n, char receipt_id_out[PREDICT_ULID_BUFSIZE],
                       char **errmsg) {
  int classify = strcmp(io->output_kind, "value") != 0;
  int rc = SQLITE_OK;

  /* every heap resource is NULL/0 here, so the single cleanup path is safe
   * from any failure below */
  predict0_result *rows = NULL;
  int nrows = 0, rcap = 0;
  f32 *batch = NULL;
  int *batch_row = NULL;
  int *amap = NULL;
  int F = 0;
  OrtMemoryInfo *mem = NULL;
  sqlite3_stmt *as = NULL;

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
  amap = sqlite3_malloc(sizeof(int) * (an - 1));
  batch_row = sqlite3_malloc(sizeof(int) * ONNX_BATCH);
  if (!amap || !batch_row) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }
  rc = map_apply_features(io, as, an, amap, &F, errmsg);
  if (rc != SQLITE_OK)
    goto done;
  batch = sqlite3_malloc(sizeof(f32) * ONNX_BATCH * F);
  if (!batch) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
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

      f32 *slot = &batch[(size_t)nbatch * F];
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

    rc = run_batch(session, mem, io, F, classify, batch, nbatch, rows,
                   batch_row, errmsg);
    if (rc != SQLITE_OK)
      goto done;
    nbatch = 0;
    if (step == SQLITE_DONE)
      break;
  }

  if (opts->receipt) {
    rc = emit_predict_receipt(db, model_id, opts, rows, nrows, NULL, apply_sql,
                              receipt_id_out, errmsg);
    if (rc != SQLITE_OK)
      goto done;
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
  if (rc == SQLITE_OK) {
    *out_rows = rows;
    *out_n = nrows;
  } else {
    predict0_results_free(rows, nrows);
  }
  return rc;
}

/* ---- the in-context inference path (teacher: train rows are context) ---- */

/* Collect the training context: x_train [Nt, F] (feature order per io), and
 * y_train as int64 class indices (classify) or f32 values (regress). */
static int collect_train(sqlite3 *db, const char *train_sql, const onnx_io *io,
                         int classify, f32 **x_out, void **y_out, int *nt_out,
                         int *F_out, char **errmsg) {
  *x_out = NULL;
  *y_out = NULL;
  *nt_out = 0;
  *F_out = 0;
  sqlite3_stmt *ts = NULL;
  int rc = SQLITE_OK;
  f32 *X = NULL;
  i64 *yi = NULL;
  f32 *yf = NULL;
  int nt = 0, cap = 0, F = 0;

  if (!train_sql) {
    *errmsg = sqlite3_mprintf(
        "%s: in-context models require a train_query", PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  if (sqlite3_prepare_v2(db, train_sql, -1, &ts, NULL) != SQLITE_OK || !ts) {
    *errmsg = sqlite3_mprintf("%s: train query does not parse: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
    return SQLITE_ERROR;
  }
  if (!sqlite3_stmt_readonly(ts)) {
    sqlite3_finalize(ts);
    *errmsg = sqlite3_mprintf("%s: train query must be a read-only SELECT",
                              PREDICT_ERR_QUERY_NOT_READONLY);
    return SQLITE_ERROR;
  }
  int tn = sqlite3_column_count(ts);
  int target_idx = -1;
  int *fmap = sqlite3_malloc(sizeof(int) * tn); /* train col -> feature slot */
  if (!fmap) {
    sqlite3_finalize(ts);
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    return SQLITE_NOMEM;
  }
  /* map train columns to feature slots. Named: match io->features. Positional
   * (no names): every non-target column becomes the next feature, in order. */
  int nfeat_seen = 0;
  for (int i = 0; i < tn; i++) {
    const char *nm = sqlite3_column_name(ts, i);
    fmap[i] = -1;
    if (nm && target_idx < 0 && strcmp(nm, io->target_name) == 0) {
      target_idx = i;
      continue;
    }
    if (io->nfeat > 0) {
      for (int f = 0; f < io->nfeat; f++)
        if (nm && strcmp(nm, io->features[f]) == 0) {
          fmap[i] = f;
          nfeat_seen++;
        }
    } else {
      fmap[i] = nfeat_seen++; /* positional slot */
    }
  }
  F = io->nfeat > 0 ? io->nfeat : nfeat_seen;
  int names_ok = io->nfeat == 0 || nfeat_seen == io->nfeat;
  int count_ok = io->nfeatures == 0 || F == io->nfeatures;
  if (target_idx < 0 || F == 0 || !names_ok || !count_ok) {
    sqlite3_free(fmap);
    sqlite3_finalize(ts);
    *errmsg = sqlite3_mprintf(
        "%s: train_query must expose target '%s' and the model's %d features",
        PREDICT_ERR_SCHEMA, io->target_name,
        io->nfeatures ? io->nfeatures : io->nfeat);
    return SQLITE_ERROR;
  }

  int step;
  while ((step = sqlite3_step(ts)) == SQLITE_ROW) {
    if (nt == cap) {
      cap = cap ? cap * 2 : 256;
      f32 *gX = sqlite3_realloc(X, sizeof(f32) * (size_t)cap * F);
      if (!gX) {
        rc = SQLITE_NOMEM;
        goto oom;
      }
      X = gX;
      if (classify) {
        i64 *g = sqlite3_realloc(yi, sizeof(i64) * cap);
        if (!g) {
          rc = SQLITE_NOMEM;
          goto oom;
        }
        yi = g;
      } else {
        f32 *g = sqlite3_realloc(yf, sizeof(f32) * cap);
        if (!g) {
          rc = SQLITE_NOMEM;
          goto oom;
        }
        yf = g;
      }
    }
    f32 *row = &X[(size_t)nt * F];
    for (int i = 0; i < tn; i++) {
      if (fmap[i] < 0)
        continue;
      int ct = sqlite3_column_type(ts, i);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf(
            "%s: train feature '%s' is not numeric", PREDICT_ERR_SCHEMA,
            sqlite3_column_name(ts, i));
        goto fail;
      }
      row[fmap[i]] = (f32)sqlite3_column_double(ts, i);
    }
    if (classify) {
      const char *lab = (const char *)sqlite3_column_text(ts, target_idx);
      int idx = -1;
      for (int k = 0; k < io->nlabels; k++)
        if (lab && strcmp(lab, io->labels[k]) == 0)
          idx = k;
      if (idx < 0) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf(
            "%s: train label '%s' is not in the model's labels[]",
            PREDICT_ERR_IO_SPEC, lab ? lab : "(null)");
        goto fail;
      }
      yi[nt] = idx;
    } else {
      int ct = sqlite3_column_type(ts, target_idx);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        rc = SQLITE_ERROR;
        *errmsg = sqlite3_mprintf("%s: regress target must be numeric",
                                  PREDICT_ERR_TARGET);
        goto fail;
      }
      yf[nt] = (f32)sqlite3_column_double(ts, target_idx);
    }
    nt++;
  }
  if (step != SQLITE_DONE) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: train query failed: %s",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
    goto fail;
  }
  if (nt == 0) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: train context is empty", PREDICT_ERR_SCHEMA);
    goto fail;
  }
  sqlite3_free(fmap);
  sqlite3_finalize(ts);
  *x_out = X;
  *y_out = classify ? (void *)yi : (void *)yf;
  *nt_out = nt;
  *F_out = F;
  return SQLITE_OK;

oom:
  *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
fail:
  sqlite3_free(fmap);
  sqlite3_finalize(ts);
  sqlite3_free(X);
  sqlite3_free(yi);
  sqlite3_free(yf);
  return rc;
}

static int onnx_incontext(sqlite3 *db, const char *model_id,
                          const char *train_sql, const char *apply_sql,
                          const predict0_backend_opts *opts, const onnx_io *io,
                          OrtSession *session, predict0_result **out_rows,
                          int *out_n, char receipt_id_out[PREDICT_ULID_BUFSIZE],
                          char **errmsg) {
  int classify = strcmp(io->output_kind, "value") != 0;
  int rc = SQLITE_OK;

  f32 *xtrain = NULL;
  void *ytrain = NULL;
  int nt = 0, F = 0, train_F = 0;
  predict0_result *rows = NULL;
  int nrows = 0, rcap = 0;
  f32 *xquery = NULL; /* [nq_num, F], only numeric query rows */
  int *qmap = NULL;   /* numeric-query index -> result index */
  int nq = 0, qcap = 0;
  int *amap = NULL;
  OrtMemoryInfo *mem = NULL;
  OrtValue *xt_val = NULL, *yt_val = NULL;
  sqlite3_stmt *as = NULL;

  rc = collect_train(db, train_sql, io, classify, &xtrain, &ytrain, &nt,
                     &train_F, errmsg);
  if (rc != SQLITE_OK)
    return rc;

  /* ---- collect the query rows ---- */
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
  amap = sqlite3_malloc(sizeof(int) * (an - 1));
  if (!amap) {
    rc = SQLITE_NOMEM;
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    goto done;
  }
  rc = map_apply_features(io, as, an, amap, &F, errmsg);
  if (rc != SQLITE_OK)
    goto done;
  if (F != train_F) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf(
        "%s: apply features (%d) must match train features (%d)",
        PREDICT_ERR_SCHEMA, F, train_F);
    goto done;
  }
  if (F > 64) { /* the per-row stack buffer below is fixed at 64 */
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: too many features (%d; max 64)",
                              PREDICT_ERR_SCHEMA, F);
    goto done;
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
      out->ref_t =
          sqlite3_mprintf("%s", (const char *)sqlite3_column_text(as, 0));

    f32 feat[64];
    int bad = 0;
    for (int i = 1; i < an; i++) {
      int ct = sqlite3_column_type(as, i);
      if (ct != SQLITE_INTEGER && ct != SQLITE_FLOAT) {
        bad = 1;
        break;
      }
      feat[amap[i - 1]] = (f32)sqlite3_column_double(as, i);
    }
    if (bad) {
      out->status = "non_numeric";
      nrows++;
      continue;
    }
    if (nq == qcap) {
      qcap = qcap ? qcap * 2 : 256;
      f32 *gx = sqlite3_realloc(xquery, sizeof(f32) * (size_t)qcap * F);
      int *gm = sqlite3_realloc(qmap, sizeof(int) * qcap);
      if (!gx || !gm) {
        sqlite3_free(gx);
        sqlite3_free(gm);
        rc = SQLITE_NOMEM;
        *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
        goto done;
      }
      xquery = gx;
      qmap = gm;
    }
    memcpy(&xquery[(size_t)nq * F], feat, sizeof(f32) * F);
    qmap[nq] = nrows;
    nq++;
    nrows++;
  }
  if (step != SQLITE_DONE) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: apply query failed: %s",
                              PREDICT_ERR_RESOURCE, sqlite3_errmsg(db));
    goto done;
  }

  /* ---- build the context tensors once, run query rows in batches ---- */
  {
    OrtStatus *st = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator,
                                               OrtMemTypeDefault, &mem);
    if (st) {
      rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateCpuMemoryInfo", errmsg);
      goto done;
    }
  }
  int64_t xt_shape[2] = {nt, F};
  OrtStatus *st = g_ort->CreateTensorWithDataAsOrtValue(
      mem, xtrain, sizeof(f32) * (size_t)nt * F, xt_shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xt_val);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateTensor(x_train)", errmsg);
    goto done;
  }
  int64_t yt_shape[1] = {nt};
  st = classify
           ? g_ort->CreateTensorWithDataAsOrtValue(
                 mem, ytrain, sizeof(i64) * (size_t)nt, yt_shape, 1,
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &yt_val)
           : g_ort->CreateTensorWithDataAsOrtValue(
                 mem, ytrain, sizeof(f32) * (size_t)nt, yt_shape, 1,
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &yt_val);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateTensor(y_train)", errmsg);
    goto done;
  }

  for (int start = 0; start < nq; start += ONNX_BATCH) {
    int nb = nq - start < ONNX_BATCH ? nq - start : ONNX_BATCH;
    int64_t xq_shape[2] = {nb, F};
    OrtValue *xq_val = NULL, *output = NULL;
    st = g_ort->CreateTensorWithDataAsOrtValue(
        mem, &xquery[(size_t)start * F], sizeof(f32) * (size_t)nb * F, xq_shape,
        2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xq_val);
    if (st) {
      rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateTensor(x_query)",
                     errmsg);
      goto done;
    }
    const char *in_names[3] = {io->x_train_name, io->y_train_name,
                               io->input_name};
    const OrtValue *inputs[3] = {xt_val, yt_val, xq_val};
    const char *out_names[1] = {io->output_name};
    st = g_ort->Run(session, NULL, in_names, inputs, 3, out_names, 1, &output);
    if (st) {
      g_ort->ReleaseValue(xq_val);
      rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "Run", errmsg);
      goto done;
    }
    rc = decode_output(output, io, classify, rows, &qmap[start], nb, errmsg);
    g_ort->ReleaseValue(output);
    g_ort->ReleaseValue(xq_val);
    if (rc != SQLITE_OK)
      goto done;
  }

  if (opts->receipt) {
    rc = emit_predict_receipt(db, model_id, opts, rows, nrows, train_sql,
                              apply_sql, receipt_id_out, errmsg);
    if (rc != SQLITE_OK)
      goto done;
  }
  rc = SQLITE_OK;

done:
  if (as)
    sqlite3_finalize(as);
  if (xt_val)
    g_ort->ReleaseValue(xt_val);
  if (yt_val)
    g_ort->ReleaseValue(yt_val);
  if (mem)
    g_ort->ReleaseMemoryInfo(mem);
  sqlite3_free(amap);
  sqlite3_free(xtrain);
  sqlite3_free(ytrain);
  sqlite3_free(xquery);
  sqlite3_free(qmap);
  if (rc == SQLITE_OK) {
    *out_rows = rows;
    *out_n = nrows;
  } else {
    predict0_results_free(rows, nrows);
  }
  return rc;
}

/* ---- router: shared setup, then dispatch by io_spec layout ---- */

int predict0_onnx_predict(sqlite3 *db, const char *model_id,
                          const char *train_sql, const char *apply_sql,
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
#ifdef SQLITE_PREDICT_ONNX_GPU
  int prec_ok = strcmp(precision, "fp32") == 0 ||
                strcmp(precision, "fp16") == 0 ||
                strcmp(precision, "int8") == 0;
#else
  int prec_ok = strcmp(precision, "fp32") == 0;
#endif
  if (!prec_ok) {
    *errmsg = sqlite3_mprintf(
        "%s: precision '%s' is not available in this build (fp32 only;"
        " fp16/int8 need the GPU build, loadable-onnx-gpu)",
        PREDICT_ERR_RUNTIME_UNAVAILABLE, precision);
    return SQLITE_ERROR;
  }
  /* fp16/int8 only take effect on a GPU device; pairing them with cpu/coreml
   * would silently compute fp32, so reject it rather than no-op quietly. */
  if (strcmp(precision, "fp32") != 0) {
    const char *dev = opts->device ? opts->device : "cpu";
    if (strcmp(dev, "cuda") != 0 && strcmp(dev, "tensorrt") != 0) {
      *errmsg = sqlite3_mprintf(
          "%s: precision '%s' requires a cuda or tensorrt device", PREDICT_ERR_OPTIONS,
          precision);
      return SQLITE_ERROR;
    }
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
  if (io.nfeat > 64) {
    onnx_io_free(&io);
    *errmsg = sqlite3_mprintf("%s: model declares %d features (max 64)",
                              PREDICT_ERR_IO_SPEC, io.nfeat);
    return SQLITE_ERROR;
  }

  OrtSession *session = NULL;
  rc = onnx_get_session(model, opts, &session, errmsg);
  if (rc != SQLITE_OK) {
    onnx_io_free(&io);
    return rc;
  }

  if (io.layout == LAYOUT_VECTOR)
    rc = onnx_vector(db, model_id, apply_sql, opts, &io, session, out_rows,
                     out_n, receipt_id_out, errmsg);
  else
    rc = onnx_incontext(db, model_id, train_sql, apply_sql, opts, &io, session,
                        out_rows, out_n, receipt_id_out, errmsg);

  onnx_io_free(&io);
  return rc;
}

#endif /* SQLITE_PREDICT_ONNX */
