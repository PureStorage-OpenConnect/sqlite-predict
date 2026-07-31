/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* ONNX runtime backend for predict() — the opt-in foundation-model
 * serving path (make loadable-onnx, -DSQLITE_PREDICT_ONNX). This file is
 * the only one that links onnxruntime; the core build never sees it.
 *
 * Two io_spec layouts are implemented: "vector" (a self-contained,
 * pre-trained model mapping a feature vector to a prediction — a
 * distilled student, or any exported tabular classifier/regressor) and
 * "in_context" (a teacher such as TabFM that ingests the training rows
 * as context on every call). The in_context path is validated against
 * real weights on the gated GPU CI job.
 *
 * Performance shape: the expensive costs are session
 * creation and weight load, so sessions are cached process-global keyed by
 * (weights, device, precision) and reused across calls; query rows are run
 * in batches, not one at a time. Execution-provider selection is explicit
 * and fails loud — a request for a provider this build lacks errors rather
 * than silently dropping to CPU. */
#include "predict-internal.h"
#include "predict-student.h" /* PREDICT0_MAX_CLASS: caps io_spec output labels */

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

/* json boolean at path -> 1 if true, else 0 (JSON true extracts as integer 1). */
static int json_flag(sqlite3 *db, const char *json, const char *path) {
  char *v = json_str(db, json, path);
  int on = v && (strcmp(v, "1") == 0 || strcmp(v, "true") == 0);
  sqlite3_free(v);
  return on;
}

/* json array at path -> sqlite3_malloc'd char*[]; *n set. SQLITE_OK on success.
 * An absent array is not an error: it yields *n == 0, so optional fields read
 * as "not declared". A present-but-malformed array is an error and is never
 * swallowed: the return code and *errmsg (ownership transfers to the caller)
 * always propagate. max caps the element count (0 = unbounded). */
static int json_arr(sqlite3 *db, const char *json, const char *path,
                    char ***out, int *n, int max, char **errmsg) {
  return predict0_json_str_array(db, json, path, out, n, max, errmsg);
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
  /* features[] is optional (absent -> nfeat 0, positional mapping), but a
   * present-but-malformed array must fail loudly rather than read as absent. */
  int rc_features = json_arr(db, io_spec, "$.features", &io->features,
                             &io->nfeat, PREDICT_MAX_FEAT, errmsg);
  if (rc_features != SQLITE_OK) {
    onnx_io_free(io);
    return rc_features;
  }
  /* Labels bound the class space; cap them and surface an oversized set instead
   * of silently truncating. Features use PREDICT_MAX_FEAT and positional
   * mapping. */
  int rc_labels = json_arr(db, io_spec, "$.output.labels", &io->labels,
                           &io->nlabels, PREDICT0_MAX_CLASS, errmsg);
  if (rc_labels != SQLITE_OK) {
    onnx_io_free(io);
    return rc_labels;
  }
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
       * not an EP flag. */
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
    /* Current state: the GPU execution providers are validated on the gated
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
    /* content-address check: the file must still hash to the
     * content_hash recorded at registration. Runs once per session-cache
     * miss, not per call. */
    char hex[PREDICT_HEX_BUFSIZE];
    char *herr = NULL;
    if (predict0_hash_file(model->weights_uri, hex, &herr) != SQLITE_OK) {
      rc = SQLITE_ERROR;
      *errmsg = herr;
      goto fail;
    }
    if (model->content_hash && strcmp(hex, model->content_hash) != 0) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: weights file does not match the registered content_hash: %s",
          PREDICT_ERR_MODEL_HASH, model->weights_uri);
      goto fail;
    }
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

/* ---- output decoding (shared by both layouts) ---- */

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

/* ---- forecast backend (sequence layout: context window -> quantile fan) ---- */

/* Run a two-head "core" sequence model once: input [1, L] -> two fans, each
 * [1, Hg, ncol] where column 0 is the point head and columns 1..nq are the
 * quantiles. Copies both into caller buffers pt_out and qs_out (each malloc'd
 * Hg*ncol, caller frees) and reports Hg. Called once or twice (flip-invariance)
 * by seq_reconstruct. Nothing here is model-specific; the shape (ncol) and how
 * the two fans combine are all declared in the io_spec. */
static int seq_core_decode(OrtSession *session, OrtMemoryInfo *mem,
                           const char *in_name, const char *pt_name,
                           const char *qs_name, const f32 *ctx, int L, int ncol,
                           int need_h, float **pt_out, float **qs_out,
                           int *hg_out, char **errmsg) {
  *pt_out = NULL;
  *qs_out = NULL;
  int rc = SQLITE_OK;
  OrtValue *input = NULL, *outs[2] = {NULL, NULL};
  OrtTensorTypeAndShapeInfo *ti = NULL;
  int64_t shape[2] = {1, L};
  OrtStatus *st = g_ort->CreateTensorWithDataAsOrtValue(
      mem, (void *)ctx, sizeof(f32) * (size_t)L, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateTensor", errmsg);
    goto done;
  }
  const char *in_names[1] = {in_name};
  const char *out_names[2] = {pt_name, qs_name};
  st = g_ort->Run(session, NULL, in_names, (const OrtValue *const *)&input, 1,
                  out_names, 2, outs);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "Run", errmsg);
    goto done;
  }
  int hg = 0;
  for (int o = 0; o < 2; o++) {
    size_t ndim = 0;
    int64_t od[8];
    if (ti) {
      g_ort->ReleaseTensorTypeAndShapeInfo(ti);
      ti = NULL;
    }
    st = g_ort->GetTensorTypeAndShape(outs[o], &ti);
    if (!st)
      st = g_ort->GetDimensionsCount(ti, &ndim);
    if (!st && ndim == 3)
      st = g_ort->GetDimensions(ti, od, ndim);
    if (st) {
      rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetShape", errmsg);
      goto done;
    }
    if (ndim != 3 || od[0] != 1 || (int)od[2] != ncol) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf(
          "%s: two-head core output must be [1, horizon, 1+quantiles=%d]",
          PREDICT_ERR_INFERENCE, ncol);
      goto done;
    }
    if (o == 0)
      hg = (int)od[1];
    else if ((int)od[1] != hg) {
      rc = SQLITE_ERROR;
      *errmsg = sqlite3_mprintf("%s: two-head core outputs disagree on horizon",
                                PREDICT_ERR_INFERENCE);
      goto done;
    }
  }
  if (hg < need_h) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: model forecasts at most %d steps, %d asked",
                              PREDICT_ERR_HORIZON, hg, need_h);
    goto done;
  }
  float *pd = NULL, *qd = NULL;
  st = g_ort->GetTensorMutableData(outs[0], (void **)&pd);
  if (!st)
    st = g_ort->GetTensorMutableData(outs[1], (void **)&qd);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetData", errmsg);
    goto done;
  }
  *pt_out = sqlite3_malloc(sizeof(float) * (size_t)hg * ncol);
  *qs_out = sqlite3_malloc(sizeof(float) * (size_t)hg * ncol);
  if (!*pt_out || !*qs_out) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  memcpy(*pt_out, pd, sizeof(float) * (size_t)hg * ncol);
  memcpy(*qs_out, qd, sizeof(float) * (size_t)hg * ncol);
  *hg_out = hg;

done:
  if (rc != SQLITE_OK) {
    sqlite3_free(*pt_out);
    sqlite3_free(*qs_out);
    *pt_out = NULL;
    *qs_out = NULL;
  }
  if (ti)
    g_ort->ReleaseTensorTypeAndShapeInfo(ti);
  if (outs[0])
    g_ort->ReleaseValue(outs[0]);
  if (outs[1])
    g_ort->ReleaseValue(outs[1]);
  if (input)
    g_ort->ReleaseValue(input);
  return rc;
}

/* Steps a two-head sequence core needs applied outside the ONNX graph, each
 * declared independently in the io_spec (none are model-specific). */
typedef struct {
  int flip_invariance;  /* average forward with reflected+flipped (TTA) */
  int continuous_head;  /* blend: quantile spread recentered on point median */
  int crossing_repair;  /* enforce non-decreasing quantiles per step */
  int denorm_instance;  /* graph emits normalized; denorm with context mean/std */
} seq_post;

/* col c of a reflected fan under flip-invariance: keep the point head (col 0),
 * reverse the quantile columns 1..nq (col c <-> ncol-c). */
#define FAN_FLIP(arr, h, c, ncol)                                              \
  ((c) == 0 ? (arr)[(size_t)(h) * (ncol)]                                      \
            : (arr)[(size_t)(h) * (ncol) + ((ncol) - (c))])

/* Reconstruct a quantile fan from a two-head sequence core by applying the
 * post-processing the ONNX graph could not carry (a second decode for flip-
 * invariance, the continuous-head blend, crossing repair, instance denorm) as
 * declared in `post`. Fills fan[horizon*nq] step-major. Column layout: col 0 is
 * the point head, cols 1..nq the quantiles in io_spec order. This mirrors
 * scripts/export_timesfm_onnx.py reconstruct() but is driven entirely by flags,
 * so any two-head quantile core exported the same way is served, not just one. */
static int seq_reconstruct(OrtSession *session, const char *in_name,
                           const char *pt_name, const char *qs_name,
                           const f64 *ctx, int L, int horizon, int nq,
                           const f32 *levels, const seq_post *post, f64 *fan,
                           char **errmsg) {
  int ncol = nq + 1;
  int med = 0; /* column of the 0.5 quantile, for the continuous-head recenter */
  if (post->continuous_head) {
    f32 best = 2.0f;
    for (int q = 0; q < nq; q++) {
      f32 d = levels[q] < 0.5f ? 0.5f - levels[q] : levels[q] - 0.5f;
      if (d < best) {
        best = d;
        med = q + 1;
      }
    }
  }
  int rc = SQLITE_OK, hg = 0, hg2 = 0;
  f32 *cf = sqlite3_malloc(sizeof(f32) * (size_t)L);
  f32 *rf = sqlite3_malloc(sizeof(f32) * (size_t)L);
  f64 *ff = sqlite3_malloc(sizeof(f64) * (size_t)ncol);
  f64 *qs = sqlite3_malloc(sizeof(f64) * (size_t)ncol);
  f64 *out = sqlite3_malloc(sizeof(f64) * (size_t)ncol);
  float *a1 = NULL, *b1 = NULL, *a2 = NULL, *b2 = NULL;
  OrtMemoryInfo *mem = NULL;
  if (!cf || !rf || !ff || !qs || !out) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  f64 mu = 0;
  for (int i = 0; i < L; i++)
    mu += ctx[i];
  mu /= L;
  f64 var = 0;
  for (int i = 0; i < L; i++)
    var += (ctx[i] - mu) * (ctx[i] - mu);
  f64 sigma = L > 1 ? sqrt(var / (L - 1)) : 0.0; /* unbiased std, matches torch */
  if (sigma < 1e-6)
    sigma = 1.0;
  for (int i = 0; i < L; i++) {
    cf[i] = (f32)ctx[i];
    rf[i] = (f32)(2.0 * mu - ctx[i]); /* reflection -> internal -x */
  }
  OrtStatus *st =
      g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateCpuMemoryInfo", errmsg);
    goto done;
  }
  if ((rc = seq_core_decode(session, mem, in_name, pt_name, qs_name, cf, L, ncol,
                            horizon, &a1, &b1, &hg, errmsg)) != SQLITE_OK)
    goto done;
  if (post->flip_invariance &&
      (rc = seq_core_decode(session, mem, in_name, pt_name, qs_name, rf, L, ncol,
                            horizon, &a2, &b2, &hg2, errmsg)) != SQLITE_OK)
    goto done;
  for (int h = 0; h < horizon; h++) {
    for (int c = 0; c < ncol; c++) {
      if (post->flip_invariance) {
        ff[c] = ((f64)a1[(size_t)h * ncol + c] - (f64)FAN_FLIP(a2, h, c, ncol)) / 2.0;
        qs[c] = ((f64)b1[(size_t)h * ncol + c] - (f64)FAN_FLIP(b2, h, c, ncol)) / 2.0;
      } else {
        ff[c] = a1[(size_t)h * ncol + c];
        qs[c] = b1[(size_t)h * ncol + c];
      }
    }
    for (int c = 0; c < ncol; c++) /* default: serve the quantile head as-is */
      out[c] = qs[c];
    if (post->continuous_head)
      for (int c = 1; c <= nq; c++) /* recenter the spread on the point median */
        out[c] = qs[c] - qs[med] + ff[med];
    if (post->crossing_repair) {
      for (int c = med - 1; c >= 1; c--)
        if (out[c] > out[c + 1])
          out[c] = out[c + 1];
      for (int c = med + 1; c <= nq; c++)
        if (out[c] < out[c - 1])
          out[c] = out[c - 1];
    }
    for (int q = 0; q < nq; q++)
      fan[(size_t)h * nq + q] =
          post->denorm_instance ? out[q + 1] * sigma + mu : out[q + 1];
  }

done:
  sqlite3_free(cf);
  sqlite3_free(rf);
  sqlite3_free(ff);
  sqlite3_free(qs);
  sqlite3_free(out);
  sqlite3_free(a1);
  sqlite3_free(b1);
  sqlite3_free(a2);
  sqlite3_free(b2);
  if (mem)
    g_ort->ReleaseMemoryInfo(mem);
  return rc;
}

/* Run the sequence model and return its raw quantile fan for the horizon:
 * *fan_out is malloc'd [horizon * nquant] step-major (fan[k*nquant + q]),
 * *levels_out is malloc'd [nquant]; the caller frees both. Used by the
 * forecast() point/interval reduction below and by distill_forecast's in-DB
 * teacher labeling. */
int predict0_onnx_forecast_fan(sqlite3 *db, const predict0_model_row *model,
                               const predict0_backend_opts *opts,
                               const f64 *context, int ctx_len, int horizon,
                               f64 **fan_out, f32 **levels_out, int *nquant_out,
                               char **errmsg) {
  *fan_out = NULL;
  *levels_out = NULL;
  *nquant_out = 0;
  int rc = SQLITE_OK, nq = 0;
  char *input_name = json_str(db, model->io_spec, "$.input");
  char *output_name = json_str(db, model->io_spec, "$.output");
  char *point_name = json_str(db, model->io_spec, "$.outputs.point");
  char *quant_name = json_str(db, model->io_spec, "$.outputs.quantile");
  char *patch_s = json_str(db, model->io_spec, "$.patch");
  char *layout = json_str(db, model->io_spec, "$.layout");
  char *denorm_s = json_str(db, model->io_spec, "$.denormalize");
  int patch = patch_s ? atoi(patch_s) : 1;
  /* A two-head core (point + quantile outputs) needs the post-processing the
   * ONNX graph could not carry, applied here as declared. Nothing is keyed on a
   * model name. */
  seq_post post = {
      .flip_invariance = json_flag(db, model->io_spec, "$.flip_invariance"),
      .continuous_head = json_flag(db, model->io_spec, "$.continuous_head"),
      .crossing_repair = json_flag(db, model->io_spec, "$.quantile_crossing_repair"),
      .denorm_instance = denorm_s && strcmp(denorm_s, "instance") == 0};
  int fixed_context = json_flag(db, model->io_spec, "$.fixed_context");
  int two_head = point_name && quant_name;
  char **lvl_s = NULL;
  f32 *levels = NULL;
  f64 *fan = NULL;
  f32 *inbuf = NULL;
  OrtMemoryInfo *mem = NULL;
  OrtValue *input = NULL, *output = NULL;
  OrtTensorTypeAndShapeInfo *ti = NULL;
  OrtSession *session = NULL;

  /* Parse quantiles only after every resource above is NULL-initialized, so the
   * error path can goto done safely. Absent quantiles -> nq 0, caught by the
   * nq <= 0 io_spec check below; a present-but-malformed array must fail loudly
   * here, not read as absent. */
  rc = json_arr(db, model->io_spec, "$.quantiles", &lvl_s, &nq, FCST_MAX_QUANT,
                errmsg);
  if (rc != SQLITE_OK)
    goto done;

  int outputs_ok = two_head ? (point_name && quant_name) : (output_name != NULL);
  if (!layout || strcmp(layout, "sequence") != 0 || !input_name || !outputs_ok ||
      nq <= 0 || patch < 1) {
    *errmsg = sqlite3_mprintf(
        "%s: forecast onnx model needs io_spec layout 'sequence' with input,"
        " output (or outputs.point + outputs.quantile), and quantiles[]",
        PREDICT_ERR_IO_SPEC);
    rc = SQLITE_ERROR;
    goto done;
  }
  /* Fail loud, never silently: these transforms are only implemented for a
   * two-head core, so reject them on a single-output model rather than ignore. */
  if (!two_head &&
      (post.flip_invariance || post.continuous_head || post.denorm_instance)) {
    *errmsg = sqlite3_mprintf(
        "%s: flip_invariance/continuous_head/denormalize require a two-head core"
        " (io_spec outputs.point + outputs.quantile)",
        PREDICT_ERR_IO_SPEC);
    rc = SQLITE_ERROR;
    goto done;
  }
  if (!license_ok(model->license, opts->accept_license)) {
    *errmsg = sqlite3_mprintf(
        "%s: model license '%s' requires a matching accept_license",
        PREDICT_ERR_LICENSE, model->license ? model->license : "(none)");
    rc = SQLITE_ERROR;
    goto done;
  }
  if ((rc = onnx_init(errmsg)) != SQLITE_OK)
    goto done;
  if ((rc = onnx_get_session(model, opts, &session, errmsg)) != SQLITE_OK)
    goto done;

  /* truncate the context to the most recent multiple of the patch size, so the
   * model never pads (see scripts/export_chronos_onnx.py). A fixed_context core
   * (e.g. TimesFM's 512-length input) takes exactly the last `patch` values. */
  int keep = fixed_context ? patch : ctx_len - (ctx_len % patch);
  if (ctx_len < patch || keep < patch) {
    *errmsg = sqlite3_mprintf("%s: context of %d is shorter than the model's"
                              " patch size %d",
                              PREDICT_ERR_SCHEMA, ctx_len, patch);
    rc = SQLITE_ERROR;
    goto done;
  }
  const f64 *ctx = context + (ctx_len - keep);
  levels = sqlite3_malloc(sizeof(f32) * nq);
  fan = sqlite3_malloc(sizeof(f64) * (size_t)horizon * nq);
  if (!levels || !fan) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  for (int q = 0; q < nq; q++)
    levels[q] = (f32)atof(lvl_s[q]);

  if (two_head) { /* two-head core: reconstruct the fan per the declared post */
    if ((rc = seq_reconstruct(session, input_name, point_name, quant_name, ctx,
                              keep, horizon, nq, levels, &post, fan, errmsg)) !=
        SQLITE_OK)
      goto done;
    *fan_out = fan;
    *levels_out = levels;
    *nquant_out = nq;
    fan = NULL;
    levels = NULL;
    goto done;
  }

  inbuf = sqlite3_malloc(sizeof(f32) * keep);
  if (!inbuf) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  for (int i = 0; i < keep; i++)
    inbuf[i] = (f32)ctx[i];

  OrtStatus *st = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                             &mem);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateCpuMemoryInfo", errmsg);
    goto done;
  }
  int64_t shape[2] = {1, keep};
  st = g_ort->CreateTensorWithDataAsOrtValue(
      mem, inbuf, sizeof(f32) * (size_t)keep, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "CreateTensor", errmsg);
    goto done;
  }
  const char *in_names[1] = {input_name};
  const char *out_names[1] = {output_name};
  st = g_ort->Run(session, NULL, in_names, (const OrtValue *const *)&input, 1,
                  out_names, 1, &output);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "Run", errmsg);
    goto done;
  }

  size_t ndim = 0;
  int64_t od[8];
  float *odata = NULL;
  st = g_ort->GetTensorTypeAndShape(output, &ti);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetTensorTypeAndShape", errmsg);
    goto done;
  }
  st = g_ort->GetDimensionsCount(ti, &ndim);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetDimensionsCount", errmsg);
    goto done;
  }
  if (ndim != 3) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: forecast output has %zu dims (expected 3:"
                              " [1, quantiles, horizon])",
                              PREDICT_ERR_INFERENCE, ndim);
    goto done;
  }
  st = g_ort->GetDimensions(ti, od, ndim);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetDimensions", errmsg);
    goto done;
  }
  int Qd = (int)od[1], Hmax = (int)od[2];
  if (od[0] != 1 || Qd != nq) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: forecast output quantile dim %d does not"
                              " match %d io_spec levels",
                              PREDICT_ERR_INFERENCE, Qd, nq);
    goto done;
  }
  if (Hmax < horizon) {
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: model forecasts at most %d steps, %d asked",
                              PREDICT_ERR_HORIZON, Hmax, horizon);
    goto done;
  }
  st = g_ort->GetTensorMutableData(output, (void **)&odata);
  if (st) {
    rc = onnx_fail(st, PREDICT_ERR_INFERENCE, "GetTensorMutableData", errmsg);
    goto done;
  }
  /* odata[q*Hmax + k] = quantile q at step k -> fan[k*nq + q] (step-major). */
  for (int k = 0; k < horizon; k++)
    for (int q = 0; q < nq; q++)
      fan[(size_t)k * nq + q] = (f64)odata[(size_t)q * Hmax + k];
  *fan_out = fan;
  *levels_out = levels;
  *nquant_out = nq;
  fan = NULL; /* ownership transferred to the caller */
  levels = NULL;

done:
  if (ti)
    g_ort->ReleaseTensorTypeAndShapeInfo(ti);
  if (output)
    g_ort->ReleaseValue(output);
  if (input)
    g_ort->ReleaseValue(input);
  if (mem)
    g_ort->ReleaseMemoryInfo(mem);
  sqlite3_free(inbuf);
  sqlite3_free(levels);
  sqlite3_free(fan);
  sqlite3_free(input_name);
  sqlite3_free(output_name);
  sqlite3_free(point_name);
  sqlite3_free(quant_name);
  sqlite3_free(patch_s);
  sqlite3_free(layout);
  sqlite3_free(denorm_s);
  for (int q = 0; q < nq; q++)
    sqlite3_free(lvl_s[q]);
  sqlite3_free(lvl_s);
  return rc;
}

/* Point + interval at `conf`, reduced from the model's quantile fan. */
int predict0_onnx_forecast(sqlite3 *db, const predict0_model_row *model,
                           const predict0_backend_opts *opts,
                           const f64 *context, int ctx_len, int horizon,
                           f64 conf, f64 *fc, f64 *lo, f64 *hi, char **errmsg) {
  f64 *fan = NULL;
  f32 *levels = NULL;
  int nq = 0;
  int rc = predict0_onnx_forecast_fan(db, model, opts, context, ctx_len, horizon,
                                      &fan, &levels, &nq, errmsg);
  if (rc != SQLITE_OK)
    return rc;
  f64 *val = sqlite3_malloc(sizeof(f64) * nq);
  if (!val) {
    sqlite3_free(fan);
    sqlite3_free(levels);
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
    return SQLITE_NOMEM;
  }
  f64 plo = (1 - conf) / 2, phi = (1 + conf) / 2;
  for (int k = 0; k < horizon; k++) {
    for (int q = 0; q < nq; q++)
      val[q] = fan[(size_t)k * nq + q];
    for (int i = 1; i < nq; i++)
      for (int j = i; j > 0 && val[j] < val[j - 1]; j--) {
        f64 t = val[j];
        val[j] = val[j - 1];
        val[j - 1] = t;
      }
    fc[k] = predict0_quantile_at(levels, val, nq, 0.5);
    lo[k] = predict0_quantile_at(levels, val, nq, plo);
    hi[k] = predict0_quantile_at(levels, val, nq, phi);
  }
  sqlite3_free(val);
  sqlite3_free(fan);
  sqlite3_free(levels);
  return SQLITE_OK;
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
                       int *out_n, char **errmsg) {
  UNUSED_PARAMETER(model_id);
  UNUSED_PARAMETER(opts);
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
                          int *out_n, char **errmsg) {
  UNUSED_PARAMETER(model_id);
  UNUSED_PARAMETER(opts);
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
  if (F > PREDICT_MAX_FEAT) { /* per-row stack buffer is sized to the cap */
    rc = SQLITE_ERROR;
    *errmsg = sqlite3_mprintf("%s: too many features (%d; max %d)",
                              PREDICT_ERR_SCHEMA, F, PREDICT_MAX_FEAT);
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

    f32 feat[PREDICT_MAX_FEAT];
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
  if (io.nfeat > PREDICT_MAX_FEAT) {
    onnx_io_free(&io);
    *errmsg = sqlite3_mprintf("%s: model declares %d features (max %d)",
                              PREDICT_ERR_IO_SPEC, io.nfeat, PREDICT_MAX_FEAT);
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
                     out_n, errmsg);
  else
    rc = onnx_incontext(db, model_id, train_sql, apply_sql, opts, &io, session,
                        out_rows, out_n, errmsg);

  onnx_io_free(&io);
  return rc;
}

#endif /* SQLITE_PREDICT_ONNX */
