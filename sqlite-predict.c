#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT1
#endif

#pragma region meta

static void predict_version_fn(sqlite3_context *context, int argc,
                               sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  char *json = sqlite3_mprintf(
      "{\"extension\":\"%s\",\"spec\":\"%s\",\"runtimes\":[\"stat\"],"
      "\"models\":[]}",
      SQLITE_PREDICT_VERSION, SQLITE_PREDICT_SPEC);
  if (!json) {
    sqlite3_result_error_nomem(context);
    return;
  }
  sqlite3_result_text(context, json, -1, sqlite3_free);
  sqlite3_result_subtype(context, 'J'); /* JSON, per json1 convention */
}

static void predict_debug_fn(sqlite3_context *context, int argc,
                             sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  char *debug =
      sqlite3_mprintf("Version: %s\nDate: %s\nCommit: %s", SQLITE_PREDICT_VERSION,
                      SQLITE_PREDICT_DATE, SQLITE_PREDICT_SOURCE);
  if (!debug) {
    sqlite3_result_error_nomem(context);
    return;
  }
  sqlite3_result_text(context, debug, -1, sqlite3_free);
}

#pragma endregion

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_predict_init(sqlite3 *db, char **pzErrMsg,
                             const sqlite3_api_routines *pApi) {
  UNUSED_PARAMETER(pzErrMsg);
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#else
  UNUSED_PARAMETER(pApi);
#endif
  int rc = SQLITE_OK;
  const int flags = SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC;

  rc = sqlite3_create_function_v2(db, "predict_version", 0, flags, NULL,
                                  predict_version_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function_v2(db, "predict_debug", 0, flags, NULL,
                                  predict_debug_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = predict0_receipts_init(db);
  if (rc != SQLITE_OK)
    return rc;
  rc = predict0_forecast_init(db);
  return rc;
}
