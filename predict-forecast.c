/* forecast() and detect_anomalies() eponymous virtual tables.
 * M0: module registration stubs; the vtabs land in M1. */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

int predict0_forecast_init(sqlite3 *db) {
  UNUSED_PARAMETER(db);
  return SQLITE_OK;
}
