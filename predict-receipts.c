/* Model registry, receipts, canonicalization, anchors, predict_replay().
 * M0: registration stub; the machinery lands in M2. */
#include "predict-internal.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

int predict0_receipts_init(sqlite3 *db) {
  UNUSED_PARAMETER(db);
  return SQLITE_OK;
}
