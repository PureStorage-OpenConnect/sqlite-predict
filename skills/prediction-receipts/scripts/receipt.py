#!/usr/bin/env python3
"""Reference implementation of the prediction-receipts convention.

Records and verifies provenance receipts for sqlite-predict calls, so
every agent that uses this script produces interoperable receipts with
identical canonicalization and hashing. Stdlib only.

Canonical result document:
  - If the prediction SQL returns exactly one row with one TEXT column
    (the aggregate document case), the document is that text verbatim.
  - Otherwise (predict/backtest rows), the document is compact JSON:
    {"columns":[...],"rows":[[...],...]} in cursor order, with floats
    serialized by Python's shortest round-trip repr.
The receipt stores sha256(document utf-8), never the document itself.

Usage:
  receipt.py record  DB "SELECT forecast(ts, value, 24) FROM t" [--operation forecast]
  receipt.py verify  DB RECEIPT_ID
  receipt.py list    DB
  Common flag: --extension PATH   loadable to use (default: env
  SQLITE_PREDICT_EXTENSION, else "predict0" on the library path)
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys

DDL = """CREATE TABLE IF NOT EXISTS _predict_receipts (
  id           INTEGER PRIMARY KEY,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  operation    TEXT NOT NULL,
  input_sql    TEXT NOT NULL,
  options      TEXT,
  model_id     TEXT NOT NULL,
  content_hash TEXT,
  extension    TEXT NOT NULL,
  result_sha256 TEXT NOT NULL
)"""

OPERATIONS = ("forecast", "detect_anomalies", "predict", "backtest")


def fail(code, msg):
    """Stable, grep-able failure codes so agents can branch on outcomes
    the way they branch on the extension's PREDICT_ERR_* codes."""
    print(f"receipt.py: {code}: {msg}", file=sys.stderr)
    sys.exit(1)


#: authorizer allow-list for replaying untrusted SQL: reads and function
#: calls only. Everything else (writes, DDL, ATTACH, PRAGMA) is denied.
_REPLAY_ALLOWED = {
    getattr(sqlite3, name, None) for name in
    ("SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE")
} - {None}


def _replay_authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _REPLAY_ALLOWED \
        else sqlite3.SQLITE_DENY


def connect(db_path, extension, untrusted=False):
    """untrusted=True is the replay posture: the database opens
    read-only at the file level and an authorizer denies everything but
    reads and function calls, so a tampered receipt's input_sql cannot
    write, drop, attach, or change pragmas on the verifier's database."""
    try:
        if untrusted:
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            db = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        fail("RECEIPT_ERR_DB", f"cannot open '{db_path}': {e}")
    db.enable_load_extension(True)
    try:
        db.load_extension(extension)
    except sqlite3.OperationalError as e:
        fail("RECEIPT_ERR_EXTENSION", f"cannot load '{extension}': {e}")
    finally:
        # a tampered receipt must not be able to load a local library
        db.enable_load_extension(False)
    if untrusted:
        db.set_authorizer(_replay_authorizer)
        # a tampered receipt must not be able to hang the verifier
        # either: abort replay after a generous VM-step budget
        steps = int(os.environ.get("RECEIPT_MAX_STEPS", 50_000_000))
        db.set_progress_handler(lambda: 1, steps)
    return db


AGGREGATE_OPS = ("forecast", "detect_anomalies")


def canonical_document(cur, operation):
    """Aggregate operations hash their single JSON document verbatim;
    row-shaped operations always hash the {"columns","rows"} form, even
    when the result happens to be one text cell. Shape must follow the
    operation or receipts stop being interoperable."""
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    if (operation in AGGREGATE_OPS and len(rows) == 1 and len(cols) == 1
            and isinstance(rows[0][0], str)):
        return rows[0][0]

    def cell(v):
        # BLOBs canonicalize as their hash: deterministic, bounded, and
        # json.dumps cannot serialize bytes anyway
        if isinstance(v, (bytes, memoryview)):
            return {"blob_sha256":
                    hashlib.sha256(bytes(v)).hexdigest()}
        return v
    return json.dumps({"columns": cols,
                       "rows": [[cell(v) for v in r] for r in rows]},
                      separators=(",", ":"), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_operation(sql):
    found = {op for op in OPERATIONS
             if re.search(r"\b" + op + r"\s*\(", sql)}
    if len(found) == 1:
        return found.pop()
    return None


def extension_version(db):
    doc = json.loads(db.execute("SELECT predict_version()").fetchone()[0])
    return doc["extension"]


def model_fields(db, document, model_flag):
    """The serving model id and, when registered, its content_hash."""
    model_id = model_flag
    if model_id is None:
        try:
            model_id = json.loads(document).get("model")
        except (json.JSONDecodeError, AttributeError):
            model_id = None
    if not model_id:
        fail("RECEIPT_ERR_MODEL_UNKNOWN",
             "the result is not a document with a \"model\" field;"
             " pass --model-id explicitly")
    content_hash = None
    try:
        row = db.execute(
            "SELECT content_hash FROM _predict_models WHERE model_id = ?",
            (model_id,)).fetchone()
        content_hash = row[0] if row else None
    except sqlite3.OperationalError:
        pass  # no registry in this database: content_hash stays NULL
    return model_id, content_hash


def cmd_record(args):
    if args.options is not None:
        try:
            json.loads(args.options)
        except json.JSONDecodeError as e:
            fail("RECEIPT_ERR_OPTIONS", f"--options is not valid JSON: {e}")
    db = connect(args.db, args.extension)
    operation = args.operation or infer_operation(args.sql)
    if operation is None:
        fail("RECEIPT_ERR_OPERATION_UNKNOWN",
             f"cannot infer the operation; pass --operation one of"
             f" {', '.join(OPERATIONS)}")
    try:
        cur = db.execute(args.sql)
        document = canonical_document(cur, operation)
    except sqlite3.Error as e:
        fail("RECEIPT_ERR_SQL", str(e))
    model_id, content_hash = model_fields(db, document, args.model_id)
    db.execute(DDL)
    cur = db.execute(
        "INSERT INTO _predict_receipts (operation, input_sql, options,"
        " model_id, content_hash, extension, result_sha256)"
        " VALUES (?,?,?,?,?,?,?)",
        (operation, args.sql, args.options, model_id, content_hash,
         extension_version(db), sha256_text(document)))
    db.commit()
    print(json.dumps({"receipt_id": cur.lastrowid, "model_id": model_id,
                      "content_hash": content_hash,
                      "result_sha256": sha256_text(document)}))
    db.close()


def cmd_verify(args):
    db = connect(args.db, args.extension, untrusted=True)
    row = db.execute(
        "SELECT operation, input_sql, model_id, content_hash, extension,"
        " result_sha256 FROM _predict_receipts WHERE id = ?",
        (args.receipt_id,)).fetchone()
    if row is None:
        fail("RECEIPT_ERR_NOT_FOUND", f"no receipt with id {args.receipt_id}")
    operation, input_sql, model_id, rec_hash, rec_ext, rec_sha = row
    try:
        document = canonical_document(db.execute(input_sql), operation)
    except sqlite3.Error as e:
        fail("RECEIPT_ERR_SQL", f"replay refused or failed: {e}")
    now_sha = sha256_text(document)
    now_ext = extension_version(db)
    # re-derive the serving model from the replayed result where the
    # result carries one (aggregate documents do; row-form results were
    # recorded via --model-id and cannot be re-derived)
    replayed_model = None
    try:
        replayed_model = json.loads(document).get("model")
    except (json.JSONDecodeError, AttributeError):
        pass
    id_matches = None if replayed_model is None \
        else replayed_model == model_id
    now_hash = None
    try:
        r = db.execute(
            "SELECT content_hash FROM _predict_models WHERE model_id = ?",
            (model_id,)).fetchone()
        now_hash = r[0] if r else None
    except sqlite3.OperationalError:
        pass
    hash_match = now_sha == rec_sha
    report = {
        "receipt_id": args.receipt_id,
        "match": hash_match and id_matches is not False,
        "result_sha256": {"recorded": rec_sha, "replayed": now_sha,
                          "match": hash_match},
        "model": {"recorded_id": model_id,
                  "replayed_id": replayed_model,
                  "id_matches": id_matches,
                  "content_hash_recorded": rec_hash,
                  "content_hash_current": now_hash,
                  "changed": rec_hash != now_hash},
        "extension": {"recorded": rec_ext, "current": now_ext,
                      "changed": rec_ext != now_ext},
        # options is informational: its authoritative copy is the options
        # text embedded in input_sql, which the replay executes verbatim
        "options_verified": False,
    }
    print(json.dumps(report, indent=2))
    db.close()
    sys.exit(0 if report["match"] else 2)


def cmd_list(args):
    db = connect(args.db, args.extension, untrusted=True)
    try:
        rows = db.execute(
            "SELECT id, created_at, operation, model_id, result_sha256"
            " FROM _predict_receipts ORDER BY id").fetchall()
    except sqlite3.OperationalError:
        rows = []
    for r in rows:
        print(f"{r[0]}  {r[1]}  {r[2]:<16} {r[3]:<24} {r[4][:16]}...")
    db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension",
                   default=os.environ.get("SQLITE_PREDICT_EXTENSION",
                                          "predict0"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="run a prediction and record a receipt")
    r.add_argument("db")
    r.add_argument("sql")
    r.add_argument("--operation", choices=OPERATIONS)
    r.add_argument("--options", help="options JSON, recorded verbatim")
    r.add_argument("--model-id",
                   help="override when the result carries no model field")
    r.set_defaults(fn=cmd_record)

    v = sub.add_parser("verify", help="replay a receipt and compare hashes")
    v.add_argument("db")
    v.add_argument("receipt_id", type=int)
    v.set_defaults(fn=cmd_verify)

    ls = sub.add_parser("list", help="list recorded receipts")
    ls.add_argument("db")
    ls.set_defaults(fn=cmd_list)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
