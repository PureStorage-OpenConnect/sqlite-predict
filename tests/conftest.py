import json
import os
import sqlite3
import pytest

EXT_PATH = os.environ.get(
    "SQLITE_PREDICT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "dist", "predict0"),
)


def connect(path=":memory:"):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(EXT_PATH)
    db.enable_load_extension(False)
    return db


@pytest.fixture
def db():
    conn = connect()
    yield conn
    conn.close()


def _doc(db, sql, params):
    """Execute one document-returning statement and parse the JSON it
    yields (None when the aggregate saw zero rows)."""
    row = db.execute(sql, params).fetchone()
    return json.loads(row[0]) if row and row[0] is not None else None


def forecast_doc(db, horizon, table="series", ts_col="ts", value_col="value",
                 **options):
    """Run the forecast aggregate over a whole table and parse the JSON
    document it returns."""
    if options:
        return _doc(db, f"SELECT forecast({ts_col}, {value_col}, ?, ?)"
                        f" FROM {table}", (horizon, json.dumps(options)))
    return _doc(db, f"SELECT forecast({ts_col}, {value_col}, ?)"
                    f" FROM {table}", (horizon,))


def anomaly_doc(db, table="series", ts_col="ts", value_col="value", **options):
    """Run the detect_anomalies aggregate over a whole table and parse the
    JSON document it returns."""
    if options:
        return _doc(db, f"SELECT detect_anomalies({ts_col}, {value_col}, ?)"
                        f" FROM {table}", (json.dumps(options),))
    return _doc(db, f"SELECT detect_anomalies({ts_col}, {value_col})"
                    f" FROM {table}", ())
