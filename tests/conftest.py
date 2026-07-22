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
