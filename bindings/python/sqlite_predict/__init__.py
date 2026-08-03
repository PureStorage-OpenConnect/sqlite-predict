"""sqlite_predict: prediction as a SQL primitive for SQLite.

    import sqlite3
    import sqlite_predict

    db = sqlite3.connect(":memory:")
    sqlite_predict.load(db)
    db.execute("SELECT forecast(ts, value, 24) FROM readings")

The wheel bundles the zero-dependency loadable extension for your platform.
"""
import os

__version__ = "0.2.1"


def loadable_path() -> str:
    """Absolute path to the bundled loadable extension, without a file suffix
    (the form SQLite's load_extension prefers: it appends the platform suffix
    and derives the entry point)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for ext in ("dylib", "so", "dll"):
        if os.path.exists(os.path.join(here, "predict0." + ext)):
            return os.path.join(here, "predict0")
    raise FileNotFoundError(
        "the sqlite-predict loadable was not bundled in this install")


def load(conn) -> None:
    """Load sqlite-predict into a connection that exposes SQLite's
    enable_load_extension / load_extension (e.g. a stdlib sqlite3.Connection)."""
    conn.enable_load_extension(True)
    conn.load_extension(loadable_path())
    conn.enable_load_extension(False)
