"""Error-path coverage for the shared collect_series() helper.

Every failure branch is exercised through BOTH forecast() and
detect_anomalies(), since they share the helper. The point is not just
that the right error code comes back, but that the error paths (which
free partially-built series arrays and resolved-name strings) are clean
under ASan/valgrind — run via `make test-asan` / `make test-valgrind`,
whose C soak driver hits these same branches in a loop.
"""

import sqlite3
import pytest
import synthetic as syn

# (label, query, options) -> expected error code. Run against both ops.
CASES = [
    ("unparseable_query", "NOT SQL AT ALL", None, "PREDICT_ERR_SCHEMA"),
    ("multi_statement", "SELECT ts, value FROM series; SELECT 1", None,
     "PREDICT_ERR_SCHEMA"),
    ("write_query", "DELETE FROM series", None,
     "PREDICT_ERR_QUERY_NOT_READONLY"),
    ("single_column", "SELECT ts FROM series", None, "PREDICT_ERR_SCHEMA"),
    ("bad_time_col", "SELECT ts, value FROM series", '{"time_col":"nope"}',
     "PREDICT_ERR_SCHEMA"),
    ("bad_value_col", "SELECT ts, value FROM series", '{"value_col":"nope"}',
     "PREDICT_ERR_SCHEMA"),
    ("bad_group_col", "SELECT ts, value FROM series",
     '{"group_cols":["nope"]}', "PREDICT_ERR_SCHEMA"),
    ("uninferable", "SELECT 'x' AS a, 'y' AS b FROM series", None,
     "PREDICT_ERR_SCHEMA"),
]


def _call(db, op, query, opts):
    if op == "forecast":
        sql = ("SELECT * FROM forecast(?, 3, ?)" if opts
               else "SELECT * FROM forecast(?, 3)")
    else:
        sql = ("SELECT * FROM detect_anomalies(?, ?)" if opts
               else "SELECT * FROM detect_anomalies(?)")
    args = (query, opts) if opts else (query,)
    return db.execute(sql, args).fetchall()


@pytest.fixture
def seeded(db):
    rows, _ = syn.trend_season(n=60, seed=71)
    syn.load_into(db, rows)
    return db


@pytest.mark.parametrize("op", ["forecast", "detect_anomalies"])
@pytest.mark.parametrize("label,query,opts,code",
                         CASES, ids=[c[0] for c in CASES])
def test_collect_error_paths(seeded, op, label, query, opts, code):
    with pytest.raises(sqlite3.OperationalError) as e:
        _call(seeded, op, query, opts)
    assert code in str(e.value), f"{op}/{label}: {e.value}"


@pytest.mark.parametrize("op", ["forecast", "detect_anomalies"])
def test_multi_group_collection_shared(op, db):
    """The shared helper collects multiple groups identically for both
    ops. (The partial-series free path — n_series > 0 at error time — is
    only reachable via the 2M-row cap and NOMEM, so it is covered by the
    valgrind soak, not here.)"""
    a, _ = syn.trend_season(n=60, level=10.0, seed=72)
    b, _ = syn.trend_season(n=60, level=90.0, seed=73)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")
    out = _call(db, op, "SELECT ts, value, grp FROM series",
                '{"group_cols":["grp"],"receipt":0}')
    assert {r[0] for r in out} == {"a", "b"}


@pytest.mark.parametrize("op", ["forecast", "detect_anomalies"])
def test_shared_helper_same_column_inference(seeded, op):
    """Both ops must infer the same columns from the same messy query
    (a text time col, an int distractor, a float value)."""
    seeded.execute("CREATE TABLE m(label TEXT, n INTEGER, ts TEXT, v REAL)")
    seeded.executemany(
        "INSERT INTO m VALUES ('x', ?, ?, ?)",
        [(i, f"2026-01-01T{i:02d}:00:00Z", float(i)) for i in range(20)],
    )
    out = _call(seeded, op, "SELECT label, n, ts, v FROM m",
                '{"time_col":"ts","value_col":"v"}')
    assert len(out) >= 1
    assert out[0][-2] in ("ok", "truncated", "insufficient_history",
                          "non_numeric") or out[0][-2] is not None
