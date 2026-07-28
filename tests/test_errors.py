"""Call-shape errors on the aggregate forecast()/detect_anomalies()
surface, plus the query-shaped validation that now lives on backtest().

Aggregate calls raise only for caller mistakes (bad options, bad
horizon, unknown model, a query string where rows belong); data-shaped
problems degrade to a status inside the result document instead.
"""

import json
import sqlite3
import pytest
import synthetic as syn

# 4-arg aggregate form; pass options=None for the no-options call
AGG = "SELECT forecast(ts, value, ?, ?) FROM series"


def expect_error(db, code, sql, params=()):
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(sql, params).fetchall()
    assert code in str(e.value), f"expected {code} in: {e.value}"


@pytest.fixture
def seeded(db):
    rows, _ = syn.trend_season(n=60, seed=21)
    syn.load_into(db, rows)
    return db


def test_unknown_option_key(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS", AGG, (3, '{"bogus": 1}'))


def test_wrong_option_type(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS", AGG,
                 (3, '{"context_limit": "many"}'))


def test_options_must_be_object(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS", AGG, (3, "[1,2]"))


def test_options_invalid_json(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS", AGG, (3, "{not json"))


def test_confidence_out_of_range(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS", AGG,
                 (3, '{"confidence_level": 1.5}'))


def test_removed_option_keys_rejected(seeded):
    """time_col/value_col/group_cols/receipt belonged to the deleted
    query form; on the aggregate they are unknown option keys."""
    for opts in ('{"time_col": "ts"}', '{"value_col": "value"}',
                 '{"group_cols": ["grp"]}', '{"receipt": 0}'):
        expect_error(seeded, "PREDICT_ERR_OPTIONS", AGG, (3, opts))


def test_anomaly_threshold_out_of_range(seeded):
    for opts in ('{"anomaly_prob_threshold": 0}',
                 '{"anomaly_prob_threshold": 1.5}'):
        expect_error(seeded, "PREDICT_ERR_THRESHOLD",
                     "SELECT detect_anomalies(ts, value, ?) FROM series",
                     (opts,))


def test_horizon_zero_and_too_big(seeded):
    expect_error(seeded, "PREDICT_ERR_HORIZON", AGG, (0, None))
    expect_error(seeded, "PREDICT_ERR_HORIZON", AGG, (1001, None))


def test_horizon_must_be_integer(seeded):
    expect_error(seeded, "PREDICT_ERR_HORIZON", AGG, (2.5, None))


def test_horizon_must_be_constant_within_group(seeded):
    with pytest.raises(sqlite3.OperationalError) as e:
        seeded.execute(
            "SELECT forecast(ts, value,"
            " CASE WHEN rowid = 1 THEN 3 ELSE 4 END) FROM series"
        ).fetchall()
    msg = str(e.value)
    assert "PREDICT_ERR_HORIZON" in msg and "constant" in msg


def test_options_must_be_constant_within_group(seeded):
    with pytest.raises(sqlite3.OperationalError) as e:
        seeded.execute(
            "SELECT forecast(ts, value, 3,"
            " CASE WHEN rowid = 1 THEN '{\"context_limit\": 40}'"
            " ELSE '{\"context_limit\": 41}' END) FROM series"
        ).fetchall()
    msg = str(e.value)
    assert "PREDICT_ERR_OPTIONS" in msg and "constant" in msg


def test_unknown_model(seeded):
    expect_error(seeded, "PREDICT_ERR_MODEL_NOT_FOUND", AGG,
                 (3, '{"model": "gpt-9000"}'))


def test_query_string_first_arg_redirects(seeded):
    """A TEXT first argument that reads as SQL means the caller tried
    the deleted query form; the error explains the aggregate shape."""
    for sql in (
        "SELECT forecast('SELECT ts, value FROM series', 3, NULL)",
        "SELECT forecast('  WITH x AS (SELECT 1) SELECT * FROM x', 3, NULL)",
        "SELECT detect_anomalies('SELECT ts, value FROM series', NULL)",
    ):
        with pytest.raises(sqlite3.OperationalError) as e:
            seeded.execute(sql).fetchall()
        msg = str(e.value)
        assert "PREDICT_ERR_SCHEMA" in msg and "aggregate over" in msg


def test_two_arg_query_call_redirects(db):
    """The BigQuery-style forecast(query, horizon) stub gives directions
    instead of 'wrong number of arguments'."""
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT forecast('SELECT ts, value FROM series', 6)").fetchall()
    msg = str(e.value)
    assert "PREDICT_ERR_SCHEMA" in msg and "aggregate over" in msg


# ---- query-string validation, now on the backtest() TVF ----

BT = "SELECT * FROM backtest(?, 3, ?)"


def test_write_query_rejected(seeded):
    expect_error(seeded, "PREDICT_ERR_QUERY_NOT_READONLY", BT,
                 ("DELETE FROM series", None))


def test_multi_statement_rejected(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA", BT,
                 ("SELECT ts, value FROM series; SELECT 1", None))


def test_unparseable_query(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA", BT, ("NOT EVEN SQL", None))


def test_single_column_query(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA", BT,
                 ("SELECT ts FROM series", None))


def test_missing_named_columns(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA", BT,
                 ("SELECT ts, value FROM series", '{"time_col": "nope"}'))
    expect_error(seeded, "PREDICT_ERR_SCHEMA", BT,
                 ("SELECT ts, value FROM series", '{"value_col": "nope"}'))
    expect_error(seeded, "PREDICT_ERR_SCHEMA", BT,
                 ("SELECT ts, value FROM series", '{"group_cols": ["nope"]}'))


# ---- data-shaped problems degrade, they don't raise ----

def test_non_numeric_value_is_status_not_error(db):
    db.execute("CREATE TABLE t(ts TEXT, v TEXT)")
    db.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(f"2026-01-01T{i:02d}:00:00Z", f"word{i}") for i in range(20)],
    )
    (raw,) = db.execute("SELECT forecast(ts, v, 3) FROM t").fetchone()
    doc = json.loads(raw)
    assert doc["status"] == "non_numeric"
    assert doc["rows"] == []


def test_forecast_requires_arguments(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT forecast()").fetchall()
