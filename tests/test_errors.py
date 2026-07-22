import sqlite3
import pytest
import synthetic as syn


def expect_error(db, code, query, horizon, opts=None):
    with pytest.raises(sqlite3.OperationalError) as e:
        if opts is None:
            db.execute("SELECT * FROM forecast(?, ?)",
                       (query, horizon)).fetchall()
        else:
            db.execute("SELECT * FROM forecast(?, ?, ?)",
                       (query, horizon, opts)).fetchall()
    assert code in str(e.value), f"expected {code} in: {e.value}"


@pytest.fixture
def seeded(db):
    rows, _ = syn.trend_season(n=60, seed=21)
    syn.load_into(db, rows)
    return db


def test_unknown_option_key(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS",
                 "SELECT ts, value FROM series", 3, '{"bogus": 1}')


def test_wrong_option_type(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS",
                 "SELECT ts, value FROM series", 3,
                 '{"context_limit": "many"}')


def test_options_must_be_object(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS",
                 "SELECT ts, value FROM series", 3, "[1,2]")


def test_options_invalid_json(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS",
                 "SELECT ts, value FROM series", 3, "{not json")


def test_confidence_out_of_range(seeded):
    expect_error(seeded, "PREDICT_ERR_OPTIONS",
                 "SELECT ts, value FROM series", 3,
                 '{"confidence_level": 1.5}')


def test_horizon_zero_and_too_big(seeded):
    expect_error(seeded, "PREDICT_ERR_HORIZON",
                 "SELECT ts, value FROM series", 0)
    expect_error(seeded, "PREDICT_ERR_HORIZON",
                 "SELECT ts, value FROM series", 1001)


def test_horizon_must_be_integer(seeded):
    with pytest.raises(sqlite3.OperationalError) as e:
        seeded.execute("SELECT * FROM forecast(?, 2.5)",
                       ("SELECT ts, value FROM series",)).fetchall()
    assert "PREDICT_ERR_HORIZON" in str(e.value)


def test_write_query_rejected(seeded):
    expect_error(seeded, "PREDICT_ERR_QUERY_NOT_READONLY",
                 "DELETE FROM series", 3)


def test_multi_statement_rejected(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA",
                 "SELECT ts, value FROM series; SELECT 1", 3)


def test_unparseable_query(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA", "NOT EVEN SQL", 3)


def test_single_column_query(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA", "SELECT ts FROM series", 3)


def test_unknown_model(seeded):
    expect_error(seeded, "PREDICT_ERR_MODEL_NOT_FOUND",
                 "SELECT ts, value FROM series", 3,
                 '{"model": "gpt-9000"}')


def test_missing_named_columns(seeded):
    expect_error(seeded, "PREDICT_ERR_SCHEMA",
                 "SELECT ts, value FROM series", 3, '{"time_col": "nope"}')
    expect_error(seeded, "PREDICT_ERR_SCHEMA",
                 "SELECT ts, value FROM series", 3, '{"value_col": "nope"}')
    expect_error(seeded, "PREDICT_ERR_SCHEMA",
                 "SELECT ts, value FROM series", 3,
                 '{"group_cols": ["nope"]}')


def test_non_numeric_value_is_status_not_error(db):
    db.execute("CREATE TABLE t(ts TEXT, v TEXT)")
    db.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(f"2026-01-01T{i:02d}:00:00Z", f"word{i}") for i in range(20)],
    )
    out = db.execute(
        "SELECT * FROM forecast(?, ?, ?)",
        ("SELECT ts, v FROM t", 3, '{"time_col":"ts","value_col":"v"}'),
    ).fetchall()
    assert len(out) == 1
    assert out[0][6] == "non_numeric"


def test_forecast_requires_arguments(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT * FROM forecast").fetchall()
