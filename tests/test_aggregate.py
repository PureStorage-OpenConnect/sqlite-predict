"""Aggregate forms of forecast/detect_anomalies (RFC 0005 §4.2.8) and the
expansion functions (§4.2.9).

forecast and detect_anomalies are aggregate-only: the enclosing statement
supplies the rows, GROUP BY splits series, and each group returns one JSON
document — exactly {"model", "status", "rows"}. The aggregate is a pure
function: it writes nothing, so views and read-only databases work by
construction.
"""

import json
import random
import sqlite3

import pytest
import synthetic as syn


def doc(db, sql, *params):
    row = db.execute(sql, params).fetchone()
    return json.loads(row[0]) if row and row[0] is not None else None


# ---- shape and semantics ----


def test_forecast_doc_shape(db):
    rows, _ = syn.trend_season(n=96, seed=21)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert set(d) == {"model", "status", "rows"}
    assert d["status"] == "ok"
    assert d["model"] == "theta-classic"
    assert len(d["rows"]) == 6
    r = d["rows"][0]
    assert set(r) == {"step", "forecast_timestamp", "forecast",
                      "lower_bound", "upper_bound"}
    assert r["step"] == 1
    assert r["lower_bound"] <= r["forecast"] <= r["upper_bound"]


def test_group_by_matches_per_group_calls(db):
    # GROUP BY splitting is pure bookkeeping: each group's document is
    # identical to a single-series call over that group's rows alone
    a, _ = syn.trend_season(n=96, seed=22)
    b, _ = syn.random_walk(n=96, seed=23)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")

    agg = {
        grp: json.loads(d)
        for grp, d in db.execute(
            "SELECT grp, forecast(ts, value, 4) FROM series GROUP BY grp")
    }
    assert set(agg) == {"a", "b"}
    for grp in ("a", "b"):
        single = doc(db, "SELECT forecast(ts, value, 4) FROM series"
                         " WHERE grp = ?", grp)
        assert agg[grp] == single


def test_input_order_does_not_matter(db):
    rows, _ = syn.trend_season(n=96, seed=24)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    syn.load_into(db, rows, table="ordered")
    syn.load_into(db, shuffled, table="shuffled")
    d1 = doc(db, "SELECT forecast(ts, value, 6) FROM ordered")
    d2 = doc(db, "SELECT forecast(ts, value, 6) FROM shuffled")
    assert d1["rows"] == d2["rows"]


def test_zero_rows_returns_null(db):
    db.execute("CREATE TABLE empty(ts TEXT, value REAL)")
    assert db.execute(
        "SELECT forecast(ts, value, 6) FROM empty").fetchone()[0] is None
    assert db.execute(
        "SELECT detect_anomalies(ts, value) FROM empty").fetchone()[0] is None


def test_degraded_series_status(db):
    # too little history: status doc with empty rows, not a query failure
    syn.load_into(db, [(f"2024-01-01T0{i}:00:00Z", float(i)) for i in range(4)])
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert set(d) == {"model", "status", "rows"}
    assert d["status"] == "insufficient_history"
    assert d["rows"] == []


def test_non_numeric_degrades_not_fails(db):
    rows, _ = syn.trend_season(n=96, seed=25)
    syn.load_into(db, rows)
    db.execute("INSERT INTO series VALUES ('2024-06-01T00:00:00Z', 'not-a-number')")
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert d["status"] == "non_numeric"
    assert d["rows"] == []


# ---- purity ----


def test_aggregate_is_pure_no_tables_no_writes(db):
    # the aggregate writes nothing: not even the registry tables
    rows, _ = syn.trend_season(n=96, seed=56)
    syn.load_into(db, rows)
    db.execute("SELECT forecast(ts, value, 6) FROM series").fetchone()
    db.execute("SELECT detect_anomalies(ts, value) FROM series").fetchone()
    assert db.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE"
                      " '_predict%'").fetchone()[0] == 0


def test_read_only_database_works(db, tmp_path):
    # the aggregate form is pure, so a read-only database serves forecasts
    path = tmp_path / "ro.db"
    setup = sqlite3.connect(path)
    rows, _ = syn.trend_season(n=96, seed=34)
    syn.load_into(setup, rows)
    setup.close()

    import conftest
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ro.enable_load_extension(True)
    ro.load_extension(conftest.EXT_PATH)
    try:
        d = json.loads(ro.execute(
            "SELECT forecast(ts, value, 6) FROM series").fetchone()[0])
        assert d["status"] == "ok"
        assert len(d["rows"]) == 6
        assert set(d) == {"model", "status", "rows"}
    finally:
        ro.close()


# ---- argument and option validation ----


def test_options_must_be_constant_within_group(db):
    rows, _ = syn.trend_season(n=96, seed=37)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError, match="constant"):
        db.execute(
            "SELECT forecast(ts, value, 4, CASE WHEN value > ("
            " SELECT avg(value) FROM series) THEN '{\"context_limit\":64}'"
            " ELSE NULL END) FROM series").fetchone()


def test_horizon_must_be_constant_within_group(db):
    rows, _ = syn.trend_season(n=96, seed=38)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError, match="HORIZON"):
        db.execute(
            "SELECT forecast(ts, value, CAST(1 + (rowid % 2) AS INTEGER))"
            " FROM series").fetchone()


def test_query_shape_and_receipt_options_rejected(db):
    # query-shape keys are replaced by GROUP BY and argument positions;
    # "receipt" is gone with the receipts machinery — all unknown now
    rows, _ = syn.trend_season(n=96, seed=39)
    syn.load_into(db, rows)
    for key in ("time_col", "value_col", "group_cols", "receipt"):
        with pytest.raises(sqlite3.OperationalError, match="OPTIONS"):
            db.execute(
                f"SELECT forecast(ts, value, 4, '{{\"{key}\": \"x\"}}')"
                " FROM series").fetchone()


def test_horizon_validation(db):
    rows, _ = syn.trend_season(n=96, seed=40)
    syn.load_into(db, rows)
    for bad in (0, -1, 1001):
        with pytest.raises(sqlite3.OperationalError, match="HORIZON"):
            db.execute(
                "SELECT forecast(ts, value, ?) FROM series", (bad,)).fetchone()
    with pytest.raises(sqlite3.OperationalError, match="HORIZON"):
        db.execute("SELECT forecast(ts, value, 1.5) FROM series").fetchone()


def test_query_string_in_expression_position_is_redirected(db):
    rows, _ = syn.trend_season(n=96, seed=41)
    syn.load_into(db, rows)
    # 2-arg scalar stub: the BigQuery-style forecast(query, horizon) attempt
    with pytest.raises(sqlite3.OperationalError,
                       match="aggregate over your rows"):
        db.execute("SELECT forecast('SELECT ts, value FROM series', 4)")
    # aggregate arity with a query string first argument
    with pytest.raises(sqlite3.OperationalError,
                       match="aggregate over your rows"):
        db.execute(
            "SELECT forecast('SELECT ts FROM series', value, 4) FROM series"
        ).fetchone()
    with pytest.raises(sqlite3.OperationalError,
                       match="aggregate over your rows"):
        db.execute(
            "SELECT detect_anomalies('  select ts FROM series', value)"
            " FROM series").fetchone()


def test_unknown_model_and_unknown_option(db):
    rows, _ = syn.trend_season(n=96, seed=42)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError, match="MODEL_NOT_FOUND"):
        db.execute("SELECT forecast(ts, value, 4, '{\"model\":\"nope\"}')"
                   " FROM series").fetchone()
    with pytest.raises(sqlite3.OperationalError, match="OPTIONS"):
        db.execute("SELECT forecast(ts, value, 4, '{\"bogus\": 1}')"
                   " FROM series").fetchone()


def test_null_and_mixed_ts_degrade(db):
    rows, _ = syn.trend_season(n=96, seed=43)
    syn.load_into(db, rows)
    db.execute("INSERT INTO series VALUES (NULL, 1.0)")
    d = doc(db, "SELECT forecast(ts, value, 4) FROM series")
    assert d["status"] == "non_numeric"


def test_epoch_integer_timestamps(db):
    base = 1704067200  # 2024-01-01T00:00:00Z
    db.execute("CREATE TABLE e(ts INTEGER, value REAL)")
    db.executemany("INSERT INTO e VALUES (?, ?)",
                   [(base + i * 3600, 50.0 + (i % 24)) for i in range(96)])
    d = doc(db, "SELECT forecast(ts, value, 4) FROM e")
    assert d["status"] == "ok"
    assert d["rows"][0]["forecast_timestamp"].startswith("2024-01-05T")


def test_single_point_series(db):
    syn.load_into(db, [("2024-01-01T00:00:00Z", 1.0)])
    d = doc(db, "SELECT forecast(ts, value, 4) FROM series")
    assert d["status"] == "insufficient_history"


# ---- anomalies aggregate ----


def test_anomaly_doc_shape_and_detection(db):
    rows, _ = syn.trend_season(n=120, noise=0.3, seed=44)
    rows, _ = syn.with_anomalies(rows, k=3, magnitude=12.0, seed=45)
    syn.load_into(db, rows)
    d = doc(db, "SELECT detect_anomalies(ts, value) FROM series")
    assert set(d) == {"model", "status", "rows"}
    assert d["status"] == "ok"
    assert len(d["rows"]) == 120
    r = d["rows"][0]
    assert set(r) == {"ts", "value", "forecast", "lower_bound", "upper_bound",
                      "is_anomaly", "anomaly_probability"}
    assert sum(x["is_anomaly"] for x in d["rows"]) >= 1


def test_anomaly_subpca_rows_have_null_intervals(db):
    rows, _ = syn.trend_season(n=250, seed=46)
    syn.load_into(db, rows)
    d = doc(db, "SELECT detect_anomalies(ts, value,"
                " '{\"model\": \"sub-pca\"}') FROM series")
    assert d["status"] == "ok"
    assert all(r["forecast"] is None and r["lower_bound"] is None
               for r in d["rows"])
    assert all(0.0 <= r["anomaly_probability"] <= 1.0 for r in d["rows"])


# ---- expansion functions ----


def test_forecast_rows_round_trip(db):
    rows, _ = syn.trend_season(n=96, seed=47)
    syn.load_into(db, rows)
    out = db.execute(
        "SELECT r.step, r.forecast_timestamp, r.forecast, r.lower_bound,"
        " r.upper_bound, r.status FROM forecast_rows("
        " (SELECT forecast(ts, value, 5) FROM series)) r").fetchall()
    assert len(out) == 5
    assert [r[0] for r in out] == [1, 2, 3, 4, 5]
    assert all(isinstance(r[2], float) for r in out)
    assert all(r[5] == "ok" for r in out)


def test_anomaly_rows_round_trip(db):
    rows, _ = syn.trend_season(n=96, seed=48)
    syn.load_into(db, rows)
    n, flagged = db.execute(
        "SELECT count(*), sum(is_anomaly) FROM anomaly_rows("
        " (SELECT detect_anomalies(ts, value) FROM series))").fetchone()
    assert n == 96
    assert flagged is not None


def test_expansion_status_doc_yields_one_status_row(db):
    syn.load_into(db, [("2024-01-01T00:00:00Z", 1.0)])
    out = db.execute(
        "SELECT step, forecast, status FROM forecast_rows("
        " (SELECT forecast(ts, value, 4) FROM series))").fetchall()
    assert out == [(None, None, "insufficient_history")]


def test_expansion_null_doc_yields_zero_rows(db):
    assert db.execute(
        "SELECT count(*) FROM forecast_rows(NULL)").fetchone()[0] == 0


def test_expansion_rejects_garbage(db):
    for bad in ("'not json'", "'[1,2,3]'", "'{\"rows\": 1}'", "42"):
        with pytest.raises(sqlite3.OperationalError, match="SCHEMA"):
            db.execute(f"SELECT * FROM forecast_rows({bad})").fetchall()


# ---- purity in expression contexts (§6.7) ----


def test_aggregate_works_in_views(db):
    # pure function: a forecast view is legal and evaluates on read
    rows, _ = syn.trend_season(n=96, seed=49)
    syn.load_into(db, rows)
    db.execute("CREATE VIEW v AS SELECT forecast(ts, value, 4) AS d"
               " FROM series")
    d = json.loads(db.execute("SELECT d FROM v").fetchone()[0])
    assert d["status"] == "ok" and len(d["rows"]) == 4


# ---- ORM-style usage (SQLAlchemy smoke) ----


def test_sqlalchemy_composes_the_aggregate(db):
    sa = pytest.importorskip("sqlalchemy")
    import conftest

    eng = sa.create_engine("sqlite://", creator=lambda: conftest.connect())
    with eng.connect() as conn:
        conn.execute(sa.text(
            "CREATE TABLE readings (city TEXT, ts TEXT, value REAL)"))
        rows, _ = syn.trend_season(n=96, seed=50)
        for city in ("SF", "LA"):
            conn.execute(
                sa.text("INSERT INTO readings VALUES (:c, :t, :v)"),
                [{"c": city, "t": t, "v": v} for t, v in rows])

        readings = sa.table(
            "readings", sa.column("city"), sa.column("ts"),
            sa.column("value"))
        stmt = (
            sa.select(
                readings.c.city,
                sa.func.forecast(
                    readings.c.ts, readings.c.value, 4).label("doc"))
            .where(readings.c.city == sa.bindparam("which"))
            .group_by(readings.c.city))
        out = conn.execute(stmt, {"which": "SF"}).fetchall()
        assert len(out) == 1
        d = json.loads(out[0].doc)
        assert d["status"] == "ok" and len(d["rows"]) == 4
