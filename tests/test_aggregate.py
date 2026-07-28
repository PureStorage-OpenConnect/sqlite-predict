"""Aggregate forms of forecast/detect_anomalies (RFC 0005 §4.2.8) and
the expansion functions (§4.2.9).

The two cross-form conformance invariants from §5 live here: an
aggregate call and a single-series query-form call over identical rows
produce identical result hashes, and an inline-series receipt still
replays with match = 1 after the source table has been mutated.
"""

import json
import random
import sqlite3

import pytest
import synthetic as syn


def doc(db, sql, *params):
    row = db.execute(sql, params).fetchone()
    return json.loads(row[0]) if row and row[0] is not None else None


def last_receipt(db, **where):
    clauses = " AND ".join(f"{k} = ?" for k in where)
    sql = "SELECT receipt_id, anchor_kind, anchor, params, input_sql," \
          " input_data, result_hash FROM _predict_receipts"
    if where:
        sql += f" WHERE {clauses}"
    sql += " ORDER BY receipt_id DESC LIMIT 1"
    row = db.execute(sql, tuple(where.values())).fetchone()
    assert row is not None
    return dict(zip(("receipt_id", "anchor_kind", "anchor", "params",
                     "input_sql", "input_data", "result_hash"), row))


# ---- shape and semantics ----


def test_forecast_doc_shape(db):
    rows, _ = syn.trend_season(n=96, seed=21)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert d["status"] == "ok"
    assert d["model"] == "theta-classic"
    assert d["receipt_id"] is not None
    assert len(d["rows"]) == 6
    r = d["rows"][0]
    assert set(r) == {"step", "forecast_timestamp", "forecast",
                      "lower_bound", "upper_bound"}
    assert r["step"] == 1
    assert r["lower_bound"] <= r["forecast"] <= r["upper_bound"]


def test_group_by_matches_query_form_group_cols(db):
    a, _ = syn.trend_season(n=96, seed=22)
    b, _ = syn.random_walk(n=96, seed=23)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")

    agg = {
        grp: json.loads(d)
        for grp, d in db.execute(
            "SELECT grp, forecast(ts, value, 4,"
            " '{\"receipt\": 0}') FROM series GROUP BY grp")
    }
    qry = db.execute(
        "SELECT series_key, step, forecast_timestamp, forecast, lower_bound,"
        " upper_bound FROM forecast('SELECT ts, value, grp FROM series"
        " ORDER BY ts', 4, '{\"group_cols\": \"grp\", \"receipt\": 0}')"
    ).fetchall()

    assert set(agg) == {"a", "b"}
    for key, step, fts, fc, lo, hi in qry:
        r = agg[key]["rows"][step - 1]
        assert r["forecast_timestamp"] == fts
        assert r["forecast"] == pytest.approx(fc, abs=0, rel=0)
        assert r["lower_bound"] == pytest.approx(lo, abs=0, rel=0)
        assert r["upper_bound"] == pytest.approx(hi, abs=0, rel=0)


def test_input_order_does_not_matter(db):
    rows, _ = syn.trend_season(n=96, seed=24)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    syn.load_into(db, rows, table="ordered")
    syn.load_into(db, shuffled, table="shuffled")
    d1 = doc(db, "SELECT forecast(ts, value, 6, '{\"receipt\":0}') FROM ordered")
    d2 = doc(db, "SELECT forecast(ts, value, 6, '{\"receipt\":0}') FROM shuffled")
    assert d1["rows"] == d2["rows"]


def test_zero_rows_returns_null(db):
    db.execute("CREATE TABLE empty(ts TEXT, value REAL)")
    assert db.execute(
        "SELECT forecast(ts, value, 6) FROM empty").fetchone()[0] is None
    assert db.execute(
        "SELECT detect_anomalies(ts, value) FROM empty").fetchone()[0] is None


def test_degraded_series_status_no_receipt(db):
    # too little history: status doc, empty rows, receipt_id null
    syn.load_into(db, [(f"2024-01-01T0{i}:00:00Z", float(i)) for i in range(4)])
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert d["status"] == "insufficient_history"
    assert d["rows"] == []
    assert d["receipt_id"] is None
    if db.execute("SELECT name FROM sqlite_master WHERE name ="
                  " '_predict_receipts'").fetchone():
        assert db.execute(
            "SELECT count(*) FROM _predict_receipts").fetchone()[0] == 0


def test_non_numeric_degrades_not_fails(db):
    rows, _ = syn.trend_season(n=96, seed=25)
    syn.load_into(db, rows)
    db.execute("INSERT INTO series VALUES ('2024-06-01T00:00:00Z', 'not-a-number')")
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert d["status"] == "non_numeric"
    assert d["rows"] == []
    assert d["receipt_id"] is None


# ---- conformance invariant 1: cross-form hash parity ----


def test_aggregate_hash_equals_single_series_query_hash(db):
    rows, _ = syn.trend_season(n=120, seed=26)
    syn.load_into(db, rows)
    db.execute("SELECT forecast(ts, value, 8) FROM series").fetchone()
    db.execute(
        "SELECT count(*) FROM forecast('SELECT ts, value FROM series"
        " ORDER BY ts', 8)").fetchone()
    inline = last_receipt(db, anchor_kind="inline-series")
    query = last_receipt(db, anchor_kind="logical-digest")
    assert inline["result_hash"] == query["result_hash"]


def test_anomaly_aggregate_hash_parity(db):
    rows, _ = syn.trend_season(n=96, seed=27)
    rows, _ = syn.with_anomalies(rows, k=3, seed=28)
    syn.load_into(db, rows)
    db.execute("SELECT detect_anomalies(ts, value) FROM series").fetchone()
    db.execute(
        "SELECT count(*) FROM detect_anomalies('SELECT ts, value FROM series"
        " ORDER BY ts')").fetchone()
    inline = last_receipt(db, anchor_kind="inline-series")
    query = last_receipt(db, anchor_kind="logical-digest")
    assert inline["result_hash"] == query["result_hash"]


# ---- conformance invariant 2: replay survives source mutation ----


def test_inline_receipt_replays_after_source_mutation(db):
    rows, _ = syn.trend_season(n=96, seed=29)
    syn.load_into(db, rows)
    db.execute("SELECT forecast(ts, value, 6) FROM series").fetchone()
    rec = last_receipt(db, anchor_kind="inline-series")

    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)",
        (rec["receipt_id"],)).fetchone()
    assert match == 1, detail

    # the whole point: replay is independent of database state
    db.execute("DELETE FROM series")
    db.execute("INSERT INTO series VALUES ('1999-01-01T00:00:00Z', -1.0)")
    db.commit()
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)",
        (rec["receipt_id"],)).fetchone()
    assert match == 1, detail


def test_anomaly_inline_replay_after_mutation(db):
    rows, _ = syn.trend_season(n=96, seed=30)
    syn.load_into(db, rows)
    db.execute("SELECT detect_anomalies(ts, value) FROM series").fetchone()
    rec = last_receipt(db, operation="detect_anomalies")
    db.execute("DROP TABLE series")
    db.commit()
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)",
        (rec["receipt_id"],)).fetchone()
    assert match == 1, detail


def test_edited_receipt_input_data_is_rejected(db):
    rows, _ = syn.trend_season(n=96, seed=31)
    syn.load_into(db, rows)
    db.execute("SELECT forecast(ts, value, 6) FROM series").fetchone()
    rec = last_receipt(db, anchor_kind="inline-series")
    tampered = rec["input_data"].replace(
        rec["input_data"][12:32], rec["input_data"][12:32], 1)
    # actually change a value: bump the first numeric char after "value":[
    head, _, tail = rec["input_data"].partition('"value":[')
    tampered = head + '"value":[9' + tail[1:]
    db.execute(
        "UPDATE _predict_receipts SET input_data = ? WHERE receipt_id = ?",
        (tampered, rec["receipt_id"]))
    db.commit()
    with pytest.raises(sqlite3.OperationalError, match="ANCHOR_UNAVAILABLE"):
        db.execute("SELECT match FROM predict_replay(?)",
                   (rec["receipt_id"],)).fetchone()


# ---- receipts ----


def test_inline_receipt_fields(db):
    rows, _ = syn.trend_season(n=96, seed=32)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    rec = last_receipt(db, anchor_kind="inline-series")
    assert rec["receipt_id"] == d["receipt_id"]
    assert rec["input_sql"] is None
    data = json.loads(rec["input_data"])
    assert set(data) == {"ts", "value"}
    assert len(data["ts"]) == len(data["value"]) == 96
    assert data["ts"] == sorted(data["ts"])
    params = json.loads(rec["params"])
    # aggregate params carry no query-shape keys (RFC §4.2.8)
    for absent in ("time_col", "value_col", "group_cols"):
        assert absent not in params
    assert params["horizon"] == 6


def test_receipt_opt_out(db):
    rows, _ = syn.trend_season(n=96, seed=33)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 6, '{\"receipt\": 0}') FROM series")
    assert d["status"] == "ok"
    assert d["receipt_id"] is None
    tables = db.execute("SELECT name FROM sqlite_master WHERE name ="
                        " '_predict_receipts'").fetchall()
    if tables:
        assert db.execute(
            "SELECT count(*) FROM _predict_receipts").fetchone()[0] == 0


def test_read_only_database_fails_loudly_without_opt_out(db, tmp_path):
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
        with pytest.raises(sqlite3.OperationalError,
                           match="receipt"):
            ro.execute("SELECT forecast(ts, value, 6) FROM series").fetchone()
        d = json.loads(ro.execute(
            "SELECT forecast(ts, value, 6, '{\"receipt\": 0}') FROM series"
        ).fetchone()[0])
        assert d["status"] == "ok"
        assert d["receipt_id"] is None
    finally:
        ro.close()


def test_one_receipt_per_group(db):
    a, _ = syn.trend_season(n=96, seed=35)
    b, _ = syn.random_walk(n=96, seed=36)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")
    docs = db.execute(
        "SELECT grp, forecast(ts, value, 4) FROM series GROUP BY grp"
    ).fetchall()
    rids = {json.loads(d)["receipt_id"] for _, d in docs}
    assert len(rids) == 2 and None not in rids
    n = db.execute("SELECT count(*) FROM _predict_receipts WHERE"
                   " anchor_kind = 'inline-series'").fetchone()[0]
    assert n == 2


# ---- argument and option validation ----


def test_options_must_be_constant_within_group(db):
    rows, _ = syn.trend_season(n=96, seed=37)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError, match="constant"):
        db.execute(
            "SELECT forecast(ts, value, 4, CASE WHEN value > ("
            " SELECT avg(value) FROM series) THEN '{\"receipt\":0}'"
            " ELSE NULL END) FROM series").fetchone()


def test_horizon_must_be_constant_within_group(db):
    rows, _ = syn.trend_season(n=96, seed=38)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError, match="HORIZON"):
        db.execute(
            "SELECT forecast(ts, value, CAST(1 + (rowid % 2) AS INTEGER))"
            " FROM series").fetchone()


def test_query_shape_options_rejected(db):
    rows, _ = syn.trend_season(n=96, seed=39)
    syn.load_into(db, rows)
    for key in ("time_col", "value_col", "group_cols"):
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
    # 2-arg scalar stub
    with pytest.raises(sqlite3.OperationalError, match="FROM clause"):
        db.execute("SELECT forecast('SELECT ts, value FROM series', 4)")
    # aggregate arity with a query string first argument
    with pytest.raises(sqlite3.OperationalError, match="FROM clause"):
        db.execute(
            "SELECT forecast('SELECT ts FROM series', value, 4) FROM series"
        ).fetchone()
    with pytest.raises(sqlite3.OperationalError, match="FROM clause"):
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
    d = doc(db, "SELECT forecast(ts, value, 4, '{\"receipt\":0}') FROM e")
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
                " '{\"model\": \"sub-pca\", \"receipt\": 0}') FROM series")
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
        " r.upper_bound, r.status, r.receipt_id FROM forecast_rows("
        " (SELECT forecast(ts, value, 5) FROM series)) r").fetchall()
    assert len(out) == 5
    assert [r[0] for r in out] == [1, 2, 3, 4, 5]
    assert all(isinstance(r[2], float) for r in out)
    assert all(r[5] == "ok" for r in out)
    assert len({r[6] for r in out}) == 1 and out[0][6] is not None


def test_anomaly_rows_round_trip(db):
    rows, _ = syn.trend_season(n=96, seed=48)
    syn.load_into(db, rows)
    n, flagged = db.execute(
        "SELECT count(*), sum(is_anomaly) FROM anomaly_rows("
        " (SELECT detect_anomalies(ts, value, '{\"receipt\":0}')"
        " FROM series))").fetchone()
    assert n == 96
    assert flagged is not None


def test_expansion_status_doc_yields_one_status_row(db):
    syn.load_into(db, [("2024-01-01T00:00:00Z", 1.0)])
    out = db.execute(
        "SELECT step, forecast, status, receipt_id FROM forecast_rows("
        " (SELECT forecast(ts, value, 4) FROM series))").fetchall()
    assert out == [(None, None, "insufficient_history", None)]


def test_expansion_null_doc_yields_zero_rows(db):
    assert db.execute(
        "SELECT count(*) FROM forecast_rows(NULL)").fetchone()[0] == 0


def test_expansion_rejects_garbage(db):
    for bad in ("'not json'", "'[1,2,3]'", "'{\"rows\": 1}'", "42"):
        with pytest.raises(sqlite3.OperationalError, match="SCHEMA"):
            db.execute(f"SELECT * FROM forecast_rows({bad})").fetchall()


# ---- side-effect containment (§6.7) ----


def test_aggregate_is_directonly(db):
    rows, _ = syn.trend_season(n=96, seed=49)
    syn.load_into(db, rows)
    db.execute("CREATE VIEW v AS SELECT forecast(ts, value, 4) AS d"
               " FROM series")
    with pytest.raises(sqlite3.OperationalError, match="unsafe"):
        db.execute("SELECT * FROM v").fetchall()


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
                    readings.c.ts, readings.c.value, 4,
                    sa.literal('{"receipt": 0}')).label("doc"))
            .where(readings.c.city == sa.bindparam("which"))
            .group_by(readings.c.city))
        out = conn.execute(stmt, {"which": "SF"}).fetchall()
        assert len(out) == 1
        d = json.loads(out[0].doc)
        assert d["status"] == "ok" and len(d["rows"]) == 4
