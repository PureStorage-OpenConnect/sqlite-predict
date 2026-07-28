"""Aggregate forms of forecast/detect_anomalies (RFC 0005 §4.2.8), the
expansion functions (§4.2.9), and predict_verify (§4.2.10).

The cross-form conformance invariants from §5 live here: an aggregate
call and a single-series query-form call over identical rows produce
identical result hashes; predict_verify over the original rows returns
match = 1 from any source that reproduces them (including after the
original table is mutated or dropped); and rows that differ from the
receipt's inputs report match = 0, never a false match.
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
          " result_hash FROM _predict_receipts"
    if where:
        sql += f" WHERE {clauses}"
    sql += " ORDER BY receipt_id DESC LIMIT 1"
    row = db.execute(sql, tuple(where.values())).fetchone()
    assert row is not None
    return dict(zip(("receipt_id", "anchor_kind", "anchor", "params",
                     "input_sql", "result_hash"), row))


def verify(db, receipt, query):
    if isinstance(receipt, dict):
        receipt = json.dumps(receipt)
    return db.execute(
        "SELECT match, detail FROM predict_verify(?, ?)",
        (receipt, query)).fetchone()


# ---- shape and semantics ----


def test_forecast_doc_shape(db):
    rows, _ = syn.trend_season(n=96, seed=21)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert d["status"] == "ok"
    assert d["model"] == "theta-classic"
    assert set(d["receipt"]) == {"op", "model", "model_hash", "params",
                                 "input_digest", "result_hash"}
    assert d["receipt"]["op"] == "forecast"
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
    assert d["receipt"] is None


def test_non_numeric_degrades_not_fails(db):
    rows, _ = syn.trend_season(n=96, seed=25)
    syn.load_into(db, rows)
    db.execute("INSERT INTO series VALUES ('2024-06-01T00:00:00Z', 'not-a-number')")
    d = doc(db, "SELECT forecast(ts, value, 6) FROM series")
    assert d["status"] == "non_numeric"
    assert d["rows"] == []
    assert d["receipt"] is None


# ---- conformance invariant 1: cross-form hash parity ----


def test_aggregate_hash_equals_single_series_query_hash(db):
    rows, _ = syn.trend_season(n=120, seed=26)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 8) FROM series")
    db.execute(
        "SELECT count(*) FROM forecast('SELECT ts, value FROM series"
        " ORDER BY ts', 8)").fetchone()
    query = last_receipt(db, anchor_kind="logical-digest")
    assert d["receipt"]["result_hash"] == query["result_hash"]


def test_anomaly_aggregate_hash_parity(db):
    rows, _ = syn.trend_season(n=96, seed=27)
    rows, _ = syn.with_anomalies(rows, k=3, seed=28)
    syn.load_into(db, rows)
    d = doc(db, "SELECT detect_anomalies(ts, value) FROM series")
    db.execute(
        "SELECT count(*) FROM detect_anomalies('SELECT ts, value FROM series"
        " ORDER BY ts')").fetchone()
    query = last_receipt(db, anchor_kind="logical-digest")
    assert d["receipt"]["result_hash"] == query["result_hash"]


# ---- conformance invariant 2: predict_verify (§4.2.10) ----


def test_verify_round_trip_and_mutation_detection(db):
    rows, _ = syn.trend_season(n=96, seed=29)
    syn.load_into(db, rows)
    rec = doc(db, "SELECT forecast(ts, value, 6) FROM series")["receipt"]

    match, detail = verify(db, rec, "SELECT ts, value FROM series")
    assert match == 1, detail

    # a whole result document is accepted too
    match, detail = db.execute(
        "SELECT match, detail FROM predict_verify("
        "(SELECT forecast(ts, value, 6) FROM series),"
        " 'SELECT ts, value FROM series')").fetchone()
    assert match == 1, detail

    # changed inputs are a finding (match = 0), never a false match
    db.execute("UPDATE series SET value = value + 1 WHERE rowid = 5")
    db.commit()
    match, detail = verify(db, rec, "SELECT ts, value FROM series")
    assert match == 0
    assert "input digest" in detail


def test_verify_survives_source_drop_via_reconstruction(db):
    rows, _ = syn.trend_season(n=96, seed=30)
    syn.load_into(db, rows)
    rec = doc(db, "SELECT detect_anomalies(ts, value) FROM series")["receipt"]
    assert rec["op"] == "detect_anomalies"

    # the durability property: verification needs the rows, not the table
    db.execute("CREATE TEMP TABLE recon AS SELECT ts, value FROM series")
    db.execute("DROP TABLE series")
    db.commit()
    match, detail = verify(db, rec, "SELECT ts, value FROM recon")
    assert match == 1, detail


def test_verify_applies_recorded_context_limit(db):
    # the receipt commits to the rows the model read (the truncated
    # window); verifying with the full history must still match
    rows, _ = syn.trend_season(n=96, seed=31)
    syn.load_into(db, rows)
    rec = doc(db, "SELECT forecast(ts, value, 6,"
                  " '{\"context_limit\": 64}') FROM series")["receipt"]
    assert rec["params"]["context_limit"] == 64
    match, detail = verify(db, rec, "SELECT ts, value FROM series")
    assert match == 1, detail


def test_verify_argument_errors(db):
    rows, _ = syn.trend_season(n=96, seed=54)
    syn.load_into(db, rows)
    rec = doc(db, "SELECT forecast(ts, value, 6) FROM series")["receipt"]
    # a receipt_id-style string is not a document
    with pytest.raises(sqlite3.OperationalError, match="SCHEMA"):
        verify(db, "01NOPE", "SELECT ts, value FROM series")
    for garbage in ("{}", "[1,2]", '{"op": "forecast"}'):
        with pytest.raises(sqlite3.OperationalError, match="SCHEMA"):
            verify(db, garbage, "SELECT ts, value FROM series")
    with pytest.raises(sqlite3.OperationalError, match="NOT_READONLY"):
        verify(db, rec, "DELETE FROM series")


def test_tampered_receipt_is_caught(db):
    rows, _ = syn.trend_season(n=96, seed=55)
    syn.load_into(db, rows)
    rec = doc(db, "SELECT forecast(ts, value, 6) FROM series")["receipt"]

    # edited input digest: a mismatch finding
    bad = dict(rec, input_digest="0" * 64)
    match, detail = verify(db, bad, "SELECT ts, value FROM series")
    assert match == 0 and "input digest" in detail

    # edited result hash: inputs confirm, result does not
    bad = dict(rec, result_hash="0" * 64)
    match, detail = verify(db, bad, "SELECT ts, value FROM series")
    assert match == 0 and "diverged" in detail

    # edited model hash: a hard error, the model is not the one attested
    bad = dict(rec, model_hash="0" * 64)
    with pytest.raises(sqlite3.OperationalError, match="MODEL_HASH"):
        verify(db, bad, "SELECT ts, value FROM series")


# ---- receipts ----


def test_document_receipt_fields(db):
    rows, _ = syn.trend_season(n=96, seed=32)
    syn.load_into(db, rows)
    rec = doc(db, "SELECT forecast(ts, value, 6) FROM series")["receipt"]
    assert rec["op"] == "forecast"
    assert len(rec["input_digest"]) == 64  # digests, never row values
    assert len(rec["result_hash"]) == 64
    assert len(rec["model_hash"]) == 64
    # aggregate params carry no query-shape keys (RFC §4.2.8)
    for absent in ("time_col", "value_col", "group_cols"):
        assert absent not in rec["params"]
    assert rec["params"]["horizon"] == 6


def test_aggregate_is_pure_no_tables_no_writes(db):
    # the aggregate form writes nothing: not even the registry tables
    rows, _ = syn.trend_season(n=96, seed=56)
    syn.load_into(db, rows)
    db.execute("SELECT forecast(ts, value, 6) FROM series").fetchone()
    db.execute("SELECT detect_anomalies(ts, value) FROM series").fetchone()
    assert db.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE"
                      " '_predict%'").fetchone()[0] == 0


def test_receipt_is_constant_size(db):
    # the document receipt's weight must not scale with the series
    import conftest
    sizes = {}
    for n in (96, 4096):
        c = conftest.connect()
        c.execute("CREATE TABLE r(ts INTEGER, value REAL)")
        c.executemany("INSERT INTO r VALUES (?,?)",
                      [(1577836800 + i * 3600, 50.0 + i % 24)
                       for i in range(n)])
        d = json.loads(c.execute(
            "SELECT detect_anomalies(ts, value) FROM r").fetchone()[0])
        sizes[n] = len(json.dumps(d["receipt"]))
        c.close()
    assert sizes[96] == sizes[4096]
    assert sizes[96] < 1000


def test_receipt_opt_out(db):
    rows, _ = syn.trend_season(n=96, seed=33)
    syn.load_into(db, rows)
    d = doc(db, "SELECT forecast(ts, value, 6, '{\"receipt\": 0}') FROM series")
    assert d["status"] == "ok"
    assert d["receipt"] is None


def test_read_only_database_works_with_receipts(db, tmp_path):
    # the aggregate form is pure, so a read-only database serves
    # forecasts with receipts intact, and verifies them too
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
        assert d["receipt"] is not None
        match, detail = verify(ro, d["receipt"],
                               "SELECT ts, value FROM series")
        assert match == 1, detail
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
    recs = [json.loads(d)["receipt"] for _, d in docs]
    assert all(r is not None for r in recs)
    assert len({r["input_digest"] for r in recs}) == 2
    assert len({r["result_hash"] for r in recs}) == 2


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
        " (SELECT detect_anomalies(ts, value, '{\"receipt\":0}')"
        " FROM series))").fetchone()
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
                    readings.c.ts, readings.c.value, 4,
                    sa.literal('{"receipt": 0}')).label("doc"))
            .where(readings.c.city == sa.bindparam("which"))
            .group_by(readings.c.city))
        out = conn.execute(stmt, {"which": "SF"}).fetchall()
        assert len(out) == 1
        d = json.loads(out[0].doc)
        assert d["status"] == "ok" and len(d["rows"]) == 4
