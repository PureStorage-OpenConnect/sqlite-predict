import json
import re
import sqlite3
import pytest
import synthetic as syn

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture
def seeded(db):
    rows, _ = syn.trend_season(n=120, seed=31)
    syn.load_into(db, rows)
    return db


def forecast_rows(db, horizon=3, opts=None):
    if opts is None:
        return db.execute(
            "SELECT * FROM forecast('SELECT ts, value FROM series', ?)",
            (horizon,),
        ).fetchall()
    return db.execute(
        "SELECT * FROM forecast('SELECT ts, value FROM series', ?, ?)",
        (horizon, opts),
    ).fetchall()


def test_receipt_written_and_stamped_on_all_rows(seeded):
    out = forecast_rows(seeded)
    rids = {r[7] for r in out}
    assert len(rids) == 1
    rid = rids.pop()
    assert ULID_RE.match(rid)
    row = seeded.execute(
        "SELECT operation, model_id, anchor_kind, params, input_sql"
        " FROM _predict_receipts WHERE receipt_id = ?",
        (rid,),
    ).fetchone()
    assert row[0] == "forecast"
    assert row[1] == "theta-classic"
    assert row[2] == "logical-digest"
    params = json.loads(row[3])
    assert params["horizon"] == 3
    assert params["time_col"] == "ts" and params["value_col"] == "value"
    assert row[4] == "SELECT ts, value FROM series"


def test_receipt_opt_out(seeded):
    out = forecast_rows(seeded, opts='{"receipt": 0}')
    assert all(r[7] is None for r in out)
    n, = seeded.execute(
        "SELECT count(*) FROM sqlite_master WHERE name = '_predict_receipts'"
    ).fetchone()
    # opt-out on a fresh db must not even create the tables
    assert n == 0


def test_bundled_models_registered(seeded):
    forecast_rows(seeded)
    rows = seeded.execute(
        "SELECT model_id, kind, runtime, license FROM _predict_models"
        " ORDER BY model_id"
    ).fetchall()
    kinds = {r[0]: r[1] for r in rows}
    assert kinds["theta-classic"] == "ts-stat"
    assert kinds["stub-seasonal-naive"] == "ts-stat"
    assert kinds["knn5-incontext"] == "tabular-stat"
    for _, _, runtime, license in rows:
        assert runtime == "bundled"
        assert license == "MIT OR Apache-2.0"


def test_replay_round_trip(seeded):
    out = forecast_rows(seeded, horizon=6)
    rid = out[0][7]
    match, rh, orig, detail = seeded.execute(
        "SELECT * FROM predict_replay(?)", (rid,)
    ).fetchone()
    assert match == 1
    assert rh == orig
    assert "reproduced" in detail


def test_replay_round_trip_grouped(db):
    a, _ = syn.trend_season(n=100, seed=32)
    b, _ = syn.random_walk(n=100, seed=33)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")
    out = db.execute(
        "SELECT * FROM forecast(?, 4, ?)",
        ("SELECT ts, value, grp FROM series",
         '{"group_cols": ["grp"]}'),
    ).fetchall()
    rid = out[0][7]
    match, = db.execute(
        "SELECT match FROM predict_replay(?)", (rid,)
    ).fetchone()
    assert match == 1


def test_replay_refuses_changed_state(seeded):
    rid = forecast_rows(seeded)[0][7]
    seeded.execute("INSERT INTO series VALUES ('2027-01-01T00:00:00Z', 1.0)")
    with pytest.raises(sqlite3.OperationalError) as e:
        seeded.execute("SELECT * FROM predict_replay(?)", (rid,)).fetchall()
    assert "PREDICT_ERR_ANCHOR_UNAVAILABLE" in str(e.value)


def test_replay_unknown_receipt(seeded):
    forecast_rows(seeded)  # ensure tables exist
    with pytest.raises(sqlite3.OperationalError) as e:
        seeded.execute("SELECT * FROM predict_replay('01NOPE')").fetchall()
    assert "PREDICT_ERR_RECEIPT_NOT_FOUND" in str(e.value)


def test_repeat_calls_same_hash_distinct_ids(seeded):
    r1 = forecast_rows(seeded)[0][7]
    r2 = forecast_rows(seeded)[0][7]
    assert r1 != r2
    h1, a1 = seeded.execute(
        "SELECT result_hash, anchor FROM _predict_receipts"
        " WHERE receipt_id = ?", (r1,)
    ).fetchone()
    h2, a2 = seeded.execute(
        "SELECT result_hash, anchor FROM _predict_receipts"
        " WHERE receipt_id = ?", (r2,)
    ).fetchone()
    # receipts are excluded from the logical digest, so both anchor and
    # result hash must be identical across back-to-back calls
    assert h1 == h2
    assert a1 == a2


def test_receipts_survive_model_landscape(seeded):
    """Receipt rows must reference the model by value (id+hash), not by
    foreign key: deleting the model row leaves the receipt intact and
    replay reports MODEL_NOT_FOUND."""
    rid = forecast_rows(seeded)[0][7]
    seeded.execute("DELETE FROM _predict_models WHERE model_id = 'theta-classic'")
    still, = seeded.execute(
        "SELECT count(*) FROM _predict_receipts WHERE receipt_id = ?",
        (rid,),
    ).fetchone()
    assert still == 1
    with pytest.raises(sqlite3.OperationalError) as e:
        seeded.execute("SELECT * FROM predict_replay(?)", (rid,)).fetchall()
    assert "PREDICT_ERR_MODEL_NOT_FOUND" in str(e.value)
