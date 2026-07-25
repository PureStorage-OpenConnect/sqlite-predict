import sqlite3
import pytest
import synthetic as syn


def detect(db, query, opts=None):
    if opts is None:
        return db.execute(
            "SELECT * FROM detect_anomalies(?)", (query,)
        ).fetchall()
    return db.execute(
        "SELECT * FROM detect_anomalies(?, ?)", (query, opts)
    ).fetchall()


def test_injected_anomalies_are_found(db):
    rows, _ = syn.trend_season(n=200, noise=0.5, amplitude=5.0, seed=41)
    noisy, indices = syn.with_anomalies(rows, k=4, magnitude=10.0, seed=42)
    syn.load_into(db, noisy)
    out = detect(db, "SELECT ts, value FROM series")
    flagged = {r[1] for r in out if r[6] == 1}
    truth = {noisy[i][0] for i in indices}
    hits = flagged & truth
    # recall: at least 3 of 4 injected anomalies; precision: at most one
    # extra flag beyond the injected set
    assert len(hits) >= 3, f"missed too many: found {flagged}, truth {truth}"
    assert len(flagged - truth) <= 1, f"false positives: {flagged - truth}"


def test_clean_series_stays_clean(db):
    rows, _ = syn.trend_season(n=200, noise=0.5, seed=43)
    syn.load_into(db, rows)
    out = detect(db, "SELECT ts, value FROM series")
    flags = sum(r[6] or 0 for r in out)
    assert flags <= 2, f"clean series produced {flags} anomaly flags"


def test_long_series_scores_every_point(db):
    """Regression: detect_anomalies must score the whole series. It once
    inherited the forecast path's 4096-point context cap and silently
    truncated to the last 4096 points (status 'truncated'), dropping most of a
    long series' anomalies."""
    rows, _ = syn.trend_season(n=6000, noise=0.5, seed=71)
    noisy, indices = syn.with_anomalies(rows, k=3, magnitude=12.0, seed=72)
    syn.load_into(db, noisy)
    out = detect(db, "SELECT ts, value FROM series")
    assert len(out) == 6000  # every point returned, not just the last 4096
    assert all(r[8] != "truncated" for r in out)
    # an anomaly injected in the FIRST 4096 (dropped under the old cap) is found
    early = [noisy[i][0] for i in indices if i < 4096]
    flagged = {r[1] for r in out if r[6] == 1}
    assert early and any(ts in flagged for ts in early)


def test_warmup_rows_have_no_prediction(db):
    rows, _ = syn.random_walk(n=50, seed=44)
    syn.load_into(db, rows)
    out = detect(db, "SELECT ts, value FROM series")
    assert len(out) == 50
    assert out[0][3] is None  # no forecast on the first row
    assert any(r[3] is not None for r in out[10:])


def test_threshold_option_monotone(db):
    rows, _ = syn.trend_season(n=200, noise=1.0, seed=45)
    noisy, _ = syn.with_anomalies(rows, k=3, magnitude=6.0, seed=46)
    syn.load_into(db, noisy)
    strict = sum(r[6] or 0 for r in
                 detect(db, "SELECT ts, value FROM series",
                        '{"anomaly_prob_threshold": 0.999}'))
    loose = sum(r[6] or 0 for r in
                detect(db, "SELECT ts, value FROM series",
                       '{"anomaly_prob_threshold": 0.9}'))
    assert loose >= strict


def test_threshold_out_of_range(db):
    rows, _ = syn.random_walk(n=50, seed=47)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError) as e:
        detect(db, "SELECT ts, value FROM series",
               '{"anomaly_prob_threshold": 2.0}')
    assert "PREDICT_ERR_THRESHOLD" in str(e.value)


def test_anomalies_receipt_and_replay(db):
    rows, _ = syn.trend_season(n=150, seed=48)
    syn.load_into(db, rows)
    out = detect(db, "SELECT ts, value FROM series")
    rid = out[0][9]
    assert rid is not None
    op, = db.execute(
        "SELECT operation FROM _predict_receipts WHERE receipt_id = ?",
        (rid,),
    ).fetchone()
    assert op == "detect_anomalies"
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)", (rid,)
    ).fetchone()
    assert match == 1, detail


def test_anomalies_replay_refuses_changed_state(db):
    rows, _ = syn.trend_season(n=100, seed=49)
    syn.load_into(db, rows)
    rid = detect(db, "SELECT ts, value FROM series")[0][9]
    db.execute("UPDATE series SET value = value + 1 WHERE rowid = 5")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict_replay(?)", (rid,)).fetchall()
    assert "PREDICT_ERR_ANCHOR_UNAVAILABLE" in str(e.value)


def test_insufficient_history_status(db):
    rows, _ = syn.random_walk(n=4, seed=50)
    syn.load_into(db, rows)
    out = detect(db, "SELECT ts, value FROM series")
    assert len(out) == 1
    assert out[0][8] == "insufficient_history"


def test_naive_model_selectable(db):
    rows, _ = syn.trend_season(n=150, seed=51)
    syn.load_into(db, rows)
    out = detect(db, "SELECT ts, value FROM series",
                 '{"model": "stub-seasonal-naive"}')
    assert len(out) == 150
    rid = out[0][9]
    mid, = db.execute(
        "SELECT model_id FROM _predict_receipts WHERE receipt_id = ?",
        (rid,),
    ).fetchone()
    assert mid == "stub-seasonal-naive"
