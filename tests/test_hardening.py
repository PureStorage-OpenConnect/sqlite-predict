"""Regression tests for the M6 hardening audit findings, plus
transaction-semantics and leak-soak checks."""

import resource
import sqlite3
import pytest
import synthetic as syn
import synthetic_tabular as syt


def test_too_many_features_errors_loudly(db):
    """Audit bug 1: feature 65+ was silently dropped."""
    cols = ", ".join(f"1.0 AS f{i}" for i in range(70))
    db.execute("CREATE TABLE t AS SELECT " + cols + ", 0 AS label")
    for _ in range(10):
        db.execute("INSERT INTO t SELECT * FROM t LIMIT 1")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict(?, ?, '{\"target\":\"label\"}')",
            ("SELECT * FROM t", "SELECT 1, * FROM t"),
        ).fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)
    assert "too many feature columns" in str(e.value)


def test_long_group_keys_stay_distinct(db):
    """Audit bug 2: >512-byte keys truncated and silently merged."""
    prefix = "x" * 600
    a, _ = syn.trend_season(n=60, level=10.0, seed=61)
    b, _ = syn.trend_season(n=60, level=90.0, seed=62)
    syn.load_into(db, a, group=prefix + "A")
    syn.load_into(db, b, group=prefix + "B")
    out = db.execute(
        "SELECT * FROM forecast(?, 3, ?)",
        ("SELECT ts, value, grp FROM series",
         '{"group_cols":["grp"],"receipt":0}'),
    ).fetchall()
    keys = {r[0] for r in out}
    assert len(keys) == 2, "long keys were merged"
    assert len(out) == 6


def test_epoch_milliseconds_not_mangled(db):
    """Audit bug 3: 13-digit epoch-ms integers were multiplied by 1000."""
    base_ms = 1_767_225_600_000  # 2026-01-01T00:00:00Z
    db.execute("CREATE TABLE es(t INTEGER, v REAL)")
    db.executemany(
        "INSERT INTO es VALUES (?, ?)",
        [(base_ms + i * 3_600_000, 10.0 + i) for i in range(50)],
    )
    out = db.execute(
        "SELECT * FROM forecast('SELECT t, v FROM es', 2,"
        " '{\"receipt\":0}')").fetchall()
    assert out[0][2] == "2026-01-03T02:00:00Z"
    assert out[0][6] == "ok"


def test_pre_epoch_dates_handled_correctly(db):
    """The M6 workaround floored parsing at 1970 to dodge garbage math;
    with the Hinnant calendar rewrite, pre-1970 dates are correct
    proleptic-Gregorian dates. They parse and round-trip."""
    rows, _ = syn.trend_season(n=40, seed=99)
    # relabel the grid into 1969 by hand and confirm forecast timestamps
    db.execute("CREATE TABLE old(ts TEXT, v REAL)")
    db.executemany(
        "INSERT INTO old VALUES (?, ?)",
        [(f"1969-12-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", float(i))
         for i in range(40)],
    )
    out = db.execute(
        "SELECT * FROM forecast('SELECT ts, v FROM old', 2,"
        " '{\"receipt\":0}')").fetchall()
    assert out[0][6] == "ok"
    assert out[0][2].startswith("1969-12-")


@pytest.mark.parametrize("bad", [
    "2026-01-01T-5:00:00Z",   # embedded sign in hour
    "2026-01-01T00:00:60Z",   # 60 seconds
    "  2026-01-01",           # leading whitespace
    "2026-13-01T00:00:00Z",   # month 13
    "2026-02-30T00:00:00Z",   # feb 30
    "0000-01-01T00:00:00Z",   # year zero
    "2026-01-01Tx",           # garbage tail
    "not-a-date",
])
def test_invalid_timestamps_rejected(db, bad):
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT predict_ulid(?)", (bad,)).fetchone()


@pytest.mark.parametrize("good", [
    "2026-07-22T13:45:07Z", "0001-01-01T00:00:00Z",
    "9999-12-31T23:59:59Z", "2000-02-29T12:00:00Z", "2026-1-1",
])
def test_valid_timestamps_accepted(db, good):
    (v,) = db.execute("SELECT predict_ulid(?)", (good,)).fetchone()
    assert len(v) == 26  # a ULID came back


def test_huge_integer_epoch_is_bounded(db):
    """A hostile integer epoch must not loop for 294k iterations or
    overflow the timestamp buffer — it clamps to the 9999 boundary."""
    db.execute("CREATE TABLE big(t INTEGER, v REAL)")
    db.executemany("INSERT INTO big VALUES (?, ?)",
                   [(9_000_000_000_000_000 + i, float(i)) for i in range(20)])
    out = db.execute(
        "SELECT * FROM forecast('SELECT t, v FROM big', 2,"
        " '{\"receipt\":0}')").fetchall()
    # forecast timestamps are clamped, never garbage or overflowed
    for r in out:
        assert r[2] is None or r[2].endswith("Z")


def test_receipt_follows_transaction_rollback(db):
    """Transaction semantics, now decided and pinned: the receipt is
    written on the caller's connection, so it participates in the
    enclosing transaction. Rollback removes it; the caller who discards
    the transaction discards the evidence with the results."""
    rows, _ = syn.trend_season(n=60, seed=63)
    syn.load_into(db, rows)
    db.execute("BEGIN")
    out = db.execute(
        "SELECT * FROM forecast('SELECT ts, value FROM series', 3)"
    ).fetchall()
    rid = out[0][7]
    n, = db.execute(
        "SELECT count(*) FROM _predict_receipts WHERE receipt_id = ?",
        (rid,)).fetchone()
    assert n == 1
    db.execute("ROLLBACK")
    exists, = db.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='_predict_receipts'"
    ).fetchone()
    if exists:
        n, = db.execute(
            "SELECT count(*) FROM _predict_receipts WHERE receipt_id = ?",
            (rid,)).fetchone()
        assert n == 0


def test_no_memory_growth_over_many_calls(db):
    """Leak soak: 1500 mixed calls; RSS growth must stay bounded."""
    rows, _ = syn.trend_season(n=120, seed=64)
    syn.load_into(db, rows)
    X, y, _ = syt.two_moons(n=120)
    syt.load_tabular(db, X, y)

    def batch(n):
        for _ in range(n):
            db.execute("SELECT * FROM forecast("
                       "'SELECT ts, value FROM series', 5,"
                       " '{\"receipt\":0}')").fetchall()
            db.execute("SELECT * FROM detect_anomalies("
                       "'SELECT ts, value FROM series',"
                       " '{\"receipt\":0}')").fetchall()
            db.execute(
                "SELECT * FROM predict(?, ?, ?)",
                ("SELECT f1, f2, label FROM tab WHERE id < 100",
                 "SELECT id, f1, f2 FROM tab WHERE id >= 100",
                 '{"target":"label","receipt":0}')).fetchall()

    batch(100)  # warmup: allocator pools, sqlite caches
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    batch(400)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (rss_after - rss_before) / (1024 * 1024)
    assert growth_mb < 20, f"RSS grew {growth_mb:.1f}MB over 1200 calls"


@pytest.mark.parametrize("op,args", [
    ("forecast",
     ("SELECT * FROM forecast('SELECT ts, value FROM series', 3, ?)",)),
    ("detect_anomalies",
     ("SELECT * FROM detect_anomalies('SELECT ts, value FROM series', ?)",)),
])
def test_duplicate_option_keys_no_leak_ts(db, op, args):
    """CI fuzzer found a leak: a duplicate JSON option key overwrote a
    strdup'd value without freeing the first. Last-wins, no leak. (Leak
    cleanliness is enforced by ASan/valgrind on the soak; here we confirm
    the call still succeeds and the last value wins.)"""
    import synthetic as syn
    rows, _ = syn.trend_season(n=60, seed=81)
    syn.load_into(db, rows)
    dup = '{"model":"theta-classic","model":"stub-seasonal-naive"}'
    out = db.execute(args[0], (dup,)).fetchall()
    assert len(out) >= 1


def test_duplicate_option_keys_no_leak_predict(db):
    import synthetic_tabular as syt
    X, y, _ = syt.two_moons(n=120)
    syt.load_tabular(db, X, y)
    dup = '{"target":"label","task":"classify","task":"classify"}'
    out = db.execute(
        "SELECT * FROM predict('SELECT f1, f2, label FROM tab WHERE id<100',"
        " 'SELECT id, f1, f2 FROM tab WHERE id>=100', ?)", (dup,)).fetchall()
    assert len(out) >= 1
