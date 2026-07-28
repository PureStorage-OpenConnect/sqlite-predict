"""distill_forecast() and the native forecast student (PSFCST, RFC 0005 §4.1.6).

distill_forecast fits a multi-output regression MLP that maps an instance-
normalized context window to a horizon of future values, reproducing a
teacher's forecast. It registers an inline-BLOB student that forecast() serves
in the zero-dependency core, with no teacher and no onnxruntime at serve time.

The tests train small and fast via the `epochs`/`hidden` options and use a
clean, learnable multi-seasonal wave so the student's forecast is checkable.
"""

import json
import math
import sqlite3

import pytest

L, H, PERIOD = 32, 8, 16


def _wave(t):
    return (10.0 + 5.0 * math.sin(2 * math.pi * t / PERIOD)
            + 2.0 * math.sin(2 * math.pi * t / (PERIOD * 2)))


def _cols(ncol):
    return ", ".join(f"c{i}" for i in range(ncol))


def _load_windows(db, n_windows=300):
    """A window table: first L columns a wave segment, next H its true
    continuation (a perfect teacher, so the student has a learnable target)."""
    ncol = L + H
    db.execute(f"CREATE TABLE w({', '.join(f'c{i} REAL' for i in range(ncol))})")
    rows = [[_wave(k + i) for i in range(L)] + [_wave(k + L + j) for j in range(H)]
            for k in range(n_windows)]
    db.executemany(f"INSERT INTO w VALUES ({','.join('?' * ncol)})", rows)
    return ncol


def _train(db, student_id="f", **over):
    ncol = _load_windows(db)
    opts = {"context": L, "horizon": H, "student_id": student_id,
            "hidden": 32, "epochs": 250, **over}
    return db.execute(
        f"SELECT model_id, content_hash, train_rows, train_rmse"
        f" FROM distill_forecast('SELECT {_cols(ncol)} FROM w', json(?))",
        (json.dumps(opts),)).fetchone()


def _load_series(db, n=64, base=1000):
    db.execute("CREATE TABLE s(ts TEXT, value REAL)")
    db.executemany("INSERT INTO s VALUES (?,?)",
                   [(f"2020-01-01T{i // 60:02d}:{i % 60:02d}:00", _wave(base + i))
                    for i in range(n)])


def test_distill_forecast_registers_a_student(db):
    model_id, chash, rows, rmse = _train(db)
    assert model_id == "f"
    assert len(chash) == 64
    assert rows == 300
    assert rmse >= 0.0
    kind, runtime, wlen, rhash = db.execute(
        "SELECT kind, runtime, length(weights), content_hash"
        " FROM _predict_models WHERE model_id='f'").fetchone()
    assert kind == "student" and runtime == "tree"
    assert wlen > 0 and rhash == chash


def test_forecast_student_serves_and_tracks_the_signal(db):
    _train(db)
    _load_series(db)
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, ?, json_object('model','f')) FROM s",
        (H,)).fetchone()[0])
    assert doc["status"] == "ok"
    assert len(doc["rows"]) == H
    fc = []
    for r in doc["rows"]:
        f, lo, hi = r["forecast"], r["lower_bound"], r["upper_bound"]
        assert f == f and lo == lo and hi == hi  # finite, not NaN
        assert lo <= f <= hi
        fc.append(f)
    truth = [_wave(1000 + 64 + j) for j in range(H)]
    mf, mt = sum(fc) / H, sum(truth) / H
    cov = sum((a - mf) * (b - mt) for a, b in zip(fc, truth))
    va = sum((a - mf) ** 2 for a in fc)
    vb = sum((b - mt) ** 2 for b in truth)
    corr = cov / ((va * vb) ** 0.5 + 1e-9)
    assert corr > 0.5  # a clean wave: the student follows the continuation


def test_linear_forecast_student_serves_and_tracks(db):
    """hidden=0 trains a pure-linear student (a DLinear-style map, the linear
    skip with no hidden path). It registers a smaller blob than the default
    TiDE student and still follows a clean wave."""
    assert _train(db, student_id="lin", hidden=0)[0] == "lin"
    db.execute("DROP TABLE w")  # _train reloads the window table
    _train(db, student_id="tide")  # default hidden (TiDE) for a size comparison
    lin_len, tide_len = (db.execute(
        "SELECT length(weights) FROM _predict_models WHERE model_id=?",
        (mid,)).fetchone()[0] for mid in ("lin", "tide"))
    assert lin_len < tide_len  # no hidden layer => smaller blob
    _load_series(db)
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, ?, json_object('model','lin')) FROM s",
        (H,)).fetchone()[0])
    assert doc["status"] == "ok"
    assert len(doc["rows"]) == H
    fc = []
    for r in doc["rows"]:
        f, lo, hi = r["forecast"], r["lower_bound"], r["upper_bound"]
        assert f == f and lo <= f <= hi
        fc.append(f)
    truth = [_wave(1000 + 64 + j) for j in range(H)]
    mf, mt = sum(fc) / H, sum(truth) / H
    cov = sum((a - mf) * (b - mt) for a, b in zip(fc, truth))
    va = sum((a - mf) ** 2 for a in fc)
    vb = sum((b - mt) ** 2 for b in truth)
    assert cov / ((va * vb) ** 0.5 + 1e-9) > 0.5


def test_forecast_horizon_may_not_exceed_the_student(db):
    _train(db)
    _load_series(db)
    with pytest.raises(sqlite3.OperationalError, match="horizon"):
        db.execute(
            "SELECT forecast(ts, value, ?, json_object('model','f'))"
            " FROM s", (H + 1,)).fetchall()


def test_predict_rejects_a_forecast_student(db):
    _train(db)
    db.execute("CREATE TABLE q(id INTEGER, f0 REAL)")
    db.execute("INSERT INTO q VALUES (1, 0.5)")
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT * FROM predict(NULL, 'SELECT id, f0 FROM q',"
                   " json_object('model','f'))").fetchall()


def test_distill_forecast_requires_student_id(db):
    ncol = _load_windows(db)
    with pytest.raises(sqlite3.OperationalError, match="student_id"):
        db.execute(f"SELECT * FROM distill_forecast('SELECT {_cols(ncol)} FROM w',"
                   f" json_object('context',{L},'horizon',{H}))").fetchall()


def test_distill_forecast_rejects_column_mismatch(db):
    ncol = _load_windows(db)
    with pytest.raises(sqlite3.OperationalError, match="columns"):
        db.execute("SELECT * FROM distill_forecast('SELECT c0, c1 FROM w',"
                   f" json_object('context',{L},'horizon',{H},'student_id','f'))"
                   ).fetchall()


QL3 = [0.25, 0.5, 0.75]  # a small quantile fan


def _load_qwindows(db, n_windows=300):
    """context + H*Q columns (horizon-major): each horizon's three teacher
    quantiles are the wave continuation with a fixed +-2 spread."""
    ncol = L + H * len(QL3)
    db.execute(f"CREATE TABLE qw({', '.join(f'c{i} REAL' for i in range(ncol))})")
    rows = []
    for k in range(n_windows):
        win = [_wave(k + i) for i in range(L)]
        fan = []
        for j in range(H):
            b = _wave(k + L + j)
            fan += [b - 2.0, b, b + 2.0]  # q0.25, q0.5, q0.75
        rows.append(win + fan)
    db.executemany(f"INSERT INTO qw VALUES ({','.join('?' * ncol)})", rows)
    return ncol


def _train_q(db, sid="qf"):
    ncol = _load_qwindows(db)
    opts = {"context": L, "horizon": H, "student_id": sid, "hidden": 32,
            "epochs": 300, "quantiles": QL3}
    return db.execute(
        f"SELECT model_id, train_rows FROM distill_forecast("
        f"'SELECT {_cols(ncol)} FROM qw', json(?))", (json.dumps(opts),)).fetchone()


def _serve(db, sid, conf):
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, ?, json_object('model',?,"
        "'confidence_level',?)) FROM s", (H, sid, conf)).fetchone()[0])
    assert doc["status"] == "ok"
    return [(r["forecast"], r["lower_bound"], r["upper_bound"])
            for r in doc["rows"]]


def test_quantile_student_serves_a_calibrated_fan(db):
    mid, rows = _train_q(db)
    assert mid == "qf" and rows == 300
    _load_series(db)
    r = _serve(db, "qf", 0.5)  # 50% band = q0.25..q0.75
    assert len(r) == H
    fc = []
    for f, lo, hi in r:
        assert lo <= f <= hi
        assert 0.5 < (hi - lo) < 12  # learned ~+-2 spread, band width ~4
        fc.append(f)
    truth = [_wave(1000 + 64 + j) for j in range(H)]
    mf, mt = sum(fc) / H, sum(truth) / H
    cov = sum((a - mf) * (b - mt) for a, b in zip(fc, truth))
    va = sum((a - mf) ** 2 for a in fc) ** 0.5
    vb = sum((b - mt) ** 2 for b in truth) ** 0.5
    assert cov / (va * vb + 1e-9) > 0.5  # the median tracks the signal


def test_quantile_interval_widens_with_confidence(db):
    _train_q(db)
    _load_series(db)
    w50 = [hi - lo for _, lo, hi in _serve(db, "qf", 0.5)]
    w90 = [hi - lo for _, lo, hi in _serve(db, "qf", 0.9)]  # extrapolated tails
    assert sum(w90) > sum(w50)  # a wider confidence -> a wider band


def test_quantile_bad_levels_rejected(db):
    ncol = L + H * len(QL3)
    db.execute(f"CREATE TABLE qw({', '.join(f'c{i} REAL' for i in range(ncol))})")
    db.execute(f"INSERT INTO qw VALUES ({','.join('0' for _ in range(ncol))})")
    for bad in ([0.5, 1.5], [0.5, 0.5], [0.0, 0.9]):
        opts = {"context": L, "horizon": H, "student_id": "x", "quantiles": bad}
        with pytest.raises(sqlite3.OperationalError, match="quantile"):
            db.execute(f"SELECT * FROM distill_forecast('SELECT {_cols(ncol)} FROM"
                       f" qw', json(?))", (json.dumps(opts),)).fetchall()


def test_forecast_rejects_a_malformed_student_blob(db):
    # the registry is writable by any SQL caller (RFC §6.2): a truncated PSFCST
    # blob (valid magic, nfeat present, nothing after) must be rejected cleanly,
    # never crash the serving path.
    _train(db)  # ensures _predict_models exists
    _load_series(db)
    db.execute(
        "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
        " content_hash, license) VALUES ('bad','student','tree',?,'x',"
        "'unspecified')", (b"PSFCST01\x05\x00\x00\x00",))
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT forecast(ts, value, ?,"
                   " json_object('model','bad')) FROM s", (H,)).fetchall()


def test_forecast_student_is_deterministic(db):
    _train(db)
    _load_series(db)
    q = "SELECT forecast(ts, value, ?, json_object('model','f')) FROM s"
    r1 = db.execute(q, (H,)).fetchone()[0]
    r2 = db.execute(q, (H,)).fetchone()[0]
    assert r1 and r1 == r2  # deterministic serving, byte for byte


# ---- implicit auto discovery (registered students compete by default) ----


def test_auto_discovers_registered_student(db):
    # once distilled, the student competes under model="auto" with no
    # candidates list; on its own training wave it wins, and the document
    # reports the winner rather than "auto"
    _train(db)
    _load_series(db)
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, ?, '{\"model\":\"auto\"}') FROM s",
        (H,)).fetchone()[0])
    assert doc["status"] == "ok"
    assert doc["model"] == "f"


def test_auto_implicit_equals_explicit_pool(db):
    _train(db)
    _load_series(db)
    implicit = db.execute(
        "SELECT forecast(ts, value, ?, '{\"model\":\"auto\"}') FROM s",
        (H,)).fetchone()[0]
    explicit = db.execute(
        "SELECT forecast(ts, value, ?, '{\"model\":\"auto\",\"candidates\":"
        "[\"theta-classic\",\"stub-seasonal-naive\",\"tsb\",\"f\"]}') FROM s",
        (H,)).fetchone()[0]
    assert implicit == explicit


def test_auto_skips_student_beyond_its_horizon(db):
    # discovery filters by eligibility: a horizon the student was not
    # trained for silently drops it from the pool (a stat model serves)
    _train(db)
    _load_series(db)
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, ?, '{\"model\":\"auto\"}') FROM s",
        (H + 4,)).fetchone()[0])
    assert doc["status"] == "ok"
    assert doc["model"] in ("theta-classic", "stub-seasonal-naive", "tsb")


def test_auto_conformal_skips_students(db):
    _train(db)
    _load_series(db)
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, 4, '{\"model\":\"auto\","
        "\"interval_method\":\"conformal\"}') FROM s").fetchone()[0])
    assert doc["status"] == "ok"
    assert doc["model"] in ("theta-classic", "stub-seasonal-naive", "tsb")


def test_auto_errors_on_corrupt_registered_student(db):
    # discovery must not paper over a corrupt registry row
    _train(db)
    _load_series(db)
    db.execute(
        "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
        " content_hash, license) VALUES ('bad','student','tree',?,'x',"
        "'unspecified')", (b"PSFCST01\x05\x00\x00\x00",))
    with pytest.raises(sqlite3.OperationalError, match="invalid forecast"):
        db.execute("SELECT forecast(ts, value, ?, '{\"model\":\"auto\"}')"
                   " FROM s", (H,)).fetchone()


def test_auto_stays_pure_without_a_registry(db):
    # no registry, no students: auto serves the stats and writes nothing
    _load_series(db)
    doc = json.loads(db.execute(
        "SELECT forecast(ts, value, 4, '{\"model\":\"auto\"}') FROM s"
    ).fetchone()[0])
    assert doc["status"] == "ok"
    assert doc["model"] in ("theta-classic", "stub-seasonal-naive", "tsb")
    assert db.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE"
                      " '_predict%'").fetchone()[0] == 0
