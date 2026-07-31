"""The backtest() TVF: rolling-origin fold metrics and their
invariants, auto selection and conformal coverage as measured through
backtest, and the distilled forecast student as a portable auto candidate.
Adversarial by design: short series, huge gaps, bad options, and the
invariants that must hold on every fold.
"""
import json
import sqlite3

import pytest
import synthetic as syn
from conftest import forecast_doc

FC = "SELECT ts, value FROM series"


def rows(db, sql, *params):
    return db.execute(sql, params).fetchall()


def bt(db, horizon, **options):
    return rows(db, "SELECT fold, model, n, mae, rmse, mase, smape, coverage,"
                    " mean_interval_width, status FROM backtest_rows("
                    "(SELECT backtest(ts, value, ?, ?) FROM series))",
                horizon, json.dumps(options))


def avg_mase(db, model, horizon=12, folds=10):
    return rows(db, "SELECT avg(mase) FROM backtest_rows("
                    "(SELECT backtest(ts, value, ?, ?) FROM series))", horizon,
                json.dumps({"model": model, "folds": folds}))[0][0]


# ---------------------------------------------------------------- auto select

def test_auto_equals_the_best_fixed_model(db):
    """auto selects per series by rolling MASE, so its backtest metrics must
    equal the better of the two fixed models exactly (no worse, and it is one
    of them)."""
    series, _ = syn.trend_season(n=200, noise=2.0, seed=7)
    syn.load_into(db, series)
    auto = avg_mase(db, "auto")
    theta = avg_mase(db, "theta-classic")
    snaive = avg_mase(db, "stub-seasonal-naive")
    assert abs(auto - min(theta, snaive)) < 1e-9, (auto, theta, snaive)


def test_auto_reports_a_real_model_name(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=8)
    syn.load_into(db, series)
    names = {r[1] for r in bt(db, 12, model="auto", folds=10)}
    assert names and names <= {"theta-classic", "stub-seasonal-naive"}


# ---------------------------------------------------------------- conformal

def test_conformal_beats_residual_coverage_on_smooth_data(db):
    """The headline: an in-sample Gaussian band is overconfident on smooth
    data, conformal calibrates to the out-of-sample error and lands at nominal.
    Measured through the extension, not in Python."""
    series, _ = syn.trend_season(n=240, noise=1.0, seed=3)
    syn.load_into(db, series)

    def coverage(conformal):
        opts = {"folds": 25, "confidence_level": 0.9}
        if conformal:
            opts["interval_method"] = "conformal"
        return rows(db, "SELECT avg(coverage) FROM backtest_rows("
                        "(SELECT backtest(ts, value, 6, ?) FROM series))",
                    json.dumps(opts))[0][0]

    conf = coverage(True)
    resid = coverage(False)
    assert 0.82 <= conf <= 1.0, conf
    assert conf > resid, (conf, resid)


# ---------------------------------------------------------------- backtest()

def test_backtest_shape_and_invariants(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=8)
    syn.load_into(db, series)
    out = bt(db, 6, folds=10)
    assert len(out) == 10
    for fold, model, n, mae, rmse, mase, smape, cov, width, status in out:
        assert status == "ok"
        assert model == "theta-classic"      # default
        assert n == 6
        assert mae >= 0
        assert rmse + 1e-12 >= mae           # RMS >= mean-abs, always
        assert 0.0 <= cov <= 1.0
        assert 0.0 <= smape <= 2.0
        assert width >= 0


def test_backtest_constant_series_is_exact(db):
    const = [(f"2024-01-01T{i // 24:02d}:{i % 24:02d}:00", 42.0)
             for i in range(48)]
    syn.load_into(db, const)
    out = bt(db, 6, folds=5)
    assert len(out) == 5
    for _fold, _m, _n, mae, rmse, mase, _sm, cov, width, _st in out:
        assert mae == 0.0 and rmse == 0.0
        # MASE = mae / naive_scale is 0/0 on a flat series: the naive scale is 0,
        # so MASE is undefined and reported as null (not 0, which would read as a
        # perfect model). mae == 0 already carries the "exact forecast" signal.
        assert mase is None
        assert cov == 1.0            # zero-width band on an exact forecast
        assert width == 0.0


def test_backtest_gap_is_a_leakage_guard(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=9)
    syn.load_into(db, series)
    assert len(bt(db, 6, folds=5, gap=3)) == 5


def test_backtest_gap_too_large_is_insufficient(db):
    series, _ = syn.trend_season(n=20, noise=0.5, seed=1)
    syn.load_into(db, series)
    out = bt(db, 6, gap=100)
    assert len(out) == 1 and out[0][9] == "insufficient_history"


def test_backtest_auto_selection_matches_best(db):
    series, _ = syn.trend_season(n=200, noise=2.0, seed=11)
    syn.load_into(db, series)
    assert abs(avg_mase(db, "auto")
               - min(avg_mase(db, "theta-classic"),
                     avg_mase(db, "stub-seasonal-naive"))) < 1e-9


def test_backtest_grouped_series(db):
    a, _ = syn.trend_season(n=120, noise=1.0, seed=1)
    b, _ = syn.trend_season(n=120, noise=1.0, seed=2)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")
    out = rows(db, "SELECT g.grp, r.fold, r.mae FROM"
                   " (SELECT grp, backtest(ts, value, 6, ?) AS d FROM series"
                   " GROUP BY grp) g, backtest_rows(g.d) r",
               json.dumps({"folds": 5}))
    assert {r[0] for r in out} == {"a", "b"}
    assert len(out) == 10  # 2 groups x 5 folds; catches truncated grouped output
    # each group produced real folds with populated metrics, not a synthesized
    # blank row from an empty $.rows
    for grp in ("a", "b"):
        maes = [r[2] for r in out if r[0] == grp]
        assert len(maes) == 5 and all(m is not None for m in maes)


def test_backtest_rejects_unknown_model(db):
    series, _ = syn.trend_season(n=100, noise=1.0, seed=1)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError) as e:
        rows(db, "SELECT * FROM backtest_rows("
                 "(SELECT backtest(ts, value, 6, ?) FROM series))",
             json.dumps({"model": "no-such-model"}))
    assert "PREDICT_ERR_MODEL_NOT_FOUND" in str(e.value)


def test_backtest_rejects_unknown_option(db):
    series, _ = syn.trend_season(n=80, noise=1.0, seed=3)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError) as e:
        rows(db, "SELECT * FROM backtest_rows("
                 "(SELECT backtest(ts, value, 6, ?) FROM series))",
             json.dumps({"no_such_option": 1}))
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_backtest_query_string_misuse_fails_loud(db):
    """backtest is an aggregate over rows, not a TVF over a query string; the
    old-TVF-syntax misuse must self-correct rather than silently do nothing."""
    series, _ = syn.trend_season(n=60, noise=1.0, seed=4)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT backtest('SELECT ts, value FROM series', 6)").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


@pytest.mark.parametrize(
    "doc", ["not a document", "42", '{"rows": 5}', '["a", "b"]'])
def test_backtest_rows_rejects_hostile_document(db, doc):
    """backtest_rows() takes a backtest() result document; a non-conforming
    document is rejected cleanly (PREDICT_ERR_SCHEMA), never a crash."""
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM backtest_rows(?)", (doc,)).fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_backtest_rows_null_document_is_empty(db):
    """NULL in, empty out: intentional and consistent with forecast_rows and
    SQLite's own json_each(NULL), which also yield zero rows. This is not a
    swallowed failure. A failed backtest() raises (the error propagates through
    the subquery); NULL only arises from an empty series, i.e. genuinely nothing
    to expand. A hostile *non-NULL* document still fails loud (see
    test_backtest_rows_rejects_hostile_document)."""
    assert db.execute("SELECT * FROM backtest_rows(NULL)").fetchall() == []


def test_backtest_rejects_candidates_option(db):
    """backtest()'s auto path searches only the bundled TS models, so it must not
    silently accept a candidates list it would ignore. The key is rejected as
    unknown (PREDICT_ERR_OPTIONS) rather than parsed and dropped."""
    series, _ = syn.trend_season(n=80, noise=1.0, seed=5)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError) as e:
        rows(db, "SELECT * FROM backtest_rows("
                 "(SELECT backtest(ts, value, 6, ?) FROM series))",
             json.dumps({"candidates": ["theta-classic", "tsb"]}))
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


# ------------------------------------------- tsb (intermittent) via backtest

def test_tsb_wins_intermittent_and_auto_picks_it(db):
    series, _ = syn.intermittent_trailing_zeros()
    syn.load_into(db, series)
    tsb = avg_mase(db, "tsb", horizon=6, folds=15)
    theta = avg_mase(db, "theta-classic", horizon=6, folds=15)
    snaive = avg_mase(db, "stub-seasonal-naive", horizon=6, folds=15)
    auto = avg_mase(db, "auto", horizon=6, folds=15)
    assert tsb < theta and tsb < snaive          # tsb is genuinely better here
    assert abs(auto - tsb) < 1e-9                 # auto selects it


def test_auto_pool_includes_tsb(db):
    series, _ = syn.trend_season(n=200, noise=1.5, seed=13)
    syn.load_into(db, series)
    best = min(avg_mase(db, m) for m in
               ("theta-classic", "stub-seasonal-naive", "tsb"))
    assert abs(avg_mase(db, "auto") - best) < 1e-9


# ------------------------- distilled student: distributable + auto candidate

# A distilled forecast student is created and served entirely in the
# zero-dependency core; the onnx runtime is only needed for a live-FM *teacher*.
# Here the training query supplies the future columns directly (a perfect
# teacher baked into the data), so no onnx is involved.

_LS, _HS = 32, 8


def _distill_student(db, student_id="stud", horizon=_HS):
    ncol = _LS + horizon
    db.execute("CREATE TABLE w(%s)" % ",".join("c%d REAL" % i for i in range(ncol)))
    db.executemany(
        "INSERT INTO w VALUES (%s)" % ",".join("?" * ncol),
        [[syn.wave(k + i) for i in range(_LS)]
         + [syn.wave(k + _LS + j) for j in range(horizon)] for k in range(300)])
    cols = ",".join("c%d" % i for i in range(ncol))
    db.execute("SELECT model_id FROM distill_forecast(?, json(?))",
               ("SELECT %s FROM w" % cols,
                json.dumps({"context": _LS, "horizon": horizon,
                            "student_id": student_id, "hidden": 32,
                            "epochs": 200}))).fetchone()


def _wave_series(db, n=64, base=1000):
    db.execute("CREATE TABLE s(ts TEXT, value REAL)")
    db.executemany(
        "INSERT INTO s VALUES (?,?)",
        [(f"2020-01-01T{i // 60:02d}:{i % 60:02d}:00", syn.wave(base + i))
         for i in range(n)])


def test_distilled_student_competes_as_auto_candidate(db):
    """No onnx: a student distilled from its own target columns competes in the
    auto pool alongside the baselines, deterministically."""
    _distill_student(db, "stud")
    _wave_series(db)
    d = forecast_doc(db, 6, table="s", model="auto",
                     candidates=["theta-classic", "stud"])
    assert d["status"] == "ok"
    assert len(d["rows"]) == 6
    assert forecast_doc(db, 6, table="s", model="auto",
                        candidates=["theta-classic", "stud"]) == d  # deterministic


def test_distilled_student_is_distributable(db):
    """Once distilled the student is a portable native row (~kilobytes): copy it
    into a fresh database with no distillation step and it serves in the
    zero-dependency core."""
    import os
    _distill_student(db, "stud")
    src = rows(db, "SELECT model_id, kind, runtime, weights, content_hash, license"
                   " FROM _predict_models WHERE model_id='stud'")[0]
    # the registry is plain data: transplant its schema and the student row
    ddl = rows(db, "SELECT sql FROM sqlite_master"
                   " WHERE name='_predict_models'")[0][0]

    db2 = sqlite3.connect(":memory:")
    db2.enable_load_extension(True)
    db2.load_extension(os.path.join(os.path.dirname(__file__), "..", "dist",
                                    "predict0"))
    try:
        _wave_series(db2)
        db2.execute(ddl)
        db2.execute(
            "INSERT INTO _predict_models"
            " (model_id, kind, runtime, weights, content_hash, license)"
            " VALUES (?,?,?,?,?,?)", src)
        d = json.loads(db2.execute(
            "SELECT forecast(ts, value, 4, '{\"model\":\"stud\"}') FROM s"
        ).fetchone()[0])
        assert d["status"] == "ok"
        assert len(d["rows"]) == 4
    finally:
        db2.close()
