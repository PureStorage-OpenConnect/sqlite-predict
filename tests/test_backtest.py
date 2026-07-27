"""Auto model selection, conformal intervals, and the backtest() TVF.

These share one rolling-origin backtest core, so the tests probe the three
surfaces and their edges together. Adversarial by design: short series, huge
gaps, bad options, and the invariants that must hold on every fold.
"""
import json
import sqlite3

import pytest
import synthetic as syn

FC = "SELECT ts, value FROM series"


def rows(db, sql, *params):
    return db.execute(sql, params).fetchall()


def bt(db, horizon, **options):
    return rows(db, "SELECT fold, model, n, mae, rmse, mase, smape, coverage,"
                    " mean_interval_width, status FROM backtest(?, ?, ?)",
                FC, horizon, json.dumps(options))


def avg_mase(db, model, horizon=12, folds=10):
    return rows(db, "SELECT avg(mase) FROM backtest(?, ?, ?)", FC, horizon,
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


def test_forecast_auto_produces_a_full_horizon(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=9)
    syn.load_into(db, series)
    out = rows(db, "SELECT step, forecast, status FROM forecast(?, 6, ?)",
               FC, json.dumps({"model": "auto"}))
    assert len(out) == 6 and all(r[2] == "ok" for r in out)


def test_auto_replays_exactly(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=10)
    syn.load_into(db, series)
    rid = rows(db, "SELECT receipt_id FROM forecast(?, 6, ?) LIMIT 1",
               FC, json.dumps({"model": "auto"}))[0][0]
    assert rows(db, "SELECT match FROM predict_replay(?)", rid)[0][0] == 1


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
        return rows(db, "SELECT avg(coverage) FROM backtest(?, 6, ?)",
                    FC, json.dumps(opts))[0][0]

    conf = coverage(True)
    resid = coverage(False)
    assert 0.82 <= conf <= 1.0, conf
    assert conf > resid, (conf, resid)


def test_conformal_band_differs_from_residual(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=4)
    syn.load_into(db, series)
    r = rows(db, "SELECT lower_bound, upper_bound FROM forecast(?, 6)", FC)
    c = rows(db, "SELECT lower_bound, upper_bound FROM forecast(?, 6, ?)",
             FC, json.dumps({"interval_method": "conformal"}))
    rw = sum(hi - lo for lo, hi in r)
    cw = sum(hi - lo for lo, hi in c)
    assert abs(rw - cw) > 1e-6


def test_conformal_replays_exactly(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=5)
    syn.load_into(db, series)
    rid = rows(db, "SELECT receipt_id FROM forecast(?, 6, ?) LIMIT 1",
               FC, json.dumps({"interval_method": "conformal"}))[0][0]
    assert rows(db, "SELECT match FROM predict_replay(?)", rid)[0][0] == 1


def test_conformal_and_auto_compose(db):
    series, _ = syn.trend_season(n=220, noise=1.5, seed=6)
    syn.load_into(db, series)
    out = rows(db, "SELECT step, lower_bound, upper_bound, status "
                   "FROM forecast(?, 6, ?)",
               FC, json.dumps({"model": "auto", "interval_method": "conformal"}))
    assert len(out) == 6
    assert all(lo <= hi for _s, lo, hi, _st in out)


def test_conformal_rejects_series_too_short_to_calibrate(db):
    """Fail loud, not silent: too few out-of-sample folds to calibrate a
    conformal band -> insufficient_history, no bogus interval."""
    series, _ = syn.trend_season(n=14, noise=0.5, seed=2)
    syn.load_into(db, series)
    out = rows(db, "SELECT status FROM forecast(?, 6, ?)",
               FC, json.dumps({"interval_method": "conformal"}))
    assert len(out) == 1 and out[0][0] == "insufficient_history"


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
        assert mase == 0.0            # naive scale is 0 on a flat series
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


def test_backtest_replays_exactly(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=12)
    syn.load_into(db, series)
    rid = rows(db, "SELECT receipt_id FROM backtest(?, 6, ?) LIMIT 1",
               FC, json.dumps({"folds": 8}))[0][0]
    assert rows(db, "SELECT match FROM predict_replay(?)", rid)[0][0] == 1


def test_backtest_grouped_series(db):
    a, _ = syn.trend_season(n=120, noise=1.0, seed=1)
    b, _ = syn.trend_season(n=120, noise=1.0, seed=2)
    syn.load_into(db, a, group="a")
    syn.load_into(db, b, group="b")
    out = rows(db, "SELECT DISTINCT series_key FROM backtest("
                   "'SELECT ts, value, grp FROM series', 6, ?)",
               json.dumps({"group_cols": ["grp"], "folds": 5}))
    assert {r[0] for r in out} == {"a", "b"}


# ---------------------------------------------------------------- bad options

@pytest.mark.parametrize("opts", [
    {"folds": 0},                       # below 1
    {"folds": 100000},                  # above cap
    {"gap": -1},                        # negative
    {"interval_method": "bogus"},       # not residual/conformal
    {"folds": "x"},                     # wrong type
])
def test_forecast_rejects_bad_options(db, opts):
    series, _ = syn.trend_season(n=100, noise=1.0, seed=1)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError):
        rows(db, "SELECT * FROM forecast(?, 6, ?)", FC, json.dumps(opts))


def test_backtest_rejects_unknown_model(db):
    series, _ = syn.trend_season(n=100, noise=1.0, seed=1)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError):
        rows(db, "SELECT * FROM backtest(?, 6, ?)", FC,
             json.dumps({"model": "no-such-model"}))


# ---------------------------------------------------------------- tsb (intermittent)

def _intermittent_trailing_zeros(n=200, active=25):
    """Early varying demand, then a long tail of zeros: seasonal-naive+drift
    extrapolates a spurious trend, tsb stays at a small non-negative rate."""
    return [(f"2024-01-01T{i // 24:02d}:{i % 24:02d}:00",
             float((i * 7 + 3) % 9 + 3) if i < active else 0.0)
            for i in range(n)]


def test_tsb_forecast_is_flat_and_nonnegative(db):
    syn.load_into(db, _intermittent_trailing_zeros())
    fcs = [r[0] for r in rows(db, "SELECT forecast FROM forecast(?, 5, ?)",
                              FC, json.dumps({"model": "tsb"}))]
    assert len(fcs) == 5
    assert all(abs(f - fcs[0]) < 1e-12 for f in fcs)   # intermittent = flat rate
    assert fcs[0] >= 0


def test_tsb_rejected_in_detect_anomalies(db):
    syn.load_into(db, _intermittent_trailing_zeros())
    with pytest.raises(sqlite3.OperationalError) as e:
        rows(db, "SELECT * FROM detect_anomalies(?, ?)", FC,
             json.dumps({"model": "tsb"}))
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_tsb_wins_intermittent_and_auto_picks_it(db):
    syn.load_into(db, _intermittent_trailing_zeros())
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


# ------------------------------------------------------------- auto candidates

def _fc_auto(db, cands, horizon=4):
    return [r[0] for r in rows(
        db, "SELECT forecast FROM forecast(?, ?, ?)", FC, horizon,
        json.dumps({"model": "auto", "candidates": cands, "receipt": 0}))]


def test_candidates_restrict_and_select_pool(db):
    """The candidate set is the auto pool: restricting it changes the winner,
    and auto picks the better of a mixed set."""
    syn.load_into(db, _intermittent_trailing_zeros())
    tsb_only = _fc_auto(db, ["tsb"])
    theta_only = _fc_auto(db, ["theta-classic"])
    both = _fc_auto(db, ["tsb", "theta-classic"])
    assert all(abs(v) < 0.05 for v in tsb_only)      # tsb rate ~ 0 on the tail
    assert any(abs(v) > 0.05 for v in theta_only)     # theta is not flat-zero
    assert both == tsb_only                           # auto picks tsb (the better)


def test_candidates_replay(db):
    syn.load_into(db, _intermittent_trailing_zeros())
    rid = rows(db, "SELECT receipt_id FROM forecast(?, 6, ?) LIMIT 1", FC,
               json.dumps({"model": "auto",
                           "candidates": ["tsb", "theta-classic"]}))[0][0]
    assert rows(db, "SELECT match FROM predict_replay(?)", rid)[0][0] == 1


def test_candidates_recorded_in_receipt(db):
    syn.load_into(db, _intermittent_trailing_zeros())
    rid = rows(db, "SELECT receipt_id FROM forecast(?, 6, ?) LIMIT 1", FC,
               json.dumps({"model": "auto",
                           "candidates": ["tsb", "theta-classic"]}))[0][0]
    params = rows(db, "SELECT params FROM _predict_receipts WHERE receipt_id=?",
                  rid)[0][0]
    assert json.loads(params)["candidates"] == ["tsb", "theta-classic"]


def test_candidates_requires_auto(db):
    series, _ = syn.trend_season(n=100, seed=1)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError) as e:
        rows(db, "SELECT * FROM forecast(?, 6, ?)", FC,
             json.dumps({"candidates": ["theta-classic"]}))
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_candidates_reject_unknown(db):
    series, _ = syn.trend_season(n=100, seed=1)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError):
        rows(db, "SELECT * FROM forecast(?, 6, ?)", FC,
             json.dumps({"model": "auto", "candidates": ["theta-classic", "nope"]}))


def test_candidates_conformal_with_stat_models_ok(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=3)
    syn.load_into(db, series)
    out = rows(db, "SELECT lower_bound, upper_bound FROM forecast(?, 6, ?)", FC,
               json.dumps({"model": "auto",
                           "candidates": ["theta-classic", "stub-seasonal-naive"],
                           "interval_method": "conformal"}))
    assert len(out) == 6 and all(lo <= hi for lo, hi in out)


# ------------------------- distilled student: distributable + auto candidate

# A distilled forecast student is created and served entirely in the
# zero-dependency core; the onnx runtime is only needed for a live-FM *teacher*.
# Here the training query supplies the future columns directly (a perfect
# teacher baked into the data), so no onnx is involved.

_LS, _HS, _PS = 32, 8, 16


def _wave(t):
    import math
    return (10.0 + 5.0 * math.sin(2 * math.pi * t / _PS)
            + 2.0 * math.sin(2 * math.pi * t / (_PS * 2)))


def _distill_student(db, student_id="stud", horizon=_HS):
    ncol = _LS + horizon
    db.execute("CREATE TABLE w(%s)" % ",".join("c%d REAL" % i for i in range(ncol)))
    db.executemany(
        "INSERT INTO w VALUES (%s)" % ",".join("?" * ncol),
        [[_wave(k + i) for i in range(_LS)]
         + [_wave(k + _LS + j) for j in range(horizon)] for k in range(300)])
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
        [(f"2020-01-01T{i // 60:02d}:{i % 60:02d}:00", _wave(base + i))
         for i in range(n)])


def test_distilled_student_competes_as_auto_candidate(db):
    """No onnx: a student distilled from its own target columns competes in the
    auto pool alongside the baselines, and the run replays deterministically."""
    _distill_student(db, "stud")
    _wave_series(db)
    out = rows(db, "SELECT step, forecast, receipt_id FROM forecast(?, 6, ?)"
                   " ORDER BY step",
               "SELECT ts, value FROM s",
               json.dumps({"model": "auto", "candidates": ["theta-classic", "stud"]}))
    assert len(out) == 6
    assert rows(db, "SELECT match FROM predict_replay(?)", out[0][2])[0][0] == 1


def test_distilled_student_is_distributable(db):
    """Once distilled the student is a portable native row (~kilobytes): copy it
    into a fresh database with no distillation step and it serves in the
    zero-dependency core."""
    import os
    _distill_student(db, "stud")
    src = rows(db, "SELECT model_id, kind, runtime, weights, content_hash, license"
                   " FROM _predict_models WHERE model_id='stud'")[0]

    db2 = sqlite3.connect(":memory:")
    db2.enable_load_extension(True)
    db2.load_extension(os.path.join(os.path.dirname(__file__), "..", "dist",
                                    "predict0"))
    try:
        _wave_series(db2)
        # a forecast ensures the registry tables exist, then transplant the row
        db2.execute("SELECT count(*) FROM forecast('SELECT ts, value FROM s', 2)")
        db2.execute(
            "INSERT INTO _predict_models"
            " (model_id, kind, runtime, weights, content_hash, license)"
            " VALUES (?,?,?,?,?,?)", src)
        n = db2.execute(
            "SELECT count(*) FROM forecast('SELECT ts, value FROM s', 4,"
            " '{\"model\":\"stud\",\"receipt\":0}')").fetchone()[0]
        assert n == 4
    finally:
        db2.close()
