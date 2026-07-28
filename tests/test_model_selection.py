"""Auto model selection and interval methods on the forecast aggregate
(RFC §4.2.1): the auto pool, explicit candidates, conformal intervals, and
option validation."""

import json
import sqlite3

import pytest
import synthetic as syn
from conftest import anomaly_doc, forecast_doc


# ---------------------------------------------------------------- auto select

def test_forecast_auto_produces_a_full_horizon(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=9)
    syn.load_into(db, series)
    d = forecast_doc(db, 6, model="auto")
    assert d["status"] == "ok"
    assert len(d["rows"]) == 6


# ---------------------------------------------------------------- conformal

def test_conformal_band_differs_from_residual(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=4)
    syn.load_into(db, series)
    r = forecast_doc(db, 6)["rows"]
    c = forecast_doc(db, 6, interval_method="conformal")["rows"]
    rw = sum(x["upper_bound"] - x["lower_bound"] for x in r)
    cw = sum(x["upper_bound"] - x["lower_bound"] for x in c)
    assert abs(rw - cw) > 1e-6


def test_conformal_and_auto_compose(db):
    series, _ = syn.trend_season(n=220, noise=1.5, seed=6)
    syn.load_into(db, series)
    d = forecast_doc(db, 6, model="auto", interval_method="conformal")
    assert d["status"] == "ok"
    assert len(d["rows"]) == 6
    assert all(x["lower_bound"] <= x["upper_bound"] for x in d["rows"])


def test_conformal_rejects_series_too_short_to_calibrate(db):
    """Fail loud, not silent: too few out-of-sample folds to calibrate a
    conformal band -> insufficient_history, no bogus interval."""
    series, _ = syn.trend_season(n=14, noise=0.5, seed=2)
    syn.load_into(db, series)
    d = forecast_doc(db, 6, interval_method="conformal")
    assert d["status"] == "insufficient_history"
    assert d["rows"] == []


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
        db.execute("SELECT forecast(ts, value, 6, ?) FROM series",
                   (json.dumps(opts),)).fetchone()


# ---------------------------------------------------- tsb (intermittent)

def test_tsb_forecast_is_flat_and_nonnegative(db):
    series, _ = syn.intermittent_trailing_zeros()
    syn.load_into(db, series)
    d = forecast_doc(db, 5, model="tsb")
    fcs = [x["forecast"] for x in d["rows"]]
    assert len(fcs) == 5
    assert all(abs(f - fcs[0]) < 1e-12 for f in fcs)   # intermittent = flat rate
    assert fcs[0] >= 0


def test_tsb_rejected_in_detect_anomalies(db):
    series, _ = syn.intermittent_trailing_zeros()
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError) as e:
        anomaly_doc(db, model="tsb")
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


# ------------------------------------------------------------- auto candidates

def _fc_auto(db, cands, horizon=4):
    d = forecast_doc(db, horizon, model="auto", candidates=cands)
    return [x["forecast"] for x in d["rows"]]


def test_candidates_restrict_and_select_pool(db):
    """The candidate set is the auto pool: restricting it changes the winner,
    and auto picks the better of a mixed set."""
    series, _ = syn.intermittent_trailing_zeros()
    syn.load_into(db, series)
    tsb_only = _fc_auto(db, ["tsb"])
    theta_only = _fc_auto(db, ["theta-classic"])
    both = _fc_auto(db, ["tsb", "theta-classic"])
    assert all(abs(v) < 0.05 for v in tsb_only)      # tsb rate ~ 0 on the tail
    assert any(abs(v) > 0.05 for v in theta_only)     # theta is not flat-zero
    assert both == tsb_only                           # auto picks tsb (the better)


def test_candidates_narrows_the_default_and_rejects_named_models(db):
    series, _ = syn.trend_season(n=100, seed=1)
    syn.load_into(db, series)
    # auto is the default, so bare candidates narrows the default pool
    doc = forecast_doc(db, 6, candidates=["theta-classic"])
    assert doc["status"] == "ok"
    assert doc["model"] == "theta-classic"
    # but candidates alongside a pinned non-auto model is a contradiction
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT forecast(ts, value, 6, ?) FROM series",
                   (json.dumps({"model": "tsb",
                                "candidates": ["theta-classic"]}),)).fetchone()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_candidates_reject_unknown(db):
    series, _ = syn.trend_season(n=100, seed=1)
    syn.load_into(db, series)
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT forecast(ts, value, 6, ?) FROM series",
                   (json.dumps({"model": "auto",
                                "candidates": ["theta-classic", "nope"]}),
                    )).fetchone()


def test_candidates_conformal_with_stat_models_ok(db):
    series, _ = syn.trend_season(n=200, noise=1.0, seed=3)
    syn.load_into(db, series)
    d = forecast_doc(db, 6, model="auto",
                     candidates=["theta-classic", "stub-seasonal-naive"],
                     interval_method="conformal")
    assert len(d["rows"]) == 6
    assert all(x["lower_bound"] <= x["upper_bound"] for x in d["rows"])
