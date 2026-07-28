import sqlite3
import pytest
import synthetic as syn
from conftest import anomaly_doc


def test_injected_anomalies_are_found(db):
    rows, _ = syn.trend_season(n=200, noise=0.5, amplitude=5.0, seed=41)
    noisy, indices = syn.with_anomalies(rows, k=4, magnitude=10.0, seed=42)
    syn.load_into(db, noisy)
    out = anomaly_doc(db)["rows"]
    flagged = {r["ts"] for r in out if r["is_anomaly"] == 1}
    truth = {noisy[i][0] for i in indices}
    hits = flagged & truth
    # recall: at least 3 of 4 injected anomalies; precision: at most one
    # extra flag beyond the injected set
    assert len(hits) >= 3, f"missed too many: found {flagged}, truth {truth}"
    assert len(flagged - truth) <= 1, f"false positives: {flagged - truth}"


def test_clean_series_stays_clean(db):
    rows, _ = syn.trend_season(n=200, noise=0.5, seed=43)
    syn.load_into(db, rows)
    out = anomaly_doc(db)["rows"]
    flags = sum(r["is_anomaly"] or 0 for r in out)
    assert flags <= 2, f"clean series produced {flags} anomaly flags"


def test_long_series_scores_every_point(db):
    """Regression: detect_anomalies must score the whole series. It once
    inherited the forecast path's 4096-point context cap and silently
    truncated to the last 4096 points (status 'truncated'), dropping most of a
    long series' anomalies."""
    rows, _ = syn.trend_season(n=6000, noise=0.5, seed=71)
    noisy, indices = syn.with_anomalies(rows, k=3, magnitude=12.0, seed=72)
    syn.load_into(db, noisy)
    doc = anomaly_doc(db)
    assert doc["status"] != "truncated"
    out = doc["rows"]
    assert len(out) == 6000  # every point returned, not just the last 4096
    # an anomaly injected in the FIRST 4096 (dropped under the old cap) is found
    early = [noisy[i][0] for i in indices if i < 4096]
    flagged = {r["ts"] for r in out if r["is_anomaly"] == 1}
    assert early and any(ts in flagged for ts in early)


def test_sub_pca_flags_injected_anomalies(db):
    """The sub-pca model is a subsequence-reconstruction detector (not a
    forecaster): no forecast/interval, and injected anomalies land in the
    high-percentile tail of the reconstruction score."""
    rows, _ = syn.trend_season(n=600, noise=0.4, amplitude=5.0, seed=61)
    noisy, indices = syn.with_anomalies(rows, k=3, magnitude=12.0, seed=62)
    syn.load_into(db, noisy)
    out = anomaly_doc(db, model="sub-pca")["rows"]
    assert len(out) == 600
    assert all(r["forecast"] is None for r in out)  # no forecast
    # every point scored
    assert all(r["anomaly_probability"] is not None for r in out)
    # each injected anomaly sits in a high-percentile neighbourhood (recall)
    for i in indices:
        peak = max(out[j]["anomaly_probability"]
                   for j in range(max(0, i - 15), min(600, i + 15)))
        assert peak > 0.9, f"anomaly at {i} not surfaced (peak {peak:.3f})"


def test_warmup_rows_have_no_prediction(db):
    rows, _ = syn.random_walk(n=50, seed=44)
    syn.load_into(db, rows)
    out = anomaly_doc(db)["rows"]
    assert len(out) == 50
    assert out[0]["forecast"] is None  # no forecast on the first row
    assert any(r["forecast"] is not None for r in out[10:])


def test_threshold_option_monotone(db):
    rows, _ = syn.trend_season(n=200, noise=1.0, seed=45)
    noisy, _ = syn.with_anomalies(rows, k=3, magnitude=6.0, seed=46)
    syn.load_into(db, noisy)
    strict = sum(r["is_anomaly"] or 0 for r in
                 anomaly_doc(db, anomaly_prob_threshold=0.999)["rows"])
    loose = sum(r["is_anomaly"] or 0 for r in
                anomaly_doc(db, anomaly_prob_threshold=0.9)["rows"])
    assert loose >= strict


def test_threshold_out_of_range(db):
    rows, _ = syn.random_walk(n=50, seed=47)
    syn.load_into(db, rows)
    with pytest.raises(sqlite3.OperationalError) as e:
        anomaly_doc(db, anomaly_prob_threshold=2.0)
    assert "PREDICT_ERR_THRESHOLD" in str(e.value)


def test_insufficient_history_status(db):
    rows, _ = syn.random_walk(n=4, seed=50)
    syn.load_into(db, rows)
    doc = anomaly_doc(db)
    assert doc["status"] == "insufficient_history"
    assert doc["rows"] == []


def test_naive_model_selectable(db):
    rows, _ = syn.trend_season(n=150, seed=51)
    syn.load_into(db, rows)
    doc = anomaly_doc(db, model="stub-seasonal-naive")
    assert len(doc["rows"]) == 150
    assert doc["model"] == "stub-seasonal-naive"
