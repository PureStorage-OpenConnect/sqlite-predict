import json
import math
import synthetic as syn
from conftest import forecast_doc


def mae(pairs):
    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def split_series(rows, holdout):
    return rows[:-holdout], rows[-holdout:]


def test_models_beat_flat_forecast_on_structured_series(db):
    rows, _ = syn.trend_season(n=200, noise=1.0, seed=11)
    train, future = split_series(rows, 24)
    syn.load_into(db, train)
    truth = [v for _, v in future]
    flat = mae([(train[-1][1], t) for t in truth])

    for model in ("theta-classic", "stub-seasonal-naive"):
        doc = forecast_doc(db, 24, model=model)
        assert doc["status"] == "ok"
        assert len(doc["rows"]) == 24
        fcs = [r["forecast"] for r in doc["rows"]]
        model_mae = mae(list(zip(fcs, truth)))
        assert model_mae < flat, (
            f"{model} mae {model_mae:.2f} not better than flat {flat:.2f}"
        )


def test_seasonal_phase_is_tracked(db):
    rows, truth_meta = syn.trend_season(n=192, noise=0.3, amplitude=6.0,
                                        seed=12)
    train, future = split_series(rows, 24)
    syn.load_into(db, train)
    truth = [v for _, v in future]
    doc = forecast_doc(db, 24)
    fcs = [r["forecast"] for r in doc["rows"]]

    # de-mean both and correlate: seasonal phase must align
    mf = sum(fcs) / len(fcs)
    mt = sum(truth) / len(truth)
    num = sum((f - mf) * (t - mt) for f, t in zip(fcs, truth))
    den = math.sqrt(sum((f - mf) ** 2 for f in fcs) *
                    sum((t - mt) ** 2 for t in truth))
    assert den > 0 and num / den > 0.7


def test_interval_coverage_reasonable(db):
    rows, _ = syn.trend_season(n=200, noise=1.0, seed=13)
    train, future = split_series(rows, 24)
    syn.load_into(db, train)
    doc = forecast_doc(db, 24, confidence_level=0.95)
    out = doc["rows"]
    covered = sum(
        1 for r, (_, t) in zip(out, future)
        if r["lower_bound"] <= t <= r["upper_bound"]
    )
    assert covered >= 15, f"95% interval covered only {covered}/24"
    # intervals must widen with horizon
    assert (out[-1]["upper_bound"] - out[-1]["lower_bound"]) > \
           (out[0]["upper_bound"] - out[0]["lower_bound"])


def test_group_by_splits_series(db):
    a, _ = syn.trend_season(n=100, level=20.0, seed=14)
    b, _ = syn.trend_season(n=100, level=80.0, seed=15)
    syn.load_into(db, a, group="alpha")
    syn.load_into(db, b, group="beta")
    out = db.execute(
        "SELECT grp, forecast(ts, value, 4) FROM series GROUP BY grp"
    ).fetchall()
    docs = {g: json.loads(d) for g, d in out}
    assert set(docs) == {"alpha", "beta"}
    assert all(len(d["rows"]) == 4 for d in docs.values())
    by_key = {k: [r["forecast"] for r in d["rows"]] for k, d in docs.items()}
    # level separation must survive into the forecasts
    assert min(by_key["beta"]) > max(by_key["alpha"])


def test_insufficient_history_is_status_not_error(db):
    rows, _ = syn.random_walk(n=5, seed=16)
    syn.load_into(db, rows)
    doc = forecast_doc(db, 6)
    assert doc["status"] == "insufficient_history"
    assert doc["rows"] == []  # no forecast rows on a degraded series


def test_context_limit_truncates_with_status(db):
    rows, _ = syn.trend_season(n=150, seed=17)
    syn.load_into(db, rows)
    doc = forecast_doc(db, 3, context_limit=50)
    assert doc["status"] == "truncated"
    assert len(doc["rows"]) == 3


def test_integer_epoch_seconds_timestamps(db):
    base = 1_767_225_600  # 2026-01-01T00:00:00Z
    db.execute("CREATE TABLE es(t INTEGER, v REAL)")
    db.executemany(
        "INSERT INTO es VALUES (?, ?)",
        [(base + i * 3600, 10.0 + i) for i in range(50)],
    )
    doc = forecast_doc(db, 2, table="es", ts_col="t", value_col="v")
    assert doc["status"] == "ok"
    # hour 50 on the grid
    assert doc["rows"][0]["forecast_timestamp"] == "2026-01-03T02:00:00Z"


def test_forecast_is_deterministic(db):
    rows, _ = syn.trend_season(n=120, seed=19)
    syn.load_into(db, rows)
    one = forecast_doc(db, 6)
    two = forecast_doc(db, 6)
    assert one == two
