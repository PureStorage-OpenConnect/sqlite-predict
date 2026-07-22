import json
import math
import synthetic as syn


def run_forecast(db, query, horizon, **options):
    opts = json.dumps(options) if options else None
    if opts:
        rows = db.execute(
            "SELECT * FROM forecast(?, ?, ?)", (query, horizon, opts)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM forecast(?, ?)", (query, horizon)
        ).fetchall()
    return rows


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
        out = run_forecast(db, "SELECT ts, value FROM series", 24, model=model)
        assert len(out) == 24
        fcs = [r[3] for r in out]
        model_mae = mae(list(zip(fcs, truth)))
        assert model_mae < flat, (
            f"{model} mae {model_mae:.2f} not better than flat {flat:.2f}"
        )
        assert all(r[6] == "ok" for r in out)


def test_seasonal_phase_is_tracked(db):
    rows, truth_meta = syn.trend_season(n=192, noise=0.3, amplitude=6.0,
                                        seed=12)
    train, future = split_series(rows, 24)
    syn.load_into(db, train)
    truth = [v for _, v in future]
    out = run_forecast(db, "SELECT ts, value FROM series", 24)
    fcs = [r[3] for r in out]

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
    out = run_forecast(db, "SELECT ts, value FROM series", 24,
                       confidence_level=0.95)
    covered = sum(
        1 for r, (_, t) in zip(out, future) if r[4] <= t <= r[5]
    )
    assert covered >= 15, f"95% interval covered only {covered}/24"
    # intervals must widen with horizon
    assert (out[-1][5] - out[-1][4]) > (out[0][5] - out[0][4])


def test_group_cols_split_series(db):
    a, _ = syn.trend_season(n=100, level=20.0, seed=14)
    b, _ = syn.trend_season(n=100, level=80.0, seed=15)
    syn.load_into(db, a, group="alpha")
    syn.load_into(db, b, group="beta")
    out = run_forecast(db, "SELECT ts, value, grp FROM series", 4,
                       group_cols=["grp"])
    keys = {r[0] for r in out}
    assert keys == {"alpha", "beta"}
    assert len(out) == 8
    by_key = {k: [r[3] for r in out if r[0] == k] for k in keys}
    # level separation must survive into the forecasts
    assert min(by_key["beta"]) > max(by_key["alpha"])


def test_insufficient_history_is_status_not_error(db):
    rows, _ = syn.random_walk(n=5, seed=16)
    syn.load_into(db, rows)
    out = run_forecast(db, "SELECT ts, value FROM series", 6)
    assert len(out) == 1
    assert out[0][6] == "insufficient_history"
    assert out[0][1] is None  # no step on status-only rows


def test_context_limit_truncates_with_status(db):
    rows, _ = syn.trend_season(n=150, seed=17)
    syn.load_into(db, rows)
    out = run_forecast(db, "SELECT ts, value FROM series", 3,
                       context_limit=50)
    assert all(r[6] == "truncated" for r in out)
    assert len(out) == 3


def test_integer_epoch_seconds_timestamps(db):
    base = 1_767_225_600  # 2026-01-01T00:00:00Z
    db.execute("CREATE TABLE es(t INTEGER, v REAL)")
    db.executemany(
        "INSERT INTO es VALUES (?, ?)",
        [(base + i * 3600, 10.0 + i) for i in range(50)],
    )
    out = run_forecast(db, "SELECT t, v FROM es", 2)
    assert out[0][2] == "2026-01-03T02:00:00Z"  # hour 50 on the grid
    assert out[0][6] == "ok"


def test_named_column_override(db):
    rows, _ = syn.trend_season(n=100, seed=18)
    db.execute("CREATE TABLE odd(junk TEXT, v REAL, ts TEXT)")
    db.executemany(
        "INSERT INTO odd VALUES ('x', ?, ?)",
        [(v, ts) for ts, v in rows],
    )
    out = run_forecast(db, "SELECT junk, v, ts FROM odd", 3,
                       time_col="ts", value_col="v")
    assert len(out) == 3 and out[0][6] == "ok"


def test_forecast_is_deterministic(db):
    rows, _ = syn.trend_season(n=120, seed=19)
    syn.load_into(db, rows)
    one = run_forecast(db, "SELECT ts, value FROM series", 6)
    two = run_forecast(db, "SELECT ts, value FROM series", 6)
    # result columns are deterministic; receipt ids are fresh per call
    assert [r[:7] for r in one] == [r[:7] for r in two]
    assert one[0][7] != two[0][7]
