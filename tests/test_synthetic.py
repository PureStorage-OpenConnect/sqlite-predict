import synthetic as syn


def test_generators_are_deterministic():
    for gen in (syn.trend_season, syn.random_walk, syn.intermittent,
                syn.level_shift):
        a, _ = gen(seed=42)
        b, _ = gen(seed=42)
        assert a == b, f"{gen.__name__} not deterministic"
        c, _ = gen(seed=43)
        assert a != c, f"{gen.__name__} ignores seed"


def test_anomaly_injection_labels_match():
    rows, _ = syn.trend_season(n=100, seed=7)
    noisy, indices = syn.with_anomalies(rows, k=4, seed=8)
    assert len(noisy) == len(rows)
    assert len(indices) == 4
    # anomalies only where labeled, and outside the warmup fifth
    changed = [i for i in range(len(rows)) if rows[i] != noisy[i]]
    assert changed == indices
    assert min(indices) >= len(rows) // 5


def test_grid_is_hourly_iso():
    rows, _ = syn.trend_season(n=3, seed=1)
    assert rows[0][0] == "2026-01-01T00:00:00Z"
    assert rows[1][0] == "2026-01-01T01:00:00Z"


def test_load_into_roundtrip(db):
    rows, _ = syn.random_walk(n=50, seed=9)
    syn.load_into(db, rows)
    n, = db.execute("SELECT count(*) FROM series").fetchone()
    assert n == 50
    first = db.execute("SELECT ts, value FROM series LIMIT 1").fetchone()
    assert first == rows[0]
