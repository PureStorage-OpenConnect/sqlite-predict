"""Seeded synthetic series with known ground truth.

Every generator returns (rows, truth) where rows is a list of
(ts_iso, value) tuples on an hourly grid from EPOCH, and truth is a dict
describing the components the generator injected. Determinism is part of
the contract: same arguments, same output, byte for byte.
"""

import math
import random
from datetime import datetime, timedelta, timezone

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _grid(n):
    return [
        (EPOCH + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n)
    ]


def trend_season(n=200, period=24, trend=0.05, amplitude=5.0, noise=0.5,
                 level=20.0, seed=1):
    rng = random.Random(seed)
    ts = _grid(n)
    rows, components = [], []
    for i in range(n):
        t = level + trend * i
        s = amplitude * math.sin(2 * math.pi * i / period)
        e = rng.gauss(0, noise)
        rows.append((ts[i], t + s + e))
        components.append({"trend": t, "season": s, "noise": e})
    truth = {
        "kind": "trend_season", "period": period, "trend": trend,
        "amplitude": amplitude, "noise": noise, "level": level,
        "components": components,
    }
    return rows, truth


def random_walk(n=200, sigma=1.0, start=100.0, seed=2):
    rng = random.Random(seed)
    ts = _grid(n)
    rows, v = [], start
    for i in range(n):
        v += rng.gauss(0, sigma)
        rows.append((ts[i], v))
    return rows, {"kind": "random_walk", "sigma": sigma, "start": start}


def intermittent(n=200, p_demand=0.2, demand_mu=8.0, seed=3):
    rng = random.Random(seed)
    ts = _grid(n)
    rows = []
    for i in range(n):
        v = rng.expovariate(1.0 / demand_mu) if rng.random() < p_demand else 0.0
        rows.append((ts[i], v))
    return rows, {"kind": "intermittent", "p_demand": p_demand,
                  "demand_mu": demand_mu}


def level_shift(n=200, shift_at=120, magnitude=15.0, noise=0.5, level=20.0,
                seed=4):
    rng = random.Random(seed)
    ts = _grid(n)
    rows = []
    for i in range(n):
        base = level + (magnitude if i >= shift_at else 0.0)
        rows.append((ts[i], base + rng.gauss(0, noise)))
    return rows, {"kind": "level_shift", "shift_at": shift_at,
                  "magnitude": magnitude}


def intermittent_trailing_zeros(n=200, active=25):
    """Early varying demand, then a long tail of zeros: seasonal-naive+drift
    extrapolates a spurious trend, tsb stays at a small non-negative rate."""
    rows = [(f"2024-01-01T{i // 24:02d}:{i % 24:02d}:00",
             float((i * 7 + 3) % 9 + 3) if i < active else 0.0)
            for i in range(n)]
    return rows, {"kind": "intermittent_trailing_zeros", "active": active}


def wave(t, period=16):
    """Point sample (not a generator) of the clean multi-seasonal wave the
    distillation tests train and serve on: two sine components over a base
    level, deterministic and learnable."""
    return (10.0 + 5.0 * math.sin(2 * math.pi * t / period)
            + 2.0 * math.sin(2 * math.pi * t / (period * 2)))


def with_anomalies(rows, k=5, magnitude=8.0, seed=5):
    """Inject k point anomalies into an existing series. Returns
    (rows, anomaly_indices). Never places anomalies in the first 20% of
    the series so context-window models have clean history."""
    rng = random.Random(seed)
    rows = list(rows)
    start = max(1, len(rows) // 5)
    indices = sorted(rng.sample(range(start, len(rows)), k))
    for i in indices:
        ts, v = rows[i]
        rows[i] = (ts, v + magnitude * (1 if rng.random() < 0.5 else -1))
    return rows, indices


def load_into(db, rows, table="series", group=None):
    """Create/append a (ts, value[, grp]) table from generated rows."""
    if group is None:
        db.execute(f"CREATE TABLE IF NOT EXISTS {table} (ts TEXT, value REAL)")
        db.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)
    else:
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (ts TEXT, value REAL, grp TEXT)"
        )
        db.executemany(
            f"INSERT INTO {table} VALUES (?, ?, ?)",
            [(ts, v, group) for ts, v in rows],
        )
    db.commit()
