"""Forecast distillation: compress a forecast FM into a native gbt student
through the real extension distill()/predict() path, per series.

The reduction turns forecasting into tabular regression, *per series* (the
product framing: you distill the series in front of you, not a global model).
Each training row is

  (3 seasonal-aligned lags, R recent lags, horizon k) -> a CORRECTION target

and one gbt student is fit per series. The target is not the forecast level but
the correction over seasonal-naive (target = value - seasonal_naive[k]); the
student's output is added back to the seasonal-naive base at serve time. That
keeps the target stationary and centered near zero even for trending series --
trees can't extrapolate a level, so regressing the level fails on trends, but
the correction is tree-friendly and it is exactly the part of the forecast
worth distilling: what the FM knows beyond the naive baseline. The
seasonal-aligned lags are the same-phase value from recent cycles; the recent
lags carry level and trend; k makes it a direct multi-horizon forecaster. Two
students are trained on identical rows:

  distill  target = FM_forecast[k]  - seasonal_naive[k]   (compress the FM)
  native   target = actual[k]       - seasonal_naive[k]   (adversarial baseline)

GENERICITY. Every hyperparameter is a fixed function of the seasonal period M
(R=2M recent lags, min-origin 3M, stride M/2), and M comes from dataset
metadata, not from tuning. The SAME pipeline runs on all three regimes. This is
the real test: on m4_hourly the FM beats the stat floor, so distillation
should help; on tourism_monthly the FM does not beat our stat model (see
forecast.md), so a sound method must NOT manufacture a win there. Nothing is
selected on the test window.

FM forecasts are cached per (dataset, series, origin) under
~/.cache/sqlite-predict/forecast-cache/chronos-distill (non-committed).

Run: uv run --with "setuptools<81" --with gluonts --with numpy --with pandas \
       --with chronos-forecasting --with torch python benchmarks/forecast_distill.py
"""
import os, sqlite3
import numpy as np
from forecast_bench import run_chronos, I_MED, FCACHE, EXT, seasonal_naive, mase

# (dataset, seasonal period M) -- M is metadata, the same m MASE is scaled by.
DATASETS = [("m4_hourly", 24), ("tourism_monthly", 12), ("m4_daily", 7)]
N_SERIES, H = 25, 48
DCACHE = os.path.join(FCACHE, "chronos-distill")


def params(M):
    """All reduction hyperparameters as fixed functions of the period M -- no
    per-dataset tuning, no test peeking."""
    R = 2 * M                       # recent lags: two cycles of context
    origin0 = 3 * M                 # min origin covers seas-lag 3 (t-3M) and R
    stride = max(1, M // 2)         # sample origins densely relative to a cycle
    nfeat = 3 + R + 1               # 3 seasonal lags + R recent + horizon
    return R, origin0, stride, nfeat


def chronos_point(name, series_i, origin, hist_upto, h):
    """Chronos median h-step forecast, cached by (dataset, series, origin)."""
    path = os.path.join(DCACHE, name, str(series_i), f"{origin}.npy")
    if os.path.exists(path):
        return np.load(path)
    pt = np.asarray(run_chronos(hist_upto, h)[I_MED], dtype=float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, pt)
    return pt


def feat_row(y, t, k, mean, std, M, R):
    """Features for horizon k off origin t, using only y[:t]."""
    def z(idx):
        return (y[idx] - mean) / std
    d = k // M                                       # cycles origin->target
    seas = [z(t + k - M * (d + c)) for c in (1, 2, 3)]  # same phase, last 3 cycles
    recent = [z(t - 1 - j) for j in range(R)]
    return seas + recent + [float(k)]


def build(name, M, h):
    from gluonts.dataset.repository import get_dataset
    R, origin0, stride, _ = params(M)
    ds = get_dataset(name)
    bundles, cnt = [], 0
    for i, entry in enumerate(ds.test):
        if cnt >= N_SERIES:
            break
        y = np.asarray(entry["target"], dtype=float)
        train_end = len(y) - h          # reserve the last h as the test window
        origins = np.arange(origin0, train_end - h, stride)
        if len(origins) < 8:
            continue
        mean = float(np.mean(y[:train_end]))
        std = float(np.std(y[:train_end])) or 1.0
        # seasonal-naive base for a target step: same-phase value, most recent
        # available cycle. Targets are corrections over this base.
        def base(tt, kk):
            return y[tt + kk - M * (kk // M + 1)]
        rows = []
        for t in origins:
            fc = chronos_point(name, i, t, y[:t], h)
            for k in range(h):
                bs = base(t, k)
                rows.append((feat_row(y, t, k, mean, std, M, R),
                             (fc[k] - bs) / std,             # distill correction
                             (y[t + k] - bs) / std))          # native correction
        teach = chronos_point(name, i, train_end, y[:train_end], h)
        feats = [feat_row(y, train_end, k, mean, std, M, R) for k in range(h)]
        test_base = np.array([base(train_end, k) for k in range(h)])
        bundles.append({"rows": rows, "truth": y[train_end:], "hist": y[:train_end],
                        "teacher": teach, "feats": feats, "std": std,
                        "base": test_base})
        cnt += 1
    return bundles


def distill_one(rows, feats, base, target_idx, std, nfeat):
    """Distill a per-series gbt on the chosen correction target, forecast the
    test window (student output added back to the seasonal-naive base).
    target_idx: 1=FM (distill), 2=native (actuals)."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    cols = ", ".join(f"f{j} REAL" for j in range(nfeat))
    ph = ",".join("?" * (nfeat + 1))
    db.execute(f"CREATE TABLE tr({cols}, target REAL)")
    db.executemany(f"INSERT INTO tr VALUES ({ph})",
                   [list(map(float, r[0])) + [float(r[target_idx])] for r in rows])
    fl = ", ".join(f"f{j}" for j in range(nfeat))
    db.execute(
        f"SELECT holdout_metric FROM distill('SELECT {fl}, target FROM tr',"
        f" json_object('target','target','task','regress','student_id','s',"
        f"'student_kind','gbt'))").fetchone()
    blob = db.execute("SELECT length(weights) FROM _predict_models WHERE"
                      " model_id='s'").fetchone()[0]
    db.execute(f"CREATE TABLE te(id INTEGER, {cols})")
    db.executemany(f"INSERT INTO te VALUES ({ph})",
                   [[k] + list(map(float, f)) for k, f in enumerate(feats)])
    out = db.execute(
        f"SELECT row_ref, prediction FROM predict(NULL, 'SELECT id, {fl} FROM"
        f" te', json_object('model','s')) ORDER BY row_ref").fetchall()
    db.close()
    return np.array([float(r[1]) for r in out]) * std + base, blob


def run_dataset(name, M):
    h = H if name == "m4_hourly" else (24 if name == "tourism_monthly" else 14)
    R, _, _, nfeat = params(M)
    bundles = build(name, M, h)
    agg = {k: [] for k in ("distill", "native", "chronos", "snaive", "fidelity")}
    blob = 0
    for b in bundles:
        dfc, blob = distill_one(b["rows"], b["feats"], b["base"], 1, b["std"], nfeat)
        nfc, _ = distill_one(b["rows"], b["feats"], b["base"], 2, b["std"], nfeat)
        truth, hist = b["truth"], b["hist"]
        agg["distill"].append(mase(dfc, truth, hist, M))
        agg["native"].append(mase(nfc, truth, hist, M))
        agg["chronos"].append(mase(b["teacher"], truth, hist, M))
        agg["snaive"].append(mase(seasonal_naive(hist, h, M), truth, hist, M))
        errs = [abs(hist[j] - hist[j - M]) for j in range(M, len(hist))]
        scale = np.mean(errs) or 1.0
        agg["fidelity"].append(float(np.mean(np.abs(dfc - b["teacher"])) / scale))
    return h, R, nfeat, blob, {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    print(f"per-series forecast distillation, {N_SERIES} series/dataset, "
          f"identical M-derived pipeline\n")
    print(f"  {'dataset':<17}{'M':>3}{'nfeat':>6}{'chronos':>9}{'distill':>9}"
          f"{'native':>9}{'snaive':>9}{'fidel':>8}")
    for name, M in DATASETS:
        h, R, nfeat, blob, a = run_dataset(name, M)
        print(f"  {name:<17}{M:>3}{nfeat:>6}{a['chronos']:>9.3f}"
              f"{a['distill']:>9.3f}{a['native']:>9.3f}{a['snaive']:>9.3f}"
              f"{a['fidelity']:>8.3f}")


if __name__ == "__main__":
    main()
