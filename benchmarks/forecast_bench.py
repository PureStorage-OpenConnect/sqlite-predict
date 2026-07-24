"""Real-data forecast benchmark on standard gluonts datasets.

Validates forecast() the way time-series foundation models are validated
(GIFT-Eval / Monash style): seasonal MASE for point accuracy, plus interval
coverage and the Winkler interval score for the probabilistic forecast, against
a seasonal-naive baseline. This is the forecast analog of the TabArena work.

Run:  uv run --with gluonts --with numpy --with pandas python benchmarks/forecast_bench.py

Slice 1: our extension models (theta-classic, stub-seasonal-naive) + a
seasonal-naive baseline. FM teachers (Chronos, TimesFM) and proper CRPS are the
next slice.
"""
import json, os, sqlite3
import numpy as np
import pandas as pd

EXT = os.path.join(os.path.dirname(__file__), "..", "dist", "predict0")
# (gluonts name, series cap, seasonal period m)
DATASETS = [("m4_hourly", 40, 24), ("tourism_monthly", 40, 12),
            ("m4_daily", 40, 7)]
OURS = ("theta-classic", "stub-seasonal-naive")


def seasonal_naive(hist, h, m):
    return [hist[-m + (i % m)] for i in range(h)]


def mase(fc, truth, hist, m):
    errs = [abs(hist[i] - hist[i - m]) for i in range(m, len(hist))]
    scale = np.mean(errs) if errs else np.mean(np.abs(np.diff(hist)))
    if not scale:
        return np.nan
    return float(np.mean(np.abs(np.asarray(fc) - np.asarray(truth))) / scale)


def cov_winkler(iv, truth, alpha=0.05):
    cov = wink = 0.0
    for (lo, hi), y in zip(iv, truth):
        cov += lo <= y <= hi
        w = hi - lo
        if y < lo:
            w += (2 / alpha) * (lo - y)
        elif y > hi:
            w += (2 / alpha) * (y - hi)
        wink += w
    n = len(truth)
    return cov / n, wink / n


def run_ext(model, hist, start, freq, h):
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    db.execute("CREATE TABLE series(ts TEXT, value REAL)")
    idx = pd.period_range(start=start, periods=len(hist), freq=freq).to_timestamp()
    db.executemany("INSERT INTO series VALUES (?,?)",
                   [(t.isoformat(), float(v)) for t, v in zip(idx, hist)])
    opts = json.dumps({"model": model, "confidence_level": 0.95, "receipt": 0})
    rows = db.execute(
        "SELECT forecast, lower_bound, upper_bound FROM forecast('SELECT ts,"
        " value FROM series', ?, ?)", (h, opts)).fetchall()
    db.close()
    return [r[0] for r in rows], [(r[1], r[2]) for r in rows]


def main():
    from gluonts.dataset.repository import get_dataset
    for name, cap, m in DATASETS:
        try:
            ds = get_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {type(e).__name__}: {e}")
            continue
        h, freq = ds.metadata.prediction_length, ds.metadata.freq
        acc = {mm: {"mase": [], "cov": [], "wink": []}
               for mm in (*OURS, "seasonal-naive")}
        cnt = 0
        for entry in ds.test:
            if cnt >= cap:
                break
            y = np.asarray(entry["target"], dtype=float)
            if len(y) < 3 * m + h:
                continue
            hist, truth, start = y[:-h], y[-h:], entry["start"]
            for mm in OURS:
                try:
                    fc, iv = run_ext(mm, hist, start, freq, h)
                    acc[mm]["mase"].append(mase(fc, truth, hist, m))
                    cv, wk = cov_winkler(iv, truth)
                    acc[mm]["cov"].append(cv)
                    acc[mm]["wink"].append(wk)
                except Exception as e:  # noqa: BLE001
                    if cnt == 0:
                        print(f"  {mm} err: {type(e).__name__}: {str(e)[:80]}")
            acc["seasonal-naive"]["mase"].append(
                mase(seasonal_naive(hist, h, m), truth, hist, m))
            cnt += 1
        print(f"\n== {name} (h={h}, freq={freq}, m={m}, {cnt} series) ==")
        for mm, d in acc.items():
            if d["mase"]:
                extra = (f"  cov {np.mean(d['cov']):.0%}  wink {np.mean(d['wink']):.1f}"
                         if d["cov"] else "")
                print(f"  {mm:<22} MASE {np.nanmean(d['mase']):.3f}{extra}")


if __name__ == "__main__":
    main()
