"""Native forecast student: distill Chronos into a zero-dependency MLP that
trains and serves entirely inside the extension.

This is the end-to-end proof that a *capable* student recovers most of a
forecast FM's edge, where a tree student could not. Chronos labels sliding
context windows (offline, cached); `distill_forecast` fits a multi-output
regression MLP (PSFCST, RFC 0005 §4.1.6) that reproduces the teacher; then
`forecast()` serves that native student with no teacher and no onnxruntime.

On m4_hourly (80 series, context 256, horizon 48):

    chronos-full (teacher)     MASE ~0.79
    native student (this)      MASE ~0.89   <- zero-dep, deterministic
    seasonal-naive (floor)     MASE ~1.00
    gbt tree student           MASE ~1.18   <- trees can't follow

The gap trees left on the table was an architecture-capacity gap, not a
distillation failure (cf. TimeDistill, DistilTS: MLP is the SOTA student for
compressing a forecast FM). The training hyperparameters (epochs/lr, exposed as
distill_forecast options) were chosen on the holdout RMSE, not the test MASE.

Run: uv run --with "setuptools<81" --with gluonts --with numpy --with pandas \
       --with chronos-forecasting --with torch python benchmarks/forecast_native.py
"""
import json, os, sqlite3, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from forecast_bench import run_chronos, I_MED, FCACHE, EXT, mase, seasonal_naive
from gluonts.dataset.repository import get_dataset

NAME, N_SERIES, L, STRIDE, H, M = "m4_hourly", 80, 256, 12, 48, 24
WCACHE = os.path.join(FCACHE, f"chronos-win{L}", NAME)


def teacher(series_i, origin, window):
    """Chronos median forecast on the context window, cached per (series, origin).
    The teacher sees the same window the student will, so the student's ceiling
    is Chronos-on-this-window (reported alongside Chronos-on-full-history)."""
    path = os.path.join(WCACHE, str(series_i), f"{origin}.npy")
    if os.path.exists(path):
        return np.load(path)
    fc = np.asarray(run_chronos(window, H)[I_MED], dtype=float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, fc)
    return fc


def build():
    ds = get_dataset(NAME)
    rows, tests, cnt = [], [], 0
    for si, entry in enumerate(ds.test):
        if cnt >= N_SERIES:
            break
        y = np.asarray(entry["target"], dtype=float)
        train_end = len(y) - H
        if train_end - H < L + 8 * STRIDE:
            continue
        for t in range(L, train_end - H, STRIDE):  # raw window + raw teacher fc
            rows.append(np.concatenate([y[t - L:t], teacher(si, t, y[t - L:t])]))
        tests.append({"hist": y[:train_end], "truth": y[train_end:], "idx": si})
        cnt += 1
    return rows, tests


def main():
    rows, tests = build()
    print(f"{len(tests)} series, {len(rows)} windows")
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    ncol = L + H
    db.execute(f"CREATE TABLE w({','.join(f'c{i} REAL' for i in range(ncol))})")
    db.executemany(f"INSERT INTO w VALUES ({','.join('?' * ncol)})",
                   [list(map(float, r)) for r in rows])
    cols = ",".join(f"c{i}" for i in range(ncol))
    opts = json.dumps({"context": L, "horizon": H, "student_id": "f", "receipt": 0})
    t0 = time.time()
    mid, trows, rmse = db.execute(
        f"SELECT model_id, train_rows, train_rmse FROM distill_forecast("
        f"'SELECT {cols} FROM w', json(?))", (opts,)).fetchone()
    blob = db.execute("SELECT length(weights) FROM _predict_models WHERE"
                      " model_id='f'").fetchone()[0]
    print(f"distilled '{mid}' on {trows} windows in {time.time()-t0:.0f}s "
          f"(holdout rmse {rmse:.3f}, blob {blob} B)")

    sopts = json.dumps({"model": "f", "receipt": 0})
    native, snaive, chronos = [], [], []
    for tb in tests:
        hist, truth = tb["hist"], tb["truth"]
        db.execute("DROP TABLE IF EXISTS s")
        db.execute("CREATE TABLE s(ts TEXT, value REAL)")
        idx = pd.date_range("2020-01-01", periods=len(hist), freq="h")
        db.executemany("INSERT INTO s VALUES (?,?)",
                       [(t.isoformat(), float(v)) for t, v in zip(idx, hist)])
        r = db.execute("SELECT forecast FROM forecast('SELECT ts, value FROM s',"
                       " ?, ?)", (H, sopts)).fetchall()
        native.append(mase(np.array([x[0] for x in r]), truth, hist, M))
        snaive.append(mase(seasonal_naive(hist, H, M), truth, hist, M))
        cpath = os.path.join(FCACHE, "chronos", NAME, f"{tb['idx']}.npy")
        if os.path.exists(cpath):
            chronos.append(mase(np.load(cpath)[I_MED], truth, hist, M))
    db.close()

    def line(name, v):
        a = np.array(v)
        print(f"  {name:28} MASE mean {a.mean():.3f}  median {np.median(a):.3f}")
    print(f"\n== {NAME}: native forecast student ==")
    if chronos:
        line("chronos-full (teacher)", chronos)
    line("native student (zero-dep)", native)
    line("seasonal-naive (floor)", snaive)


if __name__ == "__main__":
    main()
