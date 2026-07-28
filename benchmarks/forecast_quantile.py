"""Native quantile forecast student: distill Chronos's decile fan into a
zero-dependency MLP that emits a calibrated interval, and score it end-to-end
through the extension.

`distill_forecast` with a `quantiles` option fits a regression MLP whose outputs
are the teacher's quantiles per horizon step (a PSFCST blob);
`forecast()` reads the point and the interval straight off that fan. This is the
CRPS-scoreable completion of the point student in `forecast_native.py`.

Why distill the fan rather than train pinball on actuals: pinball must estimate
quantiles from limited samples and collapses the spread (under-covers ~50-65%),
while Chronos's fan is already calibrated, so distilling it inherits that
calibration (~78% coverage of an 80% band). See `results/forecast.md`.

On m4_hourly (60 series, context 256, horizon 48, deciles):

    chronos-full (teacher)     MASE ~0.79   80% cov ~79%   mwQL ~0.016
    native quantile student    MASE ~0.89   80% cov ~78%   mwQL ~0.018

Run: uv run --with "setuptools<81" --with gluonts --with numpy --with pandas \
       --with chronos-forecasting --with torch python benchmarks/forecast_quantile.py
"""
import json, os, sqlite3, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from forecast_bench import run_chronos, I_MED, QL, FCACHE, EXT, mase, cov_winkler
from gluonts.dataset.repository import get_dataset

NAME, N_SERIES, L, STRIDE, H, M = "m4_hourly", 60, 256, 12, 48, 24
QLEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q = len(QLEV)
FANC = os.path.join(FCACHE, f"chronos-fan{L}", NAME)
DEC = [i for i, q in enumerate(QL) if 0.1 <= q <= 0.9]  # decile indices in QL


def fan(series_i, origin, window):
    """Chronos's decile fan [H, Q] on the context window, cached per origin."""
    path = os.path.join(FANC, str(series_i), f"{origin}.npy")
    if os.path.exists(path):
        return np.load(path)
    f = np.asarray(run_chronos(window, H), dtype=float)[DEC].T  # [H, Q]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, f)
    return f


def mwql(dec, truth):  # dec [H, Q], normalized weighted quantile loss
    denom = float(np.sum(np.abs(truth))) or 1.0
    num = 0.0
    for qi, q in enumerate(QLEV):
        e = truth - dec[:, qi]
        num += float(np.sum(np.maximum(q * e, (q - 1) * e)))
    return num / (Q * denom)


def build():
    ds = get_dataset(NAME)
    rows, tests, cnt = [], [], 0
    for si, entry in enumerate(ds.test):
        if cnt >= N_SERIES:
            break
        y = np.asarray(entry["target"], dtype=float)
        te = len(y) - H
        if te - H < L + 8 * STRIDE:
            continue
        for t in range(L, te - H, STRIDE):  # raw window + raw teacher fan (H*Q)
            rows.append(np.concatenate([y[t - L:t],
                                        fan(si, t, y[t - L:t]).reshape(-1)]))
        cp = os.path.join(FCACHE, "chronos", NAME, f"{si}.npy")
        tests.append({"hist": y[:te], "truth": y[te:],
                      "cfan": np.load(cp) if os.path.exists(cp) else None})
        cnt += 1
    return rows, tests


def main():
    rows, tests = build()
    print(f"{len(tests)} series, {len(rows)} windows, {L + H * Q} cols")
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    ncol = L + H * Q
    db.execute(f"CREATE TABLE w({','.join(f'c{i} REAL' for i in range(ncol))})")
    db.executemany(f"INSERT INTO w VALUES ({','.join('?' * ncol)})",
                   [list(map(float, r)) for r in rows])
    cols = ",".join(f"c{i}" for i in range(ncol))
    opts = json.dumps({"context": L, "horizon": H, "student_id": "qf",
                       "receipt": 0, "quantiles": QLEV})
    t0 = time.time()
    mid, tr, rmse = db.execute(
        f"SELECT model_id, train_rows, train_rmse FROM distill_forecast("
        f"'SELECT {cols} FROM w', json(?))", (opts,)).fetchone()
    print(f"distilled '{mid}' on {tr} windows in {time.time()-t0:.0f}s "
          f"(holdout rmse {rmse:.3f})")

    def serve(hist, conf):
        db.execute("DROP TABLE IF EXISTS s")
        db.execute("CREATE TABLE s(ts TEXT, value REAL)")
        idx = pd.date_range("2020-01-01", periods=len(hist), freq="h")
        db.executemany("INSERT INTO s VALUES (?,?)",
                       [(t.isoformat(), float(v)) for t, v in zip(idx, hist)])
        o = json.dumps({"model": "qf", "confidence_level": conf, "receipt": 0})
        return np.array(db.execute(
            "SELECT forecast, lower_bound, upper_bound FROM forecast("
            "'SELECT ts, value FROM s', ?, ?)", (H, o)).fetchall())

    S = {k: [] for k in ("mase", "cov", "wink", "mwql")}
    C = {k: [] for k in ("mase", "cov", "wink", "mwql")}
    for tb in tests:
        hist, truth = tb["hist"], tb["truth"]
        r80 = serve(hist, 0.8)  # median + q0.1/q0.9
        S["mase"].append(mase(r80[:, 0], truth, hist, M))
        c, wk = cov_winkler(r80[:, 1], r80[:, 2], truth, alpha=0.2)
        S["cov"].append(c)
        S["wink"].append(wk)
        dec = np.zeros((H, Q))
        dec[:, 4] = r80[:, 0]  # median
        for conf, (a, b) in [(0.8, (0, 8)), (0.6, (1, 7)), (0.4, (2, 6)),
                             (0.2, (3, 5))]:
            rr = serve(hist, conf)
            dec[:, a] = rr[:, 1]
            dec[:, b] = rr[:, 2]
        S["mwql"].append(mwql(dec, truth))
        if tb["cfan"] is not None:
            cf = tb["cfan"]
            cdec = cf[DEC].T
            C["mase"].append(mase(cf[I_MED], truth, hist, M))
            cc, cw = cov_winkler(cdec[:, 0], cdec[:, 8], truth, alpha=0.2)
            C["cov"].append(cc)
            C["wink"].append(cw)
            C["mwql"].append(mwql(cdec, truth))

    def line(name, d):
        print(f"  {name:26}{np.mean(d['mase']):>7.3f}{np.mean(d['cov']):>7.0%}"
              f"{np.mean(d['wink']):>9.0f}{np.mean(d['mwql']):>8.4f}")
    print(f"\n== {NAME}: native quantile forecast student (80% band) ==")
    print(f"  {'model':26}{'MASE':>7}{'80cov':>7}{'Winkler':>9}{'mwQL':>8}")
    if C["mase"]:
        line("chronos (teacher)", C)
    line("native quantile student", S)


if __name__ == "__main__":
    main()
