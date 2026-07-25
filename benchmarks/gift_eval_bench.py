"""GIFT-Eval comparison: seasonal-naive vs Chronos vs TimesFM vs the native
distilled student, on GIFT-Eval datasets, with MASE + CRPS (mean weighted
quantile loss, GIFT-Eval's probabilistic metric).

This is the real generalization test of the distilled forecast student: does
distilling an FM into a native MLP hold up across GIFT-Eval's domains, or was
m4_hourly a friendly regime?

Protocol (per dataset, short term, single test window per GIFT-Eval):
  - train region = target[:-2H], test = target[-H:] from input target[:-H]
    (no leakage: the student never sees the test region).
  - distilled: sample Chronos-labeled context windows from the train region,
    distill_forecast a quantile student, serve it through the extension.
  - metrics: MASE (median, scaled by the in-sample seasonal-naive error at the
    GIFT-Eval seasonality) and mwQL over the deciles, per series, then averaged.

CAPS for a runnable first pass (noted in the output): eval is capped per dataset
and the distillation training set is bounded. FM outputs are cached under
~/.cache/sqlite-predict/gift-cache.

Setup: GIFT_EVAL must point at the downloaded data (huggingface-cli download
Salesforce/GiftEval). Run with gift_eval + gluonts + the FM stacks + the built
default extension (dist/predict0).
"""
import json, os, sqlite3, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
if os.environ.get("GIFT_SRC"):  # import gift_eval from source (skip its dep pins)
    sys.path.insert(0, os.environ["GIFT_SRC"])
from forecast_bench import run_chronos, run_timesfm, QL, EXT, mase
from gift_eval.data import Dataset
from gluonts.time_feature import get_seasonality

DATASETS = os.environ.get(
    "GIFT_DSETS",
    "m4_hourly m4_weekly m4_daily m4_monthly hospital covid_deaths"
    " car_parts_with_missing restaurant").split()
QLEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q = len(QLEV)
DEC = [i for i, q in enumerate(QL) if 0.1 <= q <= 0.9]  # decile idx in run_chronos QL
EVAL_CAP = 250          # max test series scored per dataset
TRAIN_SERIES = 120      # series sampled for distillation labeling
TRAIN_WINDOWS = 1500    # max distillation windows per dataset
GC = os.path.expanduser("~/.cache/sqlite-predict/gift-cache")


def fm_dec(model, run, name, kind, si, origin, ctx, h):
    """Cached FM decile forecast [h, Q] for a context, per (model,ds,kind,si,origin)."""
    p = os.path.join(GC, model, name, kind, str(si), f"{origin}_{h}.npy")
    if os.path.exists(p):
        return np.load(p)
    fan = np.asarray(run(ctx, h), dtype=float)[DEC].T  # run -> [len(QL),h]; deciles
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, fan)
    return fan


def targets(name):
    d = Dataset(name=name, term="short")
    H, m = d.prediction_length, get_seasonality(d.freq)
    ys = [np.asarray(e["target"], dtype=float) for e in d.hf_dataset]
    ys = [y for y in ys if np.all(np.isfinite(y))]
    return ys, H, m


def mwql(dec, truth):
    denom = float(np.sum(np.abs(truth))) or 1.0
    num = 0.0
    for qi, q in enumerate(QLEV):
        e = truth - dec[:, qi]
        num += float(np.sum(np.maximum(q * e, (q - 1) * e)))
    return num / (Q * denom)


def snaive_fan(hist, H, m):
    pt = np.array([hist[-m + (i % m)] for i in range(H)])
    return np.repeat(pt[:, None], Q, axis=1)  # point as a flat fan


def train_student(db, name, ys, H, m, L, si_eval):
    """Distill a Chronos quantile student from the train region (target[:-2H])."""
    ncol = L + H * Q
    rows = []
    rng = np.random.RandomState(0)
    pool = [i for i in range(len(ys)) if len(ys[i]) >= L + 2 * H]
    rng.shuffle(pool)
    for si in pool[:TRAIN_SERIES]:
        y = ys[si]
        hi = len(y) - 2 * H  # train region end (no test leakage)
        origins = list(range(L, hi + 1, max(1, H // 2)))
        rng.shuffle(origins)
        for t in origins[:12]:
            win = y[t - L:t]
            if win.std() < 1e-6 * (abs(win.mean()) + 1):
                continue  # degenerate (near-constant) window: instance-norm blows up
            fan = fm_dec("chronos", run_chronos, name, "train", si, t, win, H)
            rows.append(np.concatenate([win, fan.reshape(-1)]))
            if len(rows) >= TRAIN_WINDOWS:
                break
        if len(rows) >= TRAIN_WINDOWS:
            break
    if len(rows) < 50:
        return False
    db.execute(f"DROP TABLE IF EXISTS w")
    db.execute(f"CREATE TABLE w({','.join(f'c{i} REAL' for i in range(ncol))})")
    db.executemany(f"INSERT INTO w VALUES ({','.join('?' * ncol)})",
                   [list(map(float, r)) for r in rows])
    cols = ",".join(f"c{i}" for i in range(ncol))
    try:  # the registry table is created lazily by the first distill
        db.execute("DELETE FROM _predict_models WHERE model_id='qf'")
    except sqlite3.OperationalError:
        pass
    opts = json.dumps({"context": L, "horizon": H, "student_id": "qf",
                       "receipt": 0, "quantiles": QLEV, "epochs": 800})
    db.execute(f"SELECT model_id FROM distill_forecast('SELECT {cols} FROM w',"
               " json(?))", (opts,)).fetchone()
    return True


def serve_student(db, hist, H):
    db.execute("DROP TABLE IF EXISTS s")
    db.execute("CREATE TABLE s(ts TEXT, value REAL)")
    idx = pd.date_range("2020-01-01", periods=len(hist), freq="h")
    db.executemany("INSERT INTO s VALUES (?,?)",
                   [(t.isoformat(), float(v)) for t, v in zip(idx, hist)])
    dec = np.zeros((H, Q))
    for conf, (a, b) in [(0.8, (0, 8)), (0.6, (1, 7)), (0.4, (2, 6)),
                         (0.2, (3, 5))]:
        o = json.dumps({"model": "qf", "confidence_level": conf, "receipt": 0})
        r = np.array(db.execute("SELECT forecast, lower_bound, upper_bound FROM"
                                " forecast('SELECT ts, value FROM s', ?, ?)",
                                (H, o)).fetchall())
        dec[:, 4] = r[:, 0]
        dec[:, a] = r[:, 1]
        dec[:, b] = r[:, 2]
    return dec


def main():
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    have_tfm = True
    agg = {mo: {"mase": {}, "mwql": {}} for mo in ("naive", "chronos", "timesfm",
                                                   "distilled")}
    print(f"{'dataset':22}{'H':>4}{'seas':>5}{'L':>5}{'n':>6}   models: MASE / mwQL")
    for name in DATASETS:
        ys, H, m = targets(name)
        L = int(np.clip(max(2 * m, 3 * H, 32), 32, 384))
        evalable = [i for i in range(len(ys)) if len(ys[i]) >= L + H]
        np.random.RandomState(1).shuffle(evalable)
        evalable = evalable[:EVAL_CAP]
        trained = train_student(db, name, ys, H, m, L, evalable)
        res = {mo: {"mase": [], "mwql": []} for mo in agg}
        for si in evalable:
            y = ys[si]
            hist, truth = y[:-H], y[-H:]
            nf = snaive_fan(hist, H, m)
            res["naive"]["mase"].append(mase(nf[:, 4], truth, hist, m))
            res["naive"]["mwql"].append(mwql(nf, truth))
            cf = fm_dec("chronos", run_chronos, name, "test", si, len(hist), hist, H)
            res["chronos"]["mase"].append(mase(cf[:, 4], truth, hist, m))
            res["chronos"]["mwql"].append(mwql(cf, truth))
            if have_tfm:
                try:
                    tf = fm_dec("timesfm", run_timesfm, name, "test", si,
                                len(hist), hist, H)
                    res["timesfm"]["mase"].append(mase(tf[:, 4], truth, hist, m))
                    res["timesfm"]["mwql"].append(mwql(tf, truth))
                except Exception as e:  # noqa: BLE001
                    have_tfm = False
                    print(f"  timesfm unavailable: {type(e).__name__}")
            if trained:
                sf = serve_student(db, hist, H)
                res["distilled"]["mase"].append(mase(sf[:, 4], truth, hist, m))
                res["distilled"]["mwql"].append(mwql(sf, truth))
        line = f"  {name:20}{H:>4}{m:>5}{L:>5}{len(evalable):>6}   "
        for mo in ("naive", "chronos", "timesfm", "distilled"):
            if res[mo]["mase"]:
                mn = np.nanmean(res[mo]["mase"])  # MASE undefined on constant series
                wq = np.nanmean(res[mo]["mwql"])
                agg[mo]["mase"][name] = mn
                agg[mo]["mwql"][name] = wq
                line += f"{mo[:4]} {mn:.2f}/{wq:.3f}  "
        print(line)

    print("\n== aggregate (mean over datasets) ==")
    print(f"  {'model':12}{'MASE':>8}{'mwQL':>8}   datasets")
    for mo in ("naive", "chronos", "timesfm", "distilled"):
        ms = list(agg[mo]["mase"].values())
        wq = list(agg[mo]["mwql"].values())
        if ms:
            print(f"  {mo:12}{np.nanmean(ms):>8.3f}{np.nanmean(wq):>8.4f}"
                  f"   n={len(ms)}")


if __name__ == "__main__":
    main()
