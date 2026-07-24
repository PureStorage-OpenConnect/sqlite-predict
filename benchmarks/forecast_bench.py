"""Real-data forecast benchmark on standard gluonts datasets.

Validates forecast() the way time-series foundation models are validated
(GIFT-Eval / Monash style): seasonal MASE for point accuracy, 95% interval
coverage + Winkler score for the interval, and mean weighted quantile loss
(mwQL, a normalized CRPS proxy) for the full predictive distribution. Compares
our extension models against a seasonal-naive baseline and the Chronos / TimesFM
foundation models.

FM forecasts are cached per (model, dataset, series) under
~/.cache/sqlite-predict/forecast-cache (non-committed: FM outputs are a
non-commercial derivative), so the run is crash-resilient and re-runs need no
FM inference.

Run:  uv run --with "setuptools<81" --with gluonts --with numpy --with pandas \
        --with chronos-forecasting --with torch python benchmarks/forecast_bench.py
"""
import json, logging, os, sqlite3
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)  # FMs are chatty about quantile ranges

EXT = os.path.join(os.path.dirname(__file__), "..", "dist", "predict0")
FCACHE = os.path.expanduser("~/.cache/sqlite-predict/forecast-cache")
DATASETS = [("m4_hourly", 40, 24), ("tourism_monthly", 40, 12),
            ("m4_daily", 40, 7)]
# quantile levels: 0.025/0.975 bound the 95% interval, 0.5 is the point, the
# nine deciles feed mwQL.
QL = [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]
Z = {0.025: -1.9600, 0.1: -1.2816, 0.2: -0.8416, 0.3: -0.5244, 0.4: -0.2533,
     0.5: 0.0, 0.6: 0.2533, 0.7: 0.5244, 0.8: 0.8416, 0.9: 1.2816, 0.975: 1.9600}
I_LO, I_MED, I_HI = QL.index(0.025), QL.index(0.5), QL.index(0.975)
DECILES = [i for i, q in enumerate(QL) if 0.1 <= q <= 0.9]


def seasonal_naive(hist, h, m):
    return [hist[-m + (i % m)] for i in range(h)]


def mase(fc, truth, hist, m):
    errs = [abs(hist[i] - hist[i - m]) for i in range(m, len(hist))]
    scale = np.mean(errs) if errs else np.mean(np.abs(np.diff(hist)))
    return np.nan if not scale else float(
        np.mean(np.abs(np.asarray(fc) - np.asarray(truth))) / scale)


def cov_winkler(lo, hi, truth, alpha=0.05):
    cov = wink = 0.0
    for a, b, y in zip(lo, hi, truth):
        cov += a <= y <= b
        w = b - a
        w += (2 / alpha) * (a - y) if y < a else (2 / alpha) * (y - b) if y > b else 0
        wink += w
    return cov / len(truth), wink / len(truth)


def mwql(q_forecast, truth):
    """Normalized mean weighted quantile loss over the deciles (CRPS proxy)."""
    denom = float(np.sum(np.abs(truth))) or 1.0
    num = 0.0
    for qi in DECILES:
        q = QL[qi]
        for h, y in enumerate(truth):
            qh = q_forecast[qi][h]
            num += q * (y - qh) if y >= qh else (1 - q) * (qh - y)
    return num / (len(DECILES) * denom)


def run_ext(model, hist, start, freq, h):
    """Our model: one forecast() call at 95%, then the implied Gaussian
    quantiles from (point, sigma) reconstructed via sigma=(hi-point)/1.96."""
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
    pt = np.array([r[0] for r in rows])
    hi = np.array([r[2] for r in rows])
    sigma = np.maximum((hi - pt) / 1.9600, 1e-9)
    return np.array([pt + Z[q] * sigma for q in QL])  # [len(QL), h]


_DEC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def expand_deciles(dec):
    """[h, 9] decile forecasts -> [len(QL), h], linearly extrapolating the
    tails to 0.025/0.975 (the FMs emit only deciles)."""
    src = np.array(_DEC)
    h = dec.shape[0]
    out = np.zeros((len(QL), h))
    for i, ql in enumerate(QL):
        for t in range(h):
            d = dec[t]
            if ql < 0.1:
                out[i, t] = d[0] - (d[1] - d[0]) * (0.1 - ql) / 0.1
            elif ql > 0.9:
                out[i, t] = d[8] + (d[8] - d[7]) * (ql - 0.9) / 0.1
            else:
                out[i, t] = np.interp(ql, src, d)
    return out


def run_chronos(hist, h, _c={}):
    import torch
    from chronos import BaseChronosPipeline
    if "p" not in _c:
        _c["p"] = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    q, _ = _c["p"].predict_quantiles(torch.tensor([float(v) for v in hist]),
                                     prediction_length=h, quantile_levels=_DEC)
    return expand_deciles(q[0].numpy())  # [h, 9] -> [len(QL), h]


def run_timesfm(hist, h, freq, _c={}):
    if "m" not in _c:
        import timesfm
        _c["m"] = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32,
                                           horizon_len=max(64, h),
                                           num_layers=50, context_len=512),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
    fc_freq = {"H": 0, "D": 0, "W": 1, "M": 1}.get(str(freq)[0], 0)
    _, q = _c["m"].forecast([np.asarray(hist, dtype=float)], freq=[fc_freq])
    return expand_deciles(np.asarray(q)[0][:h, 1:10])  # cols 1..9 = deciles


def fm_forecast(runner, model, name, i, hist, h, freq=None):
    path = os.path.join(FCACHE, model, name, f"{i}.npy")
    if os.path.exists(path):
        return np.load(path)
    arr = runner(hist, h, freq) if freq is not None else runner(hist, h)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    return arr


def main():
    from gluonts.dataset.repository import get_dataset
    have = {"chronos": True, "timesfm": True}
    for name, cap, m in DATASETS:
        try:
            ds = get_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {e}")
            continue
        h, freq = ds.metadata.prediction_length, ds.metadata.freq
        models = [*(f"ext:{mm}" for mm in ("theta-classic", "stub-seasonal-naive")),
                  "seasonal-naive", "chronos", "timesfm"]
        acc = {mm: {"mase": [], "cov": [], "wink": [], "mwql": []} for mm in models}
        cnt = 0
        for i, entry in enumerate(ds.test):
            if cnt >= cap:
                break
            y = np.asarray(entry["target"], dtype=float)
            if len(y) < 3 * m + h:
                continue
            hist, truth, start = y[:-h], y[-h:], entry["start"]
            for mm in ("theta-classic", "stub-seasonal-naive"):
                try:
                    qf = run_ext(mm, hist, start, freq, h)
                    key = f"ext:{mm}"
                    acc[key]["mase"].append(mase(qf[I_MED], truth, hist, m))
                    cv, wk = cov_winkler(qf[I_LO], qf[I_HI], truth)
                    acc[key]["cov"].append(cv)
                    acc[key]["wink"].append(wk)
                    acc[key]["mwql"].append(mwql(qf, truth))
                except Exception as e:  # noqa: BLE001
                    if cnt == 0:
                        print(f"  {mm} err: {str(e)[:90]}")
            acc["seasonal-naive"]["mase"].append(
                mase(seasonal_naive(hist, h, m), truth, hist, m))
            for mm, run, needs_freq in (("chronos", run_chronos, False),
                                        ("timesfm", run_timesfm, True)):
                if not have[mm]:
                    continue
                try:
                    qf = fm_forecast(run, mm, name, i, hist, h,
                                     freq if needs_freq else None)
                    acc[mm]["mase"].append(mase(qf[I_MED], truth, hist, m))
                    cv, wk = cov_winkler(qf[I_LO], qf[I_HI], truth)
                    acc[mm]["cov"].append(cv)
                    acc[mm]["wink"].append(wk)
                    acc[mm]["mwql"].append(mwql(qf, truth))
                except Exception as e:  # noqa: BLE001
                    have[mm] = False
                    print(f"  {mm} unavailable: {type(e).__name__}: {str(e)[:90]}")
            cnt += 1
        print(f"\n== {name} (h={h}, freq={freq}, m={m}, {cnt} series) ==")
        print(f"  {'model':<22}{'MASE':>7}{'95%cov':>8}{'Winkler':>10}{'mwQL':>8}")
        for mm in models:
            d = acc[mm]
            if not d["mase"]:
                continue
            row = f"  {mm:<22}{np.nanmean(d['mase']):>7.3f}"
            row += (f"{np.mean(d['cov']):>8.0%}{np.mean(d['wink']):>10.1f}"
                    f"{np.mean(d['mwql']):>8.3f}" if d["cov"] else "")
            print(row)


if __name__ == "__main__":
    main()
