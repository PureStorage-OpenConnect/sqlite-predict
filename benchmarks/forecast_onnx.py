"""Serve chronos-bolt through the onnx forecast() backend and confirm it
reproduces the Python Chronos model.

Requires the opt-in onnx build: `make loadable-onnx` (dist/predict0 must be the
onnx-enabled build). The chronos-bolt ONNX graph is exported once to
~/.cache/sqlite-predict (a non-committed non-commercial derivative) via
scripts/export_chronos_onnx.py, then registered with a `sequence` io_spec and
served by `forecast()`; the point + interval come off the model's decile fan.

On m4_hourly the onnx-served MASE (~0.80) matches the Python reference (~0.79);
the small gap is the context truncation to a multiple of the patch size.

Run: uv run --with "setuptools<81" --with gluonts --with numpy --with pandas \
       --with chronos-forecasting --with torch --with onnx --with onnxscript \
       --with onnxruntime python benchmarks/forecast_onnx.py
"""
import json, os, subprocess, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import sqlite3
from forecast_bench import FCACHE, I_MED, mase, cov_winkler
from gluonts.dataset.repository import get_dataset

EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist",
                                   "predict0"))
ONNX = os.path.expanduser("~/.cache/sqlite-predict/chronos_bolt_small.onnx")
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H, M = 48, 24


def ensure_onnx():
    if os.path.exists(ONNX):
        return
    os.makedirs(os.path.dirname(ONNX), exist_ok=True)
    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "export_chronos_onnx.py")
    print("exporting chronos-bolt to onnx (one-time)...")
    subprocess.run([sys.executable, script, ONNX], check=True)


def main():
    ensure_onnx()
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    if "onnx" not in json.loads(
            db.execute("SELECT predict_version()").fetchone()[0]).get(
                "runtimes", []):
        sys.exit("dist/predict0 is not the onnx build; run `make loadable-onnx`")
    iospec = {"layout": "sequence", "input": "context", "output": "quantiles",
              "quantiles": DECILES, "patch": 16}
    db.execute("SELECT predict_register('chronos-onnx', ?)", (json.dumps(
        {"runtime": "onnx", "kind": "ts-fm", "license": "Apache-2.0",
         "weights_uri": ONNX, "io_spec": iospec}),))

    ds = get_dataset("m4_hourly")
    onx = {"mase": [], "cov": [], "wink": []}
    ref = []
    cnt = 0
    for i, e in enumerate(ds.test):
        if cnt >= 40:
            break
        y = np.asarray(e["target"], dtype=float)
        if len(y) < 3 * M + H:
            continue
        hist, truth = y[:-H], y[-H:]
        db.execute("DROP TABLE IF EXISTS s")
        db.execute("CREATE TABLE s(ts TEXT, value REAL)")
        idx = pd.date_range("2020-01-01", periods=len(hist), freq="h")
        db.executemany("INSERT INTO s VALUES (?,?)",
                       [(t.isoformat(), float(v)) for t, v in zip(idx, hist)])
        r = np.array(db.execute(
            "SELECT forecast, lower_bound, upper_bound FROM forecast("
            "'SELECT ts, value FROM s', ?, json_object('model','chronos-onnx',"
            "'confidence_level',0.8))", (H,)).fetchall())
        onx["mase"].append(mase(r[:, 0], truth, hist, M))
        c, wk = cov_winkler(r[:, 1], r[:, 2], truth, alpha=0.2)
        onx["cov"].append(c)
        onx["wink"].append(wk)
        cp = os.path.join(FCACHE, "chronos", "m4_hourly", f"{i}.npy")
        if os.path.exists(cp):
            ref.append(mase(np.load(cp)[I_MED], truth, hist, M))
        cnt += 1
    print(f"\n== m4_hourly: chronos served via the onnx forecast() backend ==")
    print(f"  onnx-served chronos   MASE {np.mean(onx['mase']):.3f}"
          f"  80cov {np.mean(onx['cov']):.0%}  Winkler {np.mean(onx['wink']):.0f}")
    if ref:
        print(f"  python chronos (ref)  MASE {np.mean(ref):.3f}")


if __name__ == "__main__":
    main()
