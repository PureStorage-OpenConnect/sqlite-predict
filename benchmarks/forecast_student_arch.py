"""Forecast student architecture comparison (in-DB, the numbers in
results/forecast.md "The student architecture was the bottleneck").

Distills an onnx Chronos teacher into native forecast students of different
shapes and measures MASE on m4_hourly, all through distill_forecast()/forecast()
so the numbers are what the extension actually produces (not a Python proxy):

  hidden=0        pure linear (DLinear-style skip, no residual)
  hidden=128..512 TiDE (linear skip + a residual of that width)

The default is hidden=256 (the m4_hourly optimum). Training on the target series
beats a larger mixed pool (dilution), so this trains and evaluates on the same 40
series. Needs the loadable-onnx build and an exported chronos-bolt graph:

  uv run --with "chronos-forecasting" ... python scripts/export_chronos_onnx.py /tmp/chronos.onnx
  CHRONOS_ONNX=/tmp/chronos.onnx uv run --with gluonts --with numpy --with pandas \
    python benchmarks/forecast_student_arch.py
"""
import json
import os
import sqlite3
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from forecast_bench import mase, seasonal_naive  # noqa: E402
from gluonts.dataset.repository import get_dataset  # noqa: E402

EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "predict0"))
ONNX = os.environ.get("CHRONOS_ONNX")
if not ONNX or not os.path.exists(ONNX):
    sys.exit("set CHRONOS_ONNX to an exported chronos-bolt graph (see the docstring)")

H, M, N = 48, 24, int(os.environ.get("N", "40"))
EPOCHS = int(os.environ.get("EPOCHS", "1500"))
HIDDENS = [int(h) for h in os.environ.get("HIDDENS", "0,128,256,512").split(",")]
IO = {"layout": "sequence", "input": "context", "output": "quantiles",
      "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "patch": 16}

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
db.load_extension(EXT)
db.execute("SELECT predict_register('chronos-onnx', ?)", (json.dumps(
    {"runtime": "onnx", "kind": "ts-fm", "license": "Apache-2.0",
     "weights_uri": ONNX, "io_spec": IO}),))

ds = get_dataset("m4_hourly")
tests, rows = [], []
for e in ds.test:
    if len(tests) >= N:
        break
    y = np.asarray(e["target"], float)
    if len(y) < 512 + 2 * H:
        continue
    sk = len(tests)
    for t, v in enumerate(y[:-H]):
        rows.append((sk, t, float(v)))
    tests.append({"hist": y[:-H], "truth": y[-H:]})
db.execute("CREATE TABLE s(series_key INTEGER, t INTEGER, value REAL)")
db.executemany("INSERT INTO s VALUES (?,?,?)", rows)


def serve(model):
    out = []
    for tb in tests:
        hist, truth = tb["hist"], tb["truth"]
        db.execute("DROP TABLE IF EXISTS q")
        db.execute("CREATE TABLE q(ts TEXT, value REAL)")
        idx = pd.date_range("2020-01-01", periods=len(hist), freq="h")
        db.executemany("INSERT INTO q VALUES (?,?)",
                       [(t.isoformat(), float(v)) for t, v in zip(idx, hist)])
        fc = [r[0] for r in db.execute(
            "SELECT forecast FROM forecast('SELECT ts,value FROM q',?,"
            "json_object('model',?,'confidence_level',0.8))",
            (H, model)).fetchall()]
        out.append(mase(np.array(fc), truth[:len(fc)], hist, M))
    return float(np.mean(out))


naive = float(np.mean([mase(seasonal_naive(t["hist"], H, M), t["truth"], t["hist"], M)
                       for t in tests]))
print(f"m4_hourly {len(tests)} series | chronos teacher {serve('chronos-onnx'):.3f}"
      f" | seasonal-naive {naive:.3f}")
for hid in HIDDENS:
    sid = f"h{hid}"
    t0 = time.time()
    opts = json.dumps({"teacher": "chronos-onnx", "context": 512, "horizon": H,
                       "student_id": sid, "hidden": hid, "epochs": EPOCHS})
    db.execute("SELECT model_id FROM distill_forecast("
               "'SELECT series_key, value FROM s ORDER BY series_key, t', json(?))",
               (opts,)).fetchone()
    kind = "linear" if hid == 0 else f"TiDE h={hid}"
    print(f"  {kind:12} MASE {serve(sid):.3f}  ({time.time()-t0:.0f}s)")
