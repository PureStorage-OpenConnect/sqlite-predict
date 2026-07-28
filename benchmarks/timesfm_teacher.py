"""Benchmark TimesFM as an in-DB distillation teacher.

The extension serves the full reconstructed TimesFM (scripts/export_timesfm_onnx.py
plus the sequence io_spec flags), then distills it into a native forecast student
in one SQL call: `distill_forecast(teacher='timesfm-onnx', ...)` re-runs the
teacher over the series' sliding windows to label them and fits the student on
those labels, no Python in the loop. This measures three points on m4_hourly:

  teacher   the reconstructed TimesFM served directly through forecast()
  student   the native PSFCST student distilled from that teacher (zero-dep serve)
  naive     the seasonal-naive floor (MASE ~ 1.0 by construction)

Needs the exported graph (kept out of git for size); point TIMESFM_ONNX at it:

  uv run --with "setuptools<81" --with "timesfm[torch]" --with torch --with onnx \
    --with onnxscript --with onnxruntime --with numpy \
    python scripts/export_timesfm_onnx.py /tmp/timesfm.onnx
  TIMESFM_ONNX=/tmp/timesfm.onnx uv run --with gluonts --with numpy --with pandas \
    python benchmarks/timesfm_teacher.py
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
ONNX = os.environ.get("TIMESFM_ONNX")
if not ONNX or not os.path.exists(ONNX):
    sys.exit("set TIMESFM_ONNX to the exported graph (see the module docstring)")

L, H, M, N = 512, 48, 24, int(os.environ.get("N", "40"))
EPOCHS = int(os.environ.get("EPOCHS", "1500"))
IO = {"layout": "sequence", "input": "context",
      "outputs": {"point": "point_fan", "quantile": "quant_fan"},
      "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      "patch": 512, "fixed_context": True, "flip_invariance": True,
      "continuous_head": True, "quantile_crossing_repair": True,
      "denormalize": "instance"}

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
db.load_extension(EXT)
db.execute("SELECT predict_register('timesfm-onnx', ?)", (json.dumps(
    {"runtime": "onnx", "kind": "ts-fm", "license": "Apache-2.0",
     "weights_uri": ONNX, "io_spec": IO}),))

ds = get_dataset("m4_hourly")
tests, rows = [], []
for e in ds.test:
    if len(tests) >= N:
        break
    y = np.asarray(e["target"], float)
    if len(y) < L + 2 * H:
        continue
    sk = len(tests)
    for t, v in enumerate(y[:-H]):
        rows.append((sk, t, float(v)))
    tests.append({"hist": y[:-H], "truth": y[-H:]})


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


print(f"m4_hourly: {len(tests)} series, context {L}, horizon {H}, epochs {EPOCHS}")
t0 = time.time()
teacher = serve("timesfm-onnx")
print(f"teacher (TimesFM served in-DB)        MASE {teacher:.3f}  ({time.time()-t0:.0f}s)")

db.execute("CREATE TABLE s(series_key INTEGER, t INTEGER, value REAL)")
db.executemany("INSERT INTO s VALUES (?,?,?)", rows)
t0 = time.time()
opts = json.dumps({"teacher": "timesfm-onnx", "context": L, "horizon": H,
                   "student_id": "tf", "epochs": EPOCHS})
mid, trows = db.execute(
    "SELECT model_id, train_rows FROM distill_forecast("
    "'SELECT series_key, value FROM s ORDER BY series_key, t', json(?))",
    (opts,)).fetchone()
train_s = time.time() - t0
student = serve("tf")

naive = float(np.mean([mase(seasonal_naive(tb["hist"], H, M), tb["truth"], tb["hist"], M)
                       for tb in tests]))
print(f"student (distilled, {trows} windows)    MASE {student:.3f}  ({train_s:.0f}s train)")
print(f"naive (seasonal-naive floor)          MASE {naive:.3f}")
gap = (naive - teacher) or 1e-9
print(f"student closes {100*(naive-student)/gap:.0f}% of the naive->teacher gap")
