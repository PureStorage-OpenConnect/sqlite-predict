"""Score detect_anomalies on TSB-AD-U, the reliable univariate anomaly-detection
benchmark (Liu & Paparrizos, "The Elephant in the Room", NeurIPS 2024), which
replaced the flawed NAB / Yahoo / NASA suites. Follows TSB_AD/main.py's
unsupervised protocol: run over the whole series, MinMax-scale the score, and
report VUS-PR (the metric the paper identifies as reliable) at the auto-detected
sliding window.

Two bundled models are worth comparing here:
  default   the theta one-step-residual z-score (a mid-pack forecast detector)
  sub-pca   the subsequence-reconstruction detector (the leaderboard family)

On TSB-AD-U sub-pca reaches ~0.5-0.6 VUS-PR, at or above the published SOTA
(~0.44), versus ~0.25-0.32 for the residual detector.

Needs the loadable build, `pip install TSB-AD`, and the dataset:
  curl -sL https://www.thedatum.org/datasets/TSB-AD-U.zip -o TSB-AD-U.zip
  unzip TSB-AD-U.zip
  TSB_AD_U=./TSB-AD-U MODEL=sub-pca uv run --with TSB-AD --with numpy \
    --with pandas --with scikit-learn python benchmarks/anomaly_tsb_ad.py
"""
import glob
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "predict0"))
DATA = os.environ.get("TSB_AD_U")
if not DATA or not os.path.isdir(DATA):
    sys.exit("set TSB_AD_U to the extracted TSB-AD-U directory (see the docstring)")
MODEL = os.environ.get("MODEL", "")  # "" = default theta detector; or "sub-pca"
N = int(os.environ.get("N", "0"))     # 0 = all series

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
db.load_extension(EXT)


def score(vals):
    db.execute("DROP TABLE IF EXISTS a")
    db.execute("CREATE TABLE a(rn INTEGER, ts TEXT, value REAL)")
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="min")
    db.executemany("INSERT INTO a VALUES (?,?,?)",
                   [(i, t.isoformat(), float(v))
                    for i, (t, v) in enumerate(zip(idx, vals))])
    opts = "'model','%s'," % MODEL if MODEL else ""
    rows = db.execute(
        "SELECT anomaly_probability FROM detect_anomalies("
        "'SELECT ts,value FROM a ORDER BY rn', json_object(%s))" % opts
    ).fetchall()
    return np.array([r[0] if r[0] is not None else 0.0 for r in rows], float)


files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
if N > 0:
    files = files[:N]
vus, skipped = [], 0
for f in files:
    d = pd.read_csv(f).dropna()
    data = d.iloc[:, 0:-1].values.astype(float)
    label = d["Label"].astype(int).to_numpy()
    if label.sum() == 0 or len(data) < 50:
        skipped += 1
        continue
    try:
        sw = int(find_length_rank(data, rank=1))
    except Exception:
        sw = 100
    sc = score(data[:, 0])
    if len(sc) != len(label):
        skipped += 1
        continue
    sc = MinMaxScaler((0, 1)).fit_transform(sc.reshape(-1, 1)).ravel()
    try:
        vus.append(get_metrics(sc, label, slidingWindow=sw)["VUS-PR"])
    except Exception:
        skipped += 1
    if len(vus) % 25 == 0 and vus:
        print(f"  [{len(vus)} scored] VUS-PR mean {np.mean(vus):.3f}", flush=True)
print(f"\nmodel={MODEL or 'default(theta)'}  series={len(vus)} (skipped {skipped})")
print(f"VUS-PR mean {np.mean(vus):.3f}  median {np.median(vus):.3f}"
      f"  (TSB-AD-U published SOTA ~0.44)")
