"""Tabular model comparison on the synthetic ground-truth suites.

Candidates: Google TabFM (local weights under models/tabfm, downloaded
from google/tabfm-1.0.0-pytorch — license tabfm-non-commercial-v1.0,
benchmark/evaluation use), plus honest floors (majority class / global
mean) and a pure-python 5-NN so wins are attributable. The extension's
predict() column joins when M4 lands. Unavailable candidates are
reported SKIPPED, never silently dropped.

Usage: uv run --with 'tabfm[pytorch]' --with safetensors python compare_tabular.py
(tabfm[pytorch] omits the safetensors dep that local from_pretrained needs)
"""

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import synthetic_tabular as syt  # noqa: E402

# TabFM weights live outside the repo (13GB, non-commercial license).
# Override with SQLITE_PREDICT_DATA; defaults to a per-user cache.
_DATA = os.environ.get(
    "SQLITE_PREDICT_DATA",
    os.path.join(os.path.expanduser("~"), ".cache", "sqlite-predict"))
MODEL_DIR = os.path.join(_DATA, "tabfm")
HOLDOUT = 100

CLS_SUITES = {"two_moons": syt.two_moons,
              "xor_categorical": syt.xor_categorical}
REG_SUITES = {"friedman1": syt.friedman1,
              "stepwise_price": syt.stepwise_price}


def encode(rows, cols):
    """categoricals -> stable integer codes (per-column vocabulary)"""
    vocab = {}
    out = []
    for r in rows:
        enc = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                enc.append(float(vocab.setdefault(c, {}).setdefault(
                    v, len(vocab[c]))))
            else:
                enc.append(float(v))
        out.append(enc)
    return out


def knn5(Xtr, ytr, Xte, classify):
    """pure-python 5-NN on z-scored features: the 'simple but real'
    baseline between the floor and the FM"""
    n_feat = len(Xtr[0])
    mu = [statistics.mean(x[i] for x in Xtr) for i in range(n_feat)]
    sd = [statistics.pstdev(x[i] for x in Xtr) or 1 for i in range(n_feat)]
    z = lambda row: [(row[i] - mu[i]) / sd[i] for i in range(n_feat)]
    Ztr = [z(x) for x in Xtr]
    preds = []
    for q in Xte:
        zq = z(q)
        d = sorted(range(len(Ztr)), key=lambda j: sum(
            (Ztr[j][k] - zq[k]) ** 2 for k in range(n_feat)))[:5]
        vals = [ytr[j] for j in d]
        if classify:
            preds.append(max(set(vals), key=vals.count))
        else:
            preds.append(sum(vals) / len(vals))
    return preds


def accuracy(pred, truth):
    return sum(p == t for p, t in zip(pred, truth)) / len(truth)


def rmse(pred, truth):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, truth))
                     / len(truth))


def main():
    skipped = []
    tabfm_cls = tabfm_reg = None
    try:
        from tabfm import TabFMClassifier, TabFMRegressor
        from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
        t0 = time.perf_counter()
        cls_model = TabFM_HF.from_pretrained(MODEL_DIR,
                                             subfolder="classification")
        reg_model = TabFM_HF.from_pretrained(MODEL_DIR,
                                             subfolder="regression")
        print(f"tabfm load: {time.perf_counter()-t0:.1f}s")
        tabfm_cls = TabFMClassifier(model=cls_model)
        tabfm_reg = TabFMRegressor(model=reg_model)
    except Exception as e:  # noqa: BLE001
        skipped.append(f"tabfm: {type(e).__name__}: {e}")

    lines = ["# Tabular model comparison (synthetic suites)", "",
             f"Holdout {HOLDOUT} rows; seeded generators"
             " (tests/synthetic_tabular.py).", "",
             "| suite | task | model | metric | value | ms/call |",
             "|---|---|---|---|---|---|"]

    def emit(suite, task, model, metric, value, ms):
        lines.append(f"| {suite} | {task} | {model} | {metric} |"
                     f" {value:.3f} | {ms:.0f} |")

    for suite, gen in CLS_SUITES.items():
        X, y, _ = gen()
        cols = list(X[0].keys())
        Xtr, ytr, Xte, yte = syt.train_test_split(X, y, HOLDOUT)
        Etr, Ete = encode(Xtr, cols), encode(Xtr + Xte, cols)[len(Xtr):]
        majority = max(set(ytr), key=ytr.count)
        emit(suite, "cls", "majority (floor)", "acc",
             accuracy([majority] * len(yte), yte), 0)
        t0 = time.perf_counter()
        emit(suite, "cls", "knn5 (baseline)", "acc",
             accuracy(knn5(Etr, ytr, Ete, True), yte),
             (time.perf_counter() - t0) * 1000)
        if tabfm_cls:
            t0 = time.perf_counter()
            tabfm_cls.fit(Etr, ytr)
            pred = list(tabfm_cls.predict(Ete))
            emit(suite, "cls", "tabfm (ref)", "acc",
                 accuracy(pred, yte), (time.perf_counter() - t0) * 1000)

    for suite, gen in REG_SUITES.items():
        X, y, _ = gen()
        cols = list(X[0].keys())
        Xtr, ytr, Xte, yte = syt.train_test_split(X, y, HOLDOUT)
        Etr, Ete = encode(Xtr, cols), encode(Xtr + Xte, cols)[len(Xtr):]
        mean = sum(ytr) / len(ytr)
        emit(suite, "reg", "mean (floor)", "rmse",
             rmse([mean] * len(yte), yte), 0)
        t0 = time.perf_counter()
        emit(suite, "reg", "knn5 (baseline)", "rmse",
             rmse(knn5(Etr, ytr, Ete, False), yte),
             (time.perf_counter() - t0) * 1000)
        if tabfm_reg:
            t0 = time.perf_counter()
            tabfm_reg.fit(Etr, ytr)
            pred = [float(p) for p in tabfm_reg.predict(Ete)]
            emit(suite, "reg", "tabfm (ref)", "rmse",
                 rmse(pred, yte), (time.perf_counter() - t0) * 1000)

    if skipped:
        lines += ["", "## Skipped (not silently dropped)", ""]
        lines += [f"- {s}" for s in skipped]

    text = "\n".join(lines) + "\n"
    with open(os.path.join(os.path.dirname(__file__), "results",
                           "comparison-tabular.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
