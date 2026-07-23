"""Benchmark sqlite-predict's models against TabFM on a TabArena-spirit subset.

TabFM was announced on TabArena (Erickson et al. 2025) — a curated OpenML
benchmark. This runs a small, honest subset of comparable OpenML tasks and
lines up, on the same train/test splits:

  - xgboost                the strong tabular baseline
  - tabfm (ref)            the zero-shot foundation model (local weights)
  - knn5-incontext (ours)  our shipping in-context model, via the extension
  - tree<-knn5 (ours)      our distill(): a native tree student, via the ext
  - tree<-tabfm (ref)      a tree fit on TabFM's train predictions (the
                           distillation principle with the real FM; our
                           extension can't serve TabFM as a teacher yet)

It is a SUBSET on a single fixed split, not TabArena's full 51-task /
30-split Elo protocol — the point is comparability and the distillation
delta, not a leaderboard. TabFM weights are non-commercial (evaluation is
permitted); nothing here is redistributed.

Run (heavy; needs the local TabFM weights):

    uv run --with torch --with tabfm --with safetensors --with numpy \
        --with pandas --with scikit-learn --with xgboost \
        python benchmarks/tabarena.py
"""

import os
import time
import warnings

# Force local-only model loading BEFORE huggingface_hub is imported: the TabFM
# weights are on disk, and a hub check over a flaky network hangs the load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
# Single-threaded everything: xgboost's OpenMP pool and torch's ATen pool
# deadlock each other on macOS when both run in one process (torch's
# parallel fill hangs in a condvar). Single-threaded is slower but reliable.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CKPT = os.path.expanduser("~/.cache/sqlite-predict/tabfm")
EXT = os.path.join(os.path.dirname(__file__), "..", "dist", "predict0")
RESULTS = os.path.join(os.path.dirname(__file__), "results", "tabarena.md")
MAX_ROWS, MAX_FEAT, SEED = 1500, 40, 0

# OpenML data_ids (stable) + a couple of sklearn built-ins for reliability.
CLS_IDS = [(31, "credit-g"), (37, "diabetes"), (1464, "blood-transfusion"),
           (54, "vehicle")]


# ---- data ----

def _prep(X, y, task):
    """Numeric matrix for the numeric models; raw frame kept for TabFM."""
    X = X.copy()
    for c in X.columns:
        if X[c].dtype.name in ("category", "object"):
            X[c] = X[c].astype("category").cat.codes.astype(float)
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if X.shape[1] > MAX_FEAT:
        X = X.iloc[:, :MAX_FEAT]
    if task == "cls":
        y = pd.Series(pd.Categorical(y).codes).astype(str)  # class labels
    else:
        y = pd.to_numeric(y, errors="coerce").astype(float)
    return X, y


def load_datasets():
    from sklearn.datasets import (fetch_california_housing, fetch_openml,
                                  load_diabetes)
    out = []
    for did, name in CLS_IDS:
        try:
            b = fetch_openml(data_id=did, as_frame=True)
            X, y = _prep(b.data, b.target, "cls")
            out.append((name, X, y, "cls"))
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {type(e).__name__}: {e}")
    for name, ld in (("diabetes-reg", load_diabetes),
                     ("california", fetch_california_housing)):
        b = ld(as_frame=True)
        X, y = _prep(b.data, b.target, "reg")
        out.append((name, X, y, "reg"))
    # subsample big datasets (TabFM context cost + wall clock)
    trimmed = []
    for name, X, y, task in out:
        if len(X) > MAX_ROWS:
            idx = np.random.default_rng(SEED).choice(len(X), MAX_ROWS, False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(
                drop=True)
        trimmed.append((name, X, y, task))
    return trimmed


def split(X, y, task):
    from sklearn.model_selection import train_test_split
    strat = y if task == "cls" else None
    return train_test_split(X, y, test_size=0.25, random_state=SEED,
                            stratify=strat)


def score(y_true, pred, task):
    from sklearn.metrics import accuracy_score, root_mean_squared_error
    if pred is None:
        return float("nan")
    if task == "cls":
        return accuracy_score(y_true.astype(str), np.asarray(pred).astype(str))
    return root_mean_squared_error(y_true.astype(float),
                                   np.asarray(pred, dtype=float))


# ---- models ----

def run_xgb(Xtr, ytr, Xte, task):
    import xgboost as xgb
    if task == "cls":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(ytr)
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, n_jobs=4,
                              verbosity=0)
        m.fit(Xtr.values, le.transform(ytr))
        return le.inverse_transform(m.predict(Xte.values))
    m = xgb.XGBRegressor(n_estimators=200, max_depth=6, n_jobs=1, verbosity=0)
    m.fit(Xtr.values, ytr.values)
    return m.predict(Xte.values)


TABFM_ESTIMATORS = 8  # reduced from the default 32 for CPU tractability


def tabfm_estimator(task, _cache={}):
    from tabfm.src.classifier_and_regressor import (TabFMClassifier,
                                                    TabFMRegressor)
    from tabfm.src.pytorch import tabfm_v1_0_0
    import torch
    torch.set_num_threads(1)  # avoid the ATen/OpenMP thread-pool deadlock
    key = "cls" if task == "cls" else "reg"
    if key not in _cache:
        mt = "classification" if task == "cls" else "regression"
        # fp32: bf16 on CPU deadlocks torch's fill kernel on this platform.
        model = tabfm_v1_0_0.load(mt, checkpoint_path=CKPT, dtype=None)
        _cache[key] = (TabFMClassifier if task == "cls" else TabFMRegressor,
                       model)
    Cls, model = _cache[key]
    return Cls(model=model, n_estimators=TABFM_ESTIMATORS)


def run_tabfm(Xtr, ytr, Xte, task):
    est = tabfm_estimator(task)
    est.fit(Xtr.values, ytr.values)
    return est.predict(Xte.values), est


def sklearn_tree(Xtr, teacher_tr, Xte, task):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    T = (DecisionTreeClassifier if task == "cls" else DecisionTreeRegressor)
    t = T(max_depth=8, min_samples_split=5, random_state=SEED)  # mirror our CART
    t.fit(Xtr.values, teacher_tr)
    return t.predict(Xte.values)


# ---- our extension, via SQL ----

def _ext():
    import sqlite3
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    return db


def _load_tables(db, Xtr, ytr, Xte, task):
    k = Xtr.shape[1]
    cols = ", ".join(f"f{i} REAL" for i in range(k))
    lab = "TEXT" if task == "cls" else "REAL"
    db.execute(f"CREATE TABLE tr({cols}, label {lab})")
    db.executemany(
        f"INSERT INTO tr VALUES ({','.join('?' * (k + 1))})",
        [list(map(float, r)) + [y] for r, y in zip(Xtr.values, ytr)])
    db.execute(f"CREATE TABLE te(id INTEGER, {cols})")
    db.executemany(
        f"INSERT INTO te VALUES ({','.join('?' * (k + 1))})",
        [[i] + list(map(float, r)) for i, r in enumerate(Xte.values)])
    feats = ", ".join(f"f{i}" for i in range(k))
    return feats


def run_ours_knn5(Xtr, ytr, Xte, task):
    db = _ext()
    feats = _load_tables(db, Xtr, ytr, Xte, task)
    rows = db.execute(
        f"SELECT row_ref, prediction FROM predict('SELECT {feats}, label FROM"
        f" tr', 'SELECT id, {feats} FROM te', json_object('target','label',"
        f"'task',?,'receipt',0)) ORDER BY row_ref",
        ("classify" if task == "cls" else "regress",)).fetchall()
    db.close()
    return [r[1] for r in rows]


def run_ours_distill(Xtr, ytr, Xte, task, kind="tree"):
    db = _ext()
    feats = _load_tables(db, Xtr, ytr, Xte, task)
    db.execute(
        f"SELECT content_hash FROM distill('SELECT {feats}, label FROM tr',"
        f" json_object('target','label','task',?,'student_id','s',"
        f"'student_kind',?))",
        ("classify" if task == "cls" else "regress", kind)).fetchone()
    blob = db.execute("SELECT length(weights) FROM _predict_models WHERE"
                      " model_id='s'").fetchone()[0]
    rows = db.execute(
        f"SELECT row_ref, prediction FROM predict(NULL, 'SELECT id, {feats}"
        f" FROM te', json_object('model','s','receipt',0)) ORDER BY row_ref"
    ).fetchall()
    db.close()
    return [r[1] for r in rows], blob


# ---- driver ----

def main():
    rng = np.random.default_rng(SEED)
    del rng
    datasets = load_datasets()
    print(f"datasets: {[d[0] for d in datasets]}")
    rows = []
    student_bytes = []
    for name, X, y, task in datasets:
        Xtr, Xte, ytr, yte = split(X, y, task)
        rec = {"dataset": name, "task": task, "n": len(X),
               "d": X.shape[1], "classes": (y.nunique() if task == "cls"
                                            else None)}
        print(f"\n== {name} ({task}, n={len(X)}, d={X.shape[1]}) ==")

        def timed(fn):
            t0 = time.time()
            r = fn()
            return r, time.time() - t0

        try:
            (xgb_p, _), t = timed(lambda: (run_xgb(Xtr, ytr, Xte, task), None))
            rec["xgboost"] = score(yte, xgb_p, task)
        except Exception as e:  # noqa: BLE001
            rec["xgboost"] = float("nan")
            print("  xgb fail:", e)

        try:
            t0 = time.time()
            tabfm_p, est = run_tabfm(Xtr, ytr, Xte, task)
            rec["tabfm"] = score(yte, tabfm_p, task)
            rec["tabfm_s"] = time.time() - t0
            # teacher train predictions for the distilled tree
            teacher_tr = est.predict(Xtr.values)
            tree_p = sklearn_tree(Xtr, teacher_tr, Xte, task)
            rec["tree<-tabfm"] = score(yte, tree_p, task)
        except Exception as e:  # noqa: BLE001
            rec["tabfm"] = rec["tree<-tabfm"] = float("nan")
            rec["tabfm_s"] = float("nan")
            print("  tabfm fail:", type(e).__name__, str(e)[:120])

        try:
            rec["knn5 (ours)"] = score(yte, run_ours_knn5(Xtr, ytr, Xte, task),
                                       task)
        except Exception as e:  # noqa: BLE001
            rec["knn5 (ours)"] = float("nan")
            print("  knn5 fail:", str(e)[:120])

        try:
            dp, blob = run_ours_distill(Xtr, ytr, Xte, task, "tree")
            rec["tree<-knn5 (ours)"] = score(yte, dp, task)
            student_bytes.append(blob)
        except Exception as e:  # noqa: BLE001
            rec["tree<-knn5 (ours)"] = float("nan")
            print("  tree-distill fail:", str(e)[:120])

        try:
            gp, gblob = run_ours_distill(Xtr, ytr, Xte, task, "gbt")
            rec["gbt<-knn5 (ours)"] = score(yte, gp, task)
            student_bytes.append(gblob)
        except Exception as e:  # noqa: BLE001
            rec["gbt<-knn5 (ours)"] = float("nan")
            print("  gbt-distill fail:", str(e)[:120])

        print("  ", {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in rec.items()})
        rows.append(rec)

    _write(rows, student_bytes)


def _fmt(v):
    return "-" if v is None or (isinstance(v, float) and v != v) else (
        f"{v:.3f}" if isinstance(v, float) else str(v))


def _write(rows, student_bytes):
    cols = ["xgboost", "tabfm", "tree<-tabfm", "knn5 (ours)",
            "tree<-knn5 (ours)", "gbt<-knn5 (ours)"]
    lines = ["# TabArena-spirit benchmark\n",
             "Curated OpenML subset, single 75/25 split (seed 0), features",
             f"capped at {MAX_FEAT}, rows capped at {MAX_ROWS}. Metric:",
             "**accuracy** (classification, higher better) / **RMSE**",
             "(regression, lower better). Not TabArena's full 51-task/30-split",
             "Elo protocol — a comparability subset. TabFM = local weights,",
             f"fp32, {TABFM_ESTIMATORS}-member ensemble (reduced from 32 for",
             "CPU tractability); `tree<-tabfm` is a depth-8 CART fit on",
             "TabFM's train predictions (the distillation principle; our",
             "extension serves the `tree<-knn5` version natively).\n",
             "| dataset | task | n | d | " + " | ".join(cols) + " | tabfm s |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in cols) +
             " | --- |"]
    for r in rows:
        cells = [r["dataset"], r["task"], str(r["n"]), str(r["d"])]
        cells += [_fmt(r.get(c)) for c in cols]
        cells.append(_fmt(r.get("tabfm_s")))
        lines.append("| " + " | ".join(cells) + " |")
    if student_bytes:
        lines.append(f"\nNative tree-student size: "
                     f"{min(student_bytes)}-{max(student_bytes)} bytes "
                     f"(runs in the zero-dependency core, ~microseconds/row; "
                     f"TabFM is tens of seconds/call).")
    print("\n".join(lines))
    print(f"\n(curated writeup with analysis lives in"
          f" {os.path.relpath(RESULTS)})")


if __name__ == "__main__":
    main()
