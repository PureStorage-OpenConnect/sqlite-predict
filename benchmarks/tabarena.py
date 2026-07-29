"""Benchmark sqlite-predict's models against TabFM on a TabArena-spirit subset.

TabFM was announced on TabArena (Erickson et al. 2025) — a curated OpenML
benchmark. This runs a small, representative subset of comparable OpenML tasks and
lines up, on the same train/test splits:

  - xgboost                the strong tabular baseline
  - tabfm (ref)            the zero-shot foundation model (local weights)
  - knn5-incontext (ours)  our shipping in-context model, via the extension
  - tree<-knn5 (ours)      our distill_predict(): a native tree student, via the ext
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
# Local cache of TabFM's per-dataset outputs (non-committed: TabFM predictions
# are a non-commercial derivative). Makes the long run crash-resilient and
# lets the student-selection analysis re-run without re-invoking TabFM.
TABFM_CACHE = os.path.expanduser("~/.cache/sqlite-predict/tabfm-cache")
EXT = os.path.join(os.path.dirname(__file__), "..", "dist", "predict0")
RESULTS = os.path.join(os.path.dirname(__file__), "results", "tabarena.md")
MAX_ROWS, MAX_FEAT, SEED = 1500, 40, 0

# The full TabArena-v0.1 suite (OpenML suite 457): 51 curated tasks, each as
# (dataset_id, name, task). Fetched by the TabArena dataset upload id so we get
# exactly the curated version. task 'reg' where the suite marks 0 classes.
TABARENA = [
    (46913, "blood-transfusion", "cls"), (46921, "diabetes", "cls"),
    (46906, "anneal", "cls"), (46954, "QSAR_fish_toxicity", "reg"),
    (46918, "credit-g", "cls"), (46941, "maternal_health_risk", "cls"),
    (46917, "concrete_compressive_strength", "reg"),
    (46952, "qsar-biodeg", "cls"), (46931, "healthcare_insurance_expenses", "reg"),
    (46963, "website_phishing", "cls"), (46927, "Fitness_Club", "cls"),
    (46904, "airfoil_self_noise", "reg"),
    (46907, "used-Fiat-500", "reg"), (46980, "MIC", "cls"),
    (46938, "Is-this-a-good-customer", "cls"), (46940, "Marketing_Campaign", "cls"),
    (46930, "hazelnut-contaminant", "cls"), (46956, "seismic-bumps", "cls"),
    (46958, "splice", "cls"), (46912, "Bioresponse", "cls"),
    (46933, "hiva_agnostic", "cls"),
    (46960, "students_dropout", "cls"), (46915, "churn", "cls"),
    (46953, "QSAR-TID-11", "reg"), (46950, "polish_bankruptcy", "cls"),
    (46964, "wine_quality", "reg"), (46962, "taiwanese_bankruptcy", "cls"),
    (46969, "NATICUSdroid", "cls"), (46916, "coil2000_insurance", "cls"),
    (46911, "Bank_Customer_Churn", "cls"), (46932, "heloc", "cls"),
    (46979, "jm1", "cls"), (46924, "E-CommerceShipping", "cls"),
    (46947, "online_shoppers_intention", "cls"),
    (46937, "in_vehicle_coupon", "cls"), (46942, "miami_housing", "reg"),
    (46935, "HR_job_change", "cls"), (46934, "houses", "reg"),
    (46961, "superconductivity", "reg"), (46919, "credit_card_default", "cls"),
    (46905, "Amazon_employee_access", "cls"), (46910, "bank-marketing", "cls"),
    (46928, "Food_Delivery_Time", "reg"), (46949, "physiochemical_protein", "reg"),
    (46939, "kddcup09_appetency", "cls"), (46923, "diamonds", "reg"),
    (46922, "Diabetes130US", "cls"), (46908, "APSFailure", "cls"),
    (46955, "SDSS17", "cls"), (46920, "airline_satisfaction", "cls"),
    (46929, "GiveMeSomeCredit", "cls"),
]


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
    """Yield (name, X, y, task) one dataset at a time (lazy: a 150k-row set is
    loaded, subsampled to MAX_ROWS, and released before the next). Each dataset
    is subsampled to <= MAX_ROWS rows and <= MAX_FEAT features so the in-context
    TabFM teacher stays tractable on CPU; every model sees the same capped data,
    so the relative comparison holds."""
    from sklearn.datasets import fetch_openml
    for did, name, task in TABARENA:
        try:
            b = fetch_openml(data_id=did, as_frame=True)
            X, y = _prep(b.data, b.target, task)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name} (did={did}): {type(e).__name__}: {e}")
            continue
        if len(X) > MAX_ROWS:
            idx = np.random.default_rng(SEED).choice(len(X), MAX_ROWS, False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
        yield name, X, y, task


def split(X, y, task):
    from sklearn.model_selection import train_test_split
    strat = y if task == "cls" else None
    try:
        return train_test_split(X, y, test_size=0.25, random_state=SEED,
                                stratify=strat)
    except ValueError:
        # a subsampled class can be too rare to stratify; fall back
        return train_test_split(X, y, test_size=0.25, random_state=SEED)


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


def oof_proba(make_estimator, Xtr, ytr, n_splits=5, seed=0):
    """Stratified out-of-fold teacher probabilities.

    In-context teachers leak labels when scoring rows already in their
    context: the soft targets collapse toward one-hot and carry no
    inter-class structure to distill. Labeling each fold with a teacher
    fitted on the other folds prevents it (Tanna et al., "Pocket
    Foundation Models", arXiv:2605.18654). Returns (proba [n, K],
    classes) aligned to a global class order.
    """
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(ytr)
    classes = sorted(set(y.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(Xtr, y):
        est = make_estimator()
        est.fit(Xtr.values[tr_i], y[tr_i])
        p = est.predict_proba(Xtr.values[te_i])
        if not np.isfinite(p).all():
            # seen from TabICL on MPS; silently binding NaN as SQL NULL
            # would corrupt the distillation, so fail loudly instead
            raise ValueError("teacher returned non-finite probabilities on"
                             " a fold; retry on cpu")
        for j, c in enumerate(est.classes_):
            out[te_i, idx[c]] += p[:, j]
    return out, [str(c) for c in classes]


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


def run_ours_distill_teacher(Xtr, teacher_tr, Xte, task, kind="gbt"):
    """Distill our native student from a precomputed teacher's train
    predictions: the label column holds the teacher's output, so the default
    distill_predict() (no teacher arg) trains directly on it. This is a real distill_predict()
    of TabFM into a student that ships in the zero-dependency core -- no
    onnxruntime, no TabFM at serve time."""
    db = _ext()
    feats = _load_tables(db, Xtr, pd.Series(teacher_tr), Xte, task)
    hold = db.execute(
        f"SELECT holdout_metric FROM distill_predict('SELECT {feats}, label FROM tr',"
        f" json_object('target','label','task',?,"
        f"'student_id','s','student_kind',?))",
        ("classify" if task == "cls" else "regress", kind)).fetchone()[0]
    blob = db.execute("SELECT length(weights) FROM _predict_models WHERE"
                      " model_id='s'").fetchone()[0]
    rows = db.execute(
        f"SELECT row_ref, prediction FROM predict(NULL, 'SELECT id, {feats}"
        f" FROM te', json_object('model','s')) ORDER BY row_ref"
    ).fetchall()
    db.close()
    return [r[1] for r in rows], blob, hold


def run_ours_distill_soft(Xtr, ytr, proba, classes, Xte, kind="gbt"):
    """Soft-label distillation of TabFM: the student (gbt or mlp) learns TabFM's
    per-class probability distribution (predict_proba) via the extension's
    proba/classes options, instead of its hard argmax. The true labels stay in
    `label` so the holdout is scored against ground truth."""
    import json
    db = _ext()
    k, K = Xtr.shape[1], proba.shape[1]
    feats = ", ".join(f"f{i} REAL" for i in range(k))
    pcols = ", ".join(f"p{j} REAL" for j in range(K))
    db.execute(f"CREATE TABLE tr({feats}, {pcols}, label TEXT)")
    db.executemany(
        f"INSERT INTO tr VALUES ({','.join('?' * (k + K + 1))})",
        [list(map(float, r)) + list(map(float, pr)) + [str(y)]
         for r, pr, y in zip(Xtr.values, proba, ytr)])
    db.execute(f"CREATE TABLE te(id INTEGER, {feats})")
    db.executemany(f"INSERT INTO te VALUES ({','.join('?' * (k + 1))})",
                   [[i] + list(map(float, r)) for i, r in enumerate(Xte.values)])
    fl = ", ".join(f"f{i}" for i in range(k))
    pl = ", ".join(f"p{j}" for j in range(K))
    hold = db.execute(
        f"SELECT holdout_metric FROM distill_predict('SELECT {fl}, {pl}, label FROM"
        f" tr', json_object('target','label','proba',json(?),'classes',json(?),"
        f"'student_kind',?,'student_id','s'))",
        (json.dumps([f"p{j}" for j in range(K)]),
         json.dumps([str(c) for c in classes]), kind)).fetchone()[0]
    blob = db.execute("SELECT length(weights) FROM _predict_models WHERE"
                      " model_id='s'").fetchone()[0]
    rows = db.execute(
        f"SELECT row_ref, prediction FROM predict(NULL,'SELECT id, {fl} FROM"
        f" te', json_object('model','s')) ORDER BY row_ref"
    ).fetchall()
    db.close()
    return [r[1] for r in rows], blob, hold


def fidelity(student_preds, teacher_preds):
    """Label-free selection signal: fraction of held-out rows where the student
    reproduces the teacher's prediction. This is all you can measure without
    ground truth."""
    s = np.asarray(student_preds).astype(str)
    t = np.asarray(teacher_preds).astype(str)
    return float((s == t).mean())


def meta_features(Xtr, ytr, Xte, yte):
    """Cheap dataset descriptors to test whether the mlp-vs-gbt winner is
    predictable: a linearity gap (tree vs logistic regression) and an
    axis-alignment gain (does rotating the features with PCA help a shallow
    tree -> the boundary is oblique -> smooth-model-favorable)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    m = {}
    try:
        lr = LogisticRegression(max_iter=300, n_jobs=1)
        lr.fit(Xtr.values, ytr)
        m["mf_logreg"] = accuracy_score(yte, lr.predict(Xte.values))
    except Exception:  # noqa: BLE001
        m["mf_logreg"] = float("nan")
    try:
        t = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        t.fit(Xtr.values, ytr)
        m["mf_tree4"] = accuracy_score(yte, t.predict(Xte.values))
        p = PCA(random_state=SEED).fit(Xtr.values)
        tr = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        tr.fit(p.transform(Xtr.values), ytr)
        m["mf_tree4_pca"] = accuracy_score(yte, tr.predict(p.transform(Xte.values)))
    except Exception:  # noqa: BLE001
        m["mf_tree4"] = m["mf_tree4_pca"] = float("nan")
    return m


def run_ours_knn5(Xtr, ytr, Xte, task):
    db = _ext()
    feats = _load_tables(db, Xtr, ytr, Xte, task)
    rows = db.execute(
        f"SELECT row_ref, prediction FROM predict('SELECT {feats}, label FROM"
        f" tr', 'SELECT id, {feats} FROM te', json_object('target','label',"
        f"'task',?)) ORDER BY row_ref",
        ("classify" if task == "cls" else "regress",)).fetchall()
    db.close()
    return [r[1] for r in rows]


def run_ours_distill(Xtr, ytr, Xte, task, kind="tree"):
    """Distill the in-context knn5 teacher into a native student (teacher named
    explicitly: distill_predict() re-runs knn5 over the rows to relabel them)."""
    db = _ext()
    feats = _load_tables(db, Xtr, ytr, Xte, task)
    hold = db.execute(
        f"SELECT holdout_metric FROM distill_predict('SELECT {feats}, label FROM tr',"
        f" json_object('target','label','task',?,'teacher','knn5-incontext',"
        f"'student_id','s','student_kind',?))",
        ("classify" if task == "cls" else "regress", kind)).fetchone()[0]
    blob = db.execute("SELECT length(weights) FROM _predict_models WHERE"
                      " model_id='s'").fetchone()[0]
    rows = db.execute(
        f"SELECT row_ref, prediction FROM predict(NULL, 'SELECT id, {feats}"
        f" FROM te', json_object('model','s')) ORDER BY row_ref"
    ).fetchall()
    db.close()
    return [r[1] for r in rows], blob, hold


# ---- driver ----

def main():
    import json
    import sys
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:]
                  if a.startswith("limit=")), 0)
    jsonl = os.path.join(os.path.dirname(RESULTS), "tabarena-full.jsonl")
    print(f"TabArena-v0.1: {len(TABARENA)} datasets"
          f"{f' (limit {limit})' if limit else ''}. Streaming results to"
          f" {os.path.relpath(jsonl)}")
    rows = []
    student_bytes = []
    jf = open(jsonl, "w")
    for i, (name, X, y, task) in enumerate(load_datasets(), 1):
        if limit and i > limit:
            break
        Xtr, Xte, ytr, yte = split(X, y, task)
        rec = {"dataset": name, "task": task, "n": len(X),
               "d": X.shape[1], "classes": (y.nunique() if task == "cls"
                                            else None)}
        print(f"\n== [{i}/{len(TABARENA)}] {name} ({task}, n={len(X)},"
              f" d={X.shape[1]}) ==", flush=True)

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
            import pickle
            cp = os.path.join(TABFM_CACHE, f"{name}.pkl")
            if os.path.exists(cp):  # TabFM outputs cached from a prior run
                with open(cp, "rb") as fh:
                    cd = pickle.load(fh)
                tabfm_p, proba, classes, teacher_tr = (
                    cd["test"], cd["proba"], cd["classes"], cd["train"])
                rec["tabfm"] = score(yte, tabfm_p, task)
                rec["tabfm_s"] = 0.0
            else:
                # one predict_proba pass gives both the hard argmax and the soft
                # distribution (no extra TabFM forward)
                t0 = time.time()
                tabfm_p, est = run_tabfm(Xtr, ytr, Xte, task)
                rec["tabfm"] = score(yte, tabfm_p, task)
                rec["tabfm_s"] = time.time() - t0
                if task == "cls":
                    proba = est.predict_proba(Xtr.values)
                    classes = list(est.classes_)
                    teacher_tr = np.asarray(classes)[proba.argmax(1)]
                else:
                    proba = classes = None
                    teacher_tr = est.predict(Xtr.values)
                os.makedirs(TABFM_CACHE, exist_ok=True)
                with open(cp, "wb") as fh:
                    pickle.dump({"test": tabfm_p, "proba": proba,
                                 "classes": classes, "train": teacher_tr}, fh)
            tree_p = sklearn_tree(Xtr, teacher_tr, Xte, task)
            rec["tree<-tabfm"] = score(yte, tree_p, task)
            gtp, gtblob, gth = run_ours_distill_teacher(Xtr, teacher_tr, Xte,
                                                        task, "gbt")
            rec["gbt<-tabfm (ours)"] = score(yte, gtp, task)
            rec["gbt<-tabfm hold"] = gth
            student_bytes.append(gtblob)
            if task == "cls":  # soft-label distillation of TabFM's distribution
                sp, sblob, sh = run_ours_distill_soft(Xtr, ytr, proba, classes,
                                                      Xte, "gbt")
                rec["gbt<-tabfm soft (ours)"] = score(yte, sp, task)
                rec["gbt<-tabfm soft hold"] = sh
                student_bytes.append(sblob)
                mp, mblob, mh = run_ours_distill_soft(Xtr, ytr, proba, classes,
                                                      Xte, "mlp")  # smooth (#3)
                rec["mlp<-tabfm soft (ours)"] = score(yte, mp, task)
                rec["mlp<-tabfm soft hold"] = mh
                student_bytes.append(mblob)
                # label-free selection signal: each student's fidelity to
                # TabFM's held-out predictions (no ground truth used)
                rec["gbt<-tabfm fid"] = fidelity(gtp, tabfm_p)
                rec["gbt<-tabfm soft fid"] = fidelity(sp, tabfm_p)
                rec["mlp<-tabfm soft fid"] = fidelity(mp, tabfm_p)
        except Exception as e:  # noqa: BLE001
            rec["tabfm"] = rec["tree<-tabfm"] = float("nan")
            rec["gbt<-tabfm (ours)"] = float("nan")
            rec["gbt<-tabfm soft (ours)"] = float("nan")
            rec["mlp<-tabfm soft (ours)"] = float("nan")
            rec["tabfm_s"] = float("nan")
            print("  tabfm fail:", type(e).__name__, str(e)[:120])

        if task == "cls":  # dataset descriptors for meta-selection analysis
            try:
                rec.update(meta_features(Xtr, ytr, Xte, yte))
            except Exception as e:  # noqa: BLE001
                print("  meta fail:", str(e)[:80])

        try:
            rec["knn5 (ours)"] = score(yte, run_ours_knn5(Xtr, ytr, Xte, task),
                                       task)
        except Exception as e:  # noqa: BLE001
            rec["knn5 (ours)"] = float("nan")
            print("  knn5 fail:", str(e)[:120])

        try:
            dp, blob, dh = run_ours_distill(Xtr, ytr, Xte, task, "tree")
            rec["tree<-knn5 (ours)"] = score(yte, dp, task)
            rec["tree<-knn5 hold"] = dh
            student_bytes.append(blob)
        except Exception as e:  # noqa: BLE001
            rec["tree<-knn5 (ours)"] = float("nan")
            print("  tree-distill fail:", str(e)[:120])

        try:
            gp, gblob, gh = run_ours_distill(Xtr, ytr, Xte, task, "gbt")
            rec["gbt<-knn5 (ours)"] = score(yte, gp, task)
            rec["gbt<-knn5 hold"] = gh
            student_bytes.append(gblob)
        except Exception as e:  # noqa: BLE001
            rec["gbt<-knn5 (ours)"] = float("nan")
            print("  gbt-distill fail:", str(e)[:120])

        print("  ", {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in rec.items()}, flush=True)
        rows.append(rec)
        jf.write(json.dumps(rec) + "\n")
        jf.flush()  # durable progress: a long run survives a mid-way failure

    jf.close()
    _write(rows, student_bytes)


def _fmt(v):
    return "-" if v is None or (isinstance(v, float) and v != v) else (
        f"{v:.3f}" if isinstance(v, float) else str(v))


def _write(rows, student_bytes):
    cols = ["xgboost", "tabfm", "tree<-tabfm", "gbt<-tabfm (ours)",
            "gbt<-tabfm soft (ours)", "mlp<-tabfm soft (ours)", "knn5 (ours)",
            "tree<-knn5 (ours)", "gbt<-knn5 (ours)"]
    lines = ["# TabArena-spirit benchmark\n",
             "Curated OpenML subset, single 75/25 split (seed 0), features",
             f"capped at {MAX_FEAT}, rows capped at {MAX_ROWS}. Metric:",
             "**accuracy** (classification, higher better) / **RMSE**",
             "(regression, lower better). Not TabArena's full 51-task/30-split",
             "Elo protocol, a comparability subset. TabFM = local weights,",
             f"fp32, {TABFM_ESTIMATORS}-member ensemble (reduced from 32 for",
             "CPU tractability); `tree<-tabfm` is a sklearn CART on TabFM's",
             "train predictions, while `gbt<-tabfm (ours)` and the `<-knn5`",
             "students all run through the extension's own distill_predict().\n",
             "| dataset | task | n | d | " + " | ".join(cols) + " | tabfm s |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in cols) +
             " | --- |"]
    for r in rows:
        cells = [r["dataset"], r["task"], str(r["n"]), str(r["d"])]
        cells += [_fmt(r.get(c)) for c in cols]
        cells.append(_fmt(r.get("tabfm_s")))
        lines.append("| " + " | ".join(cells) + " |")
    if student_bytes:
        lines.append(f"\nNative student size: "
                     f"{min(student_bytes)}-{max(student_bytes)} bytes "
                     f"(runs in the zero-dependency core, ~microseconds/row; "
                     f"TabFM is tens of seconds/call).")
    print("\n".join(lines))
    print(f"\n(curated writeup with analysis lives in"
          f" {os.path.relpath(RESULTS)})")


if __name__ == "__main__":
    main()
