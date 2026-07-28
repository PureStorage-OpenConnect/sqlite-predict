"""Benchmark the permissively licensed tabular foundation models — TabICL
(BSD-3-Clause checkpoints) and Amazon's Mitra (Apache-2.0) — on the
TabArena subset, and distill each into our native gbt students.

Motivation: the frontier tabular FMs (TabFM, TabPFN-2.5/2.6/3) are
non-commercial, so the teachers a commercial user can actually distill
and ship are TabPFN-2 (measured in tabpfn.md) and these two. Neither
license restricts distillation in any way.

Same harness, datasets, splits, and caps as tabarena.py; TabFM /
xgboost / knn5 / TabPFN columns merge from the other campaigns' jsonl
files at report time.

Run (TabICL only; Mitra needs the heavy autogluon install):

    uv run --with tabicl --with scikit-learn --with xgboost \
        --with pandas python benchmarks/permissive_arena.py

Run with Mitra too:

    TEACHERS=tabicl,mitra uv run --with tabicl --with "autogluon.tabular[mitra]" \
        --with scikit-learn --with xgboost --with pandas \
        python benchmarks/permissive_arena.py
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(os.cpu_count() or 4))
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tabarena as TA  # noqa: E402

import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CACHE = os.path.expanduser("~/.cache/sqlite-predict/permissive-cache")
JSONL = os.path.join(os.path.dirname(__file__), "results",
                     "permissive-arena.jsonl")
FULL = os.path.join(os.path.dirname(__file__), "results",
                    "tabarena-full.jsonl")
TABPFN = os.path.join(os.path.dirname(__file__), "results",
                      "tabpfn-arena.jsonl")
RESULTS = os.path.join(os.path.dirname(__file__), "results",
                       "permissive-teachers.md")
DEVICE = os.environ.get("PERMISSIVE_DEVICE", "mps")
TEACHERS = tuple(os.environ.get("TEACHERS", "tabicl").split(","))


def run_tabicl(Xtr, ytr, Xte, task):
    from tabicl import TabICLClassifier

    if task == "cls":
        est = TabICLClassifier(device=DEVICE)
    else:
        try:
            from tabicl import TabICLRegressor
            est = TabICLRegressor(device=DEVICE)
        except ImportError:
            return None  # this build has no regressor
    t0 = time.time()
    est.fit(Xtr.values, ytr.values)
    pred_te = est.predict(Xte.values)
    secs = time.time() - t0
    out = {"pred_te": pred_te, "pred_tr": est.predict(Xtr.values),
           "secs_te": secs, "proba_tr": None, "classes": None}
    if task == "cls":
        out["proba_tr"] = est.predict_proba(Xtr.values)
        out["classes"] = list(est.classes_)
    return out


def run_mitra(Xtr, ytr, Xte, task):
    from autogluon.tabular import TabularPredictor

    tr = Xtr.copy()
    tr["_target"] = ytr.values
    ptype = "binary" if task == "cls" and ytr.nunique() == 2 else (
        "multiclass" if task == "cls" else "regression")
    t0 = time.time()
    pred = TabularPredictor(label="_target", problem_type=ptype,
                            verbosity=0).fit(
        tr, hyperparameters={"MITRA": {"fine_tune": False}})
    pred_te = pred.predict(Xte).values
    secs = time.time() - t0
    out = {"pred_te": pred_te, "pred_tr": pred.predict(Xtr).values,
           "secs_te": secs, "proba_tr": None, "classes": None}
    if task == "cls":
        proba = pred.predict_proba(Xtr)
        out["proba_tr"] = proba.values
        out["classes"] = list(proba.columns)
    return out


RUNNERS = {"tabicl": run_tabicl, "mitra": run_mitra}


def run_teacher(name, Xtr, ytr, Xte, task, teacher):
    os.makedirs(os.path.join(CACHE, teacher), exist_ok=True)
    pkl = os.path.join(CACHE, teacher, f"{name}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    out = RUNNERS[teacher](Xtr, ytr, Xte, task)
    with open(pkl, "wb") as f:
        pickle.dump(out, f)
    return out


def one_dataset(name, X, y, task):
    Xtr, Xte, ytr, yte = TA.split(X, y, task)
    row = {"dataset": name, "task": task, "n": len(X), "d": X.shape[1]}
    for teacher in TEACHERS:
        t = run_teacher(name, Xtr, ytr, Xte, task, teacher)
        if t is None:
            continue  # capability gap (e.g. no regressor); recorded by absence
        row[teacher] = TA.score(yte, t["pred_te"], task)
        row[f"{teacher}_s"] = round(t["secs_te"], 2)
        try:
            preds, blob, hold = TA.run_ours_distill_teacher(
                Xtr, np.asarray(t["pred_tr"]), Xte, task, kind="gbt")
            row[f"gbt<-{teacher}"] = TA.score(yte, preds, task)
            row[f"gbt<-{teacher} fid"] = TA.fidelity(preds, t["pred_te"])
            row[f"gbt<-{teacher} blob"] = blob
        except Exception as e:  # noqa: BLE001
            if "single class" not in str(e):
                raise
            row[f"gbt<-{teacher} collapsed"] = 1
        if task == "cls" and t["proba_tr"] is not None:
            preds, blob, hold = TA.run_ours_distill_soft(
                Xtr, ytr, np.asarray(t["proba_tr"]),
                [str(c) for c in t["classes"]], Xte, kind="gbt")
            row[f"gbt<-{teacher} soft"] = TA.score(yte, preds, task)
    return row


def main():
    done = {}
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            done = {json.loads(ln)["dataset"]: json.loads(ln) for ln in f
                    if ln.strip()}
    for name, X, y, task in TA.load_datasets():
        prev = done.get(name)
        if prev is not None and all(t in prev for t in TEACHERS):
            print(f"cached {name}")
            continue
        t0 = time.time()
        try:
            row = one_dataset(name, X, y, task)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:100]}")
            continue
        done[name] = {**(prev or {}), **row}
        with open(JSONL, "w") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
        print(f"{name}: " + " ".join(
            f"{k}={row[k]:.4f}" for k in row if isinstance(row[k], float))
            + f"  ({time.time()-t0:.0f}s)")
    report()


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {json.loads(ln)["dataset"]: json.loads(ln) for ln in f
                if ln.strip()}


def report():
    rows = _load(JSONL)
    full = _load(FULL)
    pfn = _load(TABPFN)

    def better(a, b, task):
        return a > b if task == "cls" else a < b

    lines = [
        "# Permissive tabular teachers: TabICL and Mitra", "",
        "Every teacher in this table may be distilled and the student",
        "shipped commercially: TabICL checkpoints are BSD-3-Clause, Mitra",
        "is Apache-2.0, TabPFN-2 is under the Prior Labs License",
        "(distillation permitted with attribution). Same datasets, splits,",
        "and caps as `tabarena-full.md`; xgboost/TabFM/TabPFN columns come",
        "from the other campaigns.", "",
        f"Device: {DEVICE}. First calls include one-time weight download.",
        "",
        "| dataset | task | xgboost | TabPFN-2 | TabICL | Mitra |"
        " gbt<-TabICL | soft<-TabICL | gbt<-Mitra | s/call (TabICL) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    wins = {"ticl_vs_xgb": 0, "ticl_vs_pfn2": 0, "mitra_vs_xgb": 0,
            "dticl_vs_xgb": 0}
    n = {k: 0 for k in wins}
    for name, r in rows.items():
        fr, pr = full.get(name, {}), pfn.get(name, {})
        xgb, pfn2 = fr.get("xgboost"), pr.get("tabpfn_v2")

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "-"

        lines.append(
            f"| {name} | {r['task']} | {fmt(xgb)} | {fmt(pfn2)} |"
            f" {fmt(r.get('tabicl'))} | {fmt(r.get('mitra'))} |"
            f" {fmt(r.get('gbt<-tabicl'))} | {fmt(r.get('gbt<-tabicl soft'))} |"
            f" {fmt(r.get('gbt<-mitra'))} | {r.get('tabicl_s', '-')} |")
        t = r["task"]
        for key, a, b in (("ticl_vs_xgb", r.get("tabicl"), xgb),
                          ("ticl_vs_pfn2", r.get("tabicl"), pfn2),
                          ("mitra_vs_xgb", r.get("mitra"), xgb),
                          ("dticl_vs_xgb", r.get("gbt<-tabicl"), xgb)):
            if a is not None and b is not None:
                n[key] += 1
                wins[key] += better(a, b, t)

    lines += ["", "## Win counts", ""]
    for k, label in (("ticl_vs_xgb", "TabICL beats xgboost"),
                     ("ticl_vs_pfn2", "TabICL beats TabPFN-2"),
                     ("mitra_vs_xgb", "Mitra beats xgboost"),
                     ("dticl_vs_xgb", "our gbt<-TabICL beats xgboost")):
        lines.append(f"- {label}: {wins[k]}/{n[k]}")
    lines.append("")
    with open(RESULTS, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
