"""Benchmark sqlite-predict against TabPFN (v2 and the latest v3) on the
TabArena subset, and measure distilling each into our native students.

Why two TabPFN versions:

  - TabPFN-2 weights ship under the Prior Labs License (Apache 2.0 with an
    attribution provision): distillation, including for commercial use, is
    expressly permitted (Section 10). This is the commercially distillable
    in-context teacher, so its distillation numbers are the product story.
  - TabPFN-3 (the current default) and 2.5/2.6 weights are non-commercial:
    evaluation only, like TabFM. Its numbers are the reference ceiling.

Reuses the TabArena harness (task list, prep, splits, scoring, extension
distillers) from tabarena.py. TabFM / xgboost / knn5 columns are merged
from results/tabarena-full.jsonl at report time rather than re-run.

Weights download requires a one-time Prior Labs license acceptance:
set TABPFN_TOKEN, or drop the key in ~/.cache/sqlite-predict/tabpfn-token.

Run (checkpointed per dataset; safe to interrupt and re-run):

    uv run --with tabpfn --with scikit-learn --with xgboost \
        --with pandas python benchmarks/tabpfn_arena.py
"""

import os
import sys

# tabarena.py force-sets offline HF and single-thread math pools via
# setdefault at import; pre-empt what must differ here BEFORE importing it.
# TabPFN needs the hub online (first download) and benefits from threads.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(os.cpu_count() or 4))

_token_file = os.path.expanduser("~/.cache/sqlite-predict/tabpfn-token")
if "TABPFN_TOKEN" not in os.environ and os.path.exists(_token_file):
    os.environ["TABPFN_TOKEN"] = open(_token_file).read().strip()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tabarena as TA  # noqa: E402  (harness: tasks, prep, split, score, distillers)

import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

CACHE = os.path.expanduser("~/.cache/sqlite-predict/tabpfn-cache")
JSONL = os.path.join(os.path.dirname(__file__), "results", "tabpfn-arena.jsonl")
FULL = os.path.join(os.path.dirname(__file__), "results", "tabarena-full.jsonl")
RESULTS = os.path.join(os.path.dirname(__file__), "results", "tabpfn.md")
DEVICE = os.environ.get("TABPFN_DEVICE", "cpu")
# comma-separated override, e.g. TABPFN_VERSIONS=v2 while only the v2
# license is accepted on the account (each model family gates separately)
VERSIONS = tuple(os.environ.get("TABPFN_VERSIONS", "v2,v3").split(","))


def tabpfn_estimator(task, version):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    Cls = TabPFNClassifier if task == "cls" else TabPFNRegressor
    if version == "v2":
        return Cls.create_default_for_version(ModelVersion.V2, device=DEVICE)
    return Cls(device=DEVICE)  # package default = the latest model (TabPFN-3)


def run_tabpfn(name, Xtr, ytr, Xte, task, version):
    """Fit/predict with a per-dataset pickle cache: predictions on test and
    train (hard), plus train/test probabilities for soft distillation."""
    os.makedirs(os.path.join(CACHE, version), exist_ok=True)
    pkl = os.path.join(CACHE, version, f"{name}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    est = tabpfn_estimator(task, version)
    t0 = time.time()
    est.fit(Xtr.values, ytr.values)
    pred_te = est.predict(Xte.values)
    secs_te = time.time() - t0
    pred_tr = est.predict(Xtr.values)
    out = {"pred_te": pred_te, "pred_tr": pred_tr, "secs_te": secs_te,
           "proba_tr": None, "proba_te": None, "classes": None}
    if task == "cls":
        out["proba_tr"] = est.predict_proba(Xtr.values)
        out["proba_te"] = est.predict_proba(Xte.values)
        out["classes"] = list(est.classes_)
    with open(pkl, "wb") as f:
        pickle.dump(out, f)
    return out


def one_dataset(name, X, y, task):
    Xtr, Xte, ytr, yte = TA.split(X, y, task)
    row = {"dataset": name, "task": task, "n": len(X), "d": X.shape[1]}

    for v in VERSIONS:
        t = run_tabpfn(name, Xtr, ytr, Xte, task, v)
        row[f"tabpfn_{v}"] = TA.score(yte, t["pred_te"], task)
        row[f"tabpfn_{v}_s"] = round(t["secs_te"], 2)

        # hard-label distillation into our gbt student, served by the
        # extension. On heavily imbalanced sets the teacher's hard train
        # labels can collapse to one class and distill_predict refuses
        # loudly; keep the zero-shot columns and fall through to soft.
        try:
            preds, blob, hold = TA.run_ours_distill_teacher(
                Xtr, np.asarray(t["pred_tr"]), Xte, task, kind="gbt")
            row[f"gbt<-tabpfn_{v}"] = TA.score(yte, preds, task)
            row[f"gbt<-tabpfn_{v} fid"] = TA.fidelity(preds, t["pred_te"])
            row[f"gbt<-tabpfn_{v} blob"] = blob
        except Exception as e:  # noqa: BLE001
            if "single class" not in str(e):
                raise
            row[f"gbt<-tabpfn_{v} collapsed"] = 1

        # soft-label (probability) distillation, classification only
        if task == "cls" and t["proba_tr"] is not None:
            preds, blob, hold = TA.run_ours_distill_soft(
                Xtr, ytr, t["proba_tr"], t["classes"], Xte, kind="gbt")
            row[f"gbt<-tabpfn_{v} soft"] = TA.score(yte, preds, task)
            row[f"gbt<-tabpfn_{v} soft fid"] = TA.fidelity(preds, t["pred_te"])
    return row


def main():
    done = set()
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            done = {json.loads(ln)["dataset"] for ln in f if ln.strip()}
    for name, X, y, task in TA.load_datasets():
        if name in done:
            print(f"cached {name}")
            continue
        t0 = time.time()
        try:
            row = one_dataset(name, X, y, task)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "TabPFNLicenseError":
                print("\nBLOCKED: accept the Prior Labs license once at"
                      " https://ux.priorlabs.ai, then set TABPFN_TOKEN or"
                      " write the key to"
                      " ~/.cache/sqlite-predict/tabpfn-token")
                return
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            continue
        with open(JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{name}: " + " ".join(
            f"{k}={row[k]:.4f}" for k in row
            if isinstance(row[k], float)) + f"  ({time.time()-t0:.0f}s)")
    if os.path.exists(JSONL):
        report()


def report():
    """Merge with the TabFM campaign numbers and write the results doc."""
    with open(JSONL) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    full = {}
    if os.path.exists(FULL):
        with open(FULL) as f:
            full = {json.loads(ln)["dataset"]: json.loads(ln) for ln in f
                    if ln.strip()}

    def better(a, b, task):  # is a better than b
        return a > b if task == "cls" else a < b

    lines = ["# TabPFN on the TabArena subset (and distilling it)", ""]
    lines += [
        "TabPFN-2's weights are under the Prior Labs License (distillation",
        "permitted, commercial use included, with attribution when the",
        "student is distributed). TabPFN-3, the current default, is",
        "non-commercial: its numbers are the evaluation ceiling. Same",
        "datasets, splits, and row/feature caps as `tabarena-full.md`;",
        "TabFM/xgboost/knn5 columns are that campaign's numbers.",
        "",
        f"Device: {DEVICE}. TabPFN package 8.2.0.", "",
        "| dataset | task | xgboost | TabFM | TabPFN-2 | TabPFN-3 |"
        " gbt<-2 | soft<-2 | gbt<-3 | s/call (3) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    wins = {k: 0 for k in ("v2_vs_xgb", "v3_vs_xgb", "v3_vs_tabfm",
                           "soft2_vs_xgb")}
    n_cmp = {k: 0 for k in wins}
    for r in rows:
        fr = full.get(r["dataset"], {})
        xgb, tabfm = fr.get("xgboost"), fr.get("tabfm")
        soft2 = r.get("gbt<-tabpfn_v2 soft")

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

        lines.append(
            f"| {r['dataset']} | {r['task']} | {fmt(xgb)} | {fmt(tabfm)} |"
            f" {fmt(r.get('tabpfn_v2'))} | {fmt(r.get('tabpfn_v3'))} |"
            f" {fmt(r.get('gbt<-tabpfn_v2'))} | {fmt(soft2)} |"
            f" {fmt(r.get('gbt<-tabpfn_v3'))} | {r.get('tabpfn_v3_s', '-')} |")
        t = r["task"]
        if xgb is not None:
            for key, val in (("v2_vs_xgb", r.get("tabpfn_v2")),
                             ("v3_vs_xgb", r.get("tabpfn_v3")),
                             ("soft2_vs_xgb", soft2)):
                if val is not None:
                    n_cmp[key] += 1
                    wins[key] += better(val, xgb, t)
        if tabfm is not None and r.get("tabpfn_v3") is not None:
            n_cmp["v3_vs_tabfm"] += 1
            wins["v3_vs_tabfm"] += better(r["tabpfn_v3"], tabfm, t)

    lines += ["", "## Win counts", ""]
    for k, label in (("v2_vs_xgb", "TabPFN-2 beats xgboost"),
                     ("v3_vs_xgb", "TabPFN-3 beats xgboost"),
                     ("v3_vs_tabfm", "TabPFN-3 beats TabFM"),
                     ("soft2_vs_xgb", "soft gbt<-TabPFN-2 beats xgboost")):
        lines.append(f"- {label}: {wins[k]}/{n_cmp[k]}")
    lines += ["", "Built with PriorLabs-TabPFN (evaluation; students were",
              "not distributed).", ""]
    with open(RESULTS, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
