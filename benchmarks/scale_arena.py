"""Scale study: does the distillation story survive larger data?

Three questions, one overnight run over the naturally large TabArena
datasets at n in {1500, 10k, 50k, 100k}:

  1. Retention: does the distilled gbt student keep tracking its TabICL
     teacher as rows grow (our published numbers stop at 1500)?
  2. The boundary: where does tuned xgboost catch the zero-shot teacher?
     The TFM edge concentrates in small data; publish the crossover
     instead of letting readers find it.
  3. The leak: does the in-context label leak (teacher train-agreement
     minus held-out accuracy; see permissive-teachers.md and
     arXiv:2605.18654) grow with context size? Out-of-fold labels are
     compared at 10k, where five fold-fits stay affordable.

Checkpointed per (dataset, n) cell into results/scale-arena.jsonl.
Teacher outputs are not cached across runs (contexts differ per n);
each cell records wall times instead.

Run overnight:

    caffeinate -i env TEACHERS=tabicl uv run --with tabicl \
        --with scikit-learn --with xgboost --with pandas \
        python benchmarks/scale_arena.py
"""

import os
import sys

# Single-threaded math pools: xgboost's OpenMP and torch's ATen pool
# deadlock each other in one process on macOS (see tabarena.py, which
# learned this first; this file re-learned it the hard way). TabICL's
# heavy path runs on MPS, so the CPU pools cost little here.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_OFFLINE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tabarena as TA  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

JSONL = os.path.join(os.path.dirname(__file__), "results",
                     "scale-arena.jsonl")
RESULTS = os.path.join(os.path.dirname(__file__), "results", "scale.md")
DEVICE = os.environ.get("SCALE_DEVICE", "mps")
CPU_AT = int(os.environ.get("SCALE_CPU_AT", 50_000))  # mps died natively at 50k
# Metal's 2**32-byte single-tensor cap binds on roughly rows x features:
# 50k x 10 survived with batch_size=1, 50k x 40 did not. Cells over this
# budget are skipped as measured hardware scope, not attempted.
MPS_CELL_BUDGET = int(os.environ.get("SCALE_MPS_BUDGET", 1_000_000))
CELL_TIMEOUT_S = int(os.environ.get("SCALE_CELL_TIMEOUT", 2700))
ATTEMPTS = os.path.join(os.path.dirname(__file__), "results",
                        "scale-attempts")
SIZES = (1500, 10_000, 50_000, 100_000)
OOF_AT = 10_000
SEED = 0

# TabArena datasets whose full size supports the ladder (classification
# only: the leak diagnostic needs probabilities).
LARGE = [
    (46922, "Diabetes130US"), (46920, "airline_satisfaction"),
    (46929, "GiveMeSomeCredit"), (46908, "APSFailure"),
    (46955, "SDSS17"), (46962, "taiwanese_bankruptcy"),
    (46947, "online_shoppers_intention"), (46919, "credit_card_default"),
]


def load_full(did, name):
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=did, as_frame=True)
    X, y = TA._prep(b.data, b.target, "cls")
    return X, y


def make_teacher():
    from tabicl import TabICLClassifier
    return TabICLClassifier(device=DEVICE)


def teacher_with_fallback(Xtr, ytr, Xte):
    """Fit + predict with an MPS-to-CPU fallback on failure or
    non-finite output (seen from TabICL on MPS). Native-level MPS
    crashes cannot be caught here; the cell-subprocess driver absorbs
    those."""
    first = DEVICE  # batch_size=1 makes mps survive big cells; cpu is
    # still the fallback if the guard or a crash rejects the attempt
    for attempt_device in dict.fromkeys((first, "cpu")):
        try:
            from tabicl import TabICLClassifier
            kwargs = {"device": attempt_device}
            bs = os.environ.get("SCALE_BATCH")
            if bs:
                kwargs["batch_size"] = int(bs)
            elif attempt_device == "mps" and len(Xtr) >= 20_000:
                # Metal caps one NDArray at 2**32 bytes; batch_size=1
                # divides the big tensor and survives 50k (probed)
                kwargs["batch_size"] = 1
            if len(Xtr) >= CPU_AT:
                # offload_mode="auto" engages a large-data path whose
                # workers allocate MPS tensors past Metal's 2**32-byte
                # cap regardless of device=. Pin everything to cpu, no
                # worker processes.
                kwargs.update(n_jobs=1)
                for om in ("cpu", "none", None):
                    try:
                        est = TabICLClassifier(**kwargs, offload_mode=om)                             if om is not None else TabICLClassifier(**kwargs)
                        break
                    except (ValueError, TypeError):
                        continue
            else:
                est = TabICLClassifier(**kwargs)
            t0 = time.time()
            est.fit(Xtr.values, ytr.values)
            pred_te = est.predict(Xte.values)
            fit_s = time.time() - t0
            proba_tr = est.predict_proba(Xtr.values)
            pred_tr = est.predict(Xtr.values)
            if not (np.isfinite(proba_tr).all()):
                raise ValueError("non-finite probabilities")
            # wrong-but-finite guard (seen from TabICL on MPS at 10k: a
            # 0.93-accuracy teacher suddenly at chance): a teacher that
            # cannot beat the majority class ON ITS OWN CONTEXT is
            # malfunctioning, not weak. Retry on the next device.
            majority = float(ytr.value_counts(normalize=True).max())
            agree = float((np.asarray(pred_tr).astype(str)
                           == ytr.astype(str).values).mean())
            # below the mode = malfunction (a healthy teacher on heavily
            # imbalanced data may only match the majority rate; a broken
            # one falls under it, as TabICL-on-MPS did on airline)
            if agree < majority - 0.01:
                raise ValueError(
                    f"teacher below majority on its own context"
                    f" (agree={agree:.3f}, majority={majority:.3f})")
            return est, pred_te, pred_tr, proba_tr, fit_s, attempt_device
        except Exception as e:  # noqa: BLE001
            last = e
            continue
    raise last


def one_cell(name, X, y, n):
    if len(X) < n * 4 // 3:  # need n plus a test split
        return None
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(len(X), n + n // 3), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].reset_index(drop=True)
    Xtr, Xte, ytr, yte = TA.split(Xs, ys, "cls")

    row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte),
           "d": Xs.shape[1]}

    t0 = time.time()
    row["xgboost"] = TA.score(yte, TA.run_xgb(Xtr, ytr, Xte, "cls"), "cls")
    row["xgb_s"] = round(time.time() - t0, 1)

    est, pred_te, pred_tr, proba_tr, fit_s, dev = \
        teacher_with_fallback(Xtr, ytr, Xte)
    row["tabicl"] = TA.score(yte, pred_te, "cls")
    row["tabicl_s"] = round(fit_s, 1)
    row["tabicl_device"] = dev

    # the leak gap: agreement with own training labels minus held-out acc
    agree = float((np.asarray(pred_tr).astype(str)
                   == ytr.astype(str).values).mean())
    row["leak_gap"] = round(agree - row["tabicl"], 4)
    row["ic_maxp"] = round(float(np.asarray(proba_tr).max(axis=1).mean()), 4)

    t0 = time.time()
    try:
        preds, blob, hold = TA.run_ours_distill_teacher(
            Xtr, np.asarray(pred_tr), Xte, "cls", kind="gbt")
        row["gbt<-tabicl"] = TA.score(yte, preds, "cls")
        row["distill_s"] = round(time.time() - t0, 1)
        row["blob_kb"] = round(blob / 1024, 1)
    except Exception as e:  # noqa: BLE001
        if "single class" not in str(e):
            raise
        row["collapsed"] = 1

    classes = [str(c) for c in est.classes_]
    preds, _, _ = TA.run_ours_distill_soft(
        Xtr, ytr, np.asarray(proba_tr), classes, Xte, kind="gbt")
    row["gbt<-tabicl soft"] = TA.score(yte, preds, "cls")

    if len(Xtr) <= OOF_AT:
        t0 = time.time()
        proba_oof, classes_oof = TA.oof_proba(make_teacher, Xtr, ytr)
        row["oof_s"] = round(time.time() - t0, 1)
        oof_agree = float(
            (np.array(classes_oof)[np.asarray(proba_oof).argmax(1)]
             == ytr.astype(str).values).mean())
        row["oof_gap"] = round(oof_agree - row["tabicl"], 4)
        preds, _, _ = TA.run_ours_distill_soft(
            Xtr, ytr, np.asarray(proba_oof), classes_oof, Xte, kind="gbt")
        row["gbt<-tabicl soft-oof"] = TA.score(yte, preds, "cls")
    return row


def run_single_cell(name, n):
    """Child-process entry: one (dataset, n) cell, result to the jsonl."""
    if n >= CPU_AT:
        # Metal caps a single NDArray at 2**32 bytes and TabICL touches
        # MPS internally even with device="cpu" (native SIGABRT, not
        # catchable). Make torch report MPS unavailable in this child.
        import torch
        torch.backends.mps.is_available = lambda: False  # noqa: E731
        torch.backends.mps.is_built = lambda: False  # noqa: E731
    did = dict((nm, d) for d, nm in LARGE)[name]
    X, y = load_full(did, name)
    row = one_cell(name, X, y, n)
    if row is not None:
        with open(JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  n={row['n_train']}: tabicl={row['tabicl']:.4f}"
              f" xgb={row['xgboost']:.4f}"
              f" student={row.get('gbt<-tabicl', float('nan')):.4f}"
              f" leak={row['leak_gap']:+.4f}")


def main():
    """Driver: every cell runs in its own subprocess so a native crash
    (seen from torch/MPS at 50k rows) or a hang costs one cell, not the
    campaign. A cell that crashed once is skipped, not retried."""
    import subprocess
    os.makedirs(ATTEMPTS, exist_ok=True)
    done = set()
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            done = {(json.loads(ln)["dataset"], json.loads(ln)["n_train"])
                    for ln in f if ln.strip()}
    for did, name in LARGE:
        for n in SIZES:
            if any(d == name and abs(k - n) <= n // 3 for d, k in done):
                print(f"{name} n~{n}: cached")
                continue
            d_est = None
            try:
                d_est = load_full(did, name)[0].shape[1]
            except Exception:  # noqa: BLE001
                pass
            if d_est and n * d_est > MPS_CELL_BUDGET:
                print(f"{name} n={n}: skipped, over mps budget"
                      f" ({n}x{d_est} > {MPS_CELL_BUDGET})")
                continue
            marker = os.path.join(ATTEMPTS, f"{name}-{n}")
            if os.path.exists(marker):
                print(f"{name} n={n}: crashed before, skipping")
                continue
            open(marker, "w").write("attempting\n")
            t0 = time.time()
            try:
                r = subprocess.run(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--cell", name, str(n)],
                timeout=None if CELL_TIMEOUT_S == 0 else CELL_TIMEOUT_S,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                    capture_output=True, text=True)
            except subprocess.TimeoutExpired as te:
                sys.stdout.write((te.stdout or b"").decode(errors="ignore")
                                 if isinstance(te.stdout, bytes)
                                 else (te.stdout or ""))
                print(f"{name} n={n}: CELL TIMEOUT after {CELL_TIMEOUT_S}s")
                continue
            sys.stdout.write(r.stdout)
            if r.returncode == 0:
                os.remove(marker)
                # a completed run may still have skipped (too few rows)
            else:
                tailerr = (r.stderr or "").strip().splitlines()[-2:]
                print(f"{name} n={n}: CELL DIED rc={r.returncode}"
                      f" after {time.time()-t0:.0f}s :: {' | '.join(tailerr)}")
            print(f"{name} n={n}: {time.time()-t0:.0f}s")
    report()


def report():
    if not os.path.exists(JSONL):
        return
    with open(JSONL) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    rows.sort(key=lambda r: (r["dataset"], r["n_train"]))
    lines = [
        "# Scale study: distillation beyond 1500 rows", "",
        "TabICL v2 as the teacher on the naturally large TabArena",
        "datasets, at increasing training sizes. Questions: does student",
        "retention hold, where does tuned xgboost catch the zero-shot",
        "teacher, and does the in-context label leak grow with context",
        "(the gap column; see permissive-teachers.md).", "",
        f"Device: {DEVICE} with cpu fallback (per-cell `dev`).", "",
        "| dataset | n_train | xgboost | TabICL | gbt<-TabICL | soft |"
        " soft-oof | leak gap | oof gap | teacher s | distill s | blob KB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def fmt(v, nd=3):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
        lines.append(
            f"| {r['dataset']} | {r['n_train']} | {fmt(r.get('xgboost'))} |"
            f" {fmt(r.get('tabicl'))} | {fmt(r.get('gbt<-tabicl'))} |"
            f" {fmt(r.get('gbt<-tabicl soft'))} |"
            f" {fmt(r.get('gbt<-tabicl soft-oof'))} |"
            f" {fmt(r.get('leak_gap'), 4)} | {fmt(r.get('oof_gap'), 4)} |"
            f" {r.get('tabicl_s', '-')} | {r.get('distill_s', '-')} |"
            f" {r.get('blob_kb', '-')} |")
    lines.append("")
    with open(RESULTS, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--cell":
        run_single_cell(sys.argv[2], int(sys.argv[3]))
    else:
        main()
