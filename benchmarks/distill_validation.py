"""Teacher/student distillation viability, four arms per suite:

  floor          majority class (no learning)
  direct         small GBM trained on the labeled rows only (control:
                 if direct ~= distilled, the teacher added nothing)
  teacher        TabFM zero-shot (accuracy ceiling; ~30s/call on CPU)
  distilled      same small GBM trained on teacher soft labels over
                 labeled rows + an unlabeled transfer pool

Swept over label budgets: the per-tenant personalization case rests on
small-label regimes. Reports retention = (arm-floor)/(teacher-floor),
student serve latency, and student size.

sklearn students are a harness-only proxy for the C tree students the
extension's distill_predict() will ship.

Usage: uv run --with 'tabfm[pytorch]' --with safetensors --with scikit-learn \
         python distill_validation.py
"""

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import synthetic_tabular as syt  # noqa: E402
from compare_tabular import encode, accuracy  # noqa: E402

# TabFM weights live outside the repo (13GB, non-commercial license).
# Override with SQLITE_PREDICT_DATA; defaults to a per-user cache.
_DATA = os.environ.get(
    "SQLITE_PREDICT_DATA",
    os.path.join(os.path.expanduser("~"), ".cache", "sqlite-predict"))
MODEL_DIR = os.path.join(_DATA, "tabfm")
BUDGETS = [20, 50, 100, 300]
POOL = 300
HOLDOUT = 200

SUITES = {"two_moons": syt.two_moons, "xor_categorical": syt.xor_categorical}


def make_student():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                      random_state=7)


def fit_soft(student, X, soft_probs):
    """Distillation on soft labels via sample-weighted hard targets:
    each row contributes both classes weighted by the teacher's
    probability mass (works with any sklearn classifier)."""
    Xd, yd, w = [], [], []
    for x, p1 in zip(X, soft_probs):
        Xd += [x, x]
        yd += [0, 1]
        w += [1 - p1, p1]
    student.fit(Xd, yd, sample_weight=w)
    return student


def main():
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
    teacher_model = TabFM_HF.from_pretrained(MODEL_DIR,
                                             subfolder="classification")

    lines = ["# Distillation viability (classification suites)", "",
             f"Transfer pool {POOL} unlabeled rows, holdout {HOLDOUT};"
             " retention = (arm - floor) / (teacher - floor).", "",
             "| suite | n_labeled | floor | direct | teacher | distilled |"
             " retention direct | retention distilled | student µs/row |"
             " student KB |",
             "|---|---|---|---|---|---|---|---|---|"]

    for suite, gen in SUITES.items():
        X, y, _ = gen(n=BUDGETS[-1] + POOL + HOLDOUT, seed=71)
        cols = list(X[0].keys())
        E = encode(X, cols)
        # fixed slices: labeled budget from the front, then pool, then holdout
        Epool = E[BUDGETS[-1]:BUDGETS[-1] + POOL]
        Ete, yte = E[-HOLDOUT:], y[-HOLDOUT:]

        for n_lab in BUDGETS:
            Etr, ytr = E[:n_lab], y[:n_lab]
            floor_pred = max(set(ytr), key=ytr.count)
            floor = accuracy([floor_pred] * len(yte), yte)

            direct = make_student().fit(Etr, ytr)
            direct_acc = accuracy(list(direct.predict(Ete)), yte)

            teacher = TabFMClassifier(model=teacher_model)
            t0 = time.perf_counter()
            teacher.fit(Etr, ytr)
            teacher_acc = accuracy(list(teacher.predict(Ete)), yte)
            pool_soft = [float(p[1]) for p in
                         teacher.predict_proba(Epool)]
            train_soft = [float(p[1]) for p in
                          teacher.predict_proba(Etr)]
            teacher_s = time.perf_counter() - t0

            distilled = fit_soft(make_student(), Etr + Epool,
                                 train_soft + pool_soft)
            t0 = time.perf_counter()
            dist_pred = list(distilled.predict(Ete))
            serve_us = (time.perf_counter() - t0) * 1e6 / len(Ete)
            dist_acc = accuracy(dist_pred, yte)
            size_kb = len(pickle.dumps(distilled)) / 1024

            denom = (teacher_acc - floor) or 1e-9
            lines.append(
                f"| {suite} | {n_lab} | {floor:.3f} | {direct_acc:.3f} |"
                f" {teacher_acc:.3f} | {dist_acc:.3f} |"
                f" {(direct_acc - floor) / denom:.0%} |"
                f" {(dist_acc - floor) / denom:.0%} |"
                f" {serve_us:.0f} | {size_kb:.0f} |")
            print(lines[-1], f"(teacher {teacher_s:.0f}s)")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(os.path.dirname(__file__), "results",
                           "distill-viability.md"), "w") as f:
        f.write(text)
    print("\nwritten to results/distill-viability.md")


if __name__ == "__main__":
    main()
