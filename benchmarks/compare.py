"""Model comparison on the shared synthetic suites.

Runs every available candidate over the same seeded series and reports
MASE, sMAPE, 95% interval coverage, and wall-clock per call. Extension
models run through the loadable; reference models (chronos) run via pip
if installed, and are reported as SKIPPED when absent, never silently
dropped.

Usage: uv run python compare.py   (from benchmarks/)
"""

import json
import os
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import synthetic as syn  # noqa: E402

EXT = os.path.join(os.path.dirname(__file__), "..", "dist", "predict0")
HORIZON = 24

SUITES = {
    "trend_season": lambda: syn.trend_season(n=224, noise=1.0, seed=101),
    "strong_season": lambda: syn.trend_season(n=224, trend=0.0,
                                              amplitude=8.0, noise=0.8,
                                              seed=102),
    "random_walk": lambda: syn.random_walk(n=224, sigma=1.0, seed=103),
    "level_shift": lambda: syn.level_shift(n=224, shift_at=120,
                                           magnitude=12.0, seed=104),
    "intermittent": lambda: syn.intermittent(n=224, seed=105),
}


def mase(forecast, truth, train):
    naive_err = [abs(train[i + 1] - train[i]) for i in range(len(train) - 1)]
    scale = sum(naive_err) / len(naive_err)
    if scale == 0:
        return float("inf")
    return sum(abs(f - t) for f, t in zip(forecast, truth)) / len(truth) / scale


def smape(forecast, truth):
    total = 0.0
    for f, t in zip(forecast, truth):
        denom = (abs(f) + abs(t)) / 2
        total += 0 if denom == 0 else abs(f - t) / denom
    return 100 * total / len(truth)


def run_extension_model(model, train, horizon):
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXT)
    syn.load_into(db, train)
    opts = json.dumps({"model": model, "receipt": 0})
    t0 = time.perf_counter()
    rows = db.execute(
        "SELECT forecast, lower_bound, upper_bound FROM"
        " forecast('SELECT ts, value FROM series', ?, ?)",
        (horizon, opts),
    ).fetchall()
    ms = (time.perf_counter() - t0) * 1000
    db.close()
    return [r[0] for r in rows], [(r[1], r[2]) for r in rows], ms


def run_chronos(train, horizon):
    import torch  # noqa: F401
    from chronos import BaseChronosPipeline
    import numpy as np

    pipe = run_chronos._pipe
    ctx = torch.tensor([v for _, v in train])
    t0 = time.perf_counter()
    q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon,
                                  quantile_levels=[0.025, 0.5, 0.975])
    ms = (time.perf_counter() - t0) * 1000
    q = q[0].numpy()
    fc = list(np.asarray(q[:, 1], dtype=float))
    iv = [(float(a), float(b)) for a, b in zip(q[:, 0], q[:, 2])]
    return fc, iv, ms


def main():
    candidates = {
        "theta-classic (ext)": lambda tr, h: run_extension_model(
            "theta-classic", tr, h),
        "seasonal-naive (ext)": lambda tr, h: run_extension_model(
            "stub-seasonal-naive", tr, h),
    }
    skipped = []
    try:
        from chronos import BaseChronosPipeline
        import torch
        run_chronos._pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cpu",
            torch_dtype=torch.float32)
        candidates["chronos-bolt-small (ref)"] = run_chronos
    except Exception as e:  # noqa: BLE001
        skipped.append(f"chronos-bolt-small: {type(e).__name__}: {e}")

    results = []
    for suite, gen in SUITES.items():
        rows, _ = gen()
        train, future = rows[:-HORIZON], rows[-HORIZON:]
        truth = [v for _, v in future]
        train_vals = [v for _, v in train]
        for name, fn in candidates.items():
            fc, iv, ms = fn(train, HORIZON)
            cov = sum(1 for (lo, hi), t in zip(iv, truth)
                      if lo is not None and lo <= t <= hi) / len(truth)
            results.append({
                "suite": suite, "model": name,
                "mase": mase(fc, truth, train_vals),
                "smape": smape(fc, truth),
                "coverage95": cov, "ms": ms,
            })

    out = ["# Model comparison (synthetic suites)", "",
           f"Horizon {HORIZON}, hourly grid, seeded generators"
           " (tests/synthetic.py).", "",
           "| suite | model | MASE | sMAPE | 95% cov | ms/call |",
           "|---|---|---|---|---|---|"]
    for r in results:
        out.append(
            f"| {r['suite']} | {r['model']} | {r['mase']:.3f} |"
            f" {r['smape']:.1f}% | {r['coverage95']:.0%} | {r['ms']:.1f} |")
    if skipped:
        out += ["", "## Skipped (not silently dropped)", ""]
        out += [f"- {s}" for s in skipped]

    text = "\n".join(out) + "\n"
    path = os.path.join(os.path.dirname(__file__), "results",
                        "comparison.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
    # summary stat for quick eyeballing
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r["mase"])
    for m, v in by_model.items():
        finite = [x for x in v if x != float("inf")]
        print(f"median MASE {m}: {statistics.median(finite):.3f}")


if __name__ == "__main__":
    main()
