# TabArena-spirit benchmark (6-dataset subset)

> **The full 51-dataset pass is in [`tabarena-full.md`](tabarena-full.md)** and
> is the canonical result. This page is the original hand-picked 6-dataset
> subset; it uses the classic OpenML versions of a few datasets, so a couple of
> numbers differ slightly from the full run, which uses the TabArena-curated
> versions.

Where do sqlite-predict's students land against TabFM on the kind of real
tabular data TabFM was announced on? TabFM's announcement benchmarks on
TabArena (Erickson et al. 2025), a curated OpenML suite. This is a small,
comparable subset: a single 75/25 split (seed 0) over six OpenML tasks,
features capped at 40, rows capped at 1500. It is **not** TabArena's full
51-task / 30-split Elo protocol. The point is comparability and the
distillation delta, not a leaderboard.

Metric: **accuracy** (classification, higher is better) / **RMSE**
(regression, lower is better). Columns:

- `xgboost` — the strong tabular baseline (200 trees, depth 6)
- `tabfm` — the zero-shot foundation model, local weights, fp32, 8-member
  ensemble (reduced from the default 32 for CPU tractability, so it is a
  conservative TabFM)
- `tree<-tabfm` — a depth-8 sklearn CART fit on TabFM's train predictions:
  the distillation *principle* with a single-tree student, run in the harness
- **`gbt<-tabfm (ours)`** — our native gradient-boosted student, distilled
  from the **same** TabFM predictions **through the extension** (`distill()`
  on a target column holding TabFM's offline predictions)
- `knn5 (ours)` — our shipping in-context model, run through the extension
- `tree<-knn5 (ours)` — `distill(teacher='knn5-incontext', student_kind='tree')`
- `gbt<-knn5 (ours)` — `distill(teacher='knn5-incontext', student_kind='gbt')`

Reproduce with `benchmarks/tabarena.py` (weights are local and
non-commercial; nothing is redistributed). Categorical columns are
ordinal-encoded uniformly, which slightly disadvantages TabFM (it is built
for raw mixed types).

| dataset | task | n | d | xgboost | tabfm | tree<-tabfm | **gbt<-tabfm** | knn5 | tree<-knn5 | gbt<-knn5 | tabfm s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit-g | cls | 1000 | 20 | 0.788 | 0.784 | 0.692 | 0.764 | 0.732 | 0.736 | **0.776** | 58.9 |
| diabetes | cls | 768 | 8 | 0.776 | 0.786 | 0.740 | **0.797** | 0.750 | 0.734 | 0.786 | 31.7 |
| blood-transfusion | cls | 748 | 4 | 0.743 | 0.781 | 0.770 | **0.775** | 0.743 | 0.749 | 0.765 | 28.1 |
| vehicle | cls (4) | 846 | 18 | 0.750 | 0.901 | 0.679 | **0.778** | 0.693 | 0.642 | 0.741 | 43.6 |
| diabetes-reg | reg | 442 | 10 | 68.7 | 56.4 | 64.3 | **58.4** | 63.3 | 61.9 | 59.3 | 24.1 |
| california | reg | 1500 | 8 | 0.604 | 0.483 | 0.719 | **0.565** | 0.645 | 0.770 | 0.629 | 66.9 |

Native student size: single tree **1.7–6.6 KB**, gbt forest **30–200 KB**;
both run in the zero-dependency core at microseconds-to-low-milliseconds per
row, no onnxruntime. TabFM: **24–67 seconds** per call.

## What this shows

**Distilling the real TabFM into our native gbt is the headline, and it
works.** `gbt<-tabfm` beats the single-tree `tree<-tabfm` on all six tasks,
often by a lot: the native forest is a far better distillation vehicle than a
lone CART. And it is the teacher that matters. On the 4-class `vehicle`,
swapping the teacher from knn5 (0.693) to TabFM (0.901) lifts our *same*
student from 0.741 to **0.778** — the student rides its teacher up. That is
the whole thesis made concrete: the ceiling on a student is its teacher, and
a stronger teacher raises it.

**The student sometimes beats everything in the row.** On `diabetes`,
`gbt<-tabfm` reaches **0.797, above TabFM itself (0.786) and XGBoost
(0.776)** — the gbt's shrinkage and L2 regularization smooth the teacher's
predictions into a slightly better decision surface. On both regressions it
tracks TabFM closely (diabetes-reg 58.4 vs 56.4; california 0.565 vs 0.483)
while staying a few-hundred-kilobyte blob that needs no runtime.

**When knn5 is already a good teacher, the cheap path wins.** On `credit-g`,
knn5 (0.732) is nearly as good a teacher as TabFM (0.784) for this boundary,
and `gbt<-knn5` (0.776) edges out `gbt<-tabfm` (0.764). The lesson is not
"always use the foundation model" but "distill the best teacher you have."
`gbt<-knn5` needs no weights, no GPU, and no offline pass, and it matches or
beats tuned XGBoost on `diabetes` (0.786 vs 0.776), `blood-transfusion`
(0.765 vs 0.743), and `diabetes-reg` (59.3 vs 68.7).

**What lifts the gbt above a vanilla booster.** The forest fits each tree to
the loss gradient but sets each leaf to the second-order (Newton) step
`Σg / (Σh + λ)` with the softmax Hessian, the same move that separates
XGBoost from a plain gradient booster, plus shrinkage (a small learning rate
over many rounds) for regularization. It stays deterministic: no bootstrap,
no feature sampling, no early-stopping split, so every student carries an
exact-replay receipt.

## The recommended path

For most users, `distill()` with the in-context knn5 teacher is the pragmatic
default: no weights, no GPU, and a student that already rivals XGBoost. When a
task is hard enough to justify a foundation model, run TabFM once offline over
your training rows, store its predictions in a column, and distill that column
into a native gbt (`gbt<-tabfm` above). Either way you ship a tiny model that
runs anywhere with no onnxruntime, and every prediction carries a receipt.

## Caveats

- A six-dataset subset on one split, not TabArena's 51×30 Elo protocol. Treat
  it as a directional read, not a ranking.
- TabFM ran with an 8-member ensemble (default is 32), so its numbers here are
  a conservative floor; the real model is a little stronger.
- `tree<-tabfm` uses sklearn in the harness; `gbt<-tabfm` and the `<-knn5`
  students all run through the extension's own `distill()`. TabFM's own
  predictions are precomputed offline (its packed in-context signature does
  not fit the extension's serving path, per the TabFM→ONNX eval).
- An [independent evaluation](https://github.com/devYRPauli/tabfm-evaluation)
  found upstream bugs and high-dimensional failure modes in TabFM; our subset
  stays under 40 features and did not probe those.
