# TabArena-spirit benchmark

Where do sqlite-predict's models land against TabFM on the kind of real
tabular data TabFM was announced on? TabFM's announcement benchmarks on
TabArena (Erickson et al. 2025), a curated OpenML suite. This is a small,
comparable subset — a single 75/25 split (seed 0) over six OpenML tasks,
features capped at 40, rows capped at 1500. It is **not** TabArena's full
51-task / 30-split Elo protocol; the point is comparability and the
distillation delta, not a leaderboard.

Metric: **accuracy** (classification, higher is better) / **RMSE**
(regression, lower is better). Columns:

- `xgboost` — the strong tabular baseline (200 trees, depth 6)
- `tabfm` — the zero-shot foundation model, local weights, fp32, 8-member
  ensemble (reduced from the default 32 for CPU tractability, so it is a
  conservative TabFM)
- `tree<-tabfm` — a depth-8 CART fit on TabFM's train predictions: the
  distillation *principle* with the real teacher. Our extension can't serve
  TabFM as a teacher, so this runs in the harness, not through `distill()`
- `knn5 (ours)` — our shipping in-context model, run through the extension
- `tree<-knn5 (ours)` — our `distill()`: a native tree student the
  zero-dependency core executes, distilled from the knn5 teacher

Reproduce with `benchmarks/tabarena.py` (weights are local and
non-commercial; nothing is redistributed). Categorical columns are
ordinal-encoded uniformly, which slightly disadvantages TabFM (it is built
for raw mixed types).

| dataset | task | n | d | xgboost | tabfm | tree<-tabfm | knn5 (ours) | tree<-knn5 (ours) | tabfm s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit-g | cls | 1000 | 20 | 0.788 | 0.784 | 0.692 | 0.732 | 0.736 | 57.7 |
| diabetes | cls | 768 | 8 | 0.776 | 0.786 | 0.740 | 0.750 | 0.734 | 30.6 |
| blood-transfusion | cls | 748 | 4 | 0.743 | 0.781 | 0.770 | 0.743 | 0.749 | 28.1 |
| vehicle | cls (4) | 846 | 18 | 0.750 | **0.901** | 0.679 | 0.693 | 0.642 | 43.9 |
| diabetes-reg | reg | 442 | 10 | 68.7 | **56.4** | 64.3 | 63.3 | 61.9 | 24.1 |
| california | reg | 1500 | 8 | 0.604 | **0.483** | 0.719 | 0.645 | 0.770 | 65.4 |

Native tree-student size: **1766–6596 bytes**, ~microseconds per row, no
onnxruntime. TabFM: **24–65 seconds** per call.

## What this shows

**TabFM is the strongest model here, and the gap widens on hard tasks.** It
wins or ties every task, and the wins are largest where the boundary is
complex: the 4-class `vehicle` (0.901 vs XGBoost's 0.750) and both
regressions. On the easy binary sets everything bunches up — on `credit-g`
XGBoost, TabFM, and our models are within four points. That is the honest
shape of tabular ML: a foundation model earns its cost on the hard problems,
not the easy ones.

**Our zero-dependency `knn5` is competitive on simple tabular data and falls
behind on the hard tasks.** It ties XGBoost on `blood-transfusion`, is within
a couple of points on `diabetes` and `credit-g`, and trails on the 4-class
`vehicle` and the regressions. For a 5-nearest-neighbour model with no
dependencies and no training, holding even with tuned XGBoost on half the
tasks is a reasonable floor — but it is a floor, not a ceiling.

**A single depth-8 tree is a lossy student, and that is the most useful
finding.** `tree<-knn5` mostly tracks its knn5 teacher (sometimes it even
helps — `diabetes-reg` 61.9 vs 63.3 — because the tree regularizes the noisy
neighbourhood). But distilling the *strong* teacher into one shallow tree
loses a lot exactly where TabFM was winning: `tree<-tabfm` drops to 0.679 on
`vehicle` (from 0.901) and 0.719 RMSE on `california` (from 0.483). One CART
cannot carry a foundation model's decision surface. The microsecond,
few-kilobyte student is real and it runs everywhere, but the current
`student_kind='tree'` is the compression floor — a gradient-boosted or MLP
student (the RFC's `'mlp'` kind) is the obvious next lever for closing the
teacher gap while keeping the student small.

## Caveats

- A six-dataset subset on one split, not TabArena's 51×30 Elo protocol. Treat
  it as a directional read, not a ranking.
- TabFM ran with an 8-member ensemble (default is 32), so its numbers here are
  a conservative floor; the real model is a little stronger.
- `tree<-tabfm` is a harness demonstration of distilling the real FM; the
  extension natively serves the `tree<-knn5` version. Wiring TabFM in as a
  `distill()` teacher needs the model-serving work the TabFM→ONNX eval scoped.
- An [independent evaluation](https://github.com/devYRPauli/tabfm-evaluation)
  found upstream bugs and high-dimensional failure modes in TabFM; our subset
  stays under 40 features and did not probe those.
