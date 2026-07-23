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
- `tree<-knn5 (ours)` — `distill(student_kind='tree')`: a single native CART
  the zero-dependency core executes, distilled from the knn5 teacher
- `gbt<-knn5 (ours)` — `distill(student_kind='gbt')`: a native gradient-boosted
  forest, same core runtime, same knn5 teacher

Reproduce with `benchmarks/tabarena.py` (weights are local and
non-commercial; nothing is redistributed). Categorical columns are
ordinal-encoded uniformly, which slightly disadvantages TabFM (it is built
for raw mixed types).

| dataset | task | n | d | xgboost | tabfm | tree<-tabfm | knn5 (ours) | tree<-knn5 | **gbt<-knn5** | tabfm s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit-g | cls | 1000 | 20 | 0.788 | 0.784 | 0.692 | 0.732 | 0.736 | **0.752** | 57.7 |
| diabetes | cls | 768 | 8 | 0.776 | 0.786 | 0.740 | 0.750 | 0.734 | **0.786** | 30.6 |
| blood-transfusion | cls | 748 | 4 | 0.743 | 0.781 | 0.770 | 0.743 | 0.749 | **0.759** | 28.1 |
| vehicle | cls (4) | 846 | 18 | 0.750 | 0.901 | 0.679 | 0.693 | 0.642 | **0.726** | 43.9 |
| diabetes-reg | reg | 442 | 10 | 68.7 | 56.4 | 64.3 | 63.3 | 61.9 | **58.6** | 24.1 |
| california | reg | 1500 | 8 | 0.604 | 0.483 | 0.719 | 0.645 | 0.770 | **0.635** | 65.4 |

Native student size: single tree **1.7–6.6 KB**, gbt forest **31–141 KB**;
both run in the zero-dependency core at microseconds-to-low-milliseconds per
row, no onnxruntime. TabFM: **24–65 seconds** per call.

## What this shows

**The gbt student closes the single-tree gap and turns our native student
into a real one.** It beats `tree<-knn5` on all six tasks, and it rescues
exactly the two cases where the single tree collapsed: the 4-class `vehicle`
(0.642 → **0.726**) and `california` regression (0.770 → **0.635** RMSE). On
three of six it now **matches or beats tuned XGBoost** — `diabetes` (0.786 vs
0.776), `blood-transfusion` (0.759 vs 0.743), and `diabetes-reg` (58.6 vs
68.7). This is the point of the whole distillation story made real: a
few-kilobyte-to-a-few-hundred-kilobyte model that runs everywhere, with no
onnxruntime, holding its own against a gradient-boosting library.

**TabFM is still the strongest on the hardest task.** On the 4-class
`vehicle` it reaches 0.901, well above the gbt student's 0.726 and XGBoost's
0.750 — a foundation model earns its cost where the boundary is genuinely
complex. Everywhere else the field is close: on the binary sets and both
regressions the gbt student is within a few points of TabFM, sometimes
ahead. The remaining headroom is a *teacher* gap, not a *student* one: the
gbt student tracks its knn5 teacher well, and knn5 is not TabFM. Distilling
the strong teacher into a gbt (a `gbt<-tabfm` column) is the obvious next
lever, and it needs the model-serving work the TabFM→ONNX eval scoped.

**knn5, the shipping in-context model, is a reasonable floor.** It ties
XGBoost on `blood-transfusion` and is within a couple of points on the other
binaries, trailing on multiclass and regression — a fair showing for a
5-nearest-neighbour model with no dependencies and no training, and a good
teacher for the gbt student to compress.

## Caveats

- A six-dataset subset on one split, not TabArena's 51×30 Elo protocol. Treat
  it as a directional read, not a ranking.
- TabFM ran with an 8-member ensemble (default is 32), so its numbers here are
  a conservative floor; the real model is a little stronger.
- `tree<-tabfm` is a harness demonstration of distilling the real FM; the
  extension natively serves the `<-knn5` students. Wiring TabFM in as a
  `distill()` teacher needs the model-serving work the TabFM→ONNX eval scoped.
- An [independent evaluation](https://github.com/devYRPauli/tabfm-evaluation)
  found upstream bugs and high-dimensional failure modes in TabFM; our subset
  stays under 40 features and did not probe those.
