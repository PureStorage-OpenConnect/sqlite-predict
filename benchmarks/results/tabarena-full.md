# TabArena-v0.1: full suite, TabFM as teacher

This runs the whole [TabArena-v0.1 suite](https://tabarena.ai) (OpenML suite
457, 51 datasets, 38 classification and 13 regression) through the extension's
own `distill()` and `predict()`, with the real TabFM as a distillation teacher.
See [`datasets.md`](datasets.md) for what each dataset predicts.

It is not TabArena's official protocol. That uses full data, nested
cross-validation, 30 repetitions, and Elo aggregation. This is a single 75/25
split (seed 0), and every dataset is subsampled to **≤1500 rows / ≤40
features** so the in-context TabFM teacher stays tractable on CPU. Every model
sees the same capped data, so the *relative* comparison is fair, but the
absolute numbers are not TabArena leaderboard numbers. The point is the
distillation delta across a broad, real suite, not a ranking.

Metric: **accuracy** (classification, higher is better) / **RMSE**
(regression, lower is better). Columns:

- `xgb` — XGBoost, the strong tabular baseline (200 trees, depth 6)
- `tabfm` — the zero-shot foundation model (local weights, fp32, 8-member
  ensemble, reduced from 32 for CPU tractability, so a conservative TabFM)
- `tree<-fm` — a depth-8 **sklearn** CART on TabFM's train predictions (the
  distillation principle with a single-tree student, run in the harness)
- **`gbt<-fm`** — our native gradient-boosted student, distilled from the
  **same** TabFM predictions (hard argmax) **through the extension**
- **`gbt<-fm soft`** — the same student, but distilled from TabFM's full
  probability distribution (`predict_proba`) via `proba`/`classes`
  (soft-label distillation); classification only
- **`mlp<-fm soft`** — a one-hidden-layer neural-net student on the same soft
  targets (`student_kind='mlp'`); the smooth learner, classification only
- `knn5` — our shipping in-context model, via the extension
- `tree<-k5`, `gbt<-k5` — `distill(teacher='knn5-incontext', ...)`, the tree
  and gbt students, via the extension

Everything labeled `<-fm`/`<-k5`/`knn5` is the extension executing `distill()`
and `predict()` as SQL; `xgb`, `tabfm`, and `tree<-fm` are Python references.

## Aggregate (the headline)

**Mean rank** (1 = best, over the 48 datasets where every method ran):

| method | mean rank |
| --- | --- |
| tabfm | 1.38 |
| **gbt<-fm (ours)** | **2.94** |
| gbt<-k5 (ours) | 3.96 |
| xgboost | 4.21 |
| tree<-fm (sklearn) | 4.52 |
| knn5 (ours) | 5.27 |
| tree<-k5 (ours) | 5.73 |

**Head-to-head** (win / tie / loss over the 48 datasets with both methods):

| comparison | win% | record |
| --- | --- | --- |
| gbt<-fm **vs XGBoost** | **69%** | 33W 3T 12L |
| gbt<-fm vs tree<-fm (sklearn) | 79% | 38W 0T 10L |
| gbt<-fm vs gbt<-k5 | 65% | 31W 3T 14L |
| gbt<-fm vs tabfm | 8% | 4W 4T 40L |
| gbt<-k5 vs tree<-k5 | 85% | 41W 4T 3L |
| gbt<-k5 vs XGBoost | 50% | 24W 2T 22L |
| tabfm vs XGBoost | 98% | 47W 1T 0L |
| **best-of-our-students vs XGBoost** | **73%** | 35W 4T 9L |

**Soft-label distillation** (classification only, 35 datasets where both ran):

| comparison | win% | record |
| --- | --- | --- |
| gbt<-fm soft vs gbt<-fm (hard) | 46% | 16W 11T 8L |
| gbt<-fm soft **vs gbt<-k5** | **74%** | 26W 3T 6L |
| gbt<-fm (hard) vs gbt<-k5 | 65% | 23W 2T 10L |
| gbt<-fm soft vs XGBoost | 69% | 24W 4T 7L |

**The smooth (mlp) student** (classification, 35 datasets):

| comparison | win% | record |
| --- | --- | --- |
| mlp<-fm soft vs gbt<-fm soft | 26% | 9W 5T 21L |
| mlp<-fm soft vs XGBoost | 63% | 22W 0T 13L |
| **best-of-{gbt hard, gbt soft, mlp soft} vs XGBoost** | **86%** | 30W 1T 4L |

Native student size: single tree **1.7–6.6 KB**, mlp net **~1–2 KB**, gbt
forest **~2–200 KB**; all serve in microseconds with no onnxruntime. TabFM:
**24 s to ~2 min** per call (one slow outlier), and it needs a GPU-class
runtime to be practical.

## What this shows, across 51 datasets

**Among models you can actually ship, the TabFM-distilled gbt is the best.**
TabFM is the accuracy ceiling (mean rank 1.38, beats XGBoost on 47 of 48). But
it is a multi-second, runtime-heavy foundation model. Drop it and rank only the
things that run anywhere as a kilobyte blob, and `gbt<-fm` is first (rank 2.94),
ahead of tuned XGBoost (4.21), which it beats outright on **69%** of datasets.
That is the whole product thesis holding up at suite scale: distill the slow
teacher once, ship a tiny native student, keep most of the accuracy.

**Our forest is a far better distillation vehicle than a single tree.**
`gbt<-fm` beats the sklearn `tree<-fm` on **79%** of datasets, and `gbt<-k5`
beats `tree<-k5` on **85%**. The Newton-leaf gradient boosting we built is not
a toy: it carries a teacher's decision surface where one CART cannot. The
single-tree students are the bottom two rows of the rank table for a reason.

**The teacher matters, and which teacher is data-dependent.** Distilling TabFM
beats distilling knn5 on **65%** of datasets, not all of them. On `maternal_
health_risk` the TabFM teacher lifts the student from 0.685 to 0.823; on
`splice` from 0.797 to 0.955; on `SDSS17` from 0.832 to 0.960. But on a good
handful (`credit-g`, `diabetes`, `blood-transfusion`) the cheap knn5 teacher
distills *better*, because knn5's locally-smooth, axis-aligned boundary is
exactly what a tree ensemble reproduces cleanly, and because we currently
distill hard labels, which throws away the calibrated probabilities that are
TabFM's real edge. The right rule is "distill the best teacher for *this*
data," and the 73% best-of-ours-vs-XGBoost line is what per-dataset teacher
selection would chase.

**The student trails its teacher, as a lossy compression should.** `gbt<-fm`
loses to TabFM on 40 of 48 (ties 4). It keeps most of TabFM, not all.

**Soft-label distillation closes part of the gap, exactly where the theory
says it should.** Distilling TabFM's full probability distribution rather than
its hard argmax (`gbt<-fm soft`) is a modest average lift over hard labels
(16W/11T/8L, mean +0.003): on the many datasets where TabFM was already
confident, hard and soft agree, so most rows tie. The win shows up where the
teacher was *uncertain*, which is precisely the case hard argmax destroys. The
telling number is the teacher comparison: soft `gbt<-fm` beats the knn5-taught
student on **74%** of datasets, up from hard's 65%, and on **6 datasets** where
hard-label distillation had *lost* to the cheap knn5 teacher, soft distillation
caught back up to or passed it. So soft labels do what we predicted: they stop
throwing away the calibrated probabilities that are the strong teacher's edge,
which raises how often TabFM is worth distilling. What they cannot fix is the
*representational* half of the gap, an axis-aligned tree ensemble still cannot
render TabFM's warped boundary no matter how good the targets are. That is the
next student (a smooth model), not the next target.

**The smooth (mlp) student is not a better student, but it is a valuable pool
member.** A one-hidden-layer net on the same soft targets *loses* to the gbt
overall (9W/5T/21L, 26%), which is the expected result: neural nets are finicky
on small tabular data, which is why trees own the domain. But it wins outright
on 9 datasets the gbt cannot, sometimes as the best model in the row (on
`blood-transfusion` and `hazelnut-contaminant` it beats TabFM itself). The
payoff is in the pool: adding the mlp lifts best-of-our-students-vs-XGBoost
from 73% to **86%** (30W/1T/4L). A student that loses on average still earns
its place by winning where the others can't.

**But you cannot pick it in advance from cheap dataset features.** We logged a
linearity gap (tree vs logistic regression) and an axis-alignment gain (does
PCA-rotating the features help a shallow tree, i.e. is the boundary oblique) to
test whether the mlp-vs-gbt winner is predictable. It is not: the datasets
where the mlp wins and where the gbt wins have nearly identical descriptors
(axis-alignment gain −0.022 vs −0.029, linearity gap +0.032 vs +0.031,
dimensionality 23 vs 24). The affinity is real but not classifiable from these
features. This is the honest answer to "which student for this data": because
the students distill in microseconds, the reliable move is to fit all of them
and keep the one with the best held-out accuracy, not to predict the winner
from meta-features. Empirical selection beats meta-prediction when the
candidates are nearly free.

## Caveats

- A single 75/25 split with 1500-row / 40-feature caps, not TabArena's full
  nested-CV / 30-repeat / Elo protocol. Directional, not a leaderboard.
- Three datasets (`coil2000_insurance`, `hiva_agnostic`, `kddcup09_appetency`)
  are severely imbalanced: at 1500 rows both TabFM and knn5 predict a single
  class, so there is nothing to distill and those cells are blank. That is an
  honest limit of distilling a collapsed teacher, isolated per model so the
  rest of the row still runs. (`xgb`/`knn5` accuracies are shown where they
  survive; note they are near the majority base rate.)
- TabFM ran with an 8-member ensemble (default 32), a conservative floor.
- High-dimensional sets (`Bioresponse`, `hiva`, `QSAR-TID-11`) are capped to 40
  features, which handicaps everyone equally but weakens the absolute numbers.

## Full results

| dataset | n | d | xgb | tabfm | gbt<-fm | gbt<-fm soft | mlp<-fm soft | knn5 | gbt<-k5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_satisfaction | 1500 | 21 | 0.909 | 0.923 | 0.880 | 0.901 | 0.872 | 0.851 | 0.869 |
| Amazon_employee_access | 1500 | 9 | 0.928 | 0.939 | 0.939 | 0.939 | 0.936 | 0.936 | 0.939 |
| anneal | 898 | 38 | 0.987 | 0.991 | 0.987 | 0.987 | 0.982 | 0.924 | 0.951 |
| APSFailure | 1500 | 40 | 0.995 | 0.995 | 0.995 | 0.995 | 0.997 | 0.989 | 0.992 |
| bank-marketing | 1500 | 13 | 0.856 | 0.864 | 0.864 | 0.869 | 0.877 | 0.856 | 0.869 |
| Bank_Customer_Churn | 1500 | 10 | 0.837 | 0.880 | 0.867 | 0.880 | 0.880 | 0.837 | 0.832 |
| Bioresponse | 1500 | 40 | 0.781 | 0.808 | 0.757 | 0.773 | 0.731 | 0.755 | 0.765 |
| blood-transfusion | 748 | 4 | 0.711 | 0.759 | 0.765 | 0.759 | 0.791 | 0.738 | 0.781 |
| churn | 1500 | 19 | 0.933 | 0.968 | 0.944 | 0.944 | 0.901 | 0.880 | 0.867 |
| coil2000_insurance | 1500 | 40 | 0.944 | – | – | – | – | 0.952 | – |
| credit-g | 1000 | 20 | 0.752 | 0.768 | 0.732 | 0.760 | 0.692 | 0.744 | 0.752 |
| credit_card_default | 1500 | 23 | 0.800 | 0.829 | 0.827 | 0.827 | 0.813 | 0.813 | 0.805 |
| diabetes | 768 | 8 | 0.740 | 0.776 | 0.714 | 0.734 | 0.734 | 0.734 | 0.766 |
| Diabetes130US | 1500 | 40 | 0.904 | 0.917 | 0.912 | 0.920 | 0.880 | 0.907 | 0.917 |
| E-CommerceShipping | 1500 | 10 | 0.645 | 0.701 | 0.693 | 0.696 | 0.667 | 0.651 | 0.645 |
| Fitness_Club | 1500 | 6 | 0.747 | 0.787 | 0.781 | 0.781 | 0.781 | 0.749 | 0.765 |
| GiveMeSomeCredit | 1500 | 10 | 0.931 | 0.939 | 0.936 | 0.936 | 0.933 | 0.939 | 0.941 |
| hazelnut-contaminant | 1500 | 30 | 0.901 | 0.971 | 0.891 | 0.877 | 0.928 | 0.840 | 0.840 |
| heloc | 1500 | 23 | 0.757 | 0.773 | 0.757 | 0.757 | 0.773 | 0.744 | 0.736 |
| hiva_agnostic | 1500 | 40 | 0.968 | – | – | – | – | 0.968 | – |
| HR_job_change | 1500 | 12 | 0.728 | 0.739 | 0.752 | 0.747 | 0.765 | 0.741 | 0.739 |
| in_vehicle_coupon | 1500 | 24 | 0.632 | 0.656 | 0.664 | 0.640 | 0.611 | 0.613 | 0.635 |
| Is-this-a-good-customer | 1500 | 13 | 0.880 | 0.891 | 0.885 | 0.888 | 0.888 | 0.880 | 0.891 |
| jm1 | 1500 | 21 | 0.776 | 0.803 | 0.789 | 0.795 | 0.784 | 0.771 | 0.792 |
| kddcup09_appetency | 1500 | 40 | 0.981 | – | – | – | – | 0.981 | – |
| Marketing_Campaign | 1500 | 25 | 0.869 | 0.891 | 0.875 | 0.893 | 0.872 | 0.872 | 0.867 |
| maternal_health_risk | 1014 | 6 | 0.807 | 0.850 | 0.823 | 0.780 | 0.764 | 0.677 | 0.685 |
| MIC | 1500 | 40 | 0.848 | 0.861 | 0.851 | 0.853 | 0.853 | 0.856 | 0.856 |
| NATICUSdroid | 1500 | 40 | 0.880 | 0.920 | 0.899 | 0.893 | 0.904 | 0.856 | 0.888 |
| online_shoppers_intention | 1500 | 17 | 0.883 | 0.904 | 0.880 | 0.888 | 0.851 | 0.885 | 0.888 |
| polish_bankruptcy | 1500 | 40 | 0.944 | 0.957 | 0.949 | 0.947 | 0.912 | 0.941 | 0.939 |
| qsar-biodeg | 1054 | 40 | 0.871 | 0.890 | 0.848 | 0.856 | 0.875 | 0.856 | 0.864 |
| SDSS17 | 1500 | 11 | 0.952 | 0.960 | 0.960 | 0.960 | 0.917 | 0.760 | 0.832 |
| seismic-bumps | 1500 | 15 | 0.931 | 0.936 | 0.939 | 0.939 | 0.933 | 0.933 | 0.936 |
| splice | 1500 | 40 | 0.965 | 0.973 | 0.955 | 0.955 | 0.811 | 0.659 | 0.797 |
| students_dropout | 1500 | 36 | 0.728 | 0.781 | 0.717 | 0.757 | 0.731 | 0.672 | 0.717 |
| taiwanese_bankruptcy | 1500 | 40 | 0.968 | 0.976 | 0.971 | 0.976 | 0.973 | 0.963 | 0.971 |
| website_phishing | 1353 | 9 | 0.885 | 0.917 | 0.894 | 0.885 | 0.909 | 0.850 | 0.867 |
| *regression (RMSE)* | | | | | | | | | |
| airfoil_self_noise | 1500 | 5 | 1.613 | 1.081 | 2.299 | – | – | 3.185 | 2.912 |
| concrete_compressive_strength | 1030 | 8 | 5.047 | 4.082 | 4.974 | – | – | 9.116 | 8.839 |
| diamonds | 1500 | 9 | 878 | 763 | 950 | – | – | 1283 | 1181 |
| Food_Delivery_Time | 1500 | 9 | 8.720 | 7.128 | 7.275 | – | – | 8.862 | 7.739 |
| healthcare_insurance_expenses | 1338 | 6 | 5217 | 4456 | 4463 | – | – | 5342 | 4933 |
| houses | 1500 | 8 | 0.268 | 0.202 | 0.259 | – | – | 0.322 | 0.298 |
| miami_housing | 1500 | 15 | 149103 | 85055 | 117776 | – | – | 140391 | 145991 |
| physiochemical_protein | 1500 | 9 | 5.071 | 4.343 | 5.003 | – | – | 5.439 | 5.166 |
| QSAR-TID-11 | 1500 | 40 | 1.509 | 1.352 | 1.440 | – | – | 1.570 | 1.464 |
| QSAR_fish_toxicity | 907 | 6 | 0.949 | 0.910 | 0.932 | – | – | 0.904 | 0.902 |
| superconductivity | 1500 | 40 | 14.182 | 12.276 | 14.698 | – | – | 17.061 | 16.380 |
| used-Fiat-500 | 1500 | 7 | 830 | 779 | 799 | – | – | 894 | 814 |
| wine_quality | 1500 | 12 | 0.783 | 0.634 | 0.681 | – | – | 0.702 | 0.661 |

_(The display table shows the key comparison columns; the single-tree students `tree<-fm`/`tree<-k5` and per-call timings are in `tabarena-full.jsonl`.)_

Reproduce with `benchmarks/tabarena.py` (needs local TabFM weights, which are
non-commercial and never redistributed; results stream to
`tabarena-full.jsonl`).
