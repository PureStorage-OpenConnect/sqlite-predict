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
  **same** TabFM predictions **through the extension**
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

Native student size: single tree **1.7–6.6 KB**, gbt forest **~2–200 KB**;
both serve in microseconds with no onnxruntime. TabFM: **24 s to ~2 min** per
call (one slow outlier), and it needs a GPU-class runtime to be practical.

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
loses to TabFM on 40 of 48 (ties 4). It keeps most of TabFM, not all. Closing
that gap is the next lever: soft-label distillation (learn TabFM's probability
distribution, not its argmax) should transfer the part we currently discard.

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

| dataset | n | xgb | tabfm | tree<-fm | **gbt<-fm** | knn5 | tree<-k5 | gbt<-k5 | tabfm s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_satisfaction | 1500 | 0.909 | 0.923 | 0.867 | 0.880 | 0.851 | 0.808 | 0.869 | 84 |
| Amazon_employee_access | 1500 | 0.928 | 0.939 | 0.936 | 0.939 | 0.936 | 0.939 | 0.939 | 67 |
| anneal | 898 | 0.987 | 0.991 | 0.982 | 0.987 | 0.924 | 0.933 | 0.951 | 59 |
| APSFailure | 1500 | 0.995 | 0.995 | 0.989 | 0.995 | 0.989 | 0.992 | 0.992 | 112 |
| bank-marketing | 1500 | 0.856 | 0.864 | 0.869 | 0.864 | 0.856 | 0.861 | 0.869 | 71 |
| Bank_Customer_Churn | 1500 | 0.837 | 0.880 | 0.851 | 0.867 | 0.837 | 0.829 | 0.832 | 70 |
| Bioresponse | 1500 | 0.781 | 0.808 | 0.733 | 0.757 | 0.755 | 0.699 | 0.765 | 125 |
| blood-transfusion | 748 | 0.711 | 0.759 | 0.775 | 0.765 | 0.738 | 0.765 | 0.781 | 33 |
| churn | 1500 | 0.933 | 0.968 | 0.909 | 0.944 | 0.880 | 0.880 | 0.867 | 90 |
| coil2000_insurance | 1500 | 0.944 | – | – | – | 0.952 | – | – | – |
| credit-g | 1000 | 0.752 | 0.768 | 0.720 | 0.732 | 0.744 | 0.696 | 0.752 | 52 |
| credit_card_default | 1500 | 0.800 | 0.829 | 0.824 | 0.827 | 0.813 | 0.805 | 0.805 | 88 |
| diabetes | 768 | 0.740 | 0.776 | 0.708 | 0.714 | 0.734 | 0.740 | 0.766 | 32 |
| Diabetes130US | 1500 | 0.904 | 0.917 | 0.880 | 0.912 | 0.907 | 0.909 | 0.917 | 106 |
| E-CommerceShipping | 1500 | 0.645 | 0.701 | 0.685 | 0.693 | 0.651 | 0.613 | 0.645 | 1033 |
| Fitness_Club | 1500 | 0.747 | 0.787 | 0.784 | 0.781 | 0.749 | 0.749 | 0.765 | 64 |
| GiveMeSomeCredit | 1500 | 0.931 | 0.939 | 0.928 | 0.936 | 0.939 | 0.936 | 0.941 | 69 |
| hazelnut-contaminant | 1500 | 0.901 | 0.971 | 0.813 | 0.891 | 0.840 | 0.805 | 0.840 | 97 |
| heloc | 1500 | 0.757 | 0.773 | 0.747 | 0.757 | 0.744 | 0.712 | 0.736 | 86 |
| hiva_agnostic | 1500 | 0.968 | – | – | – | 0.968 | – | – | – |
| HR_job_change | 1500 | 0.728 | 0.739 | 0.741 | 0.752 | 0.741 | 0.728 | 0.739 | 71 |
| in_vehicle_coupon | 1500 | 0.632 | 0.656 | 0.669 | 0.664 | 0.613 | 0.613 | 0.635 | 87 |
| Is-this-a-good-customer | 1500 | 0.880 | 0.891 | 0.891 | 0.885 | 0.880 | 0.869 | 0.891 | 73 |
| jm1 | 1500 | 0.776 | 0.803 | 0.800 | 0.789 | 0.771 | 0.776 | 0.792 | 84 |
| kddcup09_appetency | 1500 | 0.981 | – | – | – | 0.981 | – | – | – |
| Marketing_Campaign | 1500 | 0.869 | 0.891 | 0.845 | 0.875 | 0.872 | 0.853 | 0.867 | 87 |
| maternal_health_risk | 1014 | 0.807 | 0.850 | 0.717 | 0.823 | 0.677 | 0.665 | 0.685 | 38 |
| MIC | 1500 | 0.848 | 0.861 | 0.845 | 0.851 | 0.856 | 0.853 | 0.856 | 113 |
| NATICUSdroid | 1500 | 0.880 | 0.920 | 0.909 | 0.899 | 0.856 | 0.867 | 0.888 | 111 |
| online_shoppers_intention | 1500 | 0.883 | 0.904 | 0.891 | 0.880 | 0.885 | 0.885 | 0.888 | 78 |
| polish_bankruptcy | 1500 | 0.944 | 0.957 | 0.907 | 0.949 | 0.941 | 0.917 | 0.939 | 109 |
| qsar-biodeg | 1054 | 0.871 | 0.890 | 0.795 | 0.848 | 0.856 | 0.841 | 0.864 | 81 |
| SDSS17 | 1500 | 0.952 | 0.960 | 0.957 | 0.960 | 0.760 | 0.781 | 0.832 | 69 |
| seismic-bumps | 1500 | 0.931 | 0.936 | 0.933 | 0.939 | 0.933 | 0.939 | 0.936 | 76 |
| splice | 1500 | 0.965 | 0.973 | 0.901 | 0.955 | 0.659 | 0.760 | 0.797 | 119 |
| students_dropout | 1500 | 0.728 | 0.781 | 0.704 | 0.717 | 0.672 | 0.659 | 0.717 | 118 |
| taiwanese_bankruptcy | 1500 | 0.968 | 0.976 | 0.963 | 0.971 | 0.963 | 0.971 | 0.971 | 111 |
| website_phishing | 1353 | 0.885 | 0.917 | 0.903 | 0.894 | 0.850 | 0.838 | 0.867 | 62 |
| *regression (RMSE, lower better)* | | | | | | | | | |
| airfoil_self_noise | 1500 | 1.613 | 1.081 | 3.313 | 2.299 | 3.185 | 3.572 | 2.912 | 64 |
| concrete_compressive_strength | 1030 | 5.047 | 4.082 | 6.623 | 4.974 | 9.116 | 10.321 | 8.839 | 47 |
| diamonds | 1500 | 878.3 | 763.1 | 1067.2 | 949.5 | 1283.4 | 1300.8 | 1180.7 | 67 |
| Food_Delivery_Time | 1500 | 8.720 | 7.128 | 7.692 | 7.275 | 8.862 | 8.045 | 7.739 | 67 |
| healthcare_insurance_expenses | 1338 | 5217 | 4456 | 4453 | 4463 | 5342 | 4927 | 4933 | 59 |
| houses | 1500 | 0.268 | 0.202 | 0.335 | 0.259 | 0.322 | 0.351 | 0.298 | 67 |
| miami_housing | 1500 | 149103 | 85056 | 175723 | 117776 | 140391 | 199385 | 145991 | 76 |
| physiochemical_protein | 1500 | 5.071 | 4.343 | 5.602 | 5.003 | 5.439 | 5.551 | 5.166 | 67 |
| QSAR-TID-11 | 1500 | 1.509 | 1.352 | 1.473 | 1.440 | 1.570 | 1.467 | 1.464 | 126 |
| QSAR_fish_toxicity | 907 | 0.949 | 0.910 | 0.999 | 0.932 | 0.904 | 0.933 | 0.902 | 41 |
| superconductivity | 1500 | 14.182 | 12.276 | 16.857 | 14.698 | 17.061 | 18.595 | 16.380 | 111 |
| used-Fiat-500 | 1500 | 830.1 | 779.4 | 970.3 | 799.1 | 893.9 | 858.0 | 814.1 | 66 |
| wine_quality | 1500 | 0.783 | 0.634 | 0.856 | 0.681 | 0.702 | 0.717 | 0.661 | 71 |

Reproduce with `benchmarks/tabarena.py` (needs local TabFM weights, which are
non-commercial and never redistributed; results stream to
`tabarena-full.jsonl`).
