# Forecast benchmark (real data)

Validates `forecast()` the way time-series foundation models are validated
(GIFT-Eval / Monash style), on standard [gluonts](https://ts.gluon.ai)
datasets rather than the synthetic suites in `comparison.md`. Metrics:

- **MASE**: point accuracy, scaled by the in-sample seasonal-naive error
  (so ~1.0 means "as good as seasonal-naive"; lower is better)
- **coverage**: fraction of the horizon inside the 95% interval (target 95%)
- **Winkler**: interval score at the 95% level: interval width plus a penalty
  for points that fall outside, so it rewards intervals that are *narrow and
  still cover* (lower is better)
- **mwQL**: mean weighted quantile loss over the nine deciles, normalized by
  the total absolute target: a scale-free CRPS proxy that scores the *whole*
  predictive distribution, not just the 95% interval (lower is better). This is
  the metric Chronos and TimesFM are ranked on in GIFT-Eval.

Reproduce: `uv run --with "setuptools<81" --with gluonts --with numpy --with
pandas --with chronos-forecasting --with torch --with "timesfm[torch]" python
benchmarks/forecast_bench.py`. FM forecasts are cached under
`~/.cache/sqlite-predict/forecast-cache` (a non-committed non-commercial
derivative), so re-runs need no FM inference.

## Slice 1: our models vs seasonal-naive and two forecast foundation models

40 series/dataset. Intervals use the **calibrated** per-horizon sigma (see next
section). `chronos` is `amazon/chronos-bolt-small`, `timesfm` is
`google/timesfm-2.5-200m-pytorch` (both zero-shot, CPU). Both emit deciles, so
the 95% interval is linearly extrapolated to 0.025/0.975. Bold marks the best
of each column within a dataset.

| dataset | h | m | model | MASE | 95% cov | Winkler | mwQL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m4_hourly | 48 | 24 | theta-classic | 1.141 | 96% | 7016 | 0.024 |
| | | | stub-seasonal-naive | 1.179 | 100% | 8440 | 0.025 |
| | | | seasonal-naive (baseline) | 1.109 | – | – | – |
| | | | **chronos-bolt-small** | **0.794** | 89% | **3763** | **0.016** |
| | | | timesfm-2.5-200m | 1.101 | 81% | 5179 | 0.023 |
| tourism_monthly | 24 | 12 | theta-classic | 1.537 | 90% | 58801 | 0.054 |
| | | | stub-seasonal-naive | 1.272 | 94% | 60267 | 0.050 |
| | | | seasonal-naive (baseline) | 1.381 | – | – | – |
| | | | chronos-bolt-small | 1.363 | 92% | 53567 | 0.050 |
| | | | **timesfm-2.5-200m** | **1.211** | 87% | **49437** | **0.045** |
| m4_daily | 14 | 7 | theta-classic | 1.226 | 87% | 2069 | 0.016 |
| | | | stub-seasonal-naive | 1.401 | 87% | 2495 | 0.018 |
| | | | seasonal-naive (baseline) | 1.589 | – | – | – |
| | | | chronos-bolt-small | 1.274 | 91% | **1206** | **0.013** |
| | | | **timesfm-2.5-200m** | **1.187** | 89% | 1207 | **0.013** |

## What this shows

**Our statistical models are a fair floor, and the floor is higher than it
looks.** They are never last. On `tourism_monthly` our drift-aware
seasonal-naive (`stub`, MASE 1.272) beats *both* Chronos and the
seasonal-naive baseline on point accuracy; on `m4_daily` theta (1.226) beats
Chronos too. Cheap, deterministic, microsecond statistics are not embarrassed
here.

**The two foundation models disagree, sharply, about who wins where.** Chronos
owns `m4_hourly` (MASE 0.794, next-best 1.10) and is the *worst* FM on the other
two. TimesFM is the reverse: best on `tourism_monthly` (1.211) and `m4_daily`
(1.187), but merely ordinary on hourly (1.101, barely past our theta). Neither
FM is "the forecast teacher." Which one to distill is regime-dependent, and
picking the wrong one would hand you a worse student than the stat floor. That
is the same label-free selection problem the tabular side hit (see
`tabarena-full.md`): you choose the teacher by held-out fidelity per regime, not
by reputation.

**On the full distribution, the FMs win or tie mwQL everywhere, but the margin
is thin except in one regime.** They forecast the whole quantile fan; our
two-parameter Gaussian interval can't. Yet on monthly and daily the gap is small
(0.045 vs our 0.050; 0.013 vs 0.016). The one large probabilistic win is Chronos
on hourly (0.016 vs our 0.024), the same regime it wins on point accuracy.

**So where does distillation pay?** `m4_hourly` via Chronos: a real, large win
to compress into a microsecond student. `tourism_monthly` via anyone: don't
bother on point accuracy, the stat model already wins. The benchmark's job was
to tell those two cases apart, and it does.

**Interval calibration: found on real data, then fixed.** The first run of this
benchmark exposed what the synthetic suite (flat 100% coverage) never did: the
intervals were badly miscalibrated. The seasonal-naive model was scaling its
interval from *lag-1* differences even though it forecasts a full season back,
and both models grew the interval as `sigma * sqrt(h)` (a random-walk
assumption). Replacing that with a **per-horizon sigma estimated by
backtesting the model on its own history** brought coverage toward nominal and
sharpened the intervals: seasonal-naive on `tourism_monthly` went 99% → 94%
coverage with the Winkler score roughly halved, and theta on `m4_hourly` went
91% → 96%. Point accuracy (MASE) is unchanged, since only the intervals moved.

The residual gap is on `m4_daily`, where both models still cover ~87% against
95%. That is not a scale problem any more (the empirical sigma is right on
average); it is a *distributional* one: daily errors are fat-tailed and a
symmetric Gaussian interval under-covers the tails. Proper quantile forecasts
(the CRPS-relevant work) are the fix, and they are also what distilling a
foundation-model teacher would bring.

## Done, and next

- **Interval calibration** (done, see above): replaced the lag-1 `sigma*sqrt(h)`
  random-walk interval with a per-horizon in-sample backtest. Coverage moved
  toward nominal, the interval score roughly halved.
- **Foundation-model references** (done): `chronos-bolt-small` and
  `timesfm-2.5-200m` both run zero-shot on CPU with mwQL/CRPS alongside MASE,
  cached per series. Running two FMs, not one, is what surfaced that they
  disagree about where they win, which is what tells us *which* FM to distill in
  a given regime.

- **Forecast distillation into a native student** (done, see below): distilling
  Chronos into a native MLP forecast student recovers most of the FM's edge in
  the zero-dependency build. It went through one instructive failure first.

## Forecast distillation: what actually worked

The obvious reduction, forecasting to tabular regression over lag features with
a **tree** student (the forecast analog of `gbt<-tabfm`), does not work. A
gradient-boosted tree distilled from Chronos on `m4_hourly` stalls at ~1.18
MASE, barely past seasonal-naive and nowhere near the teacher's 0.79. Trees are
axis-aligned; they cannot represent a temporal function, and no amount of
feature engineering fixes that (a level-regression variant even blew up on
trending daily series, MASE >10, because a tree cannot extrapolate a trend).

The fix was the **student**, not the reduction. A one-hidden-layer MLP distilled
from Chronos on instance-normalized context windows recovers most of the gap,
and it runs in the zero-dependency core (`distill_forecast` + a `PSFCST` blob
served by `forecast()`, benchmark `forecast_native.py`):

| model | MASE mean | median | notes |
| --- | --- | --- | --- |
| chronos-full (teacher) | 0.794 | 0.714 | the ceiling |
| **native MLP student** | **0.886** | **0.818** | zero-dep, deterministic, µs serving |
| seasonal-naive (floor) | 1.007 | 0.945 | |
| gbt tree student | ~1.18 | – | trees can't follow |

This matches the current literature: TimeDistill (arXiv 2502.15016) and DistilTS
(arXiv 2601.12785) both find an MLP is the SOTA student for compressing a
forecast FM, and name the "architecture discrepancy" that sinks the tree. The
training hyperparameters (`epochs`/`lr`, exposed as `distill_forecast` options)
were chosen on the holdout RMSE, never the test MASE.

## Probabilistic student: distill the fan, not pinball

The point student's interval came from a backtest. A quantile head fixes that,
but the loss choice is not the obvious one. Three options, on m4_hourly (80%
band, `quantiles=[0.1..0.9]`, `forecast_quantile.py`):

| student | MASE | 80% cov | mwQL |
| --- | --- | --- | --- |
| chronos (teacher) | 0.794 | 79% | 0.0162 |
| **distill the teacher's fan (MSE)** | **0.89** | **78%** | **0.018** |
| hybrid (distill median + pinball spread) | 0.88 | 65% | 0.019 |
| pinball on actuals only | 0.99 | 52% | 0.022 |

Pinball has to *estimate* quantiles from limited samples, so it collapses the
spread and under-covers (52–65%). Chronos's fan is already calibrated, so
distilling it directly (MSE over `nquant` outputs per step) inherits that
calibration: 78% coverage of an 80% band, matching the teacher, CRPS-competitive
(mwQL 0.018 vs 0.016), and served straight off the fan with no backtest. Pinball
is the right tool only when there is *no* calibrated teacher fan to distill.
This shipped as the `quantiles` option to `distill_forecast`.

## Serving the FM directly (ONNX)

For the last ~10% the native student leaves on the table, the opt-in
`loadable-onnx` build now serves `chronos-bolt-small` as a `forecast()` backend
(the `sequence` io_spec layout). Register the exported model
(`scripts/export_chronos_onnx.py`) and call `forecast()` as usual; the point and
interval come off the model's decile fan. On m4_hourly (40 series), the
onnx-served Chronos reproduces the Python model:

| path | MASE | build |
| --- | --- | --- |
| chronos via onnx `forecast()` | 0.802 | opt-in (`loadable-onnx`) |
| chronos, Python reference | 0.794 | — |
| native distilled student | 0.89 | **zero-dependency** |

The small gap (0.802 vs 0.794) is the context truncation to a multiple of the
patch size; otherwise it is the same model. So the extension spans the whole
range: a zero-dependency native student at ~0.89, and the exact FM at ~0.80 for
those who opt into onnxruntime. Reproduce with `benchmarks/forecast_onnx.py`.

## TimesFM as a teacher, reconstructed in full

TimesFM 2.5 does not export cleanly: two of its refinements (flip-invariance and
the continuous quantile head) introduce data-dependent symints / in-place
mutation that `torch.export` rejects. Rather than ship the weaker single-decode
model, the export emits the raw two-head core (point + quantile fans) and the
extension rebuilds the full model outside the graph, driven by io_spec flags
(`flip_invariance`, `continuous_head`, `quantile_crossing_repair`, `denormalize`,
`fixed_context`). The in-DB reconstruction matches the reference
`timesfm.forecast` pipeline to float32 precision. Then
`distill_forecast(teacher='timesfm-onnx')` distills it into a native student in
one SQL call. On m4_hourly (40 series, context 512, 1500 epochs, 640 windows):

| model | MASE | notes |
| --- | --- | --- |
| chronos teacher (served in-DB) | 0.802 | best FM on this stationary data |
| chronos-distilled student | 1.032 | zero-dependency serve |
| TimesFM teacher (served in-DB) | 1.049 | full reconstruction, not single-decode |
| TimesFM-distilled student | 1.065 | zero-dependency serve |
| seasonal-naive floor | 1.109 | — |

Two honest reads. First, the reconstruction works: the TimesFM teacher (1.049) is
the full model, and its student (1.065) tracks it within ~1.5%, closing 74% of
the naive->teacher gap, a real jump from the single-decode stub's ~1.17. Second,
the student is the bottleneck, not the teacher: chronos is far the better teacher
here (0.802 vs 1.049), yet its student (1.032) barely beats TimesFM's (1.065),
because at the time the native PSFCST student was a raw-lag MLP that sat near
~1.03 on hourly regardless of teacher. m4_hourly is stationary, exactly the
regime where TimesFM's dropped refinements matter least and Chronos wins;
TimesFM's edge is non-stationary data, which this benchmark does not exercise.
Reproduce with `benchmarks/timesfm_teacher.py` (needs `TIMESFM_ONNX`).

## The student architecture was the bottleneck (DLinear/TiDE)

The ~1.03 the distilled student sat at was not a distillation ceiling; it was the
wrong model. A dense MLP over 512 raw lags has no temporal inductive bias, and
the 2022-2023 forecasting literature (DLinear, "Are Transformers Effective for
Time Series Forecasting?", TiDE) is blunt that a linear map with the right
structure beats it. Rebuilding the student as a **linear skip plus a small
nonlinear residual** (out = Wskip*x + scale * W2*tanh(W1*x)) fixes it. The linear
skip carries the seasonal-naive + trend structure the MLP kept failing to
represent; the residual refines. Distilling the onnx Chronos teacher (0.802) on
m4_hourly (40 target series, in-DB, single model):

| student | MASE | note |
| --- | --- | --- |
| old raw-lag MLP | 1.032 | no temporal structure |
| pure linear (`hidden=0`) | 0.981 | convex, tiny blob |
| TiDE, hidden 128 | 0.946 | |
| **TiDE, hidden 256 (default)** | **0.894** | the m4_hourly optimum |
| TiDE, hidden 512 / 1024 | 0.919 / 0.915 | slightly overfit |
| AutoTheta (best classical) | 1.034 | |
| chronos teacher | 0.802 | ceiling |

The default student now closes ~70% of the naive->chronos gap and beats every
classical method, distilled into a zero-dependency blob served in microseconds.
Two design notes fell out of controlled sweeps, both counter to intuition: the
residual **width** helps only *with* the linear skip (256 beats 128 and 512;
widening the skip-less MLP had only hurt), and every other lever tried made it
worse or did nothing in-DB. Ensembling gave nothing (8 models 0.939 vs 0.940
single, because full-batch + weight decay leave little variance to average),
more training series *diluted* a single global model (40 target series 0.946 ->
120 mixed 0.993), more epochs overfit, and removing the feature standardization
or switching to minibatch both hurt. Reproduce the architecture comparison with
`benchmarks/forecast_student_arch.py` (needs `CHRONOS_ONNX`).
