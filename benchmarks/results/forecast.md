# Forecast benchmark (real data)

Validates `forecast()` the way time-series foundation models are validated
(GIFT-Eval / Monash style), on standard [gluonts](https://ts.gluon.ai)
datasets rather than the synthetic suites in `comparison.md`. Metrics:

- **MASE** — point accuracy, scaled by the in-sample seasonal-naive error
  (so ~1.0 means "as good as seasonal-naive"; lower is better)
- **coverage** — fraction of the horizon inside the 95% interval (target 95%)
- **Winkler** — interval score at the 95% level: interval width plus a penalty
  for points that fall outside, so it rewards intervals that are *narrow and
  still cover* (lower is better)
- **mwQL** — mean weighted quantile loss over the nine deciles, normalized by
  the total absolute target: a scale-free CRPS proxy that scores the *whole*
  predictive distribution, not just the 95% interval (lower is better). This is
  the metric Chronos and TimesFM are ranked on in GIFT-Eval.

Reproduce: `uv run --with "setuptools<81" --with gluonts --with numpy --with
pandas --with chronos-forecasting --with torch python
benchmarks/forecast_bench.py`. Chronos forecasts are cached under
`~/.cache/sqlite-predict/forecast-cache` (a non-committed non-commercial
derivative), so re-runs need no FM inference.

## Slice 1: our models vs seasonal-naive and a forecast foundation model

40 series/dataset. Intervals use the **calibrated** per-horizon sigma (see next
section). `chronos` is `amazon/chronos-bolt-small` (zero-shot, CPU); it emits
deciles, so its 95% interval is linearly extrapolated to 0.025/0.975.

| dataset | h | m | model | MASE | 95% cov | Winkler | mwQL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m4_hourly | 48 | 24 | theta-classic | 1.141 | 96% | 7016 | 0.024 |
| | | | stub-seasonal-naive | 1.179 | 100% | 8440 | 0.025 |
| | | | seasonal-naive (baseline) | 1.109 | – | – | – |
| | | | **chronos-bolt-small** | **0.794** | 89% | **3763** | **0.016** |
| tourism_monthly | 24 | 12 | theta-classic | 1.537 | 90% | 58801 | 0.054 |
| | | | **stub-seasonal-naive** | **1.272** | 94% | 60267 | 0.050 |
| | | | seasonal-naive (baseline) | 1.381 | – | – | – |
| | | | chronos-bolt-small | 1.363 | 92% | 53567 | 0.050 |
| m4_daily | 14 | 7 | theta-classic | 1.226 | 87% | 2069 | 0.016 |
| | | | stub-seasonal-naive | 1.401 | 87% | 2495 | 0.018 |
| | | | seasonal-naive (baseline) | 1.589 | – | – | – |
| | | | chronos-bolt-small | 1.274 | 91% | **1206** | **0.013** |

## What this shows

**Our statistical models are a fair floor, and the floor is higher than it
looks.** They beat seasonal-naive on `m4_daily`, and on `tourism_monthly` our
drift-aware seasonal-naive (`stub`) is the *best point forecaster in the table*
(MASE 1.272), edging out Chronos. On `m4_daily` theta ties Chronos on point
accuracy (1.226 vs 1.274). Cheap, deterministic, microsecond statistics are not
embarrassed here.

**Where the foundation model earns its keep is the shape of the distribution,
and one regime.** Chronos wins mwQL on all three datasets (it forecasts the
full quantile fan; our two-parameter Gaussian interval can't), and on
`m4_hourly` it wins outright and by a lot (MASE 0.794 vs our ~1.14). High-
frequency hourly data with rich intra-day structure is exactly where a
pretrained sequence model pulls ahead of a two-line statistical method. That is
the signal for *where distillation pays*: not everywhere, but on the regimes
and the probabilistic metric where an FM genuinely beats the floor. A student
distilled from Chronos on `m4_hourly` inherits the win at microsecond serving
cost; on `tourism_monthly` there is nothing to distill, because the stat model
already wins.

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
average); it is a *distributional* one — daily errors are fat-tailed and a
symmetric Gaussian interval under-covers the tails. Proper quantile forecasts
(the CRPS-relevant work) are the fix, and they are also what distilling a
foundation-model teacher would bring.

## Done, and next

- **Interval calibration** (done, see above): replaced the lag-1 `sigma*sqrt(h)`
  random-walk interval with a per-horizon in-sample backtest. Coverage moved
  toward nominal, the interval score roughly halved.
- **Foundation-model reference** (done for Chronos): `chronos-bolt-small` runs
  zero-shot on CPU with mwQL/CRPS alongside MASE, cached per series. This is
  what tells us *where* an FM teacher is worth distilling.

Next:

1. **TimesFM reference:** add `google/timesfm-2.0-500m-pytorch` as a second FM
   column (the harness already runs it when the `timesfm` package is present;
   it is skipped gracefully otherwise). A second FM guards against reading one
   model's quirks as a general FM/stat gap.
2. **Forecast distillation:** distill a forecast FM into a native student by
   reducing forecasting to tabular regression over lag/calendar features
   (point) and quantile regression (the probabilistic, CRPS-relevant part) —
   the forecast analog of `gbt<-tabfm`. Target the regime the benchmark flagged:
   `m4_hourly`, where Chronos beats the stat floor and there is a real win to
   compress.
