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

Reproduce: `uv run --with "setuptools<81" --with gluonts --with numpy --with
pandas python benchmarks/forecast_bench.py`.

## Slice 1: our models vs a seasonal-naive baseline (40 series/dataset)

Intervals below use the **calibrated** per-horizon sigma (see next section).

| dataset | h | m | model | MASE | 95% cov | Winkler |
| --- | --- | --- | --- | --- | --- | --- |
| m4_hourly | 48 | 24 | theta-classic | 1.141 | 96% | 7016 |
| | | | stub-seasonal-naive | 1.179 | 100% | 8440 |
| | | | seasonal-naive (baseline) | 1.109 | – | – |
| tourism_monthly | 24 | 12 | theta-classic | 1.537 | 90% | 58801 |
| | | | stub-seasonal-naive | 1.272 | 94% | 60267 |
| | | | seasonal-naive (baseline) | 1.381 | – | – |
| m4_daily | 14 | 7 | theta-classic | 1.226 | 87% | 2069 |
| | | | stub-seasonal-naive | 1.401 | 87% | 2495 |
| | | | seasonal-naive (baseline) | 1.589 | – | – |

## What this shows

**Our statistical models are a fair floor, not a winner.** They are
competitive with seasonal-naive: they beat it on `m4_daily`, roughly tie on
`tourism_monthly` (our drift-aware seasonal-naive wins, theta loses), and lose
slightly on `m4_hourly`. That is the expected place for simple statistical
methods, and it is what a foundation-model teacher would improve on.

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

## Next slices

1. **Foundation-model references:** add Chronos and TimesFM, and report CRPS /
   mean weighted quantile loss (their native probabilistic metric) alongside
   MASE. This tells us where an FM teacher is worth distilling (GIFT-Eval's
   finding: low-frequency and seasonal series).
2. **Interval calibration:** fix the over-coverage so Winkler / CRPS are
   competitive; coverage is a first-class forecast metric, not an afterthought.
3. **Forecast distillation:** distill a forecast FM into a native student by
   reducing forecasting to tabular regression over lag/calendar features
   (point) and quantile regression (the probabilistic, CRPS-relevant part) —
   the forecast analog of `gbt<-tabfm`.
