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

| dataset | h | m | model | MASE | 95% cov | Winkler |
| --- | --- | --- | --- | --- | --- | --- |
| m4_hourly | 48 | 24 | theta-classic | 1.141 | 91% | 8327 |
| | | | stub-seasonal-naive | 1.179 | 99% | 15643 |
| | | | seasonal-naive (baseline) | 1.109 | – | – |
| tourism_monthly | 24 | 12 | theta-classic | 1.537 | 98% | 77981 |
| | | | stub-seasonal-naive | 1.272 | 99% | 119968 |
| | | | seasonal-naive (baseline) | 1.381 | – | – |
| m4_daily | 14 | 7 | theta-classic | 1.226 | 88% | 1750 |
| | | | stub-seasonal-naive | 1.401 | 84% | 2189 |
| | | | seasonal-naive (baseline) | 1.589 | – | – |

## What this shows

**Our statistical models are a fair floor, not a winner.** They are
competitive with seasonal-naive: they beat it on `m4_daily`, roughly tie on
`tourism_monthly` (our drift-aware seasonal-naive wins, theta loses), and lose
slightly on `m4_hourly`. That is the expected place for simple statistical
methods, and it is what a foundation-model teacher would improve on.

**The interval calibration is the real, measured gap.** Coverage runs 84–99%
against a 95% target, mostly *over*-covering, and the Winkler scores are large
because the intervals are too wide. The synthetic suite reported a flat 100%
coverage and never surfaced this; a real benchmark does. Fixing interval
calibration is the concrete next task the endpoint validation exposes.

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
