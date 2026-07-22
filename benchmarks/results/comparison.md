# Model comparison (synthetic suites)

Horizon 24, hourly grid, seeded generators (tests/synthetic.py).

| suite | model | MASE | sMAPE | 95% cov | ms/call |
|---|---|---|---|---|---|
| trend_season | theta-classic (ext) | 0.879 | 3.7% | 100% | 0.1 |
| trend_season | seasonal-naive (ext) | 0.739 | 3.2% | 100% | 0.1 |
| strong_season | theta-classic (ext) | 0.548 | 4.6% | 100% | 0.1 |
| strong_season | seasonal-naive (ext) | 0.904 | 7.4% | 100% | 0.1 |
| random_walk | theta-classic (ext) | 2.604 | 2.4% | 100% | 0.1 |
| random_walk | seasonal-naive (ext) | 2.568 | 2.3% | 100% | 0.1 |
| level_shift | theta-classic (ext) | 1.118 | 1.9% | 100% | 0.1 |
| level_shift | seasonal-naive (ext) | 1.196 | 2.0% | 100% | 0.1 |
| intermittent | theta-classic (ext) | 1.142 | 185.7% | 100% | 0.1 |
| intermittent | seasonal-naive (ext) | 2.470 | 139.4% | 96% | 0.1 |

## Skipped (not silently dropped)

- chronos-bolt-small: ModuleNotFoundError: No module named 'chronos'
