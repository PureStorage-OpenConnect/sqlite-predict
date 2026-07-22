# Tabular model comparison (synthetic suites)

Holdout 100 rows; seeded generators (tests/synthetic_tabular.py).

| suite | task | model | metric | value | ms/call |
|---|---|---|---|---|---|
| two_moons | cls | majority (floor) | acc | 0.500 | 0 |
| two_moons | cls | knn5 (baseline) | acc | 0.990 | 12 |
| two_moons | cls | tabfm (ref) | acc | 0.990 | 32603 |
| xor_categorical | cls | majority (floor) | acc | 0.550 | 0 |
| xor_categorical | cls | knn5 (baseline) | acc | 0.980 | 14 |
| xor_categorical | cls | tabfm (ref) | acc | 0.980 | 33598 |
| friedman1 | reg | mean (floor) | rmse | 4.851 | 0 |
| friedman1 | reg | knn5 (baseline) | rmse | 2.990 | 32 |
| friedman1 | reg | tabfm (ref) | rmse | 1.184 | 42212 |
| stepwise_price | reg | mean (floor) | rmse | 28.235 | 0 |
| stepwise_price | reg | knn5 (baseline) | rmse | 2.115 | 14 |
| stepwise_price | reg | tabfm (ref) | rmse | 1.990 | 36367 |
