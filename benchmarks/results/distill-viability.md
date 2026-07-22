# Distillation viability (classification suites)

Transfer pool 300 unlabeled rows, holdout 200; retention = (arm - floor) / (teacher - floor).

| suite | n_labeled | floor | direct | teacher | distilled | retention direct | retention distilled | student µs/row | student KB |
|---|---|---|---|---|---|---|---|---|
| two_moons | 20 | 0.500 | 0.800 | 0.945 | 0.925 | 67% | 96% | 1 | 71 |
| two_moons | 50 | 0.500 | 0.945 | 0.990 | 0.975 | 91% | 97% | 1 | 71 |
| two_moons | 100 | 0.500 | 0.965 | 0.990 | 0.970 | 95% | 96% | 1 | 71 |
| two_moons | 300 | 0.500 | 0.990 | 0.990 | 0.985 | 100% | 99% | 1 | 71 |
| xor_categorical | 20 | 0.455 | 0.420 | 0.860 | 0.825 | -9% | 91% | 1 | 71 |
| xor_categorical | 50 | 0.455 | 0.405 | 0.895 | 0.895 | -11% | 100% | 1 | 70 |
| xor_categorical | 100 | 0.545 | 0.660 | 0.925 | 0.930 | 30% | 101% | 1 | 70 |
| xor_categorical | 300 | 0.455 | 0.905 | 0.935 | 0.935 | 94% | 100% | 1 | 71 |
