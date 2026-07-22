"""Seeded synthetic tabular datasets with known ground truth, for
zero-shot classification/regression validation. Same determinism
contract as synthetic.py: identical arguments, identical output.

Each generator returns (X, y, meta): X a list of feature-dict rows,
y the labels/targets, meta describing the true generative rule so tests
can assert models beat baselines for the right reason.
"""

import math
import random


def two_moons(n=400, noise=0.15, seed=11):
    """Binary classification, curved boundary — untreatable linearly."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        t = rng.random() * math.pi
        if i % 2 == 0:
            x1, x2, label = math.cos(t), math.sin(t), 0
        else:
            x1, x2, label = 1 - math.cos(t), 0.5 - math.sin(t), 1
        X.append({
            "f1": x1 + rng.gauss(0, noise),
            "f2": x2 + rng.gauss(0, noise),
        })
        y.append(label)
    return X, y, {"kind": "two_moons", "classes": 2}


def xor_categorical(n=400, p_flip=0.05, seed=12):
    """Classification where the rule is XOR of two categoricals plus a
    numeric distractor: trees find it, linear models cannot, and a
    majority-class baseline sits at ~0.5."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        a = rng.choice(["red", "blue"])
        b = rng.choice(["up", "down"])
        label = int((a == "red") != (b == "up"))
        if rng.random() < p_flip:
            label = 1 - label
        X.append({
            "color": a,
            "direction": b,
            "distractor": rng.gauss(0, 1),
        })
        y.append(label)
    return X, y, {"kind": "xor_categorical", "classes": 2}


def friedman1(n=400, noise=1.0, seed=13):
    """Friedman #1 regression: y = 10 sin(pi x1 x2) + 20 (x3-.5)^2 +
    10 x4 + 5 x5 + noise, with five useless features appended."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        xs = [rng.random() for _ in range(10)]
        target = (10 * math.sin(math.pi * xs[0] * xs[1])
                  + 20 * (xs[2] - 0.5) ** 2 + 10 * xs[3] + 5 * xs[4]
                  + rng.gauss(0, noise))
        X.append({f"x{i+1}": xs[i] for i in range(10)})
        y.append(target)
    return X, y, {"kind": "friedman1", "informative": 5, "nuisance": 5}


def stepwise_price(n=400, noise=2.0, seed=14):
    """Regression with categorical interaction: price = base[category]
    * (1 + 0.2 * premium) + size * rate[category] + noise."""
    rng = random.Random(seed)
    base = {"small": 10.0, "medium": 40.0, "large": 90.0}
    rate = {"small": 2.0, "medium": 1.0, "large": 0.5}
    X, y = [], []
    for _ in range(n):
        cat = rng.choice(list(base))
        premium = rng.random() < 0.3
        size = rng.uniform(1, 20)
        target = (base[cat] * (1 + 0.2 * premium) + size * rate[cat]
                  + rng.gauss(0, noise))
        X.append({"category": cat, "premium": int(premium), "size": size})
        y.append(target)
    return X, y, {"kind": "stepwise_price"}


def train_test_split(X, y, holdout=100):
    return X[:-holdout], y[:-holdout], X[-holdout:], y[-holdout:]


def load_tabular(db, X, y, table="tab", target="label"):
    """Create a table with an explicit id first column (the row_ref
    convention), features, and the target."""
    cols = list(X[0].keys())
    types = {
        c: "TEXT" if isinstance(X[0][c], str) else "REAL" for c in cols
    }
    ddl_cols = ", ".join(f'"{c}" {types[c]}' for c in cols)
    ttype = "TEXT" if isinstance(y[0], str) else "REAL"
    db.execute(
        f'CREATE TABLE IF NOT EXISTS {table}'
        f' (id INTEGER, {ddl_cols}, "{target}" {ttype})'
    )
    ph = ", ".join("?" for _ in range(len(cols) + 2))
    db.executemany(
        f"INSERT INTO {table} VALUES ({ph})",
        [(i, *[row[c] for c in cols], y[i]) for i, row in enumerate(X)],
    )
    db.commit()
    return cols
