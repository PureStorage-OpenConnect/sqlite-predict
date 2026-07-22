import math
import sqlite3
import pytest
import synthetic_tabular as syt


def run_predict(db, train_q, apply_q, opts='{"target":"label"}'):
    return db.execute(
        "SELECT * FROM predict(?, ?, ?)", (train_q, apply_q, opts)
    ).fetchall()


def test_two_moons_accuracy(db):
    X, y, _ = syt.two_moons()
    syt.load_tabular(db, X, y)
    out = run_predict(db, "SELECT f1, f2, label FROM tab WHERE id < 300",
                      "SELECT id, f1, f2 FROM tab WHERE id >= 300")
    truth = {i: y[i] for i in range(300, 400)}
    acc = sum(1 for r in out if float(r[1]) == truth[r[0]]) / len(out)
    assert acc >= 0.95
    assert all(r[3] == "ok" for r in out)
    # classification confidence is a vote fraction in [0.2, 1]
    assert all(0.2 <= r[2] <= 1.0 for r in out)


def test_xor_with_categoricals(db):
    X, y, _ = syt.xor_categorical()
    syt.load_tabular(db, X, y)
    out = run_predict(
        db,
        "SELECT color, direction, distractor, label FROM tab WHERE id < 300",
        "SELECT id, color, direction, distractor FROM tab WHERE id >= 300")
    truth = {i: y[i] for i in range(300, 400)}
    acc = sum(1 for r in out if float(r[1]) == truth[r[0]]) / len(out)
    assert acc >= 0.9


def test_regression(db):
    X, y, _ = syt.stepwise_price()
    syt.load_tabular(db, X, y, target="price")
    out = run_predict(
        db,
        "SELECT category, premium, size, price FROM tab WHERE id < 300",
        "SELECT id, category, premium, size FROM tab WHERE id >= 300",
        '{"target":"price","task":"regress"}')
    truth = {i: y[i] for i in range(300, 400)}
    se = [(float(r[1]) - truth[r[0]]) ** 2 for r in out]
    rmse = math.sqrt(sum(se) / len(se))
    mean = sum(y[:300]) / 300
    floor = math.sqrt(sum((mean - truth[i]) ** 2 for i in truth) / len(truth))
    assert rmse < floor / 3, f"rmse {rmse:.2f} vs floor {floor:.2f}"
    assert all(r[2] is None for r in out)  # no confidence for regression


def test_row_ref_echoed_any_type(db):
    X, y, _ = syt.two_moons(n=100)
    syt.load_tabular(db, X, y)
    out = run_predict(
        db, "SELECT f1, f2, label FROM tab WHERE id < 80",
        "SELECT 'row-' || id, f1, f2 FROM tab WHERE id >= 80")
    assert out[0][0].startswith("row-")


def test_receipt_and_replay(db):
    X, y, _ = syt.two_moons(n=200)
    syt.load_tabular(db, X, y)
    out = run_predict(db, "SELECT f1, f2, label FROM tab WHERE id < 150",
                      "SELECT id, f1, f2 FROM tab WHERE id >= 150")
    rid = out[0][4]
    op, = db.execute(
        "SELECT operation FROM _predict_receipts WHERE receipt_id = ?",
        (rid,)).fetchone()
    assert op == "predict"
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)", (rid,)).fetchone()
    assert match == 1, detail
    db.execute("UPDATE tab SET f1 = f1 + 1 WHERE id = 3")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict_replay(?)", (rid,)).fetchall()
    assert "PREDICT_ERR_ANCHOR_UNAVAILABLE" in str(e.value)


def test_null_feature_is_status_row(db):
    X, y, _ = syt.two_moons(n=100)
    syt.load_tabular(db, X, y)
    db.execute("INSERT INTO tab VALUES (999, NULL, 0.5, 0)")
    out = run_predict(db, "SELECT f1, f2, label FROM tab WHERE id < 80",
                      "SELECT id, f1, f2 FROM tab WHERE id >= 80")
    bad = [r for r in out if r[0] == 999]
    assert len(bad) == 1
    assert bad[0][3] == "non_numeric"
    assert bad[0][1] is None


def test_errors(db):
    X, y, _ = syt.two_moons(n=100)
    syt.load_tabular(db, X, y)
    train = "SELECT f1, f2, label FROM tab WHERE id < 80"
    apply = "SELECT id, f1, f2 FROM tab WHERE id >= 80"

    def expect(code, t, a, opts):
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute("SELECT * FROM predict(?, ?, ?)",
                       (t, a, opts)).fetchall()
        assert code in str(e.value), str(e.value)

    expect("PREDICT_ERR_TARGET", train, apply, '{"task":"classify"}')
    expect("PREDICT_ERR_TARGET", train, apply, '{"target":"nope"}')
    expect("PREDICT_ERR_TASK", train, apply,
           '{"target":"label","task":"cluster"}')
    expect("PREDICT_ERR_MODEL_NOT_FOUND", train, apply,
           '{"target":"label","model":"tabfm"}')
    expect("PREDICT_ERR_SCHEMA", train,
           "SELECT id, f1 FROM tab WHERE id >= 80", '{"target":"label"}')
    expect("PREDICT_ERR_SCHEMA", train,
           "SELECT id, f1, f2, distractor FROM tab, (SELECT 1 distractor)"
           " WHERE id >= 80", '{"target":"label"}')
    expect("PREDICT_ERR_QUERY_NOT_READONLY", "DELETE FROM tab", apply,
           '{"target":"label"}')
    expect("PREDICT_ERR_SCHEMA",
           "SELECT f1, f2, label FROM tab WHERE id < 3", apply,
           '{"target":"label"}')  # train smaller than k


def test_deterministic_and_distinct_receipts(db):
    X, y, _ = syt.two_moons(n=150)
    syt.load_tabular(db, X, y)
    a = run_predict(db, "SELECT f1, f2, label FROM tab WHERE id < 100",
                    "SELECT id, f1, f2 FROM tab WHERE id >= 100")
    b = run_predict(db, "SELECT f1, f2, label FROM tab WHERE id < 100",
                    "SELECT id, f1, f2 FROM tab WHERE id >= 100")
    assert [r[:4] for r in a] == [r[:4] for r in b]
    assert a[0][4] != b[0][4]
