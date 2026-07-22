"""ONNX runtime backend tests (the opt-in foundation-model serving path).

These run only against the loadable-onnx build; they self-skip when the
loaded extension reports no onnx runtime, so the core `make test` ignores
them. Fixtures are tiny hand-built ONNX models committed under fixtures/,
with expected outputs computed independently in pure Python — the tests
assert the extension reproduces them, not that it agrees with itself.
"""

import json
import os
import sqlite3

import pytest
from conftest import connect

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures")


def _has_onnx():
    db = connect()
    try:
        v = json.loads(db.execute("SELECT predict_version()").fetchone()[0])
        return "onnx" in v.get("runtimes", [])
    finally:
        db.close()


pytestmark = pytest.mark.skipif(
    not _has_onnx(), reason="extension built without the onnx runtime")


def _abs(name):
    return os.path.join(FIX, name)


def _register(db, model_id, onnx_file, io_spec, license="MIT"):
    cfg = {"runtime": "onnx", "kind": "student", "license": license,
           "weights_uri": _abs(onnx_file), "io_spec": io_spec}
    return db.execute("SELECT predict_register(?, ?)",
                      (model_id, json.dumps(cfg))).fetchone()[0]


CLF_IO = {"layout": "vector", "input": "float_input", "features": ["f1", "f2"],
          "output": {"name": "probabilities", "kind": "probs",
                     "labels": ["0", "1"]}}
REG_IO = {"layout": "vector", "input": "float_input", "features": ["f1", "f2"],
          "output": {"name": "variable", "kind": "value"}}
IC_IO = {"layout": "in_context",
         "inputs": {"x_train": "x_train", "y_train": "y_train",
                    "x_query": "x_query"},
         "features": ["f1", "f2"], "target": "label",
         "output": {"name": "probabilities", "kind": "probs",
                    "labels": ["0", "1"]}}


def _load_incontext(db, io=IC_IO, model_id="knn1"):
    _register(db, model_id, "knn_incontext.onnx", io)
    data = json.load(open(_abs("knn_incontext_cases.json")))
    db.execute("CREATE TABLE tr(id INTEGER, f1 REAL, f2 REAL, label TEXT)")
    db.executemany("INSERT INTO tr VALUES (?,?,?,?)",
                   [(c["id"], c["f1"], c["f2"], c["label"])
                    for c in data["train"]])
    db.execute("CREATE TABLE ap(id INTEGER, f1 REAL, f2 REAL)")
    db.executemany("INSERT INTO ap VALUES (?,?,?)",
                   [(c["id"], c["f1"], c["f2"]) for c in data["apply"]])
    return data


def _load_apply(db, cases):
    db.execute("CREATE TABLE apply(id INTEGER, f1 REAL, f2 REAL)")
    db.executemany("INSERT INTO apply VALUES (?,?,?)",
                   [(c["id"], c["f1"], c["f2"]) for c in cases])


def test_version_advertises_onnx(db):
    v = json.loads(db.execute("SELECT predict_version()").fetchone()[0])
    assert "onnx" in v["runtimes"]


def test_classifier_matches_reference(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    cases = json.load(open(_abs("logreg_cases.json")))
    _load_apply(db, cases)
    rows = db.execute(
        "SELECT * FROM predict(NULL, 'SELECT id, f1, f2 FROM apply',"
        " json_object('model','clf'))").fetchall()
    assert len(rows) == len(cases)
    exp = {c["id"]: c for c in cases}
    for rid, pred, conf, status, receipt, *_ in rows:
        e = exp[rid]
        assert status == "ok"
        assert pred == e["label"]
        p_pred = e["p1"] if pred == "1" else 1 - e["p1"]
        assert abs(conf - p_pred) < 1e-4
        assert receipt  # a receipt id came back


def test_regressor_matches_reference(db):
    _register(db, "reg", "linreg.onnx", REG_IO)
    cases = json.load(open(_abs("linreg_cases.json")))
    _load_apply(db, cases)
    rows = db.execute(
        "SELECT row_ref, prediction, confidence, status FROM predict("
        "NULL, 'SELECT id, f1, f2 FROM apply', json_object('model','reg'))"
    ).fetchall()
    exp = {c["id"]: c for c in cases}
    for rid, pred, conf, status in rows:
        assert status == "ok"
        assert conf is None  # regression carries no confidence
        assert abs(float(pred) - exp[rid]["value"]) < 1e-4


def test_incontext_matches_reference(db):
    """The teacher path: training rows become model context. Predictions
    must match the pure-Python 1-NN reference the fixture was built from."""
    data = _load_incontext(db)
    rows = db.execute(
        "SELECT * FROM predict('SELECT f1, f2, label FROM tr',"
        " 'SELECT id, f1, f2 FROM ap', json_object('model','knn1'))"
    ).fetchall()
    assert len(rows) == len(data["apply"])
    exp = {c["id"]: c["label"] for c in data["apply"]}
    for rid, pred, conf, status, receipt, *_ in rows:
        assert status == "ok"
        assert pred == exp[rid]
        assert abs(conf - 1.0) < 1e-6  # one-hot output
        assert receipt


def test_incontext_replays_exactly(db):
    _load_incontext(db)
    rid = db.execute(
        "SELECT receipt_id FROM predict('SELECT f1, f2, label FROM tr',"
        " 'SELECT id, f1, f2 FROM ap', json_object('model','knn1')) LIMIT 1"
    ).fetchone()[0]
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)", (rid,)).fetchone()
    assert match == 1, detail


def test_incontext_replay_detects_train_mutation(db):
    """Because the training rows are the context, changing them must break
    the receipt anchor."""
    _load_incontext(db)
    rid = db.execute(
        "SELECT receipt_id FROM predict('SELECT f1, f2, label FROM tr',"
        " 'SELECT id, f1, f2 FROM ap', json_object('model','knn1')) LIMIT 1"
    ).fetchone()[0]
    db.execute("UPDATE tr SET label = '1' WHERE label = '0'")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT match FROM predict_replay(?)", (rid,)).fetchone()
    assert "PREDICT_ERR_ANCHOR_UNAVAILABLE" in str(e.value)


def test_incontext_requires_train_query(db):
    _load_incontext(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict(NULL, 'SELECT id, f1, f2 FROM ap',"
            " json_object('model','knn1'))").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_incontext_train_missing_target_errors(db):
    _load_incontext(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict('SELECT f1, f2 FROM tr',"
            " 'SELECT id, f1, f2 FROM ap', json_object('model','knn1'))"
        ).fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_incontext_unknown_train_label_errors(db):
    _load_incontext(db)
    db.execute("UPDATE tr SET label = 'purple' WHERE id = 0")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict('SELECT f1, f2, label FROM tr',"
            " 'SELECT id, f1, f2 FROM ap', json_object('model','knn1'))"
        ).fetchall()
    assert "PREDICT_ERR_IO_SPEC" in str(e.value)


def test_incontext_nonnumeric_train_feature_errors(db):
    _register(db, "knn1", "knn_incontext.onnx", IC_IO)
    db.execute("CREATE TABLE tr(id INTEGER, f1 ANY, f2 REAL, label TEXT)")
    db.execute("INSERT INTO tr VALUES (0, 'x', 1.0, '0'), (1, 2.0, 2.0, '1')")
    db.execute("CREATE TABLE ap(id INTEGER, f1 REAL, f2 REAL)")
    db.execute("INSERT INTO ap VALUES (0, 0.5, 0.5)")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict('SELECT f1, f2, label FROM tr',"
            " 'SELECT id, f1, f2 FROM ap', json_object('model','knn1'))"
        ).fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_incontext_nonnumeric_query_row_flagged(db):
    _register(db, "knn1", "knn_incontext.onnx", IC_IO)
    db.execute("CREATE TABLE tr(id INTEGER, f1 REAL, f2 REAL, label TEXT)")
    db.execute("INSERT INTO tr VALUES (0,-1,-1,'0'),(1,1,1,'1'),(2,2,2,'1')")
    db.execute("CREATE TABLE ap(id INTEGER, f1 REAL, f2 ANY)")
    db.execute("INSERT INTO ap VALUES (0, 1.0, 1.0), (1, 0.0, 'oops')")
    rows = {r[0]: r for r in db.execute(
        "SELECT row_ref, prediction, status FROM predict("
        " 'SELECT f1, f2, label FROM tr', 'SELECT id, f1, f2 FROM ap',"
        " json_object('model','knn1','receipt',0))").fetchall()}
    assert rows[0][2] == "ok"
    assert rows[1][2] == "non_numeric" and rows[1][1] is None


def test_incontext_batch_boundary(db):
    """Many query rows against a fixed context: cross the batch boundary and
    confirm order and correctness (1-NN over the fixture's training rows)."""
    data = _load_incontext(db)
    train = [(c["f1"], c["f2"], int(c["label"])) for c in data["train"]]
    db.execute("CREATE TABLE bigq(id INTEGER, f1 REAL, f2 REAL)")
    n = 2100
    pts = [(i, (i % 9) - 4.0, ((i * 3) % 7) - 3.0) for i in range(n)]
    db.executemany("INSERT INTO bigq VALUES (?,?,?)", pts)
    rows = db.execute(
        "SELECT row_ref, prediction FROM predict("
        " 'SELECT f1, f2, label FROM tr', 'SELECT id, f1, f2 FROM bigq"
        " ORDER BY id', json_object('model','knn1','receipt',0))").fetchall()
    assert len(rows) == n
    assert [r[0] for r in rows] == list(range(n))

    def ref(qx, qy):
        best, bl = 1e30, "0"
        for tx, ty, tl in train:
            d = (qx - tx) ** 2 + (qy - ty) ** 2
            if d < best:
                best, bl = d, str(tl)
        return bl
    mism = sum(1 for rid, pred in rows if pred != ref(pts[rid][1], pts[rid][2]))
    assert mism == 0


def test_receipt_replays_exactly(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    rid = db.execute(
        "SELECT receipt_id FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
        " json_object('model','clf')) LIMIT 1").fetchone()[0]
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)", (rid,)).fetchone()
    assert match == 1, detail


def test_replay_detects_mutation(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    rid = db.execute(
        "SELECT receipt_id FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
        " json_object('model','clf')) LIMIT 1").fetchone()[0]
    db.execute("UPDATE apply SET f1 = f1 + 100 WHERE id = 0")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT match FROM predict_replay(?)", (rid,)).fetchone()
    assert "PREDICT_ERR_ANCHOR_UNAVAILABLE" in str(e.value)


def test_batch_boundary_many_rows(db):
    """Cross the 1024-row batch boundary to exercise multi-batch inference
    and confirm every row still comes back in order."""
    _register(db, "clf", "logreg.onnx", CLF_IO)
    db.execute("CREATE TABLE big(id INTEGER, f1 REAL, f2 REAL)")
    n = 2500
    db.executemany(
        "INSERT INTO big VALUES (?,?,?)",
        [(i, (i % 7) - 3.0, (i % 5) - 2.0) for i in range(n)])
    rows = db.execute(
        "SELECT row_ref, prediction FROM predict(NULL,"
        " 'SELECT id, f1, f2 FROM big ORDER BY id',"
        " json_object('model','clf','receipt',0))").fetchall()
    assert len(rows) == n
    assert [r[0] for r in rows] == list(range(n))  # order preserved
    # every prediction agrees with the boundary rule (argmax of softmax)
    for rid, pred in rows:
        f1, f2 = (rid % 7) - 3.0, (rid % 5) - 2.0
        assert pred == ("1" if (f1 + f2) > 0 else "0")


def test_non_numeric_feature_is_flagged_not_fed(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    db.execute("CREATE TABLE mixed(id INTEGER, f1 REAL, f2 ANY)")
    db.execute("INSERT INTO mixed VALUES (1, 0.5, 0.5), (2, 0.5, 'oops')")
    rows = {r[0]: r for r in db.execute(
        "SELECT row_ref, prediction, status FROM predict(NULL,"
        " 'SELECT id, f1, f2 FROM mixed', json_object('model','clf',"
        "'receipt',0))").fetchall()}
    assert rows[1][2] == "ok"
    assert rows[2][2] == "non_numeric"
    assert rows[2][1] is None


def test_license_gate_blocks_then_accepts(db):
    _register(db, "nc", "logreg.onnx", CLF_IO, license="CC-BY-NC-4.0")
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
            " json_object('model','nc'))").fetchall()
    assert "PREDICT_ERR_LICENSE" in str(e.value)
    # naming the license unlocks it
    rows = db.execute(
        "SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
        " json_object('model','nc','accept_license','CC-BY-NC-4.0'))"
    ).fetchall()
    assert len(rows) == 24


def test_unknown_device_and_gpu_fail_loud(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    # a bogus device name
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
                   " json_object('model','clf','device','banana'))").fetchall()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)
    # cuda is real but not in this CPU build: no silent fallback
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
                   " json_object('model','clf','device','cuda'))").fetchall()
    assert "PREDICT_ERR_RUNTIME_UNAVAILABLE" in str(e.value)


def test_unsupported_precision_fails_loud(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM apply',"
                   " json_object('model','clf','precision','fp16'))").fetchall()
    assert "PREDICT_ERR_RUNTIME_UNAVAILABLE" in str(e.value)


def test_feature_mismatch_errors(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    db.execute("CREATE TABLE w(id INTEGER, f1 REAL, f9 REAL)")
    db.execute("INSERT INTO w VALUES (1, 0.1, 0.2)")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f9 FROM w',"
                   " json_object('model','clf'))").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_wrong_feature_count_errors(db):
    _register(db, "clf", "logreg.onnx", CLF_IO)
    db.execute("CREATE TABLE w(id INTEGER, f1 REAL)")
    db.execute("INSERT INTO w VALUES (1, 0.1)")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict(NULL,'SELECT id, f1 FROM w',"
                   " json_object('model','clf'))").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_unknown_model_errors(db):
    db.execute("CREATE TABLE a(id INTEGER, f1 REAL, f2 REAL)")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM a',"
                   " json_object('model','ghost'))").fetchall()
    assert "PREDICT_ERR_MODEL_NOT_FOUND" in str(e.value)


def test_onnx_options_rejected_on_stat_model(db):
    """device/precision/accept_license are meaningless for the built-in
    stat model and must error, not be silently ignored."""
    import synthetic_tabular as syt
    X, y, _ = syt.two_moons(n=120)
    syt.load_tabular(db, X, y)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT * FROM predict('SELECT f1, f2, label FROM tab WHERE id<100',"
            " 'SELECT id, f1, f2 FROM tab WHERE id>=100',"
            " json_object('target','label','device','cpu'))").fetchall()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_no_memory_growth_over_many_calls(db):
    """Per-call leak soak for the onnx path: the session is cached once, so
    every later call must free its rows/batch/io_spec/receipt allocations.
    RSS growth over hundreds of calls must stay bounded."""
    import resource
    _register(db, "clf", "logreg.onnx", CLF_IO)
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    q = ("SELECT * FROM predict(NULL, 'SELECT id, f1, f2 FROM apply',"
         " json_object('model','clf'))")

    def batch(n):
        for _ in range(n):
            db.execute(q).fetchall()

    batch(50)  # warmup: session build, allocator pools
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    batch(400)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (after - before) / (1024 * 1024)
    assert growth_mb < 20, f"RSS grew {growth_mb:.1f}MB over 400 calls"


def test_session_cache_reuse_is_fast(db):
    """Second call reuses the cached session; both return identical
    predictions (a coarse check that caching doesn't corrupt state)."""
    _register(db, "clf", "logreg.onnx", CLF_IO)
    _load_apply(db, json.load(open(_abs("logreg_cases.json"))))
    q = ("SELECT row_ref, prediction FROM predict(NULL,"
         " 'SELECT id, f1, f2 FROM apply', json_object('model','clf',"
         "'receipt',0))")
    first = db.execute(q).fetchall()
    second = db.execute(q).fetchall()
    assert first == second
