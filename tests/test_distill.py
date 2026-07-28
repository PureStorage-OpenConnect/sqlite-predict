"""distill_predict() and the native tree student (RFC 0005 §4.2.4).

distill runs a teacher over the training rows, fits a CART tree on the
teacher's predictions, evaluates on a holdout, and registers the tree as an
inline-BLOB student that runs in the zero-dependency core.
"""

import sqlite3

import pytest
import synthetic_tabular as syt


def _train_table(db, n=240, seed=3):
    X, y, _ = syt.two_moons(n=n, seed=seed)
    syt.load_tabular(db, X, y)  # table tab(id, f1, f2, label)


def test_distill_produces_a_student(db):
    _train_table(db)
    r = db.execute(
        "SELECT model_id, content_hash, train_rows, holdout_metric"
        " FROM distill_predict('SELECT f1, f2, label FROM tab',"
        " json_object('target','label','student_id','s1'))").fetchone()
    model_id, chash, rows, metric = r
    assert model_id == "s1"
    assert len(chash) == 64
    assert rows == 240
    assert 0.7 <= metric <= 1.0  # holdout accuracy on a learnable boundary

    row = db.execute("SELECT kind, runtime, length(weights), content_hash"
                     " FROM _predict_models WHERE model_id='s1'").fetchone()
    assert row[0] == "student" and row[1] == "tree"
    assert row[2] > 0 and row[3] == chash


def test_student_predicts(db):
    _train_table(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
               " json_object('target','label','student_id','s1'))").fetchone()
    # the student runs with no train_query
    rows = db.execute(
        "SELECT row_ref, prediction, confidence, status FROM predict(NULL,"
        " 'SELECT id, f1, f2 FROM tab', json_object('model','s1'))").fetchall()
    assert len(rows) == 240
    labels = {str(r[0]) for r in db.execute("SELECT DISTINCT label FROM tab")}
    for _rid, pred, conf, status in rows:
        assert status == "ok"
        assert pred in labels  # student round-trips the teacher's labels
        assert 0.0 <= conf <= 1.0


def test_student_is_deterministic(db):
    _train_table(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1, f2, label FROM tab',"
               " json_object('target','label','student_id','s1'))").fetchone()
    q = ("SELECT row_ref, prediction FROM predict(NULL,"
         " 'SELECT id, f1, f2 FROM tab', json_object('model','s1'))")
    assert db.execute(q).fetchall() == db.execute(q).fetchall()


def test_distill_content_hash_is_stable(db):
    """Same data + teacher -> same student bytes -> same content hash."""
    _train_table(db)
    h1 = db.execute("SELECT content_hash FROM distill_predict('SELECT f1,f2,label FROM"
                    " tab', json_object('target','label','student_id','a'))"
                    ).fetchone()[0]
    h2 = db.execute("SELECT content_hash FROM distill_predict('SELECT f1,f2,label FROM"
                    " tab', json_object('target','label','student_id','b'))"
                    ).fetchone()[0]
    assert h1 == h2


def test_student_exists_is_not_overwritten(db):
    _train_table(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
               " json_object('target','label','student_id','s1'))").fetchone()
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
                   " json_object('target','label','student_id','s1'))"
                   ).fetchone()
    assert "PREDICT_ERR_STUDENT_EXISTS" in str(e.value)


def test_regress_distill(db):
    db.execute("CREATE TABLE r(id INTEGER, f1 REAL, f2 REAL, y REAL)")
    db.executemany(
        "INSERT INTO r VALUES (?,?,?,?)",
        [(i, i * 0.1, (i * 7) % 5, i * 0.1 + ((i * 7) % 5) * 0.5)
         for i in range(120)])
    r = db.execute(
        "SELECT train_rows, holdout_metric FROM distill_predict('SELECT f1, f2, y"
        " FROM r', json_object('target','y','task','regress',"
        "'student_id','rs'))").fetchone()
    assert r[0] == 120
    assert r[1] >= 0.0  # RMSE
    kind = db.execute("SELECT runtime FROM _predict_models WHERE model_id='rs'"
                      ).fetchone()[0]
    assert kind == "tree"
    out = db.execute(
        "SELECT prediction, status FROM predict(NULL,'SELECT id, f1, f2 FROM r',"
        " json_object('model','rs'))").fetchall()
    assert all(s == "ok" and float(p) == float(p) for p, s in out)


def test_missing_target_errors(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
                   " json_object('student_id','s1'))").fetchone()
    assert "PREDICT_ERR_TARGET" in str(e.value)


def test_missing_student_id_errors(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
                   " json_object('target','label'))").fetchone()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_bad_task_errors(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
                   " json_object('target','label','student_id','s',"
                   "'task','cluster'))").fetchone()
    assert "PREDICT_ERR_TASK" in str(e.value)


def test_mlp_student(db):
    """The mlp student: a smooth softmax net, trained deterministically, that
    predicts valid classes and is a tiny blob."""
    _train_table(db)
    metric = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,label FROM tab',"
        " json_object('target','label','student_kind','mlp','student_id','m'))"
    ).fetchone()[0]
    assert 0.6 <= metric <= 1.0
    row = db.execute("SELECT runtime, length(weights) FROM _predict_models WHERE"
                     " model_id='m'").fetchone()
    assert row[0] == "tree" and row[1] > 0  # inline student runtime
    labels = {str(x[0]) for x in db.execute("SELECT DISTINCT label FROM tab")}
    q = ("SELECT row_ref, prediction FROM predict(NULL,'SELECT id,f1,f2 FROM"
         " tab', json_object('model','m'))")
    rows = db.execute(q).fetchall()
    assert len(rows) == 240 and {p for _, p in rows} <= labels
    assert db.execute(q).fetchall() == rows  # deterministic


def test_mlp_soft_and_regress_guard(db):
    """The mlp student consumes soft targets (proba/classes) and is
    classification only."""
    _soft_table(db)
    metric = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,pA,pB,pC,label FROM"
        " st', json_object('target','label','proba',json_array('pA','pB','pC'),"
        "'classes',json_array('A','B','C'),'student_kind','mlp',"
        "'student_id','sm'))").fetchone()[0]
    assert 0.6 <= metric <= 1.0
    db.execute("CREATE TABLE rr(f1 REAL, f2 REAL, y REAL)")
    db.executemany("INSERT INTO rr VALUES (?,?,?)",
                   [(i * 0.1, (i * 7) % 5, i * 0.2) for i in range(120)])
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill_predict('SELECT f1,f2,y FROM rr',"
                   " json_object('target','y','task','regress',"
                   "'student_kind','mlp','student_id','x'))").fetchone()
    assert "classification only" in str(e.value)


def test_corrupt_mlp_blob_rejected(db):
    _train_table(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
               " json_object('target','label','student_kind','mlp',"
               "'student_id','ok'))").fetchone()
    db.execute("CREATE TABLE a(id INTEGER, f1 REAL, f2 REAL)")
    db.execute("INSERT INTO a VALUES (0, 0.1, 0.2)")
    for blob in (b"PSMLP01\x00" + b"\xff" * 40, b"PSMLP01\x00" + b"\x00" * 20):
        db.execute("DELETE FROM _predict_models WHERE model_id='bad'")
        db.execute(
            "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
            " content_hash, license) VALUES ('bad','student','tree',?,'x',"
            "'unspecified')", (blob,))
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM a',"
                       " json_object('model','bad'))").fetchall()
        assert "PREDICT_ERR" in str(e.value)


def test_too_few_rows_errors(db):
    db.execute("CREATE TABLE t(f1 REAL, f2 REAL, label TEXT)")
    db.execute("INSERT INTO t VALUES (0,0,'a'),(1,1,'b'),(2,2,'a')")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM t',"
                   " json_object('target','label','student_id','s'))"
                   ).fetchone()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def _angular3(db, n=400, seed=2):
    """A 3-class angular boundary a single shallow tree underfits."""
    import math
    import random
    rng = random.Random(seed)
    db.execute("CREATE TABLE tr3(f1 REAL, f2 REAL, label TEXT)")
    rows = []
    for _ in range(n):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        ang = math.atan2(b, a)
        rows.append((a, b, "0" if ang < -1 else ("1" if ang < 1 else "2")))
    db.executemany("INSERT INTO tr3 VALUES (?,?,?)", rows)


def test_gbt_student_beats_single_tree(db):
    """The point of the GBT student: an additive ensemble carries a curved
    boundary that one shallow tree cannot."""
    _angular3(db)
    tm = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,label FROM tr3',"
        " json_object('target','label','student_id','t','student_kind','tree'))"
    ).fetchone()[0]
    gm = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,label FROM tr3',"
        " json_object('target','label','student_id','g','student_kind','gbt'))"
    ).fetchone()[0]
    assert gm >= tm


def test_gbt_student_predicts(db):
    _angular3(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tr3',"
               " json_object('target','label','student_id','g',"
               "'student_kind','gbt'))").fetchone()
    runtime, blob = db.execute(
        "SELECT runtime, length(weights) FROM _predict_models WHERE"
        " model_id='g'").fetchone()
    assert runtime == "tree" and blob > 0
    db.execute("CREATE TABLE ap(id INTEGER, f1 REAL, f2 REAL)")
    db.execute("INSERT INTO ap VALUES (0,0.5,0.5),(1,-1.0,-1.0),(2,-1.0,1.0)")
    q = ("SELECT row_ref, prediction, confidence, status FROM predict(NULL,"
         " 'SELECT id,f1,f2 FROM ap', json_object('model','g'))")
    r1 = db.execute(q).fetchall()
    assert all(s == "ok" and p in ("0", "1", "2") and 0 <= c <= 1
               for _, p, c, s in r1)
    assert r1 == db.execute(q).fetchall()  # deterministic


def test_gbt_regression(db):
    db.execute("CREATE TABLE r(id INTEGER, f1 REAL, f2 REAL, y REAL)")
    db.executemany(
        "INSERT INTO r VALUES (?,?,?,?)",
        [(i, i * 0.1, (i * 7) % 5, i * 0.1 + ((i * 7) % 5) * 0.5)
         for i in range(120)])
    m = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,y FROM r',"
        " json_object('target','y','task','regress','student_id','rg',"
        "'student_kind','gbt'))").fetchone()[0]
    assert m >= 0
    out = db.execute(
        "SELECT prediction, status FROM predict(NULL,'SELECT id,f1,f2 FROM r',"
        " json_object('model','rg'))").fetchall()
    assert all(s == "ok" and float(p) == float(p) for p, s in out)


def test_corrupt_gbt_blob_rejected(db):
    _angular3(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tr3',"
               " json_object('target','label','student_id','good',"
               "'student_kind','gbt'))").fetchone()
    db.execute("CREATE TABLE a(id INTEGER, f1 REAL, f2 REAL)")
    db.execute("INSERT INTO a VALUES (0, 0.1, 0.2)")
    for blob in (b"PSGBT01\x00" + b"\xff" * 40, b"PSGBT01\x00" + b"\x00" * 20):
        db.execute("DELETE FROM _predict_models WHERE model_id='bad'")
        db.execute(
            "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
            " content_hash, license) VALUES ('bad','student','tree',?,'x',"
            "'unspecified')", (blob,))
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM a',"
                       " json_object('model','bad'))").fetchall()
        assert "PREDICT_ERR" in str(e.value)


def test_default_trains_on_target_column(db):
    """With no teacher (the default), distill trains the student directly on
    the target column -- the path for compressing a strong teacher's
    precomputed predictions (e.g. an offline TabFM run) into a native student,
    with no live teacher pass."""
    _train_table(db)
    metric = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1, f2, label FROM tab',"
        " json_object('target','label','student_id','s',"
        "'student_kind','gbt'))").fetchone()[0]
    assert 0.7 <= metric <= 1.0  # learnable boundary, trained on the labels
    rows = db.execute(
        "SELECT row_ref, prediction FROM predict(NULL,'SELECT id,f1,f2 FROM"
        " tab', json_object('model','s'))").fetchall()
    assert len(rows) == 240


def test_default_learns_target_not_a_live_teacher(db):
    """The distinguishing property of the default: the student learns the
    target column verbatim. Feed labels a knn5 teacher would never invent and
    confirm the student emits exactly that vocabulary."""
    db.execute("CREATE TABLE p(id INTEGER, f1 REAL, f2 REAL, teach TEXT)")
    db.executemany("INSERT INTO p VALUES (?,?,?,?)",
                   [(i, (i % 20) * 0.1, (i // 20) * 0.1,
                     "ALPHA" if (i % 20) * 0.1 < 1.0 else "OMEGA")
                    for i in range(200)])
    db.execute(
        "SELECT * FROM distill_predict('SELECT f1,f2,teach FROM p',"
        " json_object('target','teach','student_id','s',"
        "'student_kind','gbt'))").fetchone()
    preds = {r[0] for r in db.execute(
        "SELECT DISTINCT prediction FROM predict(NULL,'SELECT id,f1,f2 FROM p',"
        " json_object('model','s'))")}
    assert preds and preds <= {"ALPHA", "OMEGA"}


def test_named_teacher_relabels_via_model(db):
    """An explicit teacher names a registered predict() model, which distill
    re-runs over the rows to relabel them before fitting the student -- e.g.
    compressing the in-context knn5 into a standalone tree."""
    _train_table(db)
    metric = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,label"
        " FROM tab', json_object('target','label','teacher','knn5-incontext',"
        "'student_id','s'))").fetchone()[0]
    assert 0.5 <= metric <= 1.0
    labels = {str(x[0]) for x in db.execute("SELECT DISTINCT label FROM tab")}
    preds = {x[0] for x in db.execute(
        "SELECT DISTINCT prediction FROM predict(NULL,'SELECT id,f1,f2 FROM tab',"
        " json_object('model','s'))")}
    assert preds <= labels


def _soft_table(db, n=360, seed=5):
    """features + a soft teacher's per-class probabilities (3-class angular
    boundary) + the true label. The teacher is confident but noisy."""
    import math
    import random
    rng = random.Random(seed)
    db.execute("CREATE TABLE st(id INTEGER, f1 REAL, f2 REAL, pA REAL, pB REAL,"
               " pC REAL, label TEXT)")
    rows = []
    for i in range(n):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        ang = math.atan2(b, a)
        true = "A" if ang < -1 else ("B" if ang < 1 else "C")
        p = {"A": 0.12, "B": 0.12, "C": 0.12}
        p[true] = 0.76
        for k in p:  # noise, renormalized in the extension anyway
            p[k] = max(0.01, p[k] + rng.uniform(-0.08, 0.08))
        rows.append((i, a, b, p["A"], p["B"], p["C"], true))
    db.executemany("INSERT INTO st VALUES (?,?,?,?,?,?,?)", rows)


def test_soft_distillation_produces(db):
    """Soft-label distillation: the student learns the teacher's per-class
    probability columns (proba) instead of a hard label, and is scored on the
    holdout against the true labels (target)."""
    _soft_table(db)
    metric = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,pA,pB,"
        "pC,label FROM st', json_object('target','label','proba',"
        "json_array('pA','pB','pC'),'classes',json_array('A','B','C'),"
        "'student_id','s'))").fetchone()[0]
    assert metric >= 0.7  # recovers the boundary the confident teacher describes
    # it is a gbt forest (tree runtime) and predicts valid classes
    row = db.execute("SELECT runtime, length(weights) FROM _predict_models"
                     " WHERE model_id='s'").fetchone()
    assert row[0] == "tree" and row[1] > 0
    out = db.execute(
        "SELECT row_ref, prediction FROM predict(NULL,'SELECT id,f1,f2 FROM st',"
        " json_object('model','s'))").fetchall()
    assert len(out) == 360 and {p for _, p in out} <= {"A", "B", "C"}


def test_soft_distillation_sharper_than_hard(db):
    """Distilling the teacher's probabilities should recover the boundary at
    least as well as distilling its hard argmax, on a noisy confident teacher."""
    _soft_table(db)
    soft = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,pA,pB,pC,label FROM"
        " st', json_object('target','label','proba',json_array('pA','pB','pC'),"
        "'classes',json_array('A','B','C'),'student_id','soft'))").fetchone()[0]
    # hard: distill the argmax of the same teacher (the label column already is
    # the true class here, so train on it directly as the hard baseline)
    hard = db.execute(
        "SELECT holdout_metric FROM distill_predict('SELECT f1,f2,label FROM st',"
        " json_object('target','label','student_kind','gbt',"
        "'student_id','hard'))").fetchone()[0]
    assert soft >= hard - 0.05  # soft is competitive (usually better)


def test_soft_distillation_errors(db):
    _soft_table(db)
    base = "SELECT * FROM distill_predict('SELECT f1,f2,pA,pB,pC,label FROM st',"
    # proba requires classes
    for opts, msg in [
        ("json_object('target','label','proba',json_array('pA','pB','pC'),"
         "'student_id','x')", "PREDICT_ERR"),
        # proba + tree student is rejected
        ("json_object('target','label','proba',json_array('pA','pB','pC'),"
         "'classes',json_array('A','B','C'),'student_kind','tree',"
         "'student_id','x')", "PREDICT_ERR"),
        # a proba column not in the query
        ("json_object('target','label','proba',json_array('pA','nope'),"
         "'classes',json_array('A','B'),'student_id','x')", "PREDICT_ERR"),
        # length mismatch
        ("json_object('target','label','proba',json_array('pA','pB','pC'),"
         "'classes',json_array('A','B'),'student_id','x')", "PREDICT_ERR"),
    ]:
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute(base + " " + opts + ")").fetchone()
        assert msg in str(e.value)


def test_corrupt_student_blob_errors_not_crashes(db):
    """The registry is caller-writable, so a hand-crafted tree blob must be
    rejected by the bounds-checked deserializer, never crash."""
    db.execute("CREATE TABLE a(id INTEGER, f1 REAL, f2 REAL)")
    db.execute("INSERT INTO a VALUES (0, 0.1, 0.2)")
    # a real distill first, so _predict_models exists to write the bad row into
    _train_table(db)
    db.execute("SELECT * FROM distill_predict('SELECT f1,f2,label FROM tab',"
               " json_object('target','label','student_id','good'))").fetchone()
    for blob in (b"", b"garbage", b"PSTREE01" + b"\xff" * 40,
                 b"PSTREE01" + b"\x00" * 4):
        db.execute("DELETE FROM _predict_models WHERE model_id='bad'")
        db.execute(
            "INSERT INTO _predict_models (model_id, kind, runtime, weights,"
            " content_hash, license) VALUES ('bad','student','tree',?,"
            " 'x','unspecified')", (blob,))
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute("SELECT * FROM predict(NULL,'SELECT id, f1, f2 FROM a',"
                       " json_object('model','bad'))").fetchall()
        assert "PREDICT_ERR" in str(e.value)


def test_student_rejects_train_query(db):
    # a student carries its training in its blob: a train_query alongside
    # it would be dead weight, so it is rejected, never silently ignored
    db.execute("CREATE TABLE tt(f1 REAL, f2 REAL, label TEXT)")
    db.executemany("INSERT INTO tt VALUES (?,?,?)",
                   [(i * 0.3, (i * 7) % 5, "c%d" % (i % 2)) for i in range(60)])
    db.execute("SELECT * FROM distill_predict('SELECT f1, f2, label FROM tt',"
               " '{\"target\":\"label\",\"student_id\":\"guard-s\"}')").fetchone()
    with pytest.raises(sqlite3.OperationalError, match="no train_query"):
        db.execute("SELECT * FROM predict('SELECT f1, f2, label FROM tt',"
                   " 'SELECT rowid, f1, f2 FROM tt',"
                   " '{\"model\":\"guard-s\"}')").fetchall()
