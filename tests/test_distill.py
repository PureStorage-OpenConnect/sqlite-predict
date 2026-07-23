"""distill() and the native tree student (RFC 0005 §4.2.5).

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
        "SELECT model_id, content_hash, train_rows, holdout_metric, receipt_id"
        " FROM distill('SELECT f1, f2, label FROM tab',"
        " json_object('target','label','student_id','s1'))").fetchone()
    model_id, chash, rows, metric, receipt = r
    assert model_id == "s1"
    assert len(chash) == 64
    assert rows == 240
    assert 0.7 <= metric <= 1.0  # holdout accuracy on a learnable boundary
    assert receipt

    row = db.execute("SELECT kind, runtime, length(weights), content_hash"
                     " FROM _predict_models WHERE model_id='s1'").fetchone()
    assert row[0] == "student" and row[1] == "tree"
    assert row[2] > 0 and row[3] == chash


def test_student_predicts_and_replays(db):
    _train_table(db)
    db.execute("SELECT * FROM distill('SELECT f1, f2, label FROM tab',"
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
    # deterministic -> exact replay
    rid = db.execute(
        "SELECT receipt_id FROM predict(NULL,'SELECT id, f1, f2 FROM tab',"
        " json_object('model','s1')) LIMIT 1").fetchone()[0]
    match, detail = db.execute(
        "SELECT match, detail FROM predict_replay(?)", (rid,)).fetchone()
    assert match == 1, detail


def test_student_is_deterministic(db):
    _train_table(db)
    db.execute("SELECT * FROM distill('SELECT f1, f2, label FROM tab',"
               " json_object('target','label','student_id','s1'))").fetchone()
    q = ("SELECT row_ref, prediction FROM predict(NULL,"
         " 'SELECT id, f1, f2 FROM tab', json_object('model','s1','receipt',0))")
    assert db.execute(q).fetchall() == db.execute(q).fetchall()


def test_distill_content_hash_is_stable(db):
    """Same data + teacher -> same student bytes -> same content hash."""
    _train_table(db)
    h1 = db.execute("SELECT content_hash FROM distill('SELECT f1,f2,label FROM"
                    " tab', json_object('target','label','student_id','a'))"
                    ).fetchone()[0]
    h2 = db.execute("SELECT content_hash FROM distill('SELECT f1,f2,label FROM"
                    " tab', json_object('target','label','student_id','b'))"
                    ).fetchone()[0]
    assert h1 == h2


def test_student_exists_is_not_overwritten(db):
    _train_table(db)
    db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
               " json_object('target','label','student_id','s1'))").fetchone()
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
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
        "SELECT train_rows, holdout_metric FROM distill('SELECT f1, f2, y"
        " FROM r', json_object('target','y','task','regress',"
        "'student_id','rs'))").fetchone()
    assert r[0] == 120
    assert r[1] >= 0.0  # RMSE
    kind = db.execute("SELECT runtime FROM _predict_models WHERE model_id='rs'"
                      ).fetchone()[0]
    assert kind == "tree"
    out = db.execute(
        "SELECT prediction, status FROM predict(NULL,'SELECT id, f1, f2 FROM r',"
        " json_object('model','rs','receipt',0))").fetchall()
    assert all(s == "ok" and float(p) == float(p) for p, s in out)


def test_missing_target_errors(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
                   " json_object('student_id','s1'))").fetchone()
    assert "PREDICT_ERR_TARGET" in str(e.value)


def test_missing_student_id_errors(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
                   " json_object('target','label'))").fetchone()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_bad_task_errors(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
                   " json_object('target','label','student_id','s',"
                   "'task','cluster'))").fetchone()
    assert "PREDICT_ERR_TASK" in str(e.value)


def test_mlp_student_kind_not_available(db):
    _train_table(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
                   " json_object('target','label','student_id','s',"
                   "'student_kind','mlp'))").fetchone()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_too_few_rows_errors(db):
    db.execute("CREATE TABLE t(f1 REAL, f2 REAL, label TEXT)")
    db.execute("INSERT INTO t VALUES (0,0,'a'),(1,1,'b'),(2,2,'a')")
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM t',"
                   " json_object('target','label','student_id','s'))"
                   ).fetchone()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_corrupt_student_blob_errors_not_crashes(db):
    """The registry is caller-writable, so a hand-crafted tree blob must be
    rejected by the bounds-checked deserializer, never crash."""
    db.execute("CREATE TABLE a(id INTEGER, f1 REAL, f2 REAL)")
    db.execute("INSERT INTO a VALUES (0, 0.1, 0.2)")
    # a real distill first, so _predict_models exists to write the bad row into
    _train_table(db)
    db.execute("SELECT * FROM distill('SELECT f1,f2,label FROM tab',"
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
                       " json_object('model','bad','receipt',0))").fetchall()
        assert "PREDICT_ERR" in str(e.value)


def test_distill_receipt_records_lineage(db):
    """A distill receipt exists and anchors the training data; predict_replay
    of a distill receipt is not supported (lineage only) and says so."""
    _train_table(db)
    rid = db.execute(
        "SELECT receipt_id FROM distill('SELECT f1,f2,label FROM tab',"
        " json_object('target','label','student_id','s1'))").fetchone()[0]
    n = db.execute("SELECT count(*) FROM _predict_receipts WHERE"
                   " receipt_id=? AND operation='distill'", (rid,)).fetchone()[0]
    assert n == 1
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT * FROM predict_replay(?)", (rid,)).fetchall()
    assert "PREDICT_ERR_ANCHOR_UNAVAILABLE" in str(e.value)
