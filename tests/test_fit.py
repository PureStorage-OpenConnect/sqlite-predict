"""fit() aggregate + predict() scalar: the scikit-learn shape.

The batched / in-context TVF path is predict_batch() and is covered in
test_predict.py. Here we cover the NEW formats: training with the fit()
aggregate (register or blob) and serving with the predict() scalar.
"""
import json
import sqlite3

import pytest


def _seed(db, n=120):
    db.execute("CREATE TABLE h(region TEXT, tenure REAL, spend REAL,"
               " churned INTEGER)")
    db.executemany(
        "INSERT INTO h VALUES (?, ?, ?, ?)",
        [("us" if k % 2 else "eu", (k % 20) + 1, ((k % 20) + 1) * 10 + (k % 3),
          1 if (k % 20) + 1 < 8 else 0) for k in range(n)])
    db.execute("CREATE TABLE a(id INTEGER, tenure REAL, spend REAL)")
    db.executemany("INSERT INTO a VALUES (?, ?, ?)",
                   [(1, 2, 20), (2, 5, 55), (3, 15, 150), (4, 25, 250)])


def test_fit_register_then_scalar_predict(db):
    """fit() over the rows registers a model; predict() serves it per row,
    composing like any scalar. Low tenure churns, high does not."""
    _seed(db)
    mid = db.execute(
        "SELECT fit(tenure, spend, churned,"
        " '{\"kind\":\"gbt\",\"register\":\"churn-v1\"}') FROM h").fetchone()[0]
    assert mid == "churn-v1"
    preds = [r[1] for r in db.execute(
        "SELECT id, predict('churn-v1', tenure, spend) FROM a ORDER BY id")]
    assert preds == ["1", "1", "0", "0"]


def test_fit_leading_name_rhymes_with_predict(db):
    """The guessable mirror: a leading TEXT model name registers the model, so
    fit('id', f..., label) reads parallel to predict('id', f...). Same churn
    signal, no options object needed."""
    _seed(db)
    mid = db.execute(
        "SELECT fit('churn-lead', tenure, spend, churned) FROM h").fetchone()[0]
    assert mid == "churn-lead"
    preds = [r[1] for r in db.execute(
        "SELECT id, predict('churn-lead', tenure, spend) FROM a ORDER BY id")]
    assert preds == ["1", "1", "0", "0"]


def test_fit_leading_name_composes_with_options(db):
    """A leading name and a trailing options object coexist: the name registers,
    the options still select the student kind."""
    _seed(db)
    mid = db.execute(
        "SELECT fit('lead-tree', tenure, spend, churned, '{\"kind\":\"tree\"}')"
        " FROM h").fetchone()[0]
    assert mid == "lead-tree"
    assert db.execute("SELECT predict('lead-tree', 2, 20)").fetchone()[0] == "1"


def test_fit_leading_name_equals_options_register(db):
    """The two ways to name a model are equivalent: a model registered by a
    leading name predicts identically to one registered via {"register":...} on
    the same rows (same features, same training, deterministic)."""
    _seed(db)
    db.execute("SELECT fit('lead-eq', tenure, spend, churned) FROM h").fetchone()
    db.execute("SELECT fit(tenure, spend, churned, '{\"register\":\"opt-eq\"}')"
               " FROM h").fetchone()
    lead = [r[0] for r in db.execute(
        "SELECT predict('lead-eq', tenure, spend) FROM a ORDER BY id")]
    opt = [r[0] for r in db.execute(
        "SELECT predict('opt-eq', tenure, spend) FROM a ORDER BY id")]
    assert lead == opt


def test_fit_model_name_given_twice_fails_loud(db):
    """A leading name and a {"register":...} option name the model twice. That is
    a mistake, not a precedence to resolve silently: fail loudly
    (PREDICT_ERR_OPTIONS)."""
    _seed(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit('twice-a', tenure, spend, churned,"
                   " '{\"register\":\"twice-b\"}') FROM h").fetchall()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_fit_leading_name_still_needs_features_and_label(db):
    """A leading name does not substitute for data: fit('id', label) has a name
    and a label but no features, and must fail loudly (PREDICT_ERR_SCHEMA)."""
    _seed(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit('nofeat', churned) FROM h").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_fit_blob_served_via_cte(db):
    """No registration: fit() returns a model blob, served in one statement."""
    _seed(db)
    preds = [r[1] for r in db.execute(
        "WITH m(model) AS (SELECT fit(tenure, spend, churned) FROM h)"
        " SELECT a.id, predict((SELECT model FROM m), a.tenure, a.spend)"
        " FROM a ORDER BY a.id")]
    assert preds == ["1", "1", "0", "0"]


def test_fit_per_segment_via_group_by(db):
    """A model per segment falls out of GROUP BY; each is a blob."""
    _seed(db)
    out = db.execute("SELECT region, typeof(fit(tenure, spend, churned))"
                     " FROM h GROUP BY region").fetchall()
    assert {r[0] for r in out} == {"us", "eu"}
    assert all(r[1] == "blob" for r in out)


def test_fit_tree_kind(db):
    _seed(db)
    mid = db.execute(
        "SELECT fit(tenure, spend, churned,"
        " '{\"kind\":\"tree\",\"register\":\"t\"}') FROM h").fetchone()[0]
    assert mid == "t"
    assert db.execute("SELECT predict('t', 2, 20)").fetchone()[0] == "1"


def test_predict_proba_returns_confidence(db):
    _seed(db)
    db.execute("SELECT fit(tenure, spend, churned, '{\"register\":\"p\"}')"
               " FROM h").fetchone()
    doc = json.loads(db.execute(
        "SELECT predict('p', tenure, spend, '{\"proba\":true}')"
        " FROM a LIMIT 1").fetchone()[0])
    assert set(doc) >= {"prediction", "confidence"}
    assert 0.0 <= doc["confidence"] <= 1.0


def test_predict_proba_rejects_wrong_type(db):
    """A non-numeric proba is a typo, not false: it must fail loudly
    (PREDICT_ERR_OPTIONS), not be silently coerced to 0 and dropped."""
    _seed(db)
    db.execute("SELECT fit(tenure, spend, churned, '{\"register\":\"pt\"}')"
               " FROM h").fetchone()
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT predict('pt', tenure, spend, '{\"proba\":\"yes\"}')"
                   " FROM a").fetchall()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_predict_wrong_arity_fails_loud(db):
    """Positional features must match the model's feature count."""
    _seed(db)
    db.execute("SELECT fit(tenure, spend, churned, '{\"register\":\"q\"}')"
               " FROM h").fetchone()
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT predict('q', tenure) FROM a").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_predict_unknown_model_fails_loud(db):
    _seed(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT predict('nope', tenure, spend) FROM a").fetchall()
    assert "PREDICT_ERR_MODEL_NOT_FOUND" in str(e.value)


def test_fit_needs_features_and_a_label(db):
    """fit() with a single argument has no features to learn from."""
    _seed(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit(tenure) FROM h").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_fit_rejects_non_finite_feature(db):
    """A feature value of +/-Inf (e.g. 1e999) or NaN arrives as SQLITE_FLOAT and
    passes the numeric-type check; it must still fail loudly (PREDICT_ERR_SCHEMA)
    rather than poison the model with a non-finite value."""
    db.execute("CREATE TABLE nf(f1 REAL, f2 REAL, c INTEGER)")
    db.executemany(
        "INSERT INTO nf VALUES (?, ?, ?)",
        [((float("inf") if i == 5 else i * 1.0), (i % 3) * 1.0, i % 2)
         for i in range(20)])
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit(f1, f2, c) FROM nf").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_predict_rejects_non_finite_feature(db):
    """Symmetry with fit(): a non-finite feature at serve time fails loudly
    (PREDICT_ERR_SCHEMA), not scored on a value that could never be trained on."""
    _seed(db)
    db.execute("SELECT fit(tenure, spend, churned, '{\"register\":\"pf\"}')"
               " FROM h").fetchone()
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT predict('pf', ?, ?)", (float("inf"), 20.0)).fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_fit_json_label_with_unknown_keys_fails_loud(db):
    """The trailing '{...}' argument is always the options object (a class label
    must not be a JSON-object string). A JSON label with unknown keys therefore
    fails loudly as a bad option (PREDICT_ERR_OPTIONS), not a silent mis-split."""
    db.execute("CREATE TABLE jl(f1 REAL, f2 REAL, lbl TEXT)")
    db.executemany("INSERT INTO jl VALUES (?, ?, ?)",
                   [(i * 1.0, (i % 3) * 1.0, '{"category":"A"}') for i in range(20)])
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit(f1, f2, lbl) FROM jl").fetchall()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_fit_rejects_null_classify_label(db):
    """A NULL class label is not a real class: fit must fail loudly
    (PREDICT_ERR_TARGET) rather than silently train an empty-string class."""
    db.execute("CREATE TABLE n(f1 REAL, f2 REAL, label TEXT)")
    db.executemany("INSERT INTO n VALUES (?, ?, ?)",
                   [(1.0, 2.0, "a"), (3.0, 4.0, None), (5.0, 6.0, "b")])
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit(f1, f2, label, '{\"task\":\"classify\"}')"
                   " FROM n").fetchall()
    assert "PREDICT_ERR_TARGET" in str(e.value)


def test_fit_first_row_config_error_surfaces_its_real_code(db):
    """A configuration error is detected on the first row, before the context is
    marked configured. fit_final must surface that real diagnostic, not mask it
    as the generic 'no training rows' (PREDICT_ERR_SCHEMA). A bad task yields
    PREDICT_ERR_TASK; an unknown option yields PREDICT_ERR_OPTIONS."""
    _seed(db)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit(tenure, spend, churned, '{\"task\":\"bogus\"}')"
                   " FROM h").fetchall()
    assert "PREDICT_ERR_TASK" in str(e.value)
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute("SELECT fit(tenure, spend, churned, '{\"nope\":1}')"
                   " FROM h").fetchall()
    assert "PREDICT_ERR_OPTIONS" in str(e.value)


def test_predict_rejects_corrupt_blob(db):
    """A fit() blob is untrusted input to predict(). Corrupting its declared
    dimensions must fail cleanly (PREDICT_ERR_SCHEMA), never crash or read out
    of bounds. The gbt header is magic[8] then task, nfeat, nclass, n_score,
    n_rounds (u32 each); we stomp each dimension word with a huge value."""
    _seed(db)
    blob = db.execute("SELECT fit(tenure, spend, churned) FROM h").fetchone()[0]
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 28
    for off in (12, 16, 20, 24):  # nfeat, nclass, n_score, n_rounds
        bad = bytearray(blob)
        bad[off:off + 4] = (0x7FFFFFFF).to_bytes(4, "little")
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute("SELECT predict(?, 2, 20)", (bytes(bad),)).fetchall()
        assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_predict_rejects_nonfinite_float_in_blob(db):
    """A crafted blob can encode NaN/Inf floats. Corrupt exactly one documented
    float field (the first tree node's threshold), leaving every framing and
    integrity field intact, so the only reason predict() can fail is rd_f32()
    rejecting the non-finite value. Tree blob layout (predict-student.c):
    magic 'PSTREE01' (8), task/nfeat/nclass/n_nodes (u32 x4), then length-prefixed
    feat_names and labels, then nodes: feature (u32), threshold (f32), ..."""
    import math
    import struct

    _seed(db)
    blob = db.execute(
        "SELECT fit(tenure, spend, churned, '{\"kind\":\"tree\"}') FROM h"
    ).fetchone()[0]
    assert blob[:8] == b"PSTREE01"
    p = 8
    _task, nfeat, nclass, n_nodes = struct.unpack_from("<IIII", blob, p)
    p += 16
    for _ in range(nfeat + nclass):  # skip feat_names then labels (u32 len + bytes)
        (ln,) = struct.unpack_from("<I", blob, p)
        p += 4 + ln
    assert n_nodes >= 1
    p += 4  # skip node[0].feature (u32); p now points at node[0].threshold (f32)
    assert math.isfinite(struct.unpack_from("<f", blob, p)[0])  # a real float field
    for bad in (float("nan"), float("inf")):
        buf = bytearray(blob)
        struct.pack_into("<f", buf, p, bad)  # only the threshold changes
        with pytest.raises(sqlite3.OperationalError) as e:
            db.execute("SELECT predict(?, 2, 20)", (bytes(buf),)).fetchall()
        assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_fit_rejects_too_many_classes(db):
    """The deserializer caps classes at PREDICT0_MAX_CLASS (2048); fit() must
    reject a larger vocabulary up front rather than register a blob that
    predict() can never load."""
    db.execute("CREATE TABLE many(f1 real, f2 real, c integer)")
    db.execute(
        "INSERT INTO many(f1, f2, c)"
        " WITH RECURSIVE s(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM s WHERE i < 2048)"
        " SELECT i * 1.0, i * 2.0, i FROM s")  # 2049 distinct classes
    with pytest.raises(sqlite3.OperationalError) as e:
        db.execute(
            "SELECT fit(f1, f2, c, '{\"task\":\"classify\"}') FROM many").fetchall()
    assert "PREDICT_ERR_SCHEMA" in str(e.value)


def test_fit_accepts_exactly_max_classes(db):
    """Exactly PREDICT0_MAX_CLASS (2048) distinct classes must SUCCEED, so an
    off-by-one in the cap (>= vs >) would be caught. tree kind keeps a 2048-way
    classifier cheap to fit."""
    db.execute("CREATE TABLE max2048(f1 real, f2 real, c integer)")
    db.execute(
        "INSERT INTO max2048(f1, f2, c)"
        " WITH RECURSIVE s(i) AS (SELECT 0 UNION ALL SELECT i+1 FROM s WHERE i < 2047)"
        " SELECT i * 1.0, i * 0.5, i FROM s")  # exactly 2048 distinct classes
    model = db.execute(
        "SELECT fit(f1, f2, c,"
        " '{\"task\":\"classify\",\"kind\":\"tree\",\"register\":\"cap2048\"}')"
        " FROM max2048").fetchone()[0]
    assert model == "cap2048"  # trained + registered, cap not tripped at exactly 2048
