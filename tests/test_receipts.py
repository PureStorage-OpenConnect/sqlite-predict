"""The model registry (_predict_models): bundled rows.

Receipts are gone; the registry remains. It is created on demand by the
paths that write to it (distill/register) — a serving-path forecast never
creates it — and is seeded with the bundled model rows.
"""
import synthetic_tabular as syt


def test_bundled_models_registered(db):
    X, y, _ = syt.two_moons(n=100)
    syt.load_tabular(db, X, y)
    # distilling any student materializes the registry
    db.execute(
        "SELECT model_id FROM distill_predict('SELECT f1, f2, label FROM tab',"
        " json_object('target','label','student_id','s'))").fetchone()
    rows = db.execute(
        "SELECT model_id, kind, license FROM _predict_models"
        " WHERE runtime = 'bundled' ORDER BY model_id").fetchall()
    kinds = {r[0]: r[1] for r in rows}
    assert kinds == {
        "auto": "ts-stat",
        "knn5-incontext": "tabular-stat",
        "stub-seasonal-naive": "ts-stat",
        "sub-pca": "ts-stat",
        "theta-classic": "ts-stat",
        "tsb": "ts-stat",
    }
    assert all(r[2] == "MIT OR Apache-2.0" for r in rows)
