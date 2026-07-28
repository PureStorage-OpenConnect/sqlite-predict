import json


def test_version_is_json_with_spec(db):
    raw = db.execute("SELECT predict_version()").fetchone()[0]
    v = json.loads(raw)
    assert v["extension"].startswith("v0.")
    assert isinstance(v["runtimes"], list) and "stat" in v["runtimes"]
    assert isinstance(v["models"], list)


def test_debug_mentions_version(db):
    raw = db.execute("SELECT predict_debug()").fetchone()[0]
    assert "Version: v0." in raw
    assert "Date: " in raw
