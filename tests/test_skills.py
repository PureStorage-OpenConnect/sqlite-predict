"""The skills ship as part of the repo, so CI holds them to the
agentskills.io spec and proves the receipts reference script actually
does what the prediction-receipts skill promises: record, replay-match,
tamper-detect, and pin the model hash."""

import json
import os
import re
import sqlite3
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SKILLS = os.path.join(ROOT, "skills")
RECEIPT = os.path.join(SKILLS, "prediction-receipts", "scripts", "receipt.py")
EXT = os.path.join(ROOT, "dist", "predict0")

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def skill_dirs():
    return sorted(d for d in os.listdir(SKILLS)
                  if os.path.isdir(os.path.join(SKILLS, d)))


def frontmatter(path):
    text = open(path, encoding="utf-8").read()
    assert text.startswith("---\n"), f"{path}: missing frontmatter"
    fm = text.split("---\n")[1]
    return fm, text


@pytest.mark.parametrize("d", skill_dirs())
def test_skill_conforms_to_spec(d):
    path = os.path.join(SKILLS, d, "SKILL.md")
    fm, text = frontmatter(path)

    m = re.search(r"^name:\s*(\S+)\s*$", fm, re.M)
    assert m, "name is required"
    name = m.group(1)
    assert name == d, "name must match the directory"
    assert len(name) <= 64 and NAME_RE.match(name), f"invalid name {name!r}"

    dm = re.search(r"^description:\s*>-\n((?:  .+\n)+)", fm, re.M)
    assert dm, "description is required (folded block expected)"
    desc = " ".join(dm.group(1).split())
    assert 1 <= len(desc) <= 1024, "description must be 1..1024 chars"
    assert "when" in desc.lower(), "description should say when to use it"

    assert text.count("\n") <= 500, "SKILL.md should stay under 500 lines"
    assert "—" not in text and "–" not in text, \
        "house style: no em or en dashes"


def run_receipt(*argv):
    return subprocess.run(
        [sys.executable, RECEIPT, "--extension", EXT, *argv],
        capture_output=True, text=True)


@pytest.fixture()
def receipt_db(tmp_path):
    path = str(tmp_path / "r.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE readings(ts INTEGER, value REAL)")
    db.executemany("INSERT INTO readings VALUES (?, ?)",
                   [(1700000000 + i * 3600, 10.0 + i % 7)
                    for i in range(48)])
    db.commit()
    db.close()
    return path


def test_receipt_record_verify_and_tamper(receipt_db):
    sql = "SELECT forecast(ts, value, 6) FROM readings"
    rec = run_receipt("record", receipt_db, sql)
    assert rec.returncode == 0, rec.stderr
    out = json.loads(rec.stdout)
    assert out["receipt_id"] == 1
    assert out["model_id"] not in (None, "", "auto")

    ver = run_receipt("verify", receipt_db, "1")
    assert ver.returncode == 0, ver.stdout + ver.stderr
    report = json.loads(ver.stdout)
    assert report["match"] is True
    assert report["extension"]["changed"] is False

    # determinism claim: recording again yields the identical hash
    rec2 = run_receipt("record", receipt_db, sql)
    assert json.loads(rec2.stdout)["result_sha256"] == out["result_sha256"]

    # tamper with the data: replay must mismatch, exit code must say so
    db = sqlite3.connect(receipt_db)
    db.execute("UPDATE readings SET value = value + 100 WHERE rowid = 5")
    db.commit()
    db.close()
    bad = run_receipt("verify", receipt_db, "1")
    assert bad.returncode == 2
    assert json.loads(bad.stdout)["match"] is False


def test_pure_sql_receipt_path(receipt_db):
    """The skill's primary path: record and verify with no host language
    beyond SQL, using predict_sha256."""
    db = sqlite3.connect(receipt_db)
    db.enable_load_extension(True)
    db.load_extension(EXT)
    import hashlib
    assert db.execute("SELECT predict_sha256('abc')").fetchone()[0] == \
        hashlib.sha256(b"abc").hexdigest()
    assert db.execute("SELECT predict_sha256(NULL)").fetchone()[0] is None

    db.execute("""CREATE TABLE _predict_receipts (
      id INTEGER PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      operation TEXT NOT NULL, input_sql TEXT NOT NULL, options TEXT,
      model_id TEXT NOT NULL, content_hash TEXT,
      extension TEXT NOT NULL, result_sha256 TEXT NOT NULL)""")
    db.execute("""
      INSERT INTO _predict_receipts (operation, input_sql, options, model_id,
                                     content_hash, extension, result_sha256)
      SELECT 'forecast', 'SELECT forecast(ts, value, 6) FROM readings', NULL,
             json_extract(doc, '$.model'), NULL,
             json_extract(predict_version(), '$.extension'),
             predict_sha256(doc)
      FROM (SELECT (SELECT forecast(ts, value, 6) FROM readings) AS doc)""")
    db.commit()

    (match,) = db.execute("""
      SELECT result_sha256 = predict_sha256(
        (SELECT forecast(ts, value, 6) FROM readings))
      FROM _predict_receipts WHERE id = 1""").fetchone()
    assert match == 1

    db.execute("UPDATE readings SET value = value + 100 WHERE rowid = 3")
    (match,) = db.execute("""
      SELECT result_sha256 = predict_sha256(
        (SELECT forecast(ts, value, 6) FROM readings))
      FROM _predict_receipts WHERE id = 1""").fetchone()
    assert match == 0
    db.close()


def test_receipt_pins_a_registered_student(receipt_db):
    db = sqlite3.connect(receipt_db)
    db.enable_load_extension(True)
    db.load_extension(EXT)
    db.execute("CREATE TABLE tab(f1 REAL, f2 REAL, label TEXT)")
    db.executemany("INSERT INTO tab VALUES (?,?,?)",
                   [(i * 0.1, (i % 5) * 1.0, "a" if i % 2 else "b")
                    for i in range(60)])
    db.execute("SELECT * FROM distill_predict("
               "'SELECT f1, f2, label FROM tab',"
               " '{\"target\":\"label\",\"student_id\":\"s1\"}')").fetchall()
    db.commit()
    db.close()

    rec = run_receipt(
        "record", receipt_db,
        "SELECT forecast(ts, value, 6, '{\"model\":\"theta-classic\"}')"
        " FROM readings")
    out = json.loads(rec.stdout)
    assert out["model_id"] == "theta-classic"
    assert out["content_hash"], "registered models must record their pin"

    # rows-shaped results canonicalize too: predict() through a student
    rec2 = run_receipt(
        "record", receipt_db,
        "SELECT rowid, prediction FROM predict(NULL,"
        " 'SELECT rowid, f1, f2 FROM tab', '{\"model\":\"s1\"}')",
        "--model-id", "s1")
    assert rec2.returncode == 0, rec2.stderr
    out2 = json.loads(rec2.stdout)
    assert out2["content_hash"], "student content_hash must be recorded"
    ver = run_receipt("verify", receipt_db, str(out2["receipt_id"]))
    assert json.loads(ver.stdout)["match"] is True


def test_receipt_adversarial_paths(receipt_db):
    """Break the receipt tool the ways a hostile or careless caller
    would: single-text-cell TVF results must canonicalize as rows (not
    as an aggregate document), extension errors must surface their
    PREDICT_ERR_* code verbatim, missing receipts must fail loudly, and
    a replayed receipt must not be able to re-enable extension
    loading."""
    import hashlib

    # a backtest projected to one text column: operation-based
    # canonicalization must hash the {"columns","rows"} form, so the
    # receipt hash must NOT equal the raw-cell hash
    one_cell_sql = ("SELECT model FROM backtest("
                    "'SELECT ts, value FROM readings', 4,"
                    " '{\"model\":\"theta-classic\"}') LIMIT 1")
    rec = run_receipt("record", receipt_db, one_cell_sql,
                      "--model-id", "theta-classic")
    assert rec.returncode == 0, rec.stderr
    out = json.loads(rec.stdout)
    db = sqlite3.connect(receipt_db)
    db.enable_load_extension(True)
    db.load_extension(EXT)
    raw_cell = db.execute(one_cell_sql).fetchone()[0]
    db.close()
    assert out["result_sha256"] != hashlib.sha256(
        raw_cell.encode()).hexdigest(),         "TVF result canonicalized as an aggregate document"
    ver = run_receipt("verify", receipt_db, str(out["receipt_id"]))
    assert json.loads(ver.stdout)["match"] is True

    # unknown option key: the extension's exact error code must surface
    bad = run_receipt(
        "record", receipt_db,
        "SELECT forecast(ts, value, 6, '{\"bogus_key\":1}') FROM readings")
    assert bad.returncode != 0
    assert "PREDICT_ERR_OPTIONS" in (bad.stderr + bad.stdout)

    # numeric input to the hash primitive is rejected, never coerced
    db = sqlite3.connect(receipt_db)
    db.enable_load_extension(True)
    db.load_extension(EXT)
    import pytest as _pytest
    for bad_val in ("1.5", "42"):
        with _pytest.raises(sqlite3.OperationalError, match="PREDICT_ERR_SCHEMA"):
            db.execute(f"SELECT predict_sha256({bad_val})").fetchone()
    assert db.execute("SELECT predict_sha256(NULL)").fetchone()[0] is None
    db.close()

    # missing receipt id fails loudly
    gone = run_receipt("verify", receipt_db, "9999")
    assert gone.returncode != 0
    assert "RECEIPT_ERR_NOT_FOUND" in gone.stderr

    # a tampered receipt cannot load an arbitrary extension on replay
    db = sqlite3.connect(receipt_db)
    db.execute(
        "UPDATE _predict_receipts SET input_sql ="
        " 'SELECT load_extension(''/tmp/evil'')' WHERE id = 1")
    db.commit()
    db.close()
    evil = run_receipt("verify", receipt_db, "1")
    assert evil.returncode != 0
    assert "not authorized" in (evil.stderr + evil.stdout).lower()


def test_receipt_metadata_tampering(receipt_db):
    """A tampered model_id must fail verification via re-derivation; a
    tampered options column is documented as informational (the
    authoritative options live inside input_sql), so this test pins that
    stated limit rather than pretending it is detection."""
    rec = run_receipt("record", receipt_db,
                      "SELECT forecast(ts, value, 6) FROM readings")
    rid = str(json.loads(rec.stdout)["receipt_id"])

    db = sqlite3.connect(receipt_db)
    db.execute("UPDATE _predict_receipts SET model_id = 'impostor'"
               " WHERE id = ?", (rid,))
    db.commit()
    db.close()
    forged = run_receipt("verify", receipt_db, rid)
    assert forged.returncode == 2
    rep = json.loads(forged.stdout)
    assert rep["match"] is False
    assert rep["model"]["id_matches"] is False

    db = sqlite3.connect(receipt_db)
    db.execute("UPDATE _predict_receipts SET model_id = 'theta-classic',"
               " options = '{\"forged\":true}' WHERE id = ?", (rid,))
    db.commit()
    db.close()
    forged_opts = run_receipt("verify", receipt_db, rid)
    rep = json.loads(forged_opts.stdout)
    assert rep["match"] is True, "options is informational by design"
    assert rep["options_verified"] is False


def test_receipt_replay_cannot_write(receipt_db):
    """Replay input is untrusted: a tampered input_sql that writes,
    drops, or attaches must be refused by the read-only replay
    connection, and the verifier's data must be untouched."""
    rec = run_receipt("record", receipt_db,
                      "SELECT forecast(ts, value, 6) FROM readings")
    rid = str(json.loads(rec.stdout)["receipt_id"])

    for attack in ("DELETE FROM readings",
                   "DROP TABLE readings",
                   "ATTACH DATABASE '/tmp/exfil.db' AS x",
                   "PRAGMA journal_mode = OFF"):
        db = sqlite3.connect(receipt_db)
        db.execute("UPDATE _predict_receipts SET input_sql = ?"
                   " WHERE id = ?", (attack, rid))
        db.commit()
        db.close()
        out = run_receipt("verify", receipt_db, rid)
        assert out.returncode != 0, f"replay accepted: {attack}"
        blob = (out.stderr + out.stdout).lower()
        assert ("not authorized" in blob or "readonly" in blob
                or "read-only" in blob or "prohibited" in blob), attack

    db = sqlite3.connect(receipt_db)
    (n,) = db.execute("SELECT count(*) FROM readings").fetchone()
    db.close()
    assert n == 48, "replay modified the verifier's data"


def test_receipt_hardening_paths(receipt_db, tmp_path):
    """The audit set: runaway replay is aborted by the step budget, BLOB
    cells canonicalize deterministically instead of crashing, invalid
    --options is rejected up front, and a missing database fails with a
    clean code instead of a traceback."""
    rec = run_receipt("record", receipt_db,
                      "SELECT forecast(ts, value, 6) FROM readings")
    rid = str(json.loads(rec.stdout)["receipt_id"])

    # recursive CTE bomb: must abort via the step budget, not hang
    db = sqlite3.connect(receipt_db)
    db.execute("UPDATE _predict_receipts SET input_sql ="
               " 'WITH RECURSIVE b(x) AS (SELECT 1 UNION ALL SELECT x+1"
               "  FROM b) SELECT max(x) FROM b' WHERE id = ?", (rid,))
    db.commit()
    db.close()
    import subprocess as sp
    bomb = sp.run([sys.executable, RECEIPT, "--extension", EXT,
                   "verify", receipt_db, rid],
                  capture_output=True, text=True, timeout=60,
                  env={**os.environ, "RECEIPT_MAX_STEPS": "100000"})
    assert bomb.returncode != 0
    assert "RECEIPT_ERR_SQL" in bomb.stderr

    # BLOB cells canonicalize as their hash, deterministically
    db = sqlite3.connect(receipt_db)
    db.execute("CREATE TABLE blobs(b BLOB)")
    db.execute("INSERT INTO blobs VALUES (x'00ff10')")
    db.commit()
    db.close()
    b1 = run_receipt("record", receipt_db, "SELECT b FROM blobs",
                     "--operation", "predict", "--model-id", "n/a")
    b2 = run_receipt("record", receipt_db, "SELECT b FROM blobs",
                     "--operation", "predict", "--model-id", "n/a")
    assert b1.returncode == 0, b1.stderr
    assert json.loads(b1.stdout)["result_sha256"] == \
        json.loads(b2.stdout)["result_sha256"]

    # invalid --options rejected before anything executes
    badopt = run_receipt("record", receipt_db,
                         "SELECT forecast(ts, value, 6) FROM readings",
                         "--options", "{not json")
    assert badopt.returncode != 0
    assert "RECEIPT_ERR_OPTIONS" in badopt.stderr

    # missing database file fails cleanly on the read-only path
    gone_db = run_receipt("verify", str(tmp_path / "nope.db"), "1")
    assert gone_db.returncode != 0
    assert "RECEIPT_ERR" in gone_db.stderr
    assert "Traceback" not in gone_db.stderr
