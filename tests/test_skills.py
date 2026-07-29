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
