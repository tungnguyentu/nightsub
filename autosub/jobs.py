"""SQLite job store (KTD3). A job has failed iff `error` is set; `stage` keeps the resume point."""
import json
import sqlite3

SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, video TEXT NOT NULL, lang TEXT NOT NULL, tags INTEGER DEFAULT 0,
  stage TEXT DEFAULT 'queued', progress REAL DEFAULT 0, error TEXT, audio_min REAL,
  failed_lines INTEGER DEFAULT 0, flagged_share REAL, cues INTEGER, stage_times TEXT DEFAULT '{}')"""


def _conn(db):
    c = sqlite3.connect(db, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    return c


def add(db, video, lang, tags=False, audio_min=None):
    with _conn(db) as c:
        return c.execute("INSERT INTO jobs (video, lang, tags, audio_min) VALUES (?,?,?,?)",
                         (video, lang, int(tags), audio_min)).lastrowid


def _row(r):
    d = dict(r)
    d["stage_times"] = json.loads(d["stage_times"])
    return d


def get(db, job_id):
    with _conn(db) as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(r) if r else None


def all(db):
    with _conn(db) as c:
        return [_row(r) for r in c.execute("SELECT * FROM jobs ORDER BY id")]


def at_stage(db, stage):
    with _conn(db) as c:
        return [_row(r) for r in c.execute("SELECT * FROM jobs WHERE stage=? AND error IS NULL ORDER BY id", (stage,))]


def update(db, job_id, **fields):
    if "stage_times" in fields:
        fields["stage_times"] = json.dumps(fields["stage_times"])
    with _conn(db) as c:
        c.execute(f"UPDATE jobs SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), job_id))


def fail(db, job_id, reason):
    update(db, job_id, error=str(reason)[:500])


def delete(db, job_id):
    with _conn(db) as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
