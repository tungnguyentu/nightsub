"""SQLite job store (KTD3). A job has failed iff `error` is set; `stage` keeps the resume point."""
import json
import sqlite3

SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, video TEXT NOT NULL, lang TEXT NOT NULL, tags INTEGER DEFAULT 0,
  stage TEXT DEFAULT 'queued', progress REAL DEFAULT 0, error TEXT, audio_min REAL,
  failed_lines INTEGER DEFAULT 0, flagged_share REAL, cues INTEGER, stage_times TEXT DEFAULT '{}',
  stage_started_at REAL, control TEXT)"""


def _conn(db):
    c = sqlite3.connect(db, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    columns = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
    if "stage_started_at" not in columns:
        c.execute("ALTER TABLE jobs ADD COLUMN stage_started_at REAL")
    if "control" not in columns:  # NULL | 'pause' | 'delete' (set by the UI, honoured by running stages)
        c.execute("ALTER TABLE jobs ADD COLUMN control TEXT")
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
        return [_row(r) for r in c.execute("SELECT * FROM jobs WHERE stage=? AND error IS NULL AND control IS NULL ORDER BY id", (stage,))]


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


class Stopped(Exception):
    """Raised inside a stage when the operator paused or deleted the job; not a failure."""


def progress(db, job_id, fraction):
    """Report progress; stop this job's stage if the UI asked to pause or delete it."""
    update(db, job_id, progress=fraction)
    j = get(db, job_id)
    if j is None or j["control"]:
        raise Stopped(job_id)


def purge_deleted(db):
    """Remove jobs the UI marked for deletion; returns their ids so callers can clear their dirs."""
    with _conn(db) as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE control='delete'")]
        c.execute("DELETE FROM jobs WHERE control='delete'")
    return ids
