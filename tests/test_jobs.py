import sqlite3

from autosub import jobs


def test_roundtrip(tmp_path):
    db = str(tmp_path / "j.db")
    i = jobs.add(db, "/v/a.mp4", "en")
    jobs.update(db, i, stage="gated", stage_times={"gate": 1.5})
    j = jobs.get(db, i)
    assert (j["stage"], j["stage_times"], j["error"]) == ("gated", {"gate": 1.5}, None)
    jobs.fail(db, i, "boom")
    assert jobs.at_stage(db, "gated") == [] and jobs.get(db, i)["error"] == "boom"


def test_old_db_migrates_stage_start_column(tmp_path):
    db = str(tmp_path / "old.db")
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, video TEXT NOT NULL, lang TEXT NOT NULL, tags INTEGER DEFAULT 0,
            stage TEXT DEFAULT 'queued', progress REAL DEFAULT 0, error TEXT, audio_min REAL,
            failed_lines INTEGER DEFAULT 0, flagged_share REAL, cues INTEGER, stage_times TEXT DEFAULT '{}')""")
        c.execute("INSERT INTO jobs (video, lang) VALUES (?, ?)", ("/old.mp4", "vi"))
    job = jobs.get(db, 1)
    assert job["stage_started_at"] is None
    with sqlite3.connect(db) as c:
        assert "stage_started_at" in {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
