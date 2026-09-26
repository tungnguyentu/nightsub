import pytest
from fastapi.testclient import TestClient

from autosub import config, jobs, web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "probe_minutes", lambda p: 2.0)
    cfg = {**config.DEFAULTS, "work_dir": str(tmp_path / "w")}
    db = config.db_path(cfg)
    return TestClient(web.create_app(cfg, db)), db


def test_folder_makes_one_job_per_video(client, tmp_path):
    c, _ = client
    for n in ("a.mp4", "b.mkv", "notes.txt"):
        (tmp_path / n).write_text("x")
    r = c.post("/jobs", json={"paths": [str(tmp_path)], "lang": "vi"})
    assert r.status_code == 200 and len(r.json()["ids"]) == 2


def test_missing_path_400(client):
    c, _ = client
    r = c.post("/jobs", json={"paths": ["/nope/x.mp4"], "lang": "en"})
    assert r.status_code == 400 and "not found" in r.json()["detail"]


def test_refuses_public_bind():
    with pytest.raises(SystemExit):
        web.check_host("0.0.0.0", False)
    web.check_host("0.0.0.0", True)
    web.check_host("127.0.0.1", False)


def test_failed_reason_in_status_and_retry(client):
    c, db = client
    i = jobs.add(db, "/v/a.mp4", "en")
    jobs.fail(db, i, "GPU busy: game")
    assert c.get("/jobs").json()[0]["error"] == "GPU busy: game"
    c.post(f"/jobs/{i}/retry")
    assert c.get("/jobs").json()[0]["error"] is None


def test_eta_needs_history():
    job = {"error": None, "stage": "polished", "audio_min": 10, "stage_times": {}}
    assert web.eta(job, [job]) is None
    past = {"error": None, "stage": "done", "audio_min": 20, "stage_times": {"retime": 40}}
    assert web.eta(job, [job, past]) == 20
