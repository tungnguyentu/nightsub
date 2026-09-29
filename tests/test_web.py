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
    r = c.post("/jobs", json={"paths": [str(tmp_path)], "langs": ["vi"]})
    assert r.status_code == 200 and len(r.json()["ids"]) == 2


def test_missing_path_400(client):
    c, _ = client
    r = c.post("/jobs", json={"paths": ["/nope/x.mp4"], "langs": ["en"]})
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


def test_steps_api_and_job_state(client):
    c, db = client
    step_info = c.get("/steps").json()
    assert [s["name"] for s in step_info] == ["gate", "asr", "brief", "verify", "translate", "polish", "retime"]
    i = jobs.add(db, "/v/a.mp4", "vi")
    jobs.update(db, i, stage="gated", stage_times={"gate": 2.2}, progress=0.25)
    steps = c.get("/jobs").json()[0]["steps"]
    assert [s["state"] for s in steps[:3]] == ["done", "running", "pending"]
    assert steps[0]["secs"] == 2 and steps[1]["progress"] == 0.25


def test_lines_endpoint_reads_segments_artifact_and_skipped_brief(client):
    c, db = client
    i = jobs.add(db, "/v/a.mp4", "vi")
    assert c.get(f"/jobs/{i}/lines").status_code == 404
    d = config.job_dir({**config.DEFAULTS, "work_dir": str(db).rsplit("/", 1)[0]}, i)
    (d / "segments.json").write_text('[{"start":1.0,"end":2.0,"gender":"F","text":"日本語"}]')
    (d / "brief.json").write_text('{"skipped":"model error"}')
    data = c.get(f"/jobs/{i}/lines").json()
    assert data["brief"]["skipped"] == "model error"
    assert data["rows"] == [{"start":1.0,"end":2.0,"gender":"F","src":"日本語",
                              "text":None,"polished":False,"verified":None,"asr_alt":None,"failed":False}]


def test_lines_endpoint_prefers_polished_artifact_and_marks_failed(client):
    c, db = client
    i = jobs.add(db, "/v/a.mp4", "vi")
    d = config.job_dir({**config.DEFAULTS, "work_dir": str(db).rsplit("/", 1)[0]}, i)
    (d / "translated.json").write_text('[{"src":"源","text":"[untranslated]"}]')
    data = c.get(f"/jobs/{i}/lines").json()
    assert data["rows"][0]["src"] == "源" and data["rows"][0]["failed"]
    (d / "polished.json").write_text('[{"src":"源","text":"Đã dịch","polished":true}]')
    row = c.get(f"/jobs/{i}/lines").json()["rows"][0]
    assert row["text"] == "Đã dịch" and row["polished"] and not row["failed"]


def test_two_languages_make_two_jobs_per_video(client, tmp_path):
    c, _ = client
    (tmp_path / "a.mp4").write_text("x")
    r = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en", "vi"]})
    assert len(r.json()["ids"]) == 2
    assert sorted(j["lang"] for j in c.get("/jobs").json()) == ["en", "vi"]


def test_no_language_picked_400(client, tmp_path):
    c, _ = client
    (tmp_path / "a.mp4").write_text("x")
    assert c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": []}).status_code == 400


def test_browse_lists_folders_and_videos_only(client, tmp_path):
    c, _ = client
    root = tmp_path / "lib"  # tmp_path also holds the fixture's work dir
    (root / "sub").mkdir(parents=True)
    (root / ".hidden").mkdir()
    for n in ("a.mp4", "notes.txt"):
        (root / n).write_text("x")
    b = c.get("/browse", params={"dir": str(root)}).json()
    assert b["dirs"] == ["sub"] and b["videos"] == ["a.mp4"]


def test_deleting_a_running_job_marks_it_and_a_done_job_goes_now(client, tmp_path):
    c, db = client
    (tmp_path / "a.mp4").write_text("x")
    i = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en"]}).json()["ids"][0]
    assert c.delete(f"/jobs/{i}").json()["pending"] is True
    assert jobs.get(db, i)["control"] == "delete"
    jobs.purge_deleted(db)
    assert c.get("/jobs").json() == []
    k = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en"]}).json()["ids"][0]
    jobs.update(db, k, stage="done")
    assert c.delete(f"/jobs/{k}").json()["pending"] is False and c.get("/jobs").json() == []


def test_pause_and_resume(client, tmp_path):
    c, db = client
    (tmp_path / "a.mp4").write_text("x")
    i = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en"]}).json()["ids"][0]
    assert c.post(f"/jobs/{i}/pause").status_code == 200
    assert jobs.at_stage(db, "queued") == []  # scheduler skips it
    assert c.post(f"/jobs/{i}/resume").status_code == 200
    assert [j["id"] for j in jobs.at_stage(db, "queued")] == [i]
    assert c.post(f"/jobs/{i}/resume").status_code == 409


def test_mux_without_subtitles_is_400(client, tmp_path):
    c, db = client
    (tmp_path / "a.mp4").write_text("x")
    i = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en"]}).json()["ids"][0]
    assert c.post(f"/jobs/{i}/mux").status_code == 400


def test_progress_raises_stopped_when_paused(tmp_path):
    import pytest
    from autosub import config
    db = config.db_path({**config.DEFAULTS, "work_dir": str(tmp_path / "w2")})
    i = jobs.add(db, "/v.mp4", "vi")
    jobs.progress(db, i, 0.1)
    jobs.update(db, i, control="pause")
    with pytest.raises(jobs.Stopped):
        jobs.progress(db, i, 0.2)


def test_new_job_does_not_inherit_a_reused_ids_files(client, tmp_path):
    c, db = client
    (tmp_path / "a.mp4").write_text("x")
    i = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en"]}).json()["ids"][0]
    jobs.update(db, i, stage="done")
    from autosub import config
    stale = config.job_dir({"work_dir": str(tmp_path / "w")}, i) / "segments.partial.jsonl"
    stale.write_text("old")
    c.delete(f"/jobs/{i}")
    k = c.post("/jobs", json={"paths": [str(tmp_path / "a.mp4")], "langs": ["en"]}).json()["ids"][0]
    assert k == i and not stale.exists()
