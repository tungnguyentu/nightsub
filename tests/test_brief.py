import json

from autosub import config, jobs
from autosub.stages import brief

CFG = {**config.DEFAULTS, "work_dir": "/tmp/autosub-brief-test", "brief_model": "vi-model",
       "fallback_model": "fallback"}


def setup_batch(tmp_path, texts, count=1):
    cfg = {**CFG, "work_dir": str(tmp_path)}
    db = str(tmp_path / "jobs.db")
    ids = [jobs.add(db, f"/video/shared.mp4", lang) for lang in ("en", "vi")[:count]]
    for job_id in ids:
        d = config.job_dir(cfg, job_id)
        (d / "segments.json").write_text(json.dumps([{"text": t} for t in texts]))
    return cfg, db, [jobs.get(db, i) for i in ids]


def valid_brief():
    return json.dumps({"summary": "Cặp đôi nói chuyện.", "characters": [], "relationship": "người yêu",
                       "setting": "nhà", "vi_address": {"male_self": "anh"}})


def test_same_video_reuses_brief_cache(tmp_path, monkeypatch):
    cfg, db, batch = setup_batch(tmp_path, ["あ"], 2)
    calls = []
    monkeypatch.setattr(brief.ollama, "chat", lambda url, model, messages: calls.append(model) or valid_brief())
    monkeypatch.setattr(brief, "unload_all", lambda cfg, used: None)
    brief.run(cfg, db, batch)
    assert calls == ["vi-model", "vi-model"]
    assert [jobs.get(db, j["id"])["stage"] for j in batch] == ["briefed", "briefed"]
    assert (config.job_dir(cfg, batch[1]["id"]) / "brief.json").read_text() == (config.job_dir(cfg, batch[0]["id"]) / "brief.json").read_text()


def test_bad_merge_falls_back_then_marks_skipped(tmp_path, monkeypatch):
    cfg, db, batch = setup_batch(tmp_path, ["あ"])
    monkeypatch.setattr(brief.ollama, "chat", lambda *args: "not json")
    monkeypatch.setattr(brief, "unload_all", lambda cfg, used: None)
    brief.run(cfg, db, batch)
    data = json.loads((config.job_dir(cfg, batch[0]["id"]) / "brief.json").read_text())
    assert "skipped" in data
    assert jobs.get(db, batch[0]["id"])["stage"] == "briefed"


def test_400_lines_use_three_chunks_and_merge(tmp_path, monkeypatch):
    cfg, db, batch = setup_batch(tmp_path, [str(i) for i in range(400)])
    calls = []
    def chat(url, model, messages):
        prompt = messages[1]["content"]
        calls.append(prompt)
        return "facts" if prompt.startswith("Extract") else valid_brief()
    monkeypatch.setattr(brief.ollama, "chat", chat)
    monkeypatch.setattr(brief, "unload_all", lambda cfg, used: None)
    brief.run(cfg, db, batch)
    assert len(calls) == 4 and sum(p.startswith("Extract") for p in calls) == 3
