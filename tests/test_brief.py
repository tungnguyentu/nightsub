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
    monkeypatch.setattr(brief.ollama, "chat", lambda url, model, messages, cfg=None, **kw: calls.append(model) or valid_brief())
    monkeypatch.setattr(brief, "unload_all", lambda cfg, used: None)
    brief.run(cfg, db, batch)
    assert calls == ["vi-model"] * 3  # facts, merge, cast; second job hits the cache
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
    def chat(url, model, messages, cfg=None, **kw):
        prompt = messages[1]["content"]
        calls.append(prompt)
        return "facts" if prompt.startswith("Extract") else valid_brief()
    monkeypatch.setattr(brief.ollama, "chat", chat)
    monkeypatch.setattr(brief, "unload_all", lambda cfg, used: None)
    brief.run(cfg, db, batch)
    assert len(calls) == 9 and sum(p.startswith("Extract") for p in calls) == 3  # + merge + 5 cast windows


def test_cast_labels_each_line_and_blanks_a_bad_window(tmp_path, monkeypatch):
    cfg, db, batch = setup_batch(tmp_path, ["あ"] * 81)
    def chat(url, model, messages, cfg=None, **kw):
        p = messages[1]["content"]
        if p.startswith("Characters"):
            n = int(p.split("each of the ")[1].split()[0])
            return json.dumps({"lines": ["Nao>Shūji"] * n}) if n == 80 else "not json"
        return "facts" if p.startswith("Extract") else valid_brief()
    monkeypatch.setattr(brief.ollama, "chat", chat)
    monkeypatch.setattr(brief, "unload_all", lambda cfg, used: None)
    brief.run(cfg, db, batch)
    speakers = json.loads((config.job_dir(cfg, batch[0]["id"]) / "brief.json").read_text())["speakers"]
    assert speakers == ["Nao>Shūji"] * 80 + [""]


def test_normalize_coerces_wrong_types_instead_of_discarding():
    from autosub.stages.brief import normalize
    b = normalize({"summary": ["Nao sống cùng bố chồng", "chồng đi làm xa"], "relationship": {"Nao": "con dâu"},
                   "characters": {"name_or_role": "Nao", "gender": "F"}, "vi_address": None})
    assert "bố chồng" in b["summary"] and "con dâu" in b["relationship"]
    assert b["characters"] == [{"name_or_role": "Nao", "gender": "F"}] and b["vi_address"] == {} and b["setting"] == ""


def test_normalize_rejects_an_empty_brief():
    import pytest
    from autosub.stages.brief import normalize
    with pytest.raises(ValueError):
        normalize({"characters": []})


def test_long_notes_are_condensed_before_the_merge(monkeypatch):
    from autosub.stages import brief
    calls = []
    def chat(cfg, prompt, used, json_mode=False, max_tokens=1024):
        calls.append(prompt[:9])
        return "- short fact"
    monkeypatch.setattr(brief, "_chat", chat)
    out = brief.condense({}, ["x" * 900] * 20, set())
    assert len("\n".join(out)) <= brief.MAX_FACT_CHARS and all(c == "Condense " for c in calls)
    assert len(calls) == 4  # 20 notes in groups of 6 -> 4 condense calls, then they fit


def test_one_refused_chunk_does_not_discard_the_brief(monkeypatch):
    import json
    from autosub.stages import brief
    n = {"i": 0}
    def chat(cfg, prompt, used, json_mode=False, max_tokens=1024):
        n["i"] += 1
        if json_mode:
            return json.dumps({"summary": "s", "relationship": "r", "characters": [], "setting": "", "vi_address": {}})
        return None if n["i"] == 2 else "- fact"
    monkeypatch.setattr(brief, "_chat", chat)
    b = brief.make_brief({}, ["あ"] * (brief.CHUNK_SIZE * 3), set())
    assert b["summary"] == "s"


def test_truncated_merge_is_retried_shorter(monkeypatch):
    import json
    from autosub.stages import brief
    merges = []
    def chat(cfg, prompt, used, json_mode=False, max_tokens=1024):
        if not json_mode:
            return "- fact"
        merges.append(prompt)
        if len(merges) == 1:
            return '{"summary": "cut off mid'
        return json.dumps({"summary": "s", "relationship": "r", "characters": [], "setting": "", "vi_address": {}})
    monkeypatch.setattr(brief, "_chat", chat)
    assert brief.make_brief({}, ["あ"] * 10, set())["summary"] == "s"
    assert len(merges) == 2 and "1 sentence" in merges[1]


def test_chat_routes_agy_brief_model_to_cloud_without_ollama_options(monkeypatch):
    from autosub.stages import brief
    calls = []
    monkeypatch.setattr(brief.agy, "chat", lambda model, messages, **kw: calls.append((model, messages)) or "facts")
    monkeypatch.setattr(brief.ollama, "chat", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected local call")))
    cfg = {"brief_model": "agy/gemini-test", "models": {"vi": "local"}, "fallback_model": "local",
           "ollama_url": "http://localhost"}

    assert brief._chat(cfg, "Japanese dialogue", set(), json_mode=True, max_tokens=123) == "facts"
    assert calls[0][0] == "agy/gemini-test"
    assert calls[0][1][1]["content"] == "Japanese dialogue"


def test_first_json_ignores_trailing_text():
    from autosub.stages.brief import first_json
    assert first_json('Here you go:\n{"summary": "s"}\n\nDone. {"x": 1}') == {"summary": "s"}
    import pytest
    with pytest.raises(ValueError):
        first_json("no json here")
