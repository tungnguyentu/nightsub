import json
import sys
import threading
import time
import types

import numpy as np

from autosub import config, scheduler
from autosub.stages import verify


def cfg(tmp_path, **updates):
    value = {**config.DEFAULTS, "work_dir": str(tmp_path), "asr_model": "kotoba",
             "verify_asr_model": "large-v3", "verify_model": "agy/test",
             "asr_language": "ja", "verify_agree": 0.6,
             "models": {**config.DEFAULTS["models"], "vi": "agy/test"}}
    value.update(updates)
    return value


def segment(text="こんにちは。", **extra):
    return {"start": 1.0, "end": 2.0, "text": text, "logprob": -0.2, "gender": "F", **extra}


class FakeModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def transcribe(self, clip, **kwargs):
        self.calls.append((len(clip), kwargs))
        return iter([types.SimpleNamespace(text=self.results.pop(0))]), None


def setup_job(tmp_path, segs, video="/video/shared.mp4"):
    d = tmp_path / "job"
    d.mkdir()
    (d / "segments.json").write_text(json.dumps(segs, ensure_ascii=False))
    (d / "audio.wav").touch()
    (d / "brief.json").write_text(json.dumps({"summary": "ふたりの会話", "characters": [
        {"name_or_role": "葵"}, {"name_or_role": "先生"}, {"name_or_role": "x" * 41},
    ]}, ensure_ascii=False))
    return d, {"id": 17, "video": video}


def stub_runtime(monkeypatch):
    monkeypatch.setattr(verify, "read_wav", lambda _path: np.zeros(16000 * 5, np.float32))
    monkeypatch.setattr(verify.jobs, "progress", lambda *_args: None)


def test_agreement_keeps_kotoba_and_name_hints_reach_whisper(tmp_path, monkeypatch):
    stub_runtime(monkeypatch)
    model = FakeModel(["こん にちは"])
    d, job = setup_job(tmp_path, [segment()])
    monkeypatch.setattr(verify.agy, "chat", lambda *_args: (_ for _ in ()).throw(AssertionError("no arbitration")))

    verify.process(cfg(tmp_path), None, job, d, model)

    result = json.loads((d / "segments.json").read_text())[0]
    assert result["text"] == "こんにちは。"
    assert "verified" not in result and "asr_alt" not in result
    assert model.calls[0][1] == {"language": "ja", "condition_on_previous_text": False,
                                 "initial_prompt": "葵、先生"}


def test_disagreement_uses_arbitration_and_keeps_other_candidate(tmp_path, monkeypatch):
    stub_runtime(monkeypatch)
    model = FakeModel(["おはようございます"])
    d, job = setup_job(tmp_path, [segment()])
    messages_seen = []

    def chat(model_name, messages):
        messages_seen.append((model_name, messages))
        return json.dumps(["おはようございます"], ensure_ascii=False)

    monkeypatch.setattr(verify.agy, "chat", chat)
    original = json.loads((d / "segments.json").read_text())[0]
    verify.process(cfg(tmp_path), None, job, d, model)

    result = json.loads((d / "segments.json").read_text())[0]
    assert result["text"] == "おはようございます"
    assert result["asr_alt"] == original["text"] and result["verified"] == "second_model"
    assert {key: result[key] for key in ("start", "end", "logprob", "gender")} == {
        key: original[key] for key in ("start", "end", "logprob", "gender")}
    assert messages_seen[0][0] == "agy/test"
    assert "ふたりの会話" in messages_seen[0][1][1]["content"]
    assert "Kotoba candidate: こんにちは。" in messages_seen[0][1][1]["content"]
    assert "Second-model candidate: おはようございます" in messages_seen[0][1][1]["content"]


def test_arbitration_refusal_keeps_kotoba_text(tmp_path, monkeypatch):
    stub_runtime(monkeypatch)
    model = FakeModel(["おはようございます"])
    d, job = setup_job(tmp_path, [segment()])
    monkeypatch.setattr(verify.agy, "chat", lambda *_args: '["I cannot help with that request"]')

    verify.process(cfg(tmp_path), None, job, d, model)

    result = json.loads((d / "segments.json").read_text())[0]
    assert result["text"] == "こんにちは。"
    assert "verified" not in result and "asr_alt" not in result


def test_local_verify_model_uses_ollama_json_chat(tmp_path, monkeypatch):
    stub_runtime(monkeypatch)
    model = FakeModel(["おはようございます"])
    d, job = setup_job(tmp_path, [segment()])
    calls = []

    def chat(url, model_name, messages, options, **kwargs):
        calls.append((url, model_name, messages, kwargs))
        return json.dumps(["おはようございます"], ensure_ascii=False)

    monkeypatch.setattr(verify.ollama, "chat", chat)
    verify.process(cfg(tmp_path, verify_model="local-arb"), None, job, d, model)

    assert calls[0][1] == "local-arb" and calls[0][3] == {"json_mode": True, "max_tokens": 2048}


def test_verify_false_passes_transcript_through(tmp_path, monkeypatch):
    d, job = setup_job(tmp_path, [segment()])
    original = (d / "segments.json").read_text()
    verify.process(cfg(tmp_path, verify=False), None, job, d, None)
    assert (d / "segments.json").read_text() == original


def test_cache_reused_for_another_language_job(tmp_path, monkeypatch):
    stub_runtime(monkeypatch)
    cfg_value = cfg(tmp_path)
    first_model = FakeModel(["おはようございます"])
    d1, job1 = setup_job(tmp_path, [segment()])
    calls = []
    monkeypatch.setattr(verify.agy, "chat", lambda *_args: calls.append("agy") or
                        json.dumps(["おはようございます"], ensure_ascii=False))
    verify.process(cfg_value, None, job1, d1, first_model)

    d2 = tmp_path / "job2"
    d2.mkdir()
    (d2 / "segments.json").write_text(json.dumps([segment()], ensure_ascii=False))
    second_model = FakeModel([])
    verify.process(cfg_value, None, {**job1, "id": 18, "lang": "vi"}, d2, second_model)

    assert second_model.calls == [] and calls == ["agy"]
    assert json.loads((d2 / "segments.json").read_text())[0]["verified"] == "second_model"


def test_load_uses_configured_whisper_model_and_compute_type(tmp_path, monkeypatch):
    created = []

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    verify.load(cfg(tmp_path, verify_asr_model="large-v3-turbo", asr_compute_type="float16"))
    assert created == [(("large-v3-turbo",), {"device": "cuda", "compute_type": "float16"})]


def test_scheduler_places_verify_after_brief_before_translate():
    assert [stage[0] for stage in scheduler.STAGES] == ["gate", "asr", "brief", "verify", "translate", "polish", "retime"]
    assert scheduler.STAGES[3] == ("verify", "briefed", "verified")
    assert scheduler.STAGES[4] == ("translate", "verified", "translated")
    assert "verify" in scheduler.GPU_STAGES
    info = next(item for item in scheduler.STEP_INFO if item["name"] == "verify")
    assert info["label_vi"] == "Đối chiếu"
    assert info["help_vi"] == "Nghe lại bằng model thứ hai; câu lệch được Gemini/LLM chọn lại theo ngữ cảnh."


def test_cloud_arbitration_batches_run_concurrently(tmp_path, monkeypatch):
    stub_runtime(monkeypatch)
    segs = [segment(f"こんにちは{i}", start=float(i), end=float(i + 1)) for i in range(21)]
    d, job = setup_job(tmp_path, segs)
    model = FakeModel([f"第二候補{i}" for i in range(21)])
    active = 0
    peak = 0
    lock = threading.Lock()

    def chat(_model, messages):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        count = messages[1]["content"].count("Kotoba candidate:")
        return json.dumps(["選ばれた日本語です"] * count, ensure_ascii=False)

    monkeypatch.setattr(verify.agy, "chat", chat)
    verify.process(cfg(tmp_path, cloud_parallel=2), None, job, d, model)

    assert peak == 2
    assert all(seg["verified"] == "arbitrated" for seg in json.loads((d / "segments.json").read_text()))
