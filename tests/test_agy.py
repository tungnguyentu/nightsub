import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from autosub import agy
from autosub.stages import translate


def test_agy_models_skip_ollama_and_gpu_fit(monkeypatch):
    monkeypatch.setattr(agy, "chat", lambda model, msgs: '["xin chào"]')
    monkeypatch.setattr(translate.ollama, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ollama")))
    monkeypatch.setattr(translate, "check_fit", lambda *a: (_ for _ in ()).throw(AssertionError("fit")))
    used = set()
    assert translate.ask({"ollama_url": ""}, "agy/gemini-3.8-flash-low", ["x"], [], "vi", used) == ["xin chào"]
    assert used == set()


def test_chat_parses_print_json(monkeypatch):
    out = json.dumps({"status": "SUCCESS", "response": '["a"]\n'})
    monkeypatch.setattr(agy.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    assert agy.chat("agy/m", [{"content": "x"}]) == '["a"]\n'


def test_chat_logs_request_duration_without_prompt(monkeypatch, caplog):
    out = json.dumps({"status": "SUCCESS", "response": '["a"]'})
    monkeypatch.setattr(agy.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    with caplog.at_level(logging.INFO, logger="autosub.agy"):
        agy.chat("agy/gemini", [{"content": "private prompt text"}])
    assert "agy request model=gemini elapsed_s=" in caplog.text
    assert "private prompt text" not in caplog.text


def test_timeout_and_failure_become_oserror(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired("agy", 1)
    monkeypatch.setattr(agy.subprocess, "run", slow)
    with pytest.raises(OSError):
        agy.chat("agy/m", [{"content": "x"}], timeout=1)
    bad = json.dumps({"status": "ERROR"})
    monkeypatch.setattr(agy.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, bad, ""))
    with pytest.raises(OSError):
        agy.chat("agy/m", [{"content": "x"}])


def test_refusal_falls_back_to_local_and_is_counted(monkeypatch):
    translate.SERVED.clear()
    monkeypatch.setattr(agy, "chat", lambda model, msgs: "Tôi không thể dịch nội dung này")
    monkeypatch.setattr(translate.ollama, "chat", lambda url, model, msgs, cfg=None: '["Anh yêu em"]')
    monkeypatch.setattr(translate, "check_fit", lambda *a: None)
    cfg = {"ollama_url": "", "models": {"vi": "agy/gemini"}, "fallback_model": "gemma3:4b"}
    assert translate.with_fallback(cfg, ["x"], [], "vi", set()) == ["Anh yêu em"]
    assert translate.SERVED == {"gemma3:4b": 1}


def test_cloud_refusal_is_counted_as_job_fallback(monkeypatch):
    monkeypatch.setattr(agy, "chat", lambda model, msgs: "I cannot assist with this request." if model == "agy/gemini" else '["translated"]')
    cfg = {"ollama_url": "", "models": {"en": "agy/gemini"}, "fallback_model": "agy/fallback"}
    stats = {"cloud_fallbacks": 0}
    assert translate.with_fallback(cfg, ["source"], [], "en", set(), fallback_stats=stats) == ["translated"]
    assert stats["cloud_fallbacks"] == 1


def test_served_counter_counts_parallel_windows(monkeypatch):
    translate.SERVED.clear()
    monkeypatch.setattr(agy, "chat", lambda model, msgs: '["Anh yêu em"]')
    cfg = {"ollama_url": "", "models": {"vi": "agy/gemini"}, "fallback_model": "agy/gemini"}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: translate.with_fallback(cfg, ["x"], [], "vi", set()), range(128)))
    assert all(result == ["Anh yêu em"] for result in results)
    assert translate.SERVED["agy/gemini"] == 128
