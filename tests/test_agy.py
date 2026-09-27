import json
import subprocess

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
