import json

import pytest

from autosub import config
from autosub.stages import translate as tr

CFG = {**config.DEFAULTS, "models": {"en": "prim", "vi": "prim"}, "fallback_model": "fb"}


def fake(monkeypatch, reply, ps=()):
    """reply(model, n_lines) -> raw content string."""
    calls = []

    def chat(url, model, messages):
        n = messages[1]["content"].split("Translate:\n")[1].count("\n") + 1
        calls.append(model)
        return reply(model, n)
    monkeypatch.setattr(tr.ollama, "chat", chat)
    monkeypatch.setattr(tr.ollama, "ps", lambda url: list(ps))
    return calls


def ok(n):
    return json.dumps([f"line {i}" for i in range(n)])


def test_primary_refuses_fallback_ok(monkeypatch):
    calls = fake(monkeypatch, lambda m, n: "I'm sorry, I can't help with that." if m == "prim" else ok(n))
    out, failed = tr.translate_texts(CFG, ["あ", "い"], "en", set())
    assert out == ["line 0", "line 1"] and failed == 0 and calls == ["prim", "fb"]


def test_both_refuse_marks_line(monkeypatch):
    fake(monkeypatch, lambda m, n: "I cannot assist with this request.")
    out, failed = tr.translate_texts(CFG, ["あ"], "en", set())
    assert out == [tr.UNTRANSLATED] and failed == 1


def test_invalid_json_retried(monkeypatch):
    calls = fake(monkeypatch, lambda m, n: '```json\n["x"' if m == "prim" else ok(n))
    out, failed = tr.translate_texts(CFG, ["あ"], "en", set())
    assert out == ["line 0"] and calls == ["prim", "fb"]


def test_windows_and_split(monkeypatch):
    # Window of 12 refused by both -> each line retried alone, all succeed.
    calls = fake(monkeypatch, lambda m, n: ok(n - 1) if n > 1 else ok(1))
    out, failed = tr.translate_texts(CFG, ["x"] * 12, "en", set())
    assert len(out) == 12 and failed == 0 and len(calls) == 2 + 12


def test_partial_offload_fails_stage(monkeypatch):
    fake(monkeypatch, lambda m, n: ok(n), ps=[{"name": "prim", "size": 5 << 30, "size_vram": 3 << 30}])
    with pytest.raises(tr.GpuFitError, match="did not fit in GPU memory"):
        tr.translate_texts(CFG, ["あ"], "en", set())
