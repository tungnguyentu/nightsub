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


def test_timeout_falls_back_like_refusal(monkeypatch):
    from autosub.stages import translate
    calls = []

    def ask(cfg, model, lines, ctx, lang, used, messages=None):
        calls.append(model)
        if model == "primary":
            raise TimeoutError("timed out")
        return ["ok"] * len(lines)
    monkeypatch.setattr(translate, "ask", ask)
    cfg = {"models": {"en": "primary"}, "fallback_model": "fb"}
    assert translate.with_fallback(cfg, ["あ"], [], "en", set()) == ["ok"]
    assert calls == ["primary", "fb"]


def test_97_percent_on_gpu_is_accepted(monkeypatch):
    fake(monkeypatch, lambda m, n: ok(n), ps=[{"name": "prim", "size": 100 << 20, "size_vram": 97 << 20}])
    from autosub.stages import translate
    out, failed = translate.translate_texts(CFG, ["あ"], "en", set())
    assert failed == 0


def test_vi_goes_through_english(monkeypatch):
    from autosub.stages import translate
    seen = []

    def tt(cfg, texts, lang, used, progress=lambda f: None):
        seen.append((lang, list(texts)))
        return [f"{lang}:{t}" if t != "bad" else translate.UNTRANSLATED for t in texts], 0
    monkeypatch.setattr(translate, "translate_texts", tt)
    out, failed = translate.translate_via_pivot({"pivot": {"vi": "en"}}, ["a", "bad"], "vi", set())
    assert seen[0] == ("en", ["a", "bad"]) and seen[1][0] == "vi"
    assert out == ["vi:en:a", translate.UNTRANSLATED] and failed == 1


def test_en_has_no_pivot(monkeypatch):
    from autosub.stages import translate
    monkeypatch.setattr(translate, "translate_texts", lambda cfg, t, lang, used, progress=None: ([lang] * len(t), 0))
    assert translate.translate_via_pivot({"pivot": {"vi": "en"}}, ["a"], "en", set()) == (["en"], 0)


def test_code_fenced_json_is_parsed(monkeypatch):
    from autosub.stages import translate
    monkeypatch.setattr(translate.ollama, "chat", lambda url, model, msgs: '```json\n["a"]\n```')
    monkeypatch.setattr(translate, "check_fit", lambda cfg, model: None)
    assert translate.ask({"ollama_url": ""}, "m", ["x"], [], "vi", set()) == ["a"]


def test_vietnamese_prompts_carry_the_address_rule():
    from autosub import config
    from autosub.stages import polish, translate
    cfg = dict(config.DEFAULTS)
    rule = cfg["address"]["vi"]
    assert rule in translate.prompt(["あ"], [], "vi", cfg)[0]["content"]
    assert rule not in translate.prompt(["あ"], [], "en", cfg)[0]["content"]
    segs = [{"src": "あ", "text": "a"}]
    assert rule in polish.rewrite_prompt(segs, 0, "vi", cfg)[0]["content"]
