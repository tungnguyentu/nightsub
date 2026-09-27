import json

import pytest

from autosub import config
from autosub.stages import translate as tr

CFG = {**config.DEFAULTS, "models": {"en": "prim", "vi": "prim"}, "fallback_model": "fb"}


def fake(monkeypatch, reply, ps=()):
    """reply(model, n_lines) -> raw content string."""
    calls = []

    def chat(url, model, messages, cfg=None):
        window = messages[1]["content"].split("Translate:\n")[1].split("\n\nFollowing lines", 1)[0]
        n = window.count("\n") + 1
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

    def tt(cfg, texts, lang, used, progress=lambda f: None, **kwargs):
        seen.append((lang, list(texts)))
        return [f"{lang}:{t}" if t != "bad" else translate.UNTRANSLATED for t in texts], 0
    monkeypatch.setattr(translate, "translate_texts", tt)
    out, failed = translate.translate_via_pivot({"pivot": {"vi": "en"}}, ["a", "bad"], "vi", set())
    assert seen[0] == ("en", ["a", "bad"]) and seen[1][0] == "vi"
    assert out == ["vi:en:a", translate.UNTRANSLATED] and failed == 1


def test_en_has_no_pivot(monkeypatch):
    from autosub.stages import translate
    monkeypatch.setattr(translate, "translate_texts", lambda cfg, t, lang, used, progress=None, **kwargs: ([lang] * len(t), 0))
    assert translate.translate_via_pivot({"pivot": {"vi": "en"}}, ["a"], "en", set()) == (["en"], 0)


def test_code_fenced_json_is_parsed(monkeypatch):
    from autosub.stages import translate
    monkeypatch.setattr(translate.ollama, "chat", lambda url, model, msgs, cfg=None: '```json\n["a"]\n```')
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


def test_brief_address_and_two_sided_context_order(monkeypatch):
    cfg = {**CFG, "window": 2, "context_lines": 1, "lookahead_lines": 1}
    brief = {"summary": "Hai người là đồng nghiệp.", "vi_address": {"male_self": "tôi", "male_to_female": "chị",
                                                                   "female_self": "chị", "female_to_male": "tôi"}}
    messages_seen = []
    monkeypatch.setattr(tr.ollama, "ps", lambda url: [])
    def chat(url, model, messages, cfg=None):
        messages_seen.append(messages)
        n = messages[1]["content"].split("Translate:\n")[1].split("\n\nFollowing lines", 1)[0].count("\n") + 1
        return ok(n)
    monkeypatch.setattr(tr.ollama, "chat", chat)
    out, failed = tr.translate_texts(cfg, ["[M] A", "[F] B", "[M] C", "[F] D"], "vi", set(),
                                     sources=["JA1", "JA2", "JA3", "JA4"], brief=brief)
    first_user = messages_seen[0][1]["content"]
    user = messages_seen[1][1]["content"]
    assert failed == 0 and len(out) == 4
    assert user.index("Hai người là đồng nghiệp") < user.index("JA2 => line 1")
    assert user.index("JA2 => line 1") < user.index("Translate:")
    assert first_user.index("Hai người là đồng nghiệp") < first_user.index("Translate:") < first_user.index("JA3")
    assert "Xưng hô theo bối cảnh" in messages_seen[0][0]["content"]
    assert "tôi" in messages_seen[0][0]["content"]
    assert "Following lines (context only, do not translate)" not in messages_seen[-1][1]["content"]


def test_skipped_brief_uses_static_vi_rule_and_en_has_no_address():
    cfg = dict(config.DEFAULTS)
    rule = cfg["address"]["vi"]
    assert rule in tr.prompt(["あ"], [], "vi", cfg, brief={"skipped": "model error"})[0]["content"]
    assert not tr.address_rule(cfg, "en", {"vi_address": {"male_self": "anh"}})
    from autosub.stages import polish
    segs = [{"src": "あ", "text": "a"}]
    prompt = polish.rewrite_prompt(segs, 0, "vi", cfg, {"summary": "Hai đồng nghiệp."})
    assert "Hai đồng nghiệp." in prompt[0]["content"] and "Xưng hô theo bối cảnh" not in prompt[0]["content"]


def test_window_response_count_excludes_following_context(monkeypatch):
    cfg = {**CFG, "window": 2, "context_lines": 0, "lookahead_lines": 2}
    fake(monkeypatch, lambda model, n: ok(n))
    out, failed = tr.translate_texts(cfg, ["a", "b", "c", "d"], "en", set(), sources=["JA"] * 4)
    assert len(out) == 4 and failed == 0


def test_chat_uses_configured_context(monkeypatch):
    from autosub import ollama
    seen = {}
    def request(url, path, body, timeout):
        seen.update(body)
        return {"message": {"content": "ok"}}
    monkeypatch.setattr(ollama, "_req", request)
    ollama.chat("http://local", "model", [], {"llm_ctx": 3072})
    assert seen["options"]["num_ctx"] == 3072


def test_gender_tag_goes_in_and_is_stripped_out():
    from autosub.stages import translate
    assert translate.tagged({"text": "あ", "gender": "F"}) == "[F] あ"
    assert translate.tagged({"text": "あ", "gender": None}) == "あ"
    assert translate.TAG.sub("", "[F] Anh ơi") == "Anh ơi"
    assert translate.TAG.sub("", "Anh [M] ơi") == "Anh [M] ơi"  # only a leading tag


def test_speaker_labels_in_vi_address_fall_back_to_static_rule():
    from autosub import config
    from autosub.stages import translate
    cfg = dict(config.DEFAULTS)
    bad = {"vi_address": {"male_self": "Speaker 2", "male_to_female": "Speaker 1",
                          "female_self": "Speaker 1", "female_to_male": "Speaker 2"}}
    assert translate.address_rule(cfg, "vi", bad) == f" {cfg['address']['vi']}"
    good = {"vi_address": {"male_self": "Chú", "male_to_female": "cháu", "female_self": "cháu",
                           "female_to_male": "chú"}}
    rule = translate.address_rule(cfg, "vi", good)
    assert "'chú'" in rule and "'cháu'" in rule and "Speaker" not in rule


def test_inconsistent_or_partial_pair_falls_back():
    from autosub import config
    from autosub.stages import translate
    cfg = dict(config.DEFAULTS)
    static = f" {cfg['address']['vi']}"
    mixed = {"vi_address": {"male_self": "tôi", "male_to_female": "em", "female_self": "chị", "female_to_male": "ông"}}
    assert translate.address_rule(cfg, "vi", mixed) == static  # the real 4B output from the clip run
    assert translate.address_rule(cfg, "vi", {"vi_address": {"male_self": "anh"}}) == static
    assert translate.allowed_pronouns("vi", mixed) == {"anh", "em"}


def test_off_register_pronoun_is_flagged_for_polish():
    from autosub import config, flags
    cfg = dict(config.DEFAULTS)
    seg = {"src": "おまえ何してる", "text": "Mày đang làm gì vậy", "logprob": -0.1}
    assert flags.is_flagged(seg, cfg, {"anh", "em"})
    assert not flags.is_flagged({**seg, "text": "Anh đang làm gì vậy"}, cfg, {"anh", "em"})
    assert not flags.is_flagged(seg, cfg, None)  # English jobs: no pronoun check
    assert not flags.is_flagged({**seg, "text": "Tôi đang làm gì vậy"}, cfg, {"tôi", "chị"})
