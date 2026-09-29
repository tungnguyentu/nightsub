import json
import random
import threading
import time

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


def prompt_lines(messages):
    block = messages[1]["content"].split("Translate:\n", 1)[1].split("\n\nFollowing lines", 1)[0]
    return [line.split(". ", 1)[1] for line in block.splitlines()]


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


def test_cloud_windows_run_concurrently_and_preserve_order_with_source_context(monkeypatch):
    cfg = {**CFG, "models": {"en": "agy/cloud"}, "fallback_model": "agy/fallback",
           "cloud_window": 2, "cloud_parallel": 3, "context_lines": 2, "lookahead_lines": 2}
    rng = random.Random(42)
    delays = [0.01, 0.03, 0.06]
    rng.shuffle(delays)
    messages_seen, completed, values, callback_threads = [], [], [], []
    state_lock = threading.Lock()
    active = max_active = 0
    caller = threading.get_ident()

    def chat(model, messages):
        nonlocal active, max_active
        lines = prompt_lines(messages)
        with state_lock:
            messages_seen.append(messages)
            active += 1
            max_active = max(max_active, active)
        time.sleep(delays[int(lines[0].rsplit("-", 1)[1]) // 2])
        with state_lock:
            completed.append(int(lines[0].rsplit("-", 1)[1]) // 2)
            active -= 1
        return json.dumps([f"This is the translation: {line}" for line in lines])

    monkeypatch.setattr(tr.agy, "chat", chat)
    texts = [f"source-{i}" for i in range(6)]
    brief = {"summary": "Context shared by the windows."}
    out, failed = tr.translate_texts(cfg, texts, "en", set(),
                                     lambda f: (values.append(f), callback_threads.append(threading.get_ident())),
                                     sources=texts, brief=brief)
    assert out == [f"This is the translation: {line}" for line in texts] and failed == 0
    assert max_active == 3 and completed != [0, 1, 2]
    assert values == sorted(values) and values[-1] == 1.0 and set(callback_threads) == {caller}
    middle = next(m for m in messages_seen if "Translate:\n1. source-2" in m[1]["content"])
    user = middle[1]["content"]
    assert user.index("Context shared by the windows") < user.index("Previous Japanese source lines")
    assert user.index("Previous Japanese source lines") < user.index("Translate:") < user.index("Following lines")
    assert "source-0" in user and "source-1" in user and "source-4" in user and "=>" not in user


def test_cloud_window_24_and_local_window_12(monkeypatch):
    cloud_counts, local_counts = [], []
    monkeypatch.setattr(tr.agy, "chat", lambda model, messages: cloud_counts.append(len(prompt_lines(messages))) or
                        json.dumps(["ok"] * len(prompt_lines(messages))))
    monkeypatch.setattr(tr.ollama, "chat", lambda url, model, messages, cfg=None: local_counts.append(len(prompt_lines(messages))) or
                        json.dumps(["ok"] * len(prompt_lines(messages))))
    monkeypatch.setattr(tr.ollama, "ps", lambda url: [])
    monkeypatch.setattr(tr, "check_fit", lambda cfg, model: None)
    texts = ["source"] * 49
    base = {**CFG, "cloud_window": 24, "window": 12, "context_lines": 0, "lookahead_lines": 0}
    tr.translate_texts({**base, "models": {"en": "agy/cloud"}, "fallback_model": "agy/fallback"}, texts, "en", set())
    tr.translate_texts({**base, "models": {"en": "local"}, "fallback_model": "local"}, texts, "en", set())
    assert cloud_counts == [24, 24, 1]
    assert local_counts == [12, 12, 12, 12, 1]


def test_cloud_refused_window_retries_lines_then_marks_only_failed_lines(monkeypatch):
    cfg = {**CFG, "models": {"en": "agy/cloud"}, "fallback_model": "agy/fallback",
           "cloud_window": 2, "cloud_parallel": 2, "context_lines": 0, "lookahead_lines": 0}
    calls = []
    def chat(model, messages):
        lines = prompt_lines(messages)
        calls.append((model, len(lines)))
        if len(lines) > 1:
            return "I cannot assist with this request."
        return json.dumps([f"translated {lines[0]}"])
    monkeypatch.setattr(tr.agy, "chat", chat)
    tr.SERVED.clear()
    out, failed = tr.translate_texts(cfg, ["a", "b", "c", "d"], "en", set())
    assert out == ["translated a", "translated b", "translated c", "translated d"] and failed == 0
    assert not any(m == "agy/fallback" for m, _ in calls)  # bisect: cloud translated every single line itself
    assert calls.count(("agy/cloud", 1)) == 4


def test_cloud_config_defaults():
    assert config.DEFAULTS["cloud_window"] == 24
    assert config.DEFAULTS["cloud_parallel"] == 4


def test_local_fallback_calls_are_serialized_for_parallel_cloud_windows(monkeypatch):
    cfg = {**CFG, "models": {"vi": "agy/cloud"}, "fallback_model": "gemma3:4b",
           "cloud_window": 2, "cloud_parallel": 4, "context_lines": 0, "lookahead_lines": 0}
    active = max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(tr.agy, "chat", lambda model, messages: "I cannot assist with this request.")
    monkeypatch.setattr(tr.ollama, "ps", lambda url: [])
    monkeypatch.setattr(tr, "check_fit", lambda cfg, model: None)
    def local_chat(url, model, messages, cfg=None):
        nonlocal active, max_active
        n = len(prompt_lines(messages))
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.015)
        with lock:
            active -= 1
        return json.dumps(["Anh yêu em" for _ in range(n)])
    monkeypatch.setattr(tr.ollama, "chat", local_chat)
    out, failed = tr.translate_texts(cfg, ["source"] * 12, "vi", set())
    assert len(out) == 12 and failed == 0 and max_active == 1


def test_stopped_progress_cancels_pool_and_propagates(monkeypatch):
    from autosub import jobs
    cfg = {**CFG, "models": {"en": "agy/cloud"}, "fallback_model": "agy/fallback",
           "cloud_window": 1, "cloud_parallel": 2, "context_lines": 0, "lookahead_lines": 0}
    monkeypatch.setattr(tr.agy, "chat", lambda model, messages: json.dumps(["translated"]))
    calls = []
    def progress(fraction):
        calls.append(fraction)
        raise jobs.Stopped("pause")
    with pytest.raises(jobs.Stopped):
        tr.translate_texts(cfg, ["source"] * 8, "en", set(), progress)
    assert len(calls) == 1 and calls[0] > 0


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
    assert translate.address_rule(cfg, "vi", bad) == f" {cfg['address']['vi']}" + translate.KINSHIP_NOTE
    good = {"vi_address": {"male_self": "Chú", "male_to_female": "cháu", "female_self": "cháu",
                           "female_to_male": "chú"}}
    rule = translate.address_rule(cfg, "vi", good)
    assert "'chú'" in rule and "'cháu'" in rule and "Speaker" not in rule


def test_inconsistent_or_partial_pair_falls_back():
    from autosub import config
    from autosub.stages import translate
    cfg = dict(config.DEFAULTS)
    static = f" {cfg['address']['vi']}" + translate.KINSHIP_NOTE
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


def test_bisect_sends_only_the_line_the_cloud_refuses_alone_to_local(monkeypatch):
    cfg = {**CFG, "models": {"vi": "agy/cloud"}, "fallback_model": "local:4b",
           "cloud_window": 8, "cloud_parallel": 1, "context_lines": 0, "lookahead_lines": 0}
    def cloud(model, messages):
        lines = prompt_lines(messages)
        return "Tôi không thể dịch nội dung này" if "x" in lines else json.dumps([f"dịch {l}" for l in lines])
    local_calls = []
    def local(url, model, messages, cfg=None):
        local_calls.append(prompt_lines(messages))
        return json.dumps(["bản địa"])
    monkeypatch.setattr(tr.agy, "chat", cloud)
    monkeypatch.setattr(tr.ollama, "chat", local)
    monkeypatch.setattr(tr, "check_fit", lambda *a: None)
    stats = {"cloud_fallbacks": 0}
    texts = ["a", "b", "c", "x", "e", "f", "g", "h"]
    out, failed = tr.translate_texts(cfg, texts, "vi", set(), fallback_stats=stats)
    assert out == ["dịch a", "dịch b", "dịch c", "bản địa", "dịch e", "dịch f", "dịch g", "dịch h"]
    assert failed == 0 and local_calls == [["x"]] and stats["cloud_fallbacks"] == 1


def test_vietnamese_rule_keeps_kinship_words():
    from autosub import config
    rule = tr.address_rule(dict(config.DEFAULTS), "vi", None)
    assert "お父さん" in rule and "bố" in rule
    assert "お父さん" not in tr.address_rule(dict(config.DEFAULTS), "en", None)
