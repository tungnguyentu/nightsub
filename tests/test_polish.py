import json
import random
import re
import threading
import time

import pytest

from autosub import config, jobs, ollama
from autosub.stages import polish, translate


def make_cfg(model):
    cfg = dict(config.DEFAULTS)
    cfg["models"] = {**config.DEFAULTS["models"], "en": model}
    cfg["cloud_parallel"] = 4
    return cfg


def segments(count):
    return [{"src": f"SRC-{i}", "text": f"old-{i}", "logprob": -1.5}
            for i in range(count)]


def source_id(messages):
    match = re.search(r">> (SRC-\d+)  =>", messages[1]["content"])
    assert match
    return match.group(1)


def test_cloud_polish_is_parallel_and_assigns_results_to_their_lines(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()
    rng = random.Random(41)

    def chat(_model, messages):
        nonlocal active, max_active
        ident = source_id(messages)
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(rng.uniform(0.005, 0.04))
        with lock:
            active -= 1
        return json.dumps([f"rewritten-{ident}"])

    monkeypatch.setattr(translate.agy, "chat", chat)
    segs = segments(12)
    progress = []
    share = polish.polish(make_cfg("agy/test"), segs, "en", set(), progress=progress.append)

    assert max_active > 1
    assert share == 1.0
    assert [s["text"] for s in segs] == [f"rewritten-SRC-{i}" for i in range(12)]
    assert all(s["polished"] for s in segs)
    assert progress == sorted(progress) == [i / 12 for i in range(1, 13)]


def test_local_polish_calls_are_sequential(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def chat(_url, _model, messages, _cfg):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return json.dumps([f"rewritten-{source_id(messages)}"])

    monkeypatch.setattr(ollama, "chat", chat)
    monkeypatch.setattr(translate, "check_fit", lambda *_args: None)
    segs = segments(6)
    polish.polish(make_cfg("local-model"), segs, "en", set())

    assert max_active == 1
    assert [s["text"] for s in segs] == [f"rewritten-SRC-{i}" for i in range(6)]


def test_cloud_local_fallback_calls_share_the_gpu_lock(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(translate.agy, "chat", lambda *_args: '["I cannot help with that request"]')

    def local_chat(_url, _model, messages, _cfg):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return json.dumps([f"fallback-{source_id(messages)}"])

    monkeypatch.setattr(ollama, "chat", local_chat)
    monkeypatch.setattr(translate, "check_fit", lambda *_args: None)
    segs = segments(8)
    polish.polish(make_cfg("agy/test"), segs, "en", set())

    assert max_active == 1
    assert [s["text"] for s in segs] == [f"fallback-SRC-{i}" for i in range(8)]


def test_cloud_polish_propagates_stopped_from_progress(monkeypatch):
    monkeypatch.setattr(translate.agy, "chat", lambda _model, messages: json.dumps([f"rewritten-{source_id(messages)}"]))
    reports = []

    def stop_after_first(fraction):
        reports.append(fraction)
        raise jobs.Stopped()

    with pytest.raises(jobs.Stopped):
        polish.polish(make_cfg("agy/test"), segments(8), "en", set(), progress=stop_after_first)
    assert reports == [pytest.approx(1 / 8)]


def test_refused_cloud_rewrite_keeps_text_and_polished_false(monkeypatch):
    monkeypatch.setattr(translate.agy, "chat", lambda *_args: '["I cannot help with that request"]')
    monkeypatch.setattr(ollama, "chat", lambda *_args: "not-json")
    monkeypatch.setattr(translate, "check_fit", lambda *_args: None)
    segs = segments(1)

    polish.polish(make_cfg("agy/test"), segs, "en", set())

    assert segs[0]["text"] == "old-0"
    assert segs[0]["polished"] is False


def test_rewrite_that_brings_in_off_register_pronoun_is_rejected():
    from autosub.stages.polish import keep_rewrite
    allowed = {"bố", "con", "anh", "em"}
    assert not keep_rewrite(["Mày làm gì đấy"], "Anh làm gì đấy", allowed)
    assert keep_rewrite(["Anh đang làm gì vậy"], "Anh làm gì đấy", allowed)
    assert not keep_rewrite(["Anh làm gì đấy"], "Anh làm gì đấy", allowed)  # unchanged
    assert keep_rewrite(["I love you"], "x", None)  # English job: no pronoun rule


def test_agy_json_schema_is_passed(monkeypatch):
    import json
    import subprocess
    from autosub import agy
    seen = {}
    def run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "SUCCESS", "response": "{}"}), "")
    monkeypatch.setattr(agy.subprocess, "run", run)
    agy.chat("agy/m", [{"content": "x"}], json_schema={"type": "object"})
    assert any(a.startswith("--json-schema=") for a in seen["cmd"])
    agy.chat("agy/m", [{"content": "x"}])
    assert not any(a.startswith("--json-schema=") for a in seen["cmd"])
