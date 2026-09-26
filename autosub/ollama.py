"""Tiny ollama HTTP client (stdlib only, local only)."""
import json
import re
import time
import urllib.request


def _req(url, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tags(url):
    return [m["name"] for m in _req(url, "/api/tags", timeout=5)["models"]]


def ps(url):
    return _req(url, "/api/ps", timeout=5)["models"]


def chat(url, model, messages):
    out = _req(url, "/api/chat", {"model": model, "messages": messages, "stream": False,
                                  "options": {"temperature": 0.3}})
    # Qwen3-style models may emit a thinking block; drop it.
    return re.sub(r"<think>.*?</think>", "", out["message"]["content"], flags=re.S).strip()


def unload(url, model, wait_s=10):
    """keep_alive 0, then confirm via /api/ps (KTD2)."""
    _req(url, "/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    for _ in range(wait_s * 2):
        if not any(m["name"] == model for m in ps(url)):
            return
        time.sleep(0.5)
