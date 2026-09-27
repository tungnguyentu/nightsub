"""Opt-in cloud translation through the Antigravity CLI (`agy --print`). Text sent here LEAVES the machine.

Used only for model names starting with PREFIX, e.g. "agy/gemini-3.8-flash-low"; everything else stays on
local ollama. Refusals and timeouts fall back to the local model like any other refusal (R13).
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PREFIX = "agy/"


def is_agy(model):
    return model.startswith(PREFIX)


def binary():
    return shutil.which("agy") or next((str(p) for p in [Path.home() / ".local/bin/agy"] if p.exists()), None)


def chat(model, messages, timeout=180):
    prompt = "\n\n".join(m["content"] for m in messages)  # --print takes one message
    with tempfile.TemporaryDirectory() as empty:  # empty workspace, no auto-approved tools: it can only answer
        try:
            p = subprocess.run([binary() or "agy", "--output-format", "json", "--disable-slash-commands",
                                "--model", model[len(PREFIX):], f"--print-timeout={timeout}s", f"--print={prompt}"],
                               cwd=empty, capture_output=True, text=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:  # OSError = the refusal/fallback path in translate
            raise OSError(f"agy timed out after {timeout}s") from None
    if p.returncode != 0:
        raise OSError(f"agy failed: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
    out = json.loads(p.stdout)
    if out.get("status") != "SUCCESS":
        raise OSError(f"agy status {out.get('status')}")
    return out.get("response") or ""
