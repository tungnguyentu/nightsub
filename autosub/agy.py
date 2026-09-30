"""Opt-in cloud translation through the Antigravity CLI (`agy --print`). Text sent here LEAVES the machine.

Used only for model names starting with PREFIX, e.g. "agy/gemini-3.8-flash-low"; everything else stays on
local ollama. Refusals and timeouts fall back to the local model like any other refusal (R13).
"""
import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PREFIX = "agy/"
log = logging.getLogger(__name__)


GROK = "grok/"  # same role through the Grok CLI (`grok -p`), e.g. "grok/grok-4.7"; also leaves the machine


def is_agy(model):
    return model.startswith((PREFIX, GROK))


def binary():
    return shutil.which("agy") or next((str(p) for p in [Path.home() / ".local/bin/agy"] if p.exists()), None)


def chat(model, messages, timeout=180, json_schema=None):
    prompt = "\n\n".join(m["content"] for m in messages)  # --print takes one message
    if model.startswith(GROK):
        name, _, effort = model[len(GROK):].partition("@")  # "grok/grok-4.7@low": reasoning effort
        return _grok(name, prompt, timeout, json_schema, effort)
    with tempfile.TemporaryDirectory() as empty:  # empty workspace, no auto-approved tools: it can only answer
        started = time.perf_counter()
        try:
            p = subprocess.run([binary() or "agy", "--output-format", "json", "--disable-slash-commands",
                                "--model", model[len(PREFIX):], f"--print-timeout={timeout}s",
                                *([f"--json-schema={json.dumps(json_schema)}"] if json_schema else []),
                                f"--print={prompt}"],
                               cwd=empty, capture_output=True, text=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:  # OSError = the refusal/fallback path in translate
            raise OSError(f"agy timed out after {timeout}s") from None
        finally:
            log.info("agy request model=%s elapsed_s=%.3f", model[len(PREFIX):], time.perf_counter() - started)
    if p.returncode != 0:
        raise OSError(f"agy failed: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
    out = json.loads(p.stdout)
    if out.get("status") != "SUCCESS":
        raise OSError(f"agy status {out.get('status')}")
    if json_schema and out.get("structured_output") is not None:  # schema answers can leave "response" empty
        return json.dumps(out["structured_output"], ensure_ascii=False)
    return out.get("response") or ""


def _grok(model, prompt, timeout, json_schema, effort=""):
    with tempfile.TemporaryDirectory() as empty:
        started = time.perf_counter()
        try:
            p = subprocess.run([shutil.which("grok") or "grok", "-m", model, "--output-format", "json",
                                "--disable-web-search", "--cwd", empty, *(["--effort", effort] if effort else []),
                                *(["--json-schema", json.dumps(json_schema)] if json_schema else []), "-p", prompt],
                               cwd=empty, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise OSError(f"grok timed out after {timeout}s") from None
        finally:
            log.info("grok request model=%s elapsed_s=%.3f", model, time.perf_counter() - started)
    if p.returncode != 0:
        raise OSError(f"grok failed: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
    out = json.loads(p.stdout)
    if out.get("stopReason") not in (None, "end_turn"):
        raise OSError(f"grok stop {out.get('stopReason')}")
    return out.get("text") or ""
