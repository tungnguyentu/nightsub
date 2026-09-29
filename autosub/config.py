"""Defaults, overridable by ./autosub.toml (same keys, top level)."""
import os
import shutil
import tomllib
from pathlib import Path

DEFAULTS = {
    "work_dir": str(Path.home() / ".local/share/autosub"),
    "ollama_url": "http://127.0.0.1:11434",
    # Assumption (KTD6): verify tags/quality with scripts/bench.py.
    "models": {"en": "huihui_ai/qwen3-abliterated:8b", "vi": "gemma3:4b"},
    "fallback_model": "gemma3:4b",  # fits a 6 GB GPU fully; the 8B options do not
    "brief_model": None,  # defaults to the Vietnamese model so EN and VI jobs share one scene brief
    # Models allowed to run split between GPU and CPU (no num_gpu 99, no min_gpu_share check), e.g. a 12B
    # fallback that only sees the few lines the cloud refuses. Slower (~5x on a 6 GB card), never silent.
    "gpu_split_models": [],
    "min_gpu_share": 0.9,  # translate/polish fail if less of the model than this is on GPU (R9)
    "pivot": {},  # e.g. {"vi": "en"} to go JA -> en -> vi; measured worse with qwen3-4b (English leaks through)
    "asr_model": "kotoba-tech/kotoba-whisper-v2.0-faster",  # JA-tuned large-v3 distil: 2.6x faster, more lines caught
    "verify_asr_model": "large-v3",  # independent second ASR model for checking Kotoba transcripts
    "verify_agree": 0.6,  # normalized text similarity at or above this keeps the Kotoba result
    "verify_model": None,  # None uses models.vi to arbitrate ASR disagreements (agy/ is cloud-capable)
    "verify": True,  # set False to pass the Kotoba transcript through without a second ASR pass
    "asr_compute_type": "int8_float16",
    "asr_language": "ja",  # fixed source language; per-span auto-detect misfires on short lines (None = auto)
    "vram_needed_mb": 5000,
    "vram_wait_s": 30,
    "window": 12,
    "cloud_window": 24,
    "cloud_parallel": 4,
    "context_lines": 6,
    "lookahead_lines": 4,
    "llm_ctx": 4096,  # target RTX 3050: verify size_vram == size through Ollama /api/ps
    # Per-language style rule added to translate/polish prompts; keeps pronouns consistent across windows.
    "address": {"vi": "Xưng hô nhất quán suốt video: nhân vật nam xưng 'anh', gọi nữ là 'em'; nhân vật nữ xưng "
                      "'em', gọi nam là 'anh'. Không dùng 'tôi', 'bạn', 'mày', 'tao'."},
    "polish_logprob": -1.0,
    "polish_ratio": [0.5, 6.0],
    "max_line": 42,
    "cps": 17,
    "min_dur": 0.8,
}


def load(path="autosub.toml"):
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        cfg.update(tomllib.loads(p.read_text()))
    if cfg.get("brief_model") is None:
        cfg["brief_model"] = cfg["models"]["vi"]
    if os.environ.get("AUTOSUB_WORK_DIR"):  # set by the scheduler for stage subprocesses
        cfg["work_dir"] = os.environ["AUTOSUB_WORK_DIR"]
    return cfg


def db_path(cfg):
    Path(cfg["work_dir"]).mkdir(parents=True, exist_ok=True)
    return str(Path(cfg["work_dir"]) / "jobs.db")


def clear_job_dir(cfg, job_id):
    """SQLite reuses ids after a delete, so a new or deleted job must not inherit a previous job's files."""
    shutil.rmtree(Path(cfg["work_dir"]) / "jobs" / str(job_id), ignore_errors=True)


def job_dir(cfg, job_id):
    d = Path(cfg["work_dir"]) / "jobs" / str(job_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
