import shutil

from . import agy, gpu, ollama


def check(cfg):
    """Human-readable problems; empty list means ready (R15)."""
    problems = []
    if not shutil.which("nvidia-smi"):
        problems.append("nvidia-smi not found: NVIDIA driver missing?")
    else:
        try:
            free = gpu.free_mb()
            if free < cfg["vram_needed_mb"]:
                problems.append(f"Only {free} MiB GPU memory free, need {cfg['vram_needed_mb']}. "
                                f"Holders: {', '.join(gpu.holders()) or 'unknown'}")
        except Exception as e:
            problems.append(f"nvidia-smi failed: {e}")
    if not shutil.which("ffmpeg"):
        problems.append("ffmpeg not found on PATH")
    try:
        have = set(ollama.tags(cfg["ollama_url"]))
    except Exception as e:
        problems.append(f"ollama not reachable at {cfg['ollama_url']} ({e}). Start it with `ollama serve`.")
    else:
        wanted = {*cfg["models"].values(), cfg["fallback_model"], cfg.get("brief_model") or cfg["models"]["vi"]} | ({cfg["cloud_fallback_model"]} if cfg.get("cloud_fallback_model") else set())
        if any(agy.is_agy(m) for m in wanted) and not agy.binary():
            problems.append("an agy/ model is configured but the Antigravity CLI `agy` is not installed")
        for m in sorted(m for m in wanted if not agy.is_agy(m)):
            if m not in have and f"{m}:latest" not in have:
                problems.append(f"ollama model missing: {m} (run `ollama pull {m}`)")
    return problems
