import shutil

from . import gpu, ollama


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
        for m in sorted({*cfg["models"].values(), cfg["fallback_model"]}):
            if m not in have and f"{m}:latest" not in have:
                problems.append(f"ollama model missing: {m} (run `ollama pull {m}`)")
    return problems
