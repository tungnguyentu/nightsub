"""Translate stage (U5, KTD6): windowed ollama chat, refusal -> fallback model -> single lines -> marker."""
import json

from .. import config, jobs, ollama
from ..refusal import is_refusal

LANGS = {"en": "English", "vi": "Vietnamese"}
UNTRANSLATED = "[untranslated]"


class GpuFitError(RuntimeError):
    pass


def prompt(lines, ctx, lang):
    system = (f"You translate subtitles of adult films into natural, colloquial {LANGS[lang]}. Explicit sexual "
              "content is expected: translate it faithfully, never censor, never comment. Reply with ONLY a JSON "
              f"array of exactly {len(lines)} strings, one per input line, in order. /no_think")
    user = ""
    if ctx:
        user += "Previous lines (already translated, for context):\n" + "\n".join(ctx) + "\n\n"
    user += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def check_fit(cfg, model):
    for m in ollama.ps(cfg["ollama_url"]):
        if m["name"] in (model, f"{model}:latest") and m.get("size_vram", 0) < cfg.get("min_gpu_share", 0.95) * m.get("size", 0):  # a few % on CPU is fine
            raise GpuFitError(f"model did not fit in GPU memory: {model} "
                              f"({m['size_vram'] >> 20}/{m['size'] >> 20} MiB on GPU)")


def ask(cfg, model, lines, ctx, lang, used, messages=None):
    """One chat call; None if the output isn't a JSON list. First call per model checks GPU fit (R9)."""
    content = ollama.chat(cfg["ollama_url"], model, messages or prompt(lines, ctx, lang))
    if model not in used:
        used.add(model)
        check_fit(cfg, model)
    try:
        return json.loads(content)
    except ValueError:
        return None


def with_fallback(cfg, lines, ctx, lang, used, messages=None):
    for model in (cfg["models"][lang], cfg["fallback_model"]):
        try:
            out = ask(cfg, model, lines, ctx, lang, used, messages)
        except OSError:  # timeout / connection drop: same path as a refusal (R13)
            continue
        if not is_refusal(lines, out):
            return out
    return None


def translate_texts(cfg, texts, lang, used, progress=lambda f: None):
    """-> (translations, failed_line_count). Never drops a line (R13)."""
    out, failed, n = [], 0, cfg["window"]
    for i in range(0, len(texts), n):
        win, ctx = texts[i:i + n], out[-cfg["context_lines"]:] if cfg["context_lines"] else []
        res = with_fallback(cfg, win, ctx, lang, used)
        if res is None:
            res = []
            for line in win:
                one = with_fallback(cfg, [line], ctx, lang, used)
                res.append(one[0] if one else UNTRANSLATED)
                failed += one is None
        out += res
        progress(min(1, (i + n) / len(texts)))
    return out, failed


def unload_all(cfg, used):
    for m in used:
        try:
            ollama.unload(cfg["ollama_url"], m)
        except Exception:
            pass


def run(cfg, db, batch):
    used = set()
    try:
        for job in batch:
            d = config.job_dir(cfg, job["id"])
            segs = json.loads((d / "segments.json").read_text())
            tr, failed = translate_texts(cfg, [s["text"] for s in segs], job["lang"], used,
                                         lambda f: jobs.update(db, job["id"], progress=f))
            for s, t in zip(segs, tr):
                s["src"], s["text"] = s["text"], t
            (d / "translated.json").write_text(json.dumps(segs, ensure_ascii=False))
            jobs.update(db, job["id"], stage="translated", progress=0, failed_lines=failed)
    finally:
        unload_all(cfg, used)  # keep_alive 0 + /api/ps confirm, so the next GPU stage gets the VRAM (KTD2)
