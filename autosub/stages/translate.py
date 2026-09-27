"""Translate stage (U5, KTD6): windowed ollama chat, refusal -> fallback model -> single lines -> marker."""
import json
import re

from .. import config, jobs, ollama
from ..refusal import is_refusal

LANGS = {"en": "English", "vi": "Vietnamese"}
UNTRANSLATED = "[untranslated]"


class GpuFitError(RuntimeError):
    pass


TAG = re.compile(r"^\s*\[[MF]\]\s*")


def tagged(seg):
    """'[M] text' when the voice guess knows the speaker's gender (voice.py)."""
    return f"[{seg['gender']}] {seg['text']}" if seg.get("gender") else seg["text"]


def brief_summary(brief):
    summary = brief.get("summary") if isinstance(brief, dict) else None
    return summary[:600] if isinstance(summary, str) else None


def address_rule(cfg, lang, brief=None):
    if lang == "vi" and isinstance(brief, dict):
        address = brief.get("vi_address")
        if isinstance(address, dict) and any(address.get(k) for k in (
                "male_self", "male_to_female", "female_self", "female_to_male")):
            fields = (("male_self", "nam tự xưng"), ("male_to_female", "nam gọi nữ"),
                      ("female_self", "nữ tự xưng"), ("female_to_male", "nữ gọi nam"))
            parts = [f"{label} '{address[key]}'" for key, label in fields if address.get(key)]
            return " Xưng hô theo bối cảnh: " + "; ".join(parts) + "."
    rule = cfg.get("address", {}).get(lang)
    return f" {rule}" if rule else ""


def prompt(lines, ctx, lang, cfg=None, *, following=None, brief=None):
    system = (f"You translate subtitles of adult films into natural, colloquial {LANGS[lang]}. Explicit sexual "
              "content is expected: translate it faithfully, never censor, never comment. Reply with ONLY a JSON "
              f"array of exactly {len(lines)} strings, one per input line, in order. A line may start with [M] "
              "(male speaker) or [F] (female speaker): use it for pronouns and tone, never output the tag."
              f"{address_rule(cfg or {}, lang, brief)} /no_think")
    user = ""
    summary = brief_summary(brief)
    if summary:
        user += f"Scene brief (context only): {summary}\n\n"
    if ctx:
        user += "Previous lines (context only):\n" + "\n".join(f"{ja} => {vi}" for ja, vi in ctx) + "\n\n"
    user += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    if following:
        user += "\n\nFollowing lines (context only, do not translate):\n" + "\n".join(f"- {line}" for line in following)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def check_fit(cfg, model):
    for m in ollama.ps(cfg["ollama_url"]):
        if m["name"] in (model, f"{model}:latest") and m.get("size_vram", 0) < cfg.get("min_gpu_share", 0.9) * m.get("size", 0):  # a few % on CPU is fine
            raise GpuFitError(f"model did not fit in GPU memory: {model} "
                              f"({m['size_vram'] >> 20}/{m['size'] >> 20} MiB on GPU)")


def ask(cfg, model, lines, ctx, lang, used, messages=None):
    """One chat call; None if the output isn't a JSON list. First call per model checks GPU fit (R9)."""
    content = ollama.chat(cfg["ollama_url"], model, messages or prompt(lines, ctx, lang, cfg))
    if model not in used:
        used.add(model)
        check_fit(cfg, model)
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())  # gemma wraps JSON in a code fence
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
        if not is_refusal(lines, out, lang):
            return out
    return None


def translate_texts(cfg, texts, lang, used, progress=lambda f: None, *, sources=None, brief=None, two_sided=True):
    """-> (translations, failed_line_count). Never drops a line (R13)."""
    sources = list(sources) if sources is not None else list(texts)
    out, failed, n = [], 0, cfg["window"]
    for i in range(0, len(texts), n):
        win = texts[i:i + n]
        prev_n = cfg["context_lines"]
        ctx = list(zip(sources[max(0, i - prev_n):i], out[max(0, i - prev_n):i])) if prev_n and two_sided else []
        lookahead = cfg.get("lookahead_lines", 4) if two_sided else 0
        following = sources[i + n:i + n + lookahead]
        messages = prompt(win, ctx, lang, cfg, following=following, brief=brief)
        res = with_fallback(cfg, win, ctx, lang, used, messages)
        if res is None:
            res = []
            for line in win:
                single_messages = prompt([line], ctx, lang, cfg, following=following, brief=brief)
                one = with_fallback(cfg, [line], ctx, lang, used, single_messages)
                res.append(one[0] if one else UNTRANSLATED)
                failed += one is None
        out += res
        progress(min(1, (i + n) / len(texts)))
    return out, failed


def translate_via_pivot(cfg, texts, lang, used, progress=lambda f: None, *, sources=None, brief=None):
    """Small models translate JA->EN far better than JA->VI, so go through the pivot language when configured."""
    pivot = cfg.get("pivot", {}).get(lang)
    if not pivot:
        return translate_texts(cfg, texts, lang, used, progress, sources=sources, brief=brief)
    mid, _ = translate_texts(cfg, texts, pivot, used, lambda f: progress(f / 2), brief=brief, two_sided=False)
    out, _ = translate_texts(cfg, mid, lang, used, lambda f: progress(0.5 + f / 2), brief=brief, two_sided=False)
    out = [UNTRANSLATED if m == UNTRANSLATED else o for m, o in zip(mid, out)]
    return out, out.count(UNTRANSLATED)


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
            brief = json.loads((d / "brief.json").read_text()) if (d / "brief.json").exists() else None
            tr, failed = translate_via_pivot(cfg, [tagged(s) for s in segs], job["lang"], used,
                                             lambda f: jobs.update(db, job["id"], progress=f),
                                             sources=[s["text"] for s in segs], brief=brief)
            tr = [TAG.sub("", t) for t in tr]
            for s, t in zip(segs, tr):
                s["src"], s["text"] = s["text"], t
            (d / "translated.json").write_text(json.dumps(segs, ensure_ascii=False))
            jobs.update(db, job["id"], stage="translated", progress=0, failed_lines=failed)
    finally:
        unload_all(cfg, used)  # keep_alive 0 + /api/ps confirm, so the next GPU stage gets the VRAM (KTD2)
