"""Polish stage (U6): one rewrite call per flagged line, with +-2 lines of context."""
import json

from .. import config, jobs
from ..flags import is_flagged
from .translate import LANGS, UNTRANSLATED, address_rule, allowed_pronouns, brief_summary, unload_all, with_fallback


def rewrite_prompt(segs, i, lang, cfg=None, brief=None):
    ctx = "\n".join(f"{'>>' if j == i else '  '} {s['src']}  =>  {s['text']}"
                    for j, s in enumerate(segs[max(0, i - 2):i + 3], start=max(0, i - 2)))
    system = (f"You polish {LANGS[lang]} subtitles for adult films. Explicit content is expected; never censor. "
              "Rewrite the line marked >> so it is natural, colloquial and faithful to the source. "
              f"Reply with ONLY a JSON array of 1 string.{address_rule(cfg or {}, lang, brief)} /no_think")
    summary = brief_summary(brief)
    if summary:
        system += f" Scene brief (context only): {summary}"
    return [{"role": "system", "content": system}, {"role": "user", "content": ctx}]


def polish(cfg, segs, lang, used, progress=lambda f: None, brief=None):
    """Rewrites flagged lines in place; returns the flagged share. Refused rewrites keep the old line."""
    # ponytail: one call per flagged line; batch several per call if this stage dominates the bench.
    allowed = allowed_pronouns(lang, brief)
    idx = [i for i, s in enumerate(segs) if s["text"] != UNTRANSLATED and is_flagged(s, cfg, allowed)]
    for n, i in enumerate(idx):
        old = segs[i]["text"]
        segs[i]["polished"] = False
        out = with_fallback(cfg, [segs[i]["src"]], None, lang, used, rewrite_prompt(segs, i, lang, cfg, brief))
        if out and out[0] != old:
            segs[i]["text"] = out[0]
            segs[i]["polished"] = True
        progress((n + 1) / len(idx))
    return len(idx) / len(segs) if segs else 0.0


def run(cfg, db, batch):
    used = set()
    try:
        for job in batch:
            if jobs.get(db, job["id"])["control"]:
                continue
            try:
                d = config.job_dir(cfg, job["id"])
                segs = json.loads((d / "translated.json").read_text())
                brief = json.loads((d / "brief.json").read_text()) if (d / "brief.json").exists() else None
                share = polish(cfg, segs, job["lang"], used, lambda f: jobs.progress(db, job["id"], f), brief)
                (d / "polished.json").write_text(json.dumps(segs, ensure_ascii=False))
                jobs.update(db, job["id"], stage="polished", progress=0, flagged_share=share)
            except jobs.Stopped:  # paused/deleted from the UI: leave the stage to redo later
                continue
    finally:
        unload_all(cfg, used)
