"""Polish stage (U6): one rewrite call per flagged line, with +-2 lines of context."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .. import agy, config, jobs
from ..flags import is_flagged, wrong_pronoun
from .translate import LANGS, UNTRANSLATED, address_rule, allowed_pronouns, with_kinship, brief_summary, unload_all, with_fallback


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


def keep_rewrite(out, old, allowed):
    """Accept a rewrite only if it changed the line and did not bring in an off-register pronoun
    (HMN-904: 17 of 21 mày/tao lines came from polish rewrites)."""
    if not out or out[0] == old:
        return False
    return allowed is None or not wrong_pronoun(out[0], allowed)


def polish(cfg, segs, lang, used, progress=lambda f: None, brief=None):
    """Rewrites flagged lines in place; returns the flagged share. Refused rewrites keep the old line."""
    allowed = allowed_pronouns(lang, brief)
    idx = [i for i, s in enumerate(segs) if s["text"] != UNTRANSLATED and is_flagged(s, cfg, allowed)]
    if not idx:
        return len(idx) / len(segs) if segs else 0.0

    cloud = agy.is_agy(cfg["models"][lang])
    if not cloud:
        # Keep local-model behavior sequential: prompts see prior rewrites as before.
        for n, i in enumerate(idx):
            old = segs[i]["text"]
            segs[i]["polished"] = False
            out = with_fallback(cfg, [segs[i]["src"]], None, lang, used,
                                rewrite_prompt(segs, i, lang, cfg, brief))
            if keep_rewrite(out, old, allowed):
                segs[i]["text"] = out[0]
                segs[i]["polished"] = True
            progress((n + 1) / len(idx))
        return len(idx) / len(segs) if segs else 0.0

    # Build all prompts before workers start, so each rewrite gets its own original
    # line plus the same neighboring context regardless of which requests finish first.
    tasks = [(i, segs[i]["text"], rewrite_prompt(segs, i, lang, cfg, brief)) for i in idx]
    for i, _, _ in tasks:
        segs[i]["polished"] = False

    progress_lock = threading.Lock()
    last_progress = 0.0

    def report_progress(fraction):
        nonlocal last_progress
        with progress_lock:
            last_progress = max(last_progress, fraction)
            progress(last_progress)

    def rewrite(i, old, messages):
        out = with_fallback(cfg, [segs[i]["src"]], None, lang, used, messages)
        return i, old, out

    pool = ThreadPoolExecutor(max_workers=max(1, int(cfg.get("cloud_parallel", 4))))
    futures = [pool.submit(rewrite, i, old, messages) for i, old, messages in tasks]
    try:
        for finished, future in enumerate(as_completed(futures), start=1):
            i, old, out = future.result()
            if keep_rewrite(out, old, allowed):
                segs[i]["text"] = out[0]
                segs[i]["polished"] = True
            report_progress(finished / len(idx))
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
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
                brief = with_kinship(brief, segs)
                share = polish(cfg, segs, job["lang"], used, lambda f: jobs.progress(db, job["id"], f), brief)
                (d / "polished.json").write_text(json.dumps(segs, ensure_ascii=False))
                jobs.update(db, job["id"], stage="polished", progress=0, flagged_share=share)
            except jobs.Stopped:  # paused/deleted from the UI: leave the stage to redo later
                continue
    finally:
        unload_all(cfg, used)
