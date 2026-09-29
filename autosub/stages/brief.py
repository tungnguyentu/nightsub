"""Language-independent per-video scene brief, cached across language jobs."""
import hashlib
import json
import re
from pathlib import Path

from .. import config, jobs, ollama
from ..refusal import PHRASES
from .translate import unload_all

BRIEF_VERSION = "scene-brief-v1"
CHUNK_SIZE = 150
MAX_CHARS = 600


def cache_path(cfg, video):
    model = cfg.get("brief_model") or cfg["models"]["vi"]
    key = hashlib.sha1(
        f"{Path(video).resolve()}|{cfg['asr_model']}|{model}|{BRIEF_VERSION}".encode()
    ).hexdigest()[:16]
    p = Path(cfg["work_dir"]) / "brief-cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"


def _content(raw):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)


def _chat(cfg, prompt, used, json_mode=False, max_tokens=1024):
    primary = cfg.get("brief_model") or cfg["models"]["vi"]
    for model in dict.fromkeys((primary, cfg["fallback_model"])):
        try:
            out = ollama.chat(cfg["ollama_url"], model, [
                {"role": "system", "content": "Summarize the Japanese dialogue faithfully. Do not invent facts. /no_think"},
                {"role": "user", "content": prompt},
            ], cfg, json_mode=json_mode, max_tokens=max_tokens)
        except OSError:
            continue
        used.add(model)
        if not out or PHRASES.search(out):
            continue
        return out
    return None


# Chunk facts for a 2 h video (~20 chunks x ~900 chars) filled the whole 4096-token context, leaving the
# merge ~40 tokens of output (done_reason=length). Condense in groups until the notes fit comfortably.
MAX_FACT_CHARS = 5000
GROUP = 6


def condense(cfg, facts, used):
    while len("\n".join(facts)) > MAX_FACT_CHARS and len(facts) > 1:
        groups = [facts[i:i + GROUP] for i in range(0, len(facts), GROUP)]
        facts = [_chat(cfg, "Condense these notes about one video into at most eight short bullet facts about "
                            "characters, relationships, setting and forms of address. Keep names and who "
                            "addresses whom how.\n\n" + "\n".join(g), used) or "\n".join(g)[:MAX_FACT_CHARS // 2]
                 for g in groups]
    return facts


def make_brief(cfg, texts, used):
    facts = []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start:start + CHUNK_SIZE]
        prompt = ("Extract at most five short bullet facts about characters, relationships, setting, "
                  "and any explicit forms of address in this Japanese dialogue. Preserve uncertainty.\n\n" +
                  "\n".join(chunk))
        result = _chat(cfg, prompt, used)
        if result is not None:  # a refused chunk just contributes no facts
            facts.append(result)
    if not facts:
        raise ValueError("every chunk summary was refused or unavailable")
    facts = condense(cfg, facts, used)
    notes = "\n".join(facts)
    last = None
    for limit in ("summary: at most 2 sentences; relationship: 1 sentence; setting: a few words; "
                  "at most 6 characters",
                  "summary: 1 sentence; relationship: 1 short sentence; setting: 3 words; at most 3 characters"):
        merged = _chat(cfg, "Return only one JSON object matching this schema: "
                       '{"summary":"...","characters":[{"name_or_role":"...","gender":"...",'
                       '"age_hint":"..."}],"relationship":"...","setting":"...",'
                       '"vi_address":{"male_self":"...","male_to_female":"...",'
                       '"female_self":"...","female_to_male":"..."}}. '
                       f"Keep it short ({limit}). "
                       "Each vi_address value must be ONE Vietnamese pronoun such as anh, em, chị, cô, chú, "
                       "ông, bà, cháu, tôi, mình, sếp, chồng, vợ, chosen from the characters' relationship and "
                       "ages; never a speaker label. Use null for unknown address fields. Facts:\n" + notes,
                       used, json_mode=True, max_tokens=2048)
        if merged is None:
            raise ValueError("brief merge refused or unavailable")
        try:
            data = json.loads(_content(merged))
        except ValueError as e:  # cut off mid-string: ask again, shorter
            last = e
            continue
        if not isinstance(data, dict):
            raise ValueError("brief has invalid shape")
        return normalize(data)
    raise last


def _text(v):
    """4B models return lists/dicts where strings were asked for; flatten instead of discarding the brief."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "; ".join(filter(None, map(_text, v)))
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_text(x)}" for k, x in v.items() if _text(x))
    return "" if v is None else str(v)


def normalize(data):
    """Coerce a model-made brief to the expected shape; only an empty brief is rejected."""
    chars = data.get("characters")
    chars = [c for c in (chars if isinstance(chars, list) else [chars]) if isinstance(c, dict)]
    out = {"summary": _text(data.get("summary"))[:MAX_CHARS], "characters": chars,
           "relationship": _text(data.get("relationship")), "setting": _text(data.get("setting")),
           "vi_address": data.get("vi_address") if isinstance(data.get("vi_address"), dict) else {}}
    if not (out["summary"] or out["relationship"]):
        raise ValueError("brief has no summary or relationship")
    return out


def run(cfg, db, batch):
    used = set()
    try:
        for job in batch:
            if jobs.get(db, job["id"])["control"]:
                continue
            try:
                d = config.job_dir(cfg, job["id"])
                cache = cache_path(cfg, job["video"])
                if cache.exists():
                    (d / "brief.json").write_text(cache.read_text())
                else:
                    try:
                        segs = json.loads((d / "segments.json").read_text())
                        brief = make_brief(cfg, [s["text"] for s in segs], used)
                    except Exception as e:
                        brief = {"skipped": f"{type(e).__name__}: {e}"[:300]}
                    rendered = json.dumps(brief, ensure_ascii=False)
                    (d / "brief.json").write_text(rendered)
                    cache.write_text(rendered)
                jobs.update(db, job["id"], stage="briefed", progress=0)
            except jobs.Stopped:  # paused/deleted from the UI: leave the stage to redo later
                continue
    finally:
        unload_all(cfg, used)
