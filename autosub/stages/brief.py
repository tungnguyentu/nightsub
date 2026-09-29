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


def _chat(cfg, prompt, used, json_mode=False):
    primary = cfg.get("brief_model") or cfg["models"]["vi"]
    for model in dict.fromkeys((primary, cfg["fallback_model"])):
        try:
            out = ollama.chat(cfg["ollama_url"], model, [
                {"role": "system", "content": "Summarize the Japanese dialogue faithfully. Do not invent facts. /no_think"},
                {"role": "user", "content": prompt},
            ], cfg, json_mode=json_mode)
        except OSError:
            continue
        used.add(model)
        if not out or PHRASES.search(out):
            continue
        return out
    return None


def make_brief(cfg, texts, used):
    facts = []
    for start in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[start:start + CHUNK_SIZE]
        prompt = ("Extract at most five short bullet facts about characters, relationships, setting, "
                  "and any explicit forms of address in this Japanese dialogue. Preserve uncertainty.\n\n" +
                  "\n".join(chunk))
        result = _chat(cfg, prompt, used)
        if result is None:
            raise ValueError("chunk summary refused or unavailable")
        facts.append(result)
    merged = _chat(cfg, "Return only one JSON object matching this schema: "
                   '{"summary":"...","characters":[{"name_or_role":"...","gender":"...",'
                   '"age_hint":"..."}],"relationship":"...","setting":"...",'
                   '"vi_address":{"male_self":"...","male_to_female":"...",'
                   '"female_self":"...","female_to_male":"..."}}. '
                   "Each vi_address value must be ONE Vietnamese pronoun such as anh, em, chị, cô, chú, "
                   "ông, bà, cháu, tôi, mình, sếp, chồng, vợ, chosen from the characters' relationship and ages; "
                   "never a speaker label. Use null for unknown address fields; summary must be concise. Facts:\n" + "\n".join(facts), used, json_mode=True)
    if merged is None:
        raise ValueError("brief merge refused or unavailable")
    data = json.loads(_content(merged))
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise ValueError("brief has invalid shape")
    if not isinstance(data.get("characters"), list) or not isinstance(data.get("relationship"), str):
        raise ValueError("brief has invalid shape")
    if not isinstance(data.get("setting"), str) or not isinstance(data.get("vi_address"), dict):
        raise ValueError("brief has invalid shape")
    data["summary"] = data["summary"][:MAX_CHARS]
    return data


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
