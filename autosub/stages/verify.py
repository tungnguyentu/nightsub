"""Second-model ASR check with optional LLM arbitration for disagreements."""
import difflib
import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import agy, jobs, ollama
from ..refusal import PHRASES
from . import cli, read_wav

SR = 16000
VERIFY_VERSION = "asr-crosscheck-v1"
ARBITRATION_BATCH = 20
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def load(cfg):
    if not cfg.get("verify", True):
        return None
    from faster_whisper import WhisperModel
    return WhisperModel(cfg.get("verify_asr_model", "large-v3"), device="cuda",
                        compute_type=cfg["asr_compute_type"])


def cache_path(cfg, video):
    """Cache the verified transcript per video and both model choices."""
    asr_model = cfg["asr_model"]
    verify_asr = cfg.get("verify_asr_model", "large-v3")
    verify_model = cfg.get("verify_model") or cfg["models"]["vi"]
    key = hashlib.sha1(
        f"{Path(video).resolve()}|{asr_model}|{verify_asr}|{verify_model}|"
        f"{cfg['asr_language']}|{cfg.get('verify_agree', 0.6)}|{VERIFY_VERSION}".encode()
    ).hexdigest()[:16]
    directory = Path(cfg["work_dir"]) / "verify-cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def normalize(text):
    """Remove Unicode punctuation and whitespace before comparing ASR candidates."""
    return "".join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def similarity(left, right):
    return difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def name_hints(brief):
    chars = brief.get("characters", []) if isinstance(brief, dict) else []
    if isinstance(chars, dict):
        chars = [chars]
    names = []
    for character in chars if isinstance(chars, list) else []:
        value = character.get("name_or_role") if isinstance(character, dict) else None
        if isinstance(value, str) and 0 < len(value.strip()) <= 40:
            name = value.strip()
            if name not in names:
                names.append(name)
    return "、".join(names[:8])


def too_short(text, cfg):
    """Interjections (64% of HMN-904's lines) are skipped; re-hearing them cost most of a 75-minute verify pass."""
    return len(re.sub(r"[\W_]+", "", text or "")) <= cfg.get("verify_min_chars", 4)


def transcribe_segment(model, audio, segment, cfg, prompt=None):
    start, end = float(segment["start"]), float(segment["end"])
    clip = audio[max(0, int(start * SR)):max(0, int(end * SR))]
    kwargs = {"language": cfg["asr_language"], "condition_on_previous_text": False,
              "beam_size": cfg.get("verify_beam", 1)}  # a disagreement check, not the final transcript: greedy is enough
    if prompt:
        kwargs["initial_prompt"] = prompt
    try:
        parts = list(model.transcribe(clip, **kwargs)[0])
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        parts = list(model.transcribe(clip, **{**kwargs, "beam_size": 1})[0])
    return "".join(part.text.strip() for part in parts).strip()


def arbitration_prompt(brief, items, segs, accepted):
    summary = brief.get("summary", "") if isinstance(brief, dict) else ""
    lines = []
    for number, (index, second_text) in enumerate(items, 1):
        previous = [segs[i]["text"] for i in accepted if i < index][-3:]
        following = [segs[i]["text"] for i in accepted if i > index][:3]
        lines.append(
            f"{number}. Previous accepted Japanese lines (context only): {json.dumps(previous, ensure_ascii=False)}\n"
            f"Kotoba candidate: {segs[index]['text']}\nSecond-model candidate: {second_text}\n"
            f"Following accepted Japanese lines (context only): {json.dumps(following, ensure_ascii=False)}"
        )
    system = ("Choose the most plausible original Japanese line for each item using the scene brief and context. "
              "You may select either candidate verbatim or merge them, but output Japanese only, never translate. "
              f"Reply with ONLY a JSON array of exactly {len(items)} strings in order.")
    user = (f"Scene brief: {summary or '(none)'}\n\n" + "\n\n".join(lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def arbitrate(cfg, model, messages, count):
    """Return validated Japanese selections, or None per item on refusal/error/invalid output."""
    try:
        if agy.is_agy(model):
            raw = agy.chat(model, messages)
        else:
            raw = ollama.chat(cfg["ollama_url"], model, messages, cfg, json_mode=True, max_tokens=2048)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
        result = json.loads(raw)
        if (not isinstance(result, list) or len(result) != count or
                not all(isinstance(text, str) for text in result)):
            return [None] * count
        result = [text.strip() for text in result]
        if PHRASES.search("\n".join(result)):
            return [None] * count
        return [text if JAPANESE.search(text) else None for text in result]
    except Exception:
        return [None] * count


def _read_partial(path, key):
    second, batches = {}, {}
    if not path.exists():
        return second, batches
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("key") != key:
            continue
        if row.get("phase") == "asr":
            second[row["i"]] = row["text"]
        elif row.get("phase") == "batch":
            batches[row["batch"]] = row["selected"]
    return second, batches


def _apply(segs, items, selected):
    for (index, second), chosen in zip(items, selected):
        if not chosen or chosen == segs[index]["text"]:
            continue
        kotoba = segs[index]["text"]
        segs[index]["text"] = chosen
        segs[index]["asr_alt"] = kotoba if chosen == second else second
        segs[index]["verified"] = "second_model" if chosen == second else "arbitrated"


def process(cfg, db, job, d, model):
    segments_path = d / "segments.json"
    if not cfg.get("verify", True):
        # The artifact is already the Kotoba transcript; the stage advances unchanged.
        return

    cache = cache_path(cfg, job["video"])
    if cache.exists():
        segments_path.write_text(cache.read_text())
        jobs.progress(db, job["id"], 1.0)
        return

    segs = json.loads(segments_path.read_text())
    if not segs:
        cache.write_text("[]")
        return
    brief_path = d / "brief.json"
    brief = json.loads(brief_path.read_text()) if brief_path.exists() else None
    prompt = name_hints(brief)
    audio = read_wav(d / "audio.wav")
    key = cache.stem
    partial = d / "verify.partial.jsonl"
    second, completed = _read_partial(partial, key)
    progress_floor = float(job.get("progress") or 0.0)

    def report_progress(fraction):
        nonlocal progress_floor
        progress_floor = max(progress_floor, fraction)
        jobs.progress(db, job["id"], progress_floor)

    with partial.open("a") as checkpoint:
        for i, segment in enumerate(segs):
            if i not in second and too_short(segment["text"], cfg):
                second[i] = segment["text"]  # うん / あっ / はい: a second hearing cannot change anything useful
            if i not in second:
                second[i] = transcribe_segment(model, audio, segment, cfg, prompt)
                checkpoint.write(json.dumps({"key": key, "phase": "asr", "i": i,
                                             "text": second[i]}, ensure_ascii=False) + "\n")
                checkpoint.flush()
            report_progress(0.7 * (i + 1) / len(segs))

        disagreements = [i for i, seg in enumerate(segs)
                         if similarity(seg["text"], second[i]) < cfg.get("verify_agree", 0.6)]
        batches = [disagreements[i:i + ARBITRATION_BATCH]
                   for i in range(0, len(disagreements), ARBITRATION_BATCH)]
        items_by_batch = [[(i, second[i]) for i in indices] for indices in batches]

        # Restore completed arbitration from an earlier paused/interrupted attempt first.
        for batch_index, items in enumerate(items_by_batch):
            if batch_index in completed:
                _apply(segs, items, completed[batch_index])

        accepted = {i for i in range(len(segs)) if i not in disagreements or
                    any(i in indices and b in completed for b, indices in enumerate(batches))}
        unresolved = [(batch_index, items) for batch_index, items in enumerate(items_by_batch)
                      if batch_index not in completed]
        verify_model = cfg.get("verify_model") or cfg["models"]["vi"]
        messages = [(batch_index, items, arbitration_prompt(brief, items, segs, accepted))
                    for batch_index, items in unresolved]

        def finish(batch_index, items, selected):
            _apply(segs, items, selected)
            checkpoint.write(json.dumps({"key": key, "phase": "batch", "batch": batch_index,
                                         "selected": selected}, ensure_ascii=False) + "\n")
            checkpoint.flush()

        done_count = len(completed)
        if agy.is_agy(verify_model) and messages:
            max_workers = max(1, int(cfg.get("cloud_parallel", 4)))
            pool = ThreadPoolExecutor(max_workers=min(max_workers, len(messages)))
            futures = {pool.submit(arbitrate, cfg, verify_model, prompt_messages, len(items)):
                       (batch_index, items)
                       for batch_index, items, prompt_messages in messages}
            try:
                for future in as_completed(futures):
                    batch_index, items = futures[future]
                    selected = future.result()
                    finish(batch_index, items, selected)
                    done_count += 1
                    report_progress(0.7 + 0.3 * done_count / max(1, len(batches)))
            except BaseException:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        else:
            for batch_index, items, prompt_messages in messages:
                selected = arbitrate(cfg, verify_model, prompt_messages, len(items))
                finish(batch_index, items, selected)
                done_count += 1
                report_progress(0.7 + 0.3 * done_count / max(1, len(batches)))
        report_progress(1.0)

    rendered = json.dumps(segs, ensure_ascii=False)
    segments_path.write_text(rendered)
    cache.write_text(rendered)
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    cli(load, process, "verified")
