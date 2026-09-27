"""ASR stage (U4, KTD5): faster-whisper per speech span, absolute timestamps -> segments.json."""
import hashlib
import json
from pathlib import Path

from .. import jobs, voice
from . import cli, read_wav

SR = 16000


def offset_segments(span_start, segs, clip=None):
    """Shift span-relative segments to absolute time (no drift, R7); drop empty text; guess speaker gender."""
    out = []
    for s in segs:
        if s.text.strip():
            g = voice.gender(clip[int(s.start * SR):int(s.end * SR)]) if clip is not None else None
            out.append({"start": round(span_start + s.start, 3), "end": round(span_start + s.end, 3),
                        "text": s.text.strip(), "logprob": s.avg_logprob, "gender": g})
    return out


def load(cfg):
    from faster_whisper import WhisperModel
    return WhisperModel(cfg["asr_model"], device="cuda", compute_type=cfg["asr_compute_type"])


def cache_path(cfg, video):
    """Transcript is per video, not per language: a second job on the same file reuses it."""
    key = hashlib.sha1(f"{Path(video).resolve()}|{cfg['asr_model']}|{cfg['asr_language']}|g1".encode()).hexdigest()[:16]
    p = Path(cfg["work_dir"]) / "asr-cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"


def transcribe(model, clip, cfg):
    kw = dict(language=cfg["asr_language"], condition_on_previous_text=False, vad_filter=False)
    try:
        return list(model.transcribe(clip, **kw)[0])
    except RuntimeError as e:  # VRAM squeezed by other apps mid-run: retry lighter before failing the job
        if "out of memory" not in str(e):
            raise
        return list(model.transcribe(clip, beam_size=1, **kw)[0])


def process(cfg, db, job, d, model):
    cache = cache_path(cfg, job["video"])
    if cache.exists():
        (d / "segments.json").write_text(cache.read_text())
        return
    audio = read_wav(d / "audio.wav")
    speech = [s for s in json.loads((d / "spans.json").read_text()) if s["kind"] == "speech"]
    partial = d / "segments.partial.jsonl"  # one line per finished span, so a crash resumes mid-video (R10)
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            r = json.loads(line)
            done[r["i"]] = r["segs"]
    with partial.open("a") as f:
        for i, sp in enumerate(speech):
            if i not in done:
                clip = audio[int(sp["start"] * SR):int(sp["end"] * SR)]
                done[i] = offset_segments(sp["start"], transcribe(model, clip, cfg), clip)
                f.write(json.dumps({"i": i, "segs": done[i]}, ensure_ascii=False) + "\n")
                f.flush()
            jobs.progress(db, job["id"], (i + 1) / len(speech))
    out = json.dumps([s for i in range(len(speech)) for s in done[i]], ensure_ascii=False)
    (d / "segments.json").write_text(out)
    cache.write_text(out)


if __name__ == "__main__":
    cli(load, process, "transcribed")
