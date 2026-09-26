"""ASR stage (U4, KTD5): faster-whisper per speech span, absolute timestamps -> segments.json."""
import json

from .. import jobs
from . import cli, read_wav

SR = 16000


def offset_segments(span_start, segs):
    """Shift span-relative segments to absolute time (no drift, R7); drop empty text."""
    return [{"start": round(span_start + s.start, 3), "end": round(span_start + s.end, 3),
             "text": s.text.strip(), "logprob": s.avg_logprob} for s in segs if s.text.strip()]


def load(cfg):
    from faster_whisper import WhisperModel
    return WhisperModel(cfg["asr_model"], device="cuda", compute_type=cfg["asr_compute_type"])


def process(cfg, db, job, d, model):
    audio = read_wav(d / "audio.wav")
    speech = [s for s in json.loads((d / "spans.json").read_text()) if s["kind"] == "speech"]
    out = []
    for i, sp in enumerate(speech):
        segs, _ = model.transcribe(audio[int(sp["start"] * SR):int(sp["end"] * SR)], language=cfg["asr_language"],
                                   condition_on_previous_text=False, vad_filter=False)
        out += offset_segments(sp["start"], segs)
        jobs.update(db, job["id"], progress=(i + 1) / len(speech))
    (d / "segments.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    cli(load, process, "transcribed")
