"""Gate stage (U3, KTD4): fsmn-vad spans, each classified by SenseVoice tags -> spans.json."""
import json
import re
import subprocess

from .. import jobs
from ..tags import EVENTS
from . import cli, read_wav

SR = 16000


def parse(raw):
    """'<|ja|><|NEUTRAL|><|Speech|><|woitn|>text' -> (['ja','NEUTRAL','Speech','woitn'], 'text')."""
    return re.findall(r"<\|(.*?)\|>", raw), re.sub(r"<\|.*?\|>", "", raw).strip()


def classify(raw):
    """(kind, tag): event-tagged or textless spans are non-speech; everything else is speech."""
    tags, text = parse(raw)
    event = next((t for t in tags if t in EVENTS), None)
    if event or not re.sub(r"[\W_]+", "", text):
        return "nonspeech", event
    return "speech", None


def load(cfg):
    from funasr import AutoModel
    return (AutoModel(model="fsmn-vad", device="cuda:0", disable_update=True),
            AutoModel(model="iic/SenseVoiceSmall", device="cuda:0", disable_update=True))


def process(cfg, db, job, d, models):
    vad, sv = models
    wav = d / "audio.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", job["video"],
                    "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    audio = read_wav(wav)
    ranges = vad.generate(input=str(wav))[0]["value"]
    spans = []
    for i, (a, b) in enumerate(ranges):
        raw = sv.generate(input=audio[a * SR // 1000:b * SR // 1000], language="auto", use_itn=False)[0]["text"]
        kind, tag = classify(raw)
        spans.append({"start": a / 1000, "end": b / 1000, "kind": kind, "tag": tag})
        jobs.update(db, job["id"], progress=(i + 1) / len(ranges))
    (d / "spans.json").write_text(json.dumps(spans))
    jobs.update(db, job["id"], audio_min=len(audio) / SR / 60)


if __name__ == "__main__":
    cli(load, process, "gated")
