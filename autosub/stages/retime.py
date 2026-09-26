"""Retime stage (U7): polished lines (+ optional tags) -> <stem>.<lang>.srt next to the video."""
import json

from .. import config, jobs, srt
from ..retime import retime
from ..tags import tag_cues


def run(cfg, db, batch):
    for job in batch:
        d = config.job_dir(cfg, job["id"])
        segs = json.loads((d / "polished.json").read_text())
        spans = json.loads((d / "spans.json").read_text())
        cues = retime([{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]
                      + tag_cues(spans, job["lang"], job["tags"]), cfg)
        dialogue = sum(1 for c in cues if not c.get("tag"))
        reason = srt.near_empty(dialogue, sum(s["end"] - s["start"] for s in spans if s["kind"] == "speech"))
        if reason:
            jobs.fail(db, job["id"], reason)
            continue
        srt.write(srt.output_path(job["video"], job["lang"]), cues)
        jobs.update(db, job["id"], stage="done", progress=1, cues=len(cues))
