"""Readable cues (R14): split long cues at punctuation, wrap to 2 lines, extend to reading speed."""
import re

BREAK = re.compile(r"(?<=[.,!?;:…。、，！？])\s*")


def _split(cue, max_chars):
    t = cue["text"]
    if len(t) <= 2 * max_chars:
        return [cue]
    cuts = [m.end() for m in BREAK.finditer(t) if 0 < m.end() < len(t)] or [m.start() for m in re.finditer(" ", t)]
    if not cuts:
        return [cue]
    k = min(cuts, key=lambda c: abs(c - len(t) / 2))
    mid = cue["start"] + (cue["end"] - cue["start"]) * k / len(t)
    return (_split({**cue, "text": t[:k].strip(), "end": mid}, max_chars)
            + _split({**cue, "text": t[k:].strip(), "start": mid}, max_chars))


def wrap(text, max_chars):
    if len(text) <= max_chars:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    k = min(spaces, key=lambda i: abs(i - len(text) / 2)) if spaces else len(text) // 2
    return text[:k].rstrip() + "\n" + text[k:].lstrip()


def retime(cues, cfg):
    cues = sorted((c for cue in cues for c in _split(cue, cfg["max_line"])), key=lambda c: c["start"])
    out = []
    for i, c in enumerate(cues):
        end = max(c["end"], c["start"] + max(cfg["min_dur"], len(c["text"]) / cfg["cps"]))
        if i + 1 < len(cues):
            end = min(end, cues[i + 1]["start"])
        out.append({**c, "end": end, "text": wrap(c["text"], cfg["max_line"])})
    return out
