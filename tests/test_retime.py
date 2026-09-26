from autosub import config
from autosub.retime import retime

CFG = config.DEFAULTS


def test_long_line_wrapped_to_two():
    text = "I really want you to keep going like that, please don't stop"
    assert len(text) == 60
    [c] = retime([{"start": 0, "end": 5, "text": text}], CFG)
    lines = c["text"].split("\n")
    assert len(lines) == 2 and all(len(x) <= 42 for x in lines)


def test_fast_cue_extended_not_past_next():
    fast = {"start": 0, "end": 1, "text": "x" * 30}  # 30 CPS
    nxt = {"start": 1.5, "end": 3, "text": "ok"}
    a, b = retime([fast, nxt], CFG)
    assert a["end"] == 1.5


def test_min_duration():
    [c] = retime([{"start": 0, "end": 0.1, "text": "Hi"}], CFG)
    assert c["end"] == 0.8


def test_overlong_cue_split_at_punctuation():
    t = "This is the first long sentence of the cue, and here is the second long part of it."
    out = retime([{"start": 0, "end": 10, "text": t * 2}], CFG)
    assert len(out) > 1 and out[0]["end"] <= out[1]["start"]
