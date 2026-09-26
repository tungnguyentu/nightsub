from types import SimpleNamespace as S

from autosub.stages.asr import offset_segments


def test_offset_math():
    out = offset_segments(3600, [S(start=1.2, end=2.0, text=" はい ", avg_logprob=-0.3)])
    assert out == [{"start": 3601.2, "end": 3602.0, "text": "はい", "logprob": -0.3}]


def test_empty_skipped():
    assert offset_segments(0, [S(start=0, end=1, text="  ", avg_logprob=-2)]) == []
