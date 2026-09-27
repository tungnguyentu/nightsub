from autosub import config
from autosub.flags import is_flagged
from autosub.stages import polish

CFG = config.DEFAULTS


def seg(text="I want you so much", src="すごく欲しい", logprob=-0.3):
    return {"text": text, "src": src, "logprob": logprob}


def test_low_logprob():
    assert is_flagged(seg(logprob=-1.5), CFG)


def test_normal_line():
    assert not is_flagged(seg(), CFG)


def test_repeats():
    assert is_flagged(seg(text="Ah", src="あああああ"), CFG)
    assert is_flagged(seg(text="ha ha ha ha ha"), CFG)


def test_ratio_out_of_bounds():
    assert is_flagged(seg(text="a", src="すごく欲しいよ本当に"), CFG)


def test_polished_flag_only_when_rewrite_changes_text(monkeypatch):
    monkeypatch.setattr(polish, "with_fallback", lambda *args, **kwargs: ["Rewritten"])
    changed = [seg(text="x", logprob=-1.5)]
    polish.polish(CFG, changed, "en", set())
    assert changed[0]["text"] == "Rewritten" and changed[0]["polished"] is True
    monkeypatch.setattr(polish, "with_fallback", lambda *args, **kwargs: None)
    refused = [seg(text="x", logprob=-1.5)]
    polish.polish(CFG, refused, "en", set())
    assert refused[0]["text"] == "x" and refused[0]["polished"] is False
