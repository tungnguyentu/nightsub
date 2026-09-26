"""Which lines get the polish rewrite (KTD8)."""
import re


def repeats(text):
    """'あああああ', 'ha ha ha ha' -> True: a 1-4 char unit repeated 4+ times."""
    return bool(re.search(r"(.{1,4})\1{3,}", re.sub(r"[\W_]+", "", text)))


def is_flagged(seg, cfg):
    """seg: {'src', 'text', 'logprob'}."""
    if seg["logprob"] < cfg["polish_logprob"]:
        return True
    lo, hi = cfg["polish_ratio"]
    ratio = len(seg["text"]) / max(1, len(seg["src"]))
    return not lo <= ratio <= hi or repeats(seg["text"]) or repeats(seg["src"])
