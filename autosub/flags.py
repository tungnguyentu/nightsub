"""Which lines get the polish rewrite (KTD8)."""
import re


def repeats(text):
    """'あああああ', 'ha ha ha ha' -> True: a 1-4 char unit repeated 4+ times."""
    return bool(re.search(r"(.{1,4})\1{3,}", re.sub(r"[\W_]+", "", text)))


# Vietnamese personal pronouns that signal the wrong register if they aren't the chosen pair.
VI_OFF_REGISTER = {"mày", "tao", "tôi", "bạn", "tớ", "cậu"}


def wrong_pronoun(text, allowed):
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words & (VI_OFF_REGISTER - allowed))


def is_flagged(seg, cfg, allowed=None):
    """seg: {'src', 'text', 'logprob'}. allowed: Vietnamese pronoun set for this video (None = not Vietnamese)."""
    if seg["logprob"] < cfg["polish_logprob"]:
        return True
    if allowed is not None and wrong_pronoun(seg["text"], allowed):
        return True
    lo, hi = cfg["polish_ratio"]
    ratio = len(seg["text"]) / max(1, len(seg["src"]))
    return not lo <= ratio <= hi or repeats(seg["text"]) or repeats(seg["src"])
