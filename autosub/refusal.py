"""Heuristic refusal detection (KTD7): wrong shape, refusal phrasing, or still in the source script."""
import re

from .flags import is_sound, repeats

PHRASES = re.compile("|".join([
    r"\bI can(?:'|no)t\b", r"\bI(?: am|'m) (?:not able|unable)", r"\bI apologi[sz]e\b", r"\bas an AI\b",
    r"\bcannot (?:assist|help|translate|provide)\b", r"\binappropriate\b", r"\bexplicit content\b",
    # bare "xin lỗi" is normal dialogue ("sorry"); only count it next to a refusal
    r"xin lỗi,? (?:tôi|mình) không", r"tôi không thể",
    r"không thể (?:dịch|hỗ trợ|giúp)", r"nội dung (?:khiêu dâm|nhạy cảm|không phù hợp)",
]), re.I)
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def in_source_script(text):
    letters = re.sub(r"[\W\d_]+", "", text)
    return bool(letters) and len(CJK.findall(letters)) / len(letters) > 0.3


VI_MARKS = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹáàãéèíìóòõúùý]", re.I)


def not_vietnamese(text):
    """3+ words with no Vietnamese diacritic: the model left it in English (short interjections are fine).
    Repeated-syllable sounds ("Ha ha ha ha", "A ha ha") are the same in every language, not a leak."""
    return len(text.split()) >= 3 and not VI_MARKS.search(text) and not repeats(text) and not is_sound(text)


def is_refusal(src_lines, out, lang=None):
    if not isinstance(out, list) or len(out) != len(src_lines) or not all(isinstance(o, str) for o in out):
        return True
    return any(PHRASES.search(o) or in_source_script(o) or (lang == "vi" and not_vietnamese(o)) for o in out)
