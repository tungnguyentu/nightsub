"""Heuristic refusal detection (KTD7): wrong shape, refusal phrasing, or still in the source script."""
import re

PHRASES = re.compile("|".join([
    r"\bI can(?:'|no)t\b", r"\bI(?: am|'m) (?:not able|unable|sorry)", r"\bI apologi[sz]e\b", r"\bas an AI\b",
    r"\bcannot (?:assist|help|translate|provide)\b", r"\binappropriate\b", r"\bexplicit content\b",
    r"xin lỗi", r"tôi không thể", r"không thể (?:dịch|hỗ trợ|giúp)", r"nội dung (?:khiêu dâm|nhạy cảm|không phù hợp)",
]), re.I)
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def in_source_script(text):
    letters = re.sub(r"[\W\d_]+", "", text)
    return bool(letters) and len(CJK.findall(letters)) / len(letters) > 0.3


VI_MARKS = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹáàãéèíìóòõúùý]", re.I)


def not_vietnamese(text):
    """3+ words with no Vietnamese diacritic: the model left it in English (short interjections are fine)."""
    return len(text.split()) >= 3 and not VI_MARKS.search(text)


def is_refusal(src_lines, out, lang=None):
    if not isinstance(out, list) or len(out) != len(src_lines) or not all(isinstance(o, str) for o in out):
        return True
    return any(PHRASES.search(o) or in_source_script(o) or (lang == "vi" and not_vietnamese(o)) for o in out)
