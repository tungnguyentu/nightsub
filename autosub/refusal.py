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


def is_refusal(src_lines, out):
    if not isinstance(out, list) or len(out) != len(src_lines) or not all(isinstance(o, str) for o in out):
        return True
    return any(PHRASES.search(o) or in_source_script(o) for o in out)
