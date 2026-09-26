from autosub.refusal import is_refusal

SRC = ["a"] * 3


def test_ok():
    assert not is_refusal(SRC, ["Yes", "More", "Don't stop"])


def test_refusal_phrase_en_vi():
    assert is_refusal(SRC, ["Yes", "I'm sorry, but I can't help with that.", "x"])
    assert is_refusal(SRC, ["Có", "Xin lỗi, tôi không thể dịch nội dung này.", "x"])


def test_count_mismatch():
    assert is_refusal(["a"] * 12, ["b"] * 11)


def test_still_kana():
    assert is_refusal(SRC, ["Yes", "きもちいい", "ok"])


def test_not_a_list():
    assert is_refusal(SRC, None) and is_refusal(SRC, {"a": 1})
