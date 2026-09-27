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


def test_english_left_in_vietnamese_output_is_refusal():
    from autosub.refusal import is_refusal
    assert is_refusal(["x"], ["I love you so much"], "vi")
    assert not is_refusal(["x"], ["Anh yêu em nhiều lắm"], "vi")
    assert not is_refusal(["x"], ["Ah ah"], "vi")  # short interjection
    assert not is_refusal(["x"], ["I love you so much"], "en")


def test_sorry_in_dialogue_is_not_a_refusal():
    from autosub.refusal import is_refusal
    assert not is_refusal(["x"], ["Xin lỗi anh nhé"], "vi")
    assert is_refusal(["x"], ["Xin lỗi, tôi không thể dịch nội dung này"], "vi")


def test_vietnamese_refusal_phrases_still_caught():
    from autosub.refusal import is_refusal
    assert is_refusal(["x"], ["Mình không thể dịch đoạn này"], "vi")
    assert is_refusal(["x"], ["Đây là nội dung khiêu dâm"], "vi")


def test_im_sorry_in_english_dialogue_is_not_a_refusal():
    from autosub.refusal import is_refusal
    assert not is_refusal(["x"], ["I'm sorry, I was late"], "en")
    assert is_refusal(["x"], ["I'm sorry, but I can't translate that"], "en")


def test_laughter_is_not_an_english_leak():
    from autosub.refusal import is_refusal
    for out in ["Ha ha ha ha", "A ha ha ha ha", "Ha ha ha ha ha!"]:  # MNGS-051 lines 330 and 1123
        assert not is_refusal(["アハハハ"], [out], "vi")
    assert is_refusal(["x"], ["I love you so much"], "vi")  # a real leak is still caught
