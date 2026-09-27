from autosub.stages.gate import classify


def test_event_only_is_nonspeech():
    assert classify("<|ja|><|EMO_UNKNOWN|><|Laughter|><|woitn|>") == ("nonspeech", "Laughter")


def test_empty_text_is_nonspeech():
    assert classify("<|ja|><|NEUTRAL|><|Speech|><|woitn|> 。") == ("nonspeech", None)


def test_text_with_emotion_is_speech():
    assert classify("<|ja|><|HAPPY|><|Speech|><|woitn|>気持ちいい") == ("speech", None)


def test_dialogue_over_music_is_speech():
    from autosub.stages.gate import classify
    assert classify("<|ja|><|NEUTRAL|><|BGM|><|woitn|>ちょっと待って") == ("speech", None)
    assert classify("<|ja|><|NEUTRAL|><|BGM|><|woitn|>") == ("nonspeech", "BGM")  # music only
    assert classify("<|ja|><|HAPPY|><|Laughter|><|woitn|>あはは")[0] == "nonspeech"  # vocal event stays out
