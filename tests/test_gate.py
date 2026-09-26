from autosub.stages.gate import classify


def test_event_only_is_nonspeech():
    assert classify("<|ja|><|EMO_UNKNOWN|><|Laughter|><|woitn|>") == ("nonspeech", "Laughter")


def test_empty_text_is_nonspeech():
    assert classify("<|ja|><|NEUTRAL|><|Speech|><|woitn|> 。") == ("nonspeech", None)


def test_text_with_emotion_is_speech():
    assert classify("<|ja|><|HAPPY|><|Speech|><|woitn|>気持ちいい") == ("speech", None)
