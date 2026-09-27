import numpy as np

from autosub import voice


def tone(hz, secs=1.0):
    t = np.arange(int(voice.SR * secs)) / voice.SR
    # a few harmonics, like a voice
    return (0.3 * np.sin(2 * np.pi * hz * t) + 0.1 * np.sin(4 * np.pi * hz * t)).astype(np.float32)


def test_low_voice_is_male():
    assert voice.gender(tone(120)) == "M"


def test_high_voice_is_female():
    assert voice.gender(tone(220)) == "F"


def test_silence_is_unknown():
    assert voice.gender(np.zeros(voice.SR, np.float32)) is None


def test_pitch_is_close():
    assert abs(voice.pitch_hz(tone(200)) - 200) < 10
