"""Cheap speaker-gender guess from pitch, so the translator knows who is talking (anh/em, tone).

ponytail: median F0 threshold, not real diarization; add pyannote if two same-gender speakers matter.
"""
import numpy as np

SR = 16000
MALE_MAX_HZ = 160  # typical adult F0: male ~85-155 Hz, female ~165-255 Hz


def pitch_hz(clip, frame=1024, hop=512, fmin=60, fmax=400):
    """Median F0 over voiced frames via autocorrelation; None if nothing voiced."""
    lo, hi = SR // fmax, SR // fmin
    f0 = []
    for s in range(0, len(clip) - frame, hop):
        x = clip[s:s + frame] - clip[s:s + frame].mean()
        if np.sqrt((x ** 2).mean()) < 0.01:  # silence
            continue
        ac = np.correlate(x, x, "full")[frame - 1:]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:  # periodic enough to be voiced
            f0.append(SR / lag)
    return float(np.median(f0)) if f0 else None


def gender(clip):
    """'M', 'F' or None."""
    f = pitch_hz(clip)
    return None if f is None else ("M" if f < MALE_MAX_HZ else "F")
