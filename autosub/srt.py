from pathlib import Path


def ts(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def output_path(video, lang):
    return Path(video).with_suffix(f".{lang}.srt")


def near_empty(n_cues, speech_s):
    """Reason string if output is too thin to be real (R16): < 5 cues per 10 min of speech."""
    if n_cues < max(1, 5 * speech_s / 600):
        return f"empty/near-empty output: {n_cues} cues for {speech_s / 60:.1f} min of speech"
    return None


def write(path, cues):
    Path(path).write_text("".join(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n\n"
                                  for i, c in enumerate(cues, 1)), encoding="utf-8")
