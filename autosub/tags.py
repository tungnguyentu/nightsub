"""SenseVoice event tags -> short bracketed labels (R6). Used only when a job has tags on."""
EVENTS = {"BGM", "Applause", "Laughter", "Cry", "Sneeze", "Cough", "Breath"}

LABELS = {
    "music": {"en": "[music]", "vi": "[nhạc]"},
    "laughs": {"en": "[laughs]", "vi": "[cười]"},
    "moans": {"en": "[moans]", "vi": "[rên]"},
    "coughs": {"en": "[coughs]", "vi": "[ho]"},
}
TAG_KIND = {"BGM": "music", "Applause": "music", "Laughter": "laughs", "Cough": "coughs", "Sneeze": "coughs"}


def label(tag, lang):
    # Cry/Breath/untagged non-speech in this domain is almost always moaning.
    return LABELS[TAG_KIND.get(tag, "moans")][lang]


def tag_cues(spans, lang, enabled):
    if not enabled:
        return []
    return [{"start": s["start"], "end": s["end"], "text": label(s["tag"], lang), "tag": True}
            for s in spans if s["kind"] == "nonspeech"]
