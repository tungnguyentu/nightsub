from autosub.tags import label, tag_cues

SPANS = [{"start": 0, "end": 5, "kind": "nonspeech", "tag": "BGM"},
         {"start": 5, "end": 8, "kind": "speech", "tag": None},
         {"start": 8, "end": 20, "kind": "nonspeech", "tag": None}]


def test_labels_localized():
    assert label("BGM", "vi") == "[nhạc]" and label(None, "en") == "[moans]"


def test_tags_off_no_lines():
    assert tag_cues(SPANS, "en", False) == []


def test_tags_on_only_nonspeech():
    assert [c["text"] for c in tag_cues(SPANS, "en", True)] == ["[music]", "[moans]"]
