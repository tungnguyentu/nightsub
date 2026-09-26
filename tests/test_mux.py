from autosub import mux


def test_finds_only_known_language_srts(tmp_path):
    v = tmp_path / "clip.mp4"
    for n in ("clip.en.srt", "clip.vi.srt", "clip.srt", "other.en.srt", "clip.xx.srt"):
        (tmp_path / n).write_text("")
    assert [p.name for p in mux.srts_for(v)] == ["clip.en.srt", "clip.vi.srt"]


def test_cmd_tags_each_track_language(tmp_path):
    v = tmp_path / "clip.mp4"
    cmd = mux.mux_cmd(v, [tmp_path / "clip.en.srt", tmp_path / "clip.vi.srt"], tmp_path / "o.mkv")
    assert "language=eng" in cmd and "language=vie" in cmd
    assert cmd[cmd.index("-c") + 1] == "copy"
