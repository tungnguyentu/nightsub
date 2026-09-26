import json

from autosub import config, jobs, srt
from autosub.stages import retime as stage


def test_timestamp_format():
    assert srt.ts(62.345) == "00:01:02,345" and srt.ts(3723.0) == "01:02:03,000"


def test_output_name():
    assert str(srt.output_path("/v/a.b.mp4", "vi")) == "/v/a.b.vi.srt"


def test_zero_cues_fails_and_writes_nothing(tmp_path):
    cfg = {**config.DEFAULTS, "work_dir": str(tmp_path)}
    db = config.db_path(cfg)
    video = tmp_path / "v.mp4"
    i = jobs.add(db, str(video), "en")
    d = config.job_dir(cfg, i)
    (d / "polished.json").write_text("[]")
    (d / "spans.json").write_text(json.dumps([{"start": 0, "end": 600, "kind": "speech", "tag": None}]))
    stage.run(cfg, db, [jobs.get(db, i)])
    assert "near-empty" in jobs.get(db, i)["error"]
    assert not srt.output_path(video, "en").exists()
