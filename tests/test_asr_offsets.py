from types import SimpleNamespace as S

from autosub.stages.asr import offset_segments


def test_offset_math():
    out = offset_segments(3600, [S(start=1.2, end=2.0, text=" はい ", avg_logprob=-0.3)])
    assert out == [{"start": 3601.2, "end": 3602.0, "text": "はい", "logprob": -0.3, "gender": None}]


def test_empty_skipped():
    assert offset_segments(0, [S(start=0, end=1, text="  ", avg_logprob=-2)]) == []


class Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text, self.avg_logprob = start, end, text, -0.1


class FakeModel:
    def __init__(self, fail_at=None, oom_once=False):
        self.calls, self.fail_at, self.oom_once = 0, fail_at, oom_once

    def transcribe(self, clip, beam_size=5, **kw):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("boom")
        if self.oom_once and beam_size != 1:
            raise RuntimeError("CUDA failed with error out of memory")
        return iter([Seg(0.0, 1.0, "こんにちは")]), None


def _job(tmp_path, monkeypatch, n_spans=3):
    import json
    import numpy as np
    from autosub.stages import asr
    d = tmp_path / "job"
    d.mkdir()
    (d / "spans.json").write_text(json.dumps([{"start": i * 2.0, "end": i * 2.0 + 1, "kind": "speech"}
                                              for i in range(n_spans)]))
    monkeypatch.setattr(asr, "read_wav", lambda p: np.zeros(16000 * 10, np.float32))
    monkeypatch.setattr(asr.jobs, "update", lambda *a, **k: None)
    cfg = {"work_dir": str(tmp_path), "asr_model": "m", "asr_language": None}
    return asr, cfg, d, {"id": 1, "video": str(tmp_path / "v.mp4")}


def test_crash_mid_video_resumes_from_checkpoint(tmp_path, monkeypatch):
    import json
    import pytest
    asr, cfg, d, job = _job(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        asr.process(cfg, None, job, d, FakeModel(fail_at=3))
    m = FakeModel()
    asr.process(cfg, None, job, d, m)
    assert m.calls == 1  # only the unfinished span
    assert [s["start"] for s in json.loads((d / "segments.json").read_text())] == [0.0, 2.0, 4.0]


def test_oom_retries_with_beam_1(tmp_path, monkeypatch):
    asr, cfg, d, job = _job(tmp_path, monkeypatch, n_spans=1)
    asr.process(cfg, None, job, d, FakeModel(oom_once=True))
    assert (d / "segments.json").exists()


def test_second_language_reuses_transcript(tmp_path, monkeypatch):
    asr, cfg, d, job = _job(tmp_path, monkeypatch)
    asr.process(cfg, None, job, d, FakeModel())
    d2 = tmp_path / "job2"
    d2.mkdir()
    m = FakeModel()
    asr.process(cfg, None, {**job, "id": 2}, d2, m)
    assert m.calls == 0 and (d2 / "segments.json").read_text() == (d / "segments.json").read_text()
