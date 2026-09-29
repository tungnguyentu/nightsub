import types

from autosub import jobs, scheduler, web


CFG = {"vram_needed_mb": 5000, "vram_wait_s": 0}


def make(tmp_path, monkeypatch, stages):
    db = str(tmp_path / "j.db")
    monkeypatch.setattr(scheduler.preflight, "check", lambda cfg: [])
    monkeypatch.setattr(scheduler.gpu, "wait_free", lambda need, timeout_s: None)
    ids = [jobs.add(db, f"/v/{i}.mp4", "en", audio_min=10) for i in range(len(stages))]
    for i, s in zip(ids, stages):
        jobs.update(db, i, stage=s)
    return db, ids


def recorder(calls):
    def runner_for(name):
        to = next(s[2] for s in scheduler.STAGES if s[0] == name)

        def run(cfg, db, batch):
            calls.append((name, [j["id"] for j in batch]))
            for j in batch:
                jobs.update(db, j["id"], stage=to)
        return run
    return {n: runner_for(n) for n, _, _ in scheduler.STAGES}


def test_asr_called_once_for_all(tmp_path, monkeypatch):
    db, ids = make(tmp_path, monkeypatch, ["gated"] * 3)
    calls = []
    scheduler.run_pass(CFG, db, recorder(calls))
    assert calls[0] == ("asr", ids)
    assert all(j["stage"] == "done" for j in jobs.all(db))


def test_resume_skips_asr(tmp_path, monkeypatch):
    db, ids = make(tmp_path, monkeypatch, ["transcribed"])
    calls = []
    scheduler.run_pass(CFG, db, recorder(calls))
    assert [c[0] for c in calls] == ["brief", "verify", "translate", "polish", "retime"]


def test_old_translated_job_is_not_rebriefed(tmp_path, monkeypatch):
    db, ids = make(tmp_path, monkeypatch, ["translated"])
    calls = []
    scheduler.run_pass(CFG, db, recorder(calls))
    assert [c[0] for c in calls] == ["polish", "retime"]


def test_stage_started_at_is_set_while_runner_executes_and_cleared_after(tmp_path, monkeypatch):
    db, ids = make(tmp_path, monkeypatch, ["gated"])
    monkeypatch.setattr(scheduler.time, "time", lambda: 1234.0)
    seen = []
    runners = recorder([])
    run_asr = runners["asr"]
    def asr_runner(cfg, db, batch):
        seen.append(jobs.get(db, batch[0]["id"])["stage_started_at"])
        run_asr(cfg, db, batch)
    runners["asr"] = asr_runner
    scheduler.run_pass(CFG, db, runners)
    assert seen == [1234.0]
    assert jobs.get(db, ids[0])["stage_started_at"] is None


def test_step_states_for_running_and_failed_brief():
    base = {"stage": "gated", "error": None, "progress": 0.4, "stage_times": {"gate": 3.8},
            "stage_started_at": 100.0}
    steps = web.steps_for(base, now=109.0)
    assert steps[0] == {"name": "gate", "state": "done", "secs": 4}
    assert steps[1] == {"name": "asr", "state": "running", "secs": 9, "progress": 0.4}
    assert all(s["state"] == "pending" for s in steps[2:])
    failed = web.steps_for({**base, "stage": "transcribed", "error": "bad brief", "stage_started_at": None})
    assert [s["state"] for s in failed[:4]] == ["done", "done", "failed", "pending"]


def test_gpu_busy_fails_job(tmp_path, monkeypatch):
    db, ids = make(tmp_path, monkeypatch, ["transcribed"])
    monkeypatch.setattr(scheduler.gpu, "wait_free", lambda need, timeout_s: "GPU busy: game.exe")
    scheduler.run_pass(CFG, db, recorder([]))
    j = jobs.get(db, ids[0])
    assert j["stage"] == "transcribed" and "GPU busy: game.exe" in j["error"]


def test_subprocess_exit_fails_only_its_jobs(tmp_path, monkeypatch):
    db, ids = make(tmp_path, monkeypatch, ["gated", "transcribed"])
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stderr="Traceback...\nRuntimeError: CUDA oops\n"))
    calls = []
    runners = recorder(calls)
    runners["asr"] = scheduler.subprocess_stage("asr")
    scheduler.run_pass(CFG, db, runners)
    a, b = jobs.get(db, ids[0]), jobs.get(db, ids[1])
    assert a["error"] == "asr: RuntimeError: CUDA oops"
    assert b["error"] is None and b["stage"] == "done"


def test_gpu_lock_is_exclusive_across_holders(tmp_path):
    import fcntl
    from autosub import gpu
    path = str(tmp_path / "gpu.lock")
    with gpu.lock(path):
        with open(path, "w") as other:
            import pytest
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(path, "w") as other:  # released after the block
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
