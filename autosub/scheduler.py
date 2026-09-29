"""Stage-major worker (KTD10): each stage runs once over every job waiting for it."""
import contextlib
import os
import subprocess
import sys
import threading
import time

from . import config, gpu, jobs, preflight

# (name, from_stage, to_stage). Stage runners take (cfg, db, batch) and advance/fail jobs themselves.
STAGES = [("gate", "queued", "gated"), ("asr", "gated", "transcribed"), ("brief", "transcribed", "briefed"),
          ("verify", "briefed", "verified"), ("translate", "verified", "translated"),
          ("polish", "translated", "polished"), ("retime", "polished", "done")]
STEP_INFO = [
    {"name": "gate", "label_vi": "Lọc âm thanh", "help_vi": "Tách lời thoại khỏi đoạn không lời."},
    {"name": "asr", "label_vi": "Nhận giọng nói", "help_vi": "Chuyển lời thoại tiếng Nhật thành văn bản."},
    {"name": "brief", "label_vi": "Tóm tắt bối cảnh", "help_vi": "Tóm tắt nhân vật, quan hệ và cách xưng hô."},
    {"name": "verify", "label_vi": "Đối chiếu", "help_vi": "Nghe lại bằng model thứ hai; câu lệch được Gemini/LLM chọn lại theo ngữ cảnh."},
    {"name": "translate", "label_vi": "Dịch", "help_vi": "Dịch phụ đề theo bối cảnh và hội thoại lân cận."},
    {"name": "polish", "label_vi": "Chỉnh câu", "help_vi": "Rà soát câu dịch để tự nhiên, đúng nghĩa."},
    {"name": "retime", "label_vi": "Căn thời gian", "help_vi": "Điều chỉnh thời lượng và tốc độ đọc."},
]

GPU_STAGES = {"gate", "asr", "brief", "verify", "translate", "polish"}  # everything but retime
LLM_STAGES = {"brief", "verify", "translate", "polish"}


def subprocess_stage(name):
    """GPU stage in its own process so VRAM is returned on exit (KTD2)."""
    def run(cfg, db, batch):
        env = {**os.environ, "LD_LIBRARY_PATH": _cuda12_libs()}
        if "work_dir" in cfg:
            env["AUTOSUB_WORK_DIR"] = cfg["work_dir"]
        p = subprocess.run([sys.executable, "-m", f"autosub.stages.{name}", *[str(j["id"]) for j in batch]],
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            last = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
            frm = next(s[1] for s in STAGES if s[0] == name)
            for j in batch:
                if jobs.get(db, j["id"])["stage"] == frm:
                    jobs.fail(db, j["id"], f"{name}: {last}")
    return run


def _cuda12_libs():
    """faster-whisper's ctranslate2 wants CUDA 12 cuBLAS/cuDNN; torch ships CUDA 13, so point at the cu12 wheels."""
    import glob
    import site
    dirs = [d for sp in site.getsitepackages() for lib in ("cublas", "cudnn") for d in glob.glob(f"{sp}/nvidia/{lib}/lib")]
    return os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)


def inprocess_stage(name):
    def run(cfg, db, batch):
        import importlib
        importlib.import_module(f"autosub.stages.{name}").run(cfg, db, batch)
    return run


RUNNERS = {"gate": subprocess_stage("gate"), "asr": subprocess_stage("asr"),
           "verify": subprocess_stage("verify"),
           "brief": inprocess_stage("brief"),
           "translate": inprocess_stage("translate"), "polish": inprocess_stage("polish"),
           "retime": inprocess_stage("retime")}


def run_pass(cfg, db, runners=RUNNERS):
    """One stage-major pass. Returns True if there was work."""
    for i in jobs.purge_deleted(db):
        config.clear_job_dir(cfg, i)
    active = [j for j in jobs.all(db) if j["error"] is None and j["stage"] != "done" and not j["control"]]
    if not active:
        return False
    problems = preflight.check(cfg)
    if problems:
        for j in active:
            jobs.fail(db, j["id"], "preflight: " + "; ".join(problems))
        return True
    for name, frm, _ in STAGES:
        batch = jobs.at_stage(db, frm)
        if not batch:
            continue
        with (gpu.lock(cfg.get("gpu_lock", gpu.LOCK_PATH)) if name in GPU_STAGES else contextlib.nullcontext()):
            if name in LLM_STAGES:
                reason = gpu.wait_free(cfg["vram_needed_mb"], cfg["vram_wait_s"])  # R9, AE2
                if reason:
                    for j in batch:
                        jobs.fail(db, j["id"], reason)
                    continue
            started_at = time.time()
            for j in batch:
                jobs.update(db, j["id"], stage_started_at=started_at)
            t0 = time.monotonic()
            try:
                runners[name](cfg, db, batch)
            except Exception as e:  # one stage blowing up fails its batch, not the worker
                for j in batch:
                    jobs.fail(db, j["id"], f"{name}: {e}")
            _record_time(db, name, batch, time.monotonic() - t0)
        for i in jobs.purge_deleted(db):
            config.clear_job_dir(cfg, i)
    return True


def _record_time(db, name, batch, elapsed):
    """Split batch wall time across jobs by audio length, for ETA (U8)."""
    fresh = [jobs.get(db, j["id"]) for j in batch]
    total = sum(j["audio_min"] or 0 for j in fresh)
    for j in fresh:
        if j["error"] is None and total:
            jobs.update(db, j["id"], stage_times={**j["stage_times"], name: elapsed * (j["audio_min"] or 0) / total},
                        stage_started_at=None)
        else:
            jobs.update(db, j["id"], stage_started_at=None)


def start_worker(cfg, db):
    def loop():
        while True:
            try:
                busy = run_pass(cfg, db)
            except Exception as e:
                print("worker error:", e, file=sys.stderr)
                busy = False
            if not busy:
                time.sleep(2)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
