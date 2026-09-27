"""Local web UI (U8, KTD9): FastAPI on 127.0.0.1, one static page polling GET /jobs."""
import json
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import agy, config, jobs, mux, preflight, scheduler

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v", ".ts", ".flv"}
STAGE_NAMES = [s[0] for s in scheduler.STAGES]
STAGE_FROM = {s[1]: i for i, s in enumerate(scheduler.STAGES)}


class NewJobs(BaseModel):
    paths: list[str]
    langs: list[str]
    tags: bool = False


def probe_minutes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip()) / 60
    except ValueError:
        return None


def expand(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        elif p.is_file():
            files.append(p)
        else:
            raise HTTPException(400, f"not found: {p}")
    if not files:
        raise HTTPException(400, "no video files found")
    return files


def eta(job, all_jobs):
    """Measured sec per audio-minute per remaining stage x audio minutes; None until every stage has data."""
    if job["error"] or job["stage"] == "done" or not job["audio_min"]:
        return None
    total = 0.0
    for name in STAGE_NAMES[STAGE_FROM[job["stage"]]:]:
        done = [j for j in all_jobs if name in j["stage_times"] and j["audio_min"]]
        if not done:
            return None
        total += sum(j["stage_times"][name] for j in done) / sum(j["audio_min"] for j in done)
    return round(total * job["audio_min"])


def steps_for(job, now=None):
    """Build ordered step state from the persisted last-completed stage."""
    now = time.time() if now is None else now
    current = next((i for i, (_, frm, _) in enumerate(scheduler.STAGES) if frm == job["stage"]),
                   len(scheduler.STAGES) if job["stage"] == "done" else 0)
    result = []
    for i, (name, _, _) in enumerate(scheduler.STAGES):
        if i < current:
            state = "done"
        elif i == current:
            state = "failed" if job["error"] else "running"
        else:
            state = "pending"
        secs = job["stage_times"].get(name) if state == "done" else None
        if state == "running" and job.get("stage_started_at") is not None:
            secs = max(0, round(now - job["stage_started_at"]))
        row = {"name": name, "state": state, "secs": round(secs) if secs is not None else None}
        if state == "running":
            row["progress"] = job["progress"]
        result.append(row)
    return result


def create_app(cfg, db):
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/jobs")
    def add(req: NewJobs):
        bad = [l for l in req.langs if l not in cfg["models"]]
        if bad or not req.langs:
            raise HTTPException(400, f"unsupported language: {', '.join(bad) or 'none picked'}")
        files = expand(req.paths)
        # one job per (video, language); the scheduler transcribes each video once and reuses it (ASR cache)
        ids = [jobs.add(db, str(f.resolve()), lang, req.tags, probe_minutes(f))
               for f in files for lang in req.langs]
        for i in ids:
            config.clear_job_dir(cfg, i)
        return {"ids": ids}

    @app.get("/jobs")
    def status():
        rows = jobs.all(db)
        return [{**j, "cloud_model": cfg["models"].get(j["lang"])
                 if agy.is_agy(cfg["models"].get(j["lang"], "")) else None,
                 "eta_s": eta(j, rows), "steps": steps_for(j)} for j in rows]

    @app.get("/steps")
    def step_info():
        return scheduler.STEP_INFO

    @app.get("/jobs/{job_id}/lines")
    def lines(job_id: int):
        job = jobs.get(db, job_id)
        if not job:
            raise HTTPException(404, "no such job")
        d = config.job_dir(cfg, job_id)
        artifact = next((d / name for name in ("polished.json", "translated.json", "segments.json")
                         if (d / name).exists()), None)
        if artifact is None:
            raise HTTPException(404, "subtitle lines are not available until ASR finishes")
        raw = json.loads(artifact.read_text())
        has_translation = artifact.name != "segments.json"
        brief_path = d / "brief.json"
        brief = json.loads(brief_path.read_text()) if brief_path.exists() else None
        rows = [{"start": s.get("start"), "end": s.get("end"), "gender": s.get("gender"),
                 "src": s.get("src", s.get("text")), "text": s.get("text") if has_translation else None,
                 "polished": bool(s.get("polished")),
                 "failed": has_translation and s.get("text") == "[untranslated]"}
                for s in raw]
        model = cfg["models"].get(job["lang"], "")
        return {"brief": brief, "rows": rows, "cloud_model": model if agy.is_agy(model) else None,
                "cloud_fallbacks": job.get("cloud_fallbacks", 0)}

    @app.post("/jobs/{job_id}/retry")
    def retry(job_id: int):
        if not jobs.get(db, job_id):
            raise HTTPException(404, "no such job")
        jobs.update(db, job_id, error=None)
        return {"ok": True}

    def _job(job_id):
        j = jobs.get(db, job_id)
        if not j:
            raise HTTPException(404, "no such job")
        return j

    @app.delete("/jobs/{job_id}")
    def remove(job_id: int):
        j = _job(job_id)
        if j["error"] is None and j["stage"] != "done" and not j["control"]:
            jobs.update(db, job_id, control="delete")  # running: its stage stops at the next progress tick
            return {"ok": True, "pending": True}
        jobs.delete(db, job_id)
        config.clear_job_dir(cfg, job_id)
        return {"ok": True, "pending": False}

    @app.post("/jobs/{job_id}/pause")
    def pause(job_id: int):
        j = _job(job_id)
        if j["error"] is not None or j["stage"] == "done":
            raise HTTPException(409, "only a queued or running job can be paused")
        jobs.update(db, job_id, control="pause")
        return {"ok": True}

    @app.post("/jobs/{job_id}/resume")
    def resume(job_id: int):
        if _job(job_id)["control"] != "pause":
            raise HTTPException(409, "job is not paused")
        jobs.update(db, job_id, control=None)
        return {"ok": True}

    @app.post("/jobs/{job_id}/mux")
    def mux_job(job_id: int):
        j = jobs.get(db, job_id)
        if not j:
            raise HTTPException(404, "no such job")
        try:
            return {"path": str(mux.mux(j["video"]))}
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))

    @app.get("/browse")
    def browse(dir: str = str(Path.home())):
        d = Path(dir).expanduser()
        if not d.is_dir():
            raise HTTPException(400, f"not a folder: {d}")
        items = [p for p in sorted(d.iterdir(), key=lambda p: p.name.lower()) if not p.name.startswith(".")]
        return {"dir": str(d.resolve()), "parent": str(d.resolve().parent),
                "dirs": [p.name for p in items if p.is_dir()],
                "videos": [p.name for p in items if p.is_file() and p.suffix.lower() in VIDEO_EXT]}

    @app.get("/doctor")
    def doctor():
        return {"problems": preflight.check(cfg)}

    return app


def check_host(host, allow_remote):
    if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
        raise SystemExit(f"refusing to bind {host}: autosub is local-only (R4). Pass --allow-remote to override.")


def serve(cfg, host="127.0.0.1", port=8765, allow_remote=False):
    import uvicorn
    check_host(host, allow_remote)
    db = config.db_path(cfg)
    scheduler.start_worker(cfg, db)
    print(f"autosub UI: http://{host}:{port}")
    uvicorn.run(create_app(cfg, db), host=host, port=port, log_level="warning")
