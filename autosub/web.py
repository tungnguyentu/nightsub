"""Local web UI (U8, KTD9): FastAPI on 127.0.0.1, one static page polling GET /jobs."""
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, jobs, scheduler

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v", ".ts", ".flv"}
STAGE_NAMES = [s[0] for s in scheduler.STAGES]
STAGE_FROM = {s[1]: i for i, s in enumerate(scheduler.STAGES)}


class NewJobs(BaseModel):
    paths: list[str]
    lang: str
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


def create_app(cfg, db):
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/jobs")
    def add(req: NewJobs):
        if req.lang not in cfg["models"]:
            raise HTTPException(400, f"unsupported language: {req.lang}")
        files = expand(req.paths)
        return {"ids": [jobs.add(db, str(f.resolve()), req.lang, req.tags, probe_minutes(f)) for f in files]}

    @app.get("/jobs")
    def status():
        rows = jobs.all(db)
        return [{**j, "eta_s": eta(j, rows)} for j in rows]

    @app.post("/jobs/{job_id}/retry")
    def retry(job_id: int):
        if not jobs.get(db, job_id):
            raise HTTPException(404, "no such job")
        jobs.update(db, job_id, error=None)
        return {"ok": True}

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
