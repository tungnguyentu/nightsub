import sys
import traceback
import wave

from .. import config, jobs


def cli(load, process, to):
    """Subprocess entry: load models once, process each job id in argv, advance or fail it."""
    cfg = config.load()
    db = config.db_path(cfg)
    models = load(cfg)
    for job_id in map(int, sys.argv[1:]):
        job = jobs.get(db, job_id)
        try:
            process(cfg, db, job, config.job_dir(cfg, job_id), models)
            jobs.update(db, job_id, stage=to, progress=0)
        except Exception as e:
            traceback.print_exc()
            jobs.fail(db, job_id, f"{type(e).__name__}: {e}")


def read_wav(path):
    import numpy as np
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
