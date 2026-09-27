import contextlib
import fcntl
import subprocess
import time


def _smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]


def free_mb():
    return int(_smi("gpu=memory.free")[0][0])


def holders():
    return [f"{r[1].split()[0].rsplit('/', 1)[-1]} (pid {r[0]}, {r[2]} MiB)" for r in _smi("compute-apps=pid,process_name,used_memory") if len(r) == 3]


def wait_free(need_mb, timeout_s=30):
    """None if enough VRAM frees up in time, else a 'GPU busy' reason (R9, AE2)."""
    deadline = time.monotonic() + timeout_s
    while True:
        if free_mb() >= need_mb:
            return None
        if time.monotonic() >= deadline:
            return "GPU busy: " + (", ".join(holders()) or f"less than {need_mb} MiB free")
        time.sleep(1)


LOCK_PATH = "/tmp/nightsub-gpu.lock"


@contextlib.contextmanager
def lock(path=LOCK_PATH):
    """One GPU stage at a time across processes (web worker, bench, CLI). Blocks until free."""
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
