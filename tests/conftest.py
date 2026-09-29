import pytest

from autosub import gpu


@pytest.fixture(autouse=True)
def private_gpu_lock(tmp_path, monkeypatch):
    """Tests must never wait on the real /tmp/nightsub-gpu.lock held by a running server job."""
    monkeypatch.setattr(gpu, "LOCK_PATH", str(tmp_path / "gpu.lock"))
