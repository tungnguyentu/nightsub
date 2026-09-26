from autosub import config, preflight


def setup(monkeypatch, tags=None, free=6000):
    monkeypatch.setattr(preflight.shutil, "which", lambda n: "/usr/bin/" + n)
    monkeypatch.setattr(preflight.gpu, "free_mb", lambda: free)
    monkeypatch.setattr(preflight.gpu, "holders", lambda: [])

    def fake_tags(url):
        if tags is None:
            raise ConnectionRefusedError("refused")
        return tags
    monkeypatch.setattr(preflight.ollama, "tags", fake_tags)
    return dict(config.DEFAULTS)


def test_missing_model_named(monkeypatch):
    cfg = setup(monkeypatch, tags=["dolphin3:8b"])
    probs = preflight.check(cfg)
    assert len(probs) == 1 and cfg["models"]["en"] in probs[0]


def test_ollama_unreachable_one_problem(monkeypatch):
    probs = preflight.check(setup(monkeypatch, tags=None))
    assert len(probs) == 1 and "ollama not reachable" in probs[0]


def test_all_present(monkeypatch):
    cfg = setup(monkeypatch, tags=[cfg_m for cfg_m in [config.DEFAULTS["models"]["en"], "dolphin3:8b"]])
    assert preflight.check(cfg) == []


def test_low_vram(monkeypatch):
    cfg = setup(monkeypatch, tags=[config.DEFAULTS["models"]["en"], "dolphin3:8b"], free=1000)
    assert "GPU memory" in preflight.check(cfg)[0]
