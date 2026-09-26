# nightsub

Local-only auto-subtitles for JA/ZH/KO video, translated to English or Vietnamese, sized for a 6 GB GPU.
Nothing leaves your machine.

Pipeline: non-speech gate (SenseVoice + VAD) → Whisper large-v3 → local LLM translation via ollama → selective polish → retime → `.srt`.

```sh
uv sync --extra gpu
ollama pull huihui_ai/qwen3-abliterated:4b
uv run python -m autosub doctor          # checks GPU, ffmpeg, ollama, models
uv run python -m autosub serve           # http://127.0.0.1:8765
uv run python -m autosub mux video.mp4   # embed .en/.vi.srt -> video.subs.mkv
```

Per-machine overrides go in `./autosub.toml` (see `autosub/config.py` for keys).
