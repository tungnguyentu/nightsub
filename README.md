# nightsub

Automatic subtitles for Japanese video, translated to English or Vietnamese, on a consumer NVIDIA GPU (built and tested on an RTX 3050 6 GB). Everything runs on your machine by default; cloud translation is an explicit opt-in.

**Pipeline:** non-speech gate (SenseVoice + VAD) → speech recognition (kotoba-whisper) → scene brief → translation (local LLM via ollama) → polish of weak lines → retime → `.srt`, optionally embedded into an `.mkv`.

## Requirements

- Linux with an NVIDIA GPU (6 GB VRAM is enough for the 4B models below) and a recent driver
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you)
- `ffmpeg` / `ffprobe` on `PATH`
- [ollama](https://ollama.com) running locally (`ollama serve`)

## Install

```sh
git clone https://github.com/tungnguyentu/nightsub.git
cd nightsub
uv sync --extra gpu            # torch, faster-whisper, funasr, CUDA 12 libs (~3 GB)

# translation models (pick what fits your GPU, see Configuration)
ollama pull gemma3:4b                           # Vietnamese (default) + fallback
ollama pull huihui_ai/qwen3-abliterated:4b      # English on a 6 GB card
```

The speech models (SenseVoiceSmall, fsmn-vad, kotoba-whisper, ~3 GB) download automatically on the first run.

## Quick start

```sh
uv run python -m autosub doctor                 # checks GPU memory, ffmpeg, ollama, models
uv run python -m autosub serve                  # web UI on http://127.0.0.1:8765
uv run python -m autosub serve --port 8790      # if 8765 is taken by another app
```

In the web UI:

1. Browse to a folder, click the videos you want (or **Chọn cả thư mục này** for a whole folder).
2. Tick **Tiếng Anh** and/or **Tiếng Việt**. Each video is transcribed once even when both are picked.
3. Optional: **Ghi chú âm thanh** adds `[nhạc]`, `[cười]`… for non-speech parts.
4. Click **Bắt đầu**. Each job shows its six steps, progress and time per step.
5. When a job is done: **Xem phụ đề** shows Japanese source and translation side by side, **Gắn vào video** writes `<video>.subs.mkv`.

Jobs can be paused (**Tạm dừng** / **Tiếp tục**), retried after an error (**Thử lại**) and deleted (**Xóa**) at any time. Subtitles are written next to the video as `<name>.en.srt` / `<name>.vi.srt`.

## Commands

| Command | What it does |
|---|---|
| `uv run python -m autosub doctor` | Lists problems (missing models, low VRAM, ollama down); prints `OK: ready` otherwise |
| `uv run python -m autosub serve [--port N] [--host H]` | Web UI + background worker. Binds `127.0.0.1` only; `--allow-remote` to bind another host |
| `uv run python -m autosub mux <video>` | Embeds every `<video>.en.srt` / `.vi.srt` as soft subtitle tracks into `<video>.subs.mkv` (no re-encode) |
| `uv run python scripts/bench.py pipeline <video> --lang en,vi` | Runs the whole pipeline without the UI and prints time per stage, cues, failed lines, and which model translated each window |
| `uv run python scripts/bench.py pipeline <video> --lang vi --work-dir <dir>` | Same, reusing a previous run's transcript cache |
| `uv run python scripts/bench.py models sample.txt [model …]` | Translates a text file (one source line per line) with each model into EN and VI for side-by-side comparison |
| `uv run pytest` | Test suite (no GPU or models needed) |

## Configuration

Defaults live in `autosub/config.py`. Override any key in `./autosub.toml` (gitignored, per machine). A working config for a 6 GB card with a desktop running:

```toml
vram_needed_mb = 2800          # free VRAM required before an LLM stage starts
fallback_model = "gemma3:4b"   # used when the main model refuses or fails
brief_model = "gemma3:4b"      # scene brief (characters, relationship, pronouns)

[models]
en = "huihui_ai/qwen3-abliterated:4b"
vi = "gemma3:4b"
```

Keys you are most likely to touch:

| Key | Default | Meaning |
|---|---|---|
| `models.en`, `models.vi` | qwen3 8B / gemma3 4B | Translation model per language. 8B models do not fit fully on a 6 GB GPU |
| `fallback_model` | `gemma3:4b` | Second try for refused / broken windows (and lines the cloud refuses) |
| `asr_model` | `kotoba-tech/kotoba-whisper-v2.0-faster` | Japanese-tuned Whisper; `large-v3` also works |
| `asr_language` | `ja` | Source language; `None` auto-detects per span (unreliable on short lines) |
| `address.vi` | anh/em rule | Vietnamese pronoun rule used when the scene brief has no consistent pair |
| `window`, `context_lines`, `lookahead_lines` | 12, 6, 4 | Lines per translation call, previous and following lines shown as context |
| `llm_ctx` | 4096 | ollama context size; both 4B models stay 100% on GPU at this size |
| `min_gpu_share` | 0.9 | Fail instead of silently running an LLM mostly on CPU |
| `work_dir` | `~/.local/share/autosub` | Job database, per-job files, transcript cache |

## Optional: cloud translation via Antigravity (`agy`)

Any model name starting with `agy/` is sent through the Antigravity CLI (`agy`) instead of ollama, for example:

```toml
[models]
vi = "agy/gemini-3.8-flash-low"    # `agy models` lists the available names
```

- **Subtitle text leaves your machine** and goes to the provider under your Antigravity account. Provider content policies apply; refused windows fall back to `fallback_model` locally, and the UI shows how many did.
- Cloud windows use `cloud_window` (24) lines and run `cloud_parallel` (4) calls at a time.
- Requires `agy` on `PATH` (or at `~/.local/bin/agy`) and a logged-in Antigravity account.

## Troubleshooting

- **`address already in use` on start:** another app owns port 8765. Use `serve --port 8790` (or any free port).
- **`Only N MiB GPU memory free`:** close GPU-heavy apps (browsers, OBS, other ollama models: `ollama ps`, `ollama stop <model>`), or lower `vram_needed_mb`.
- **`model did not fit in GPU memory`:** the model was split between GPU and CPU. Use a 4B model or free VRAM; raising `min_gpu_share` tolerance only makes it slower.
- **`CUDA failed with error out of memory` during recognition:** another GPU job was running. Only one GPU stage runs at a time across the UI and `bench.py` (`/tmp/nightsub-gpu.lock`); retry the job and it resumes from its checkpoint.
- **Changed the code or `autosub.toml`:** restart `serve`; config and in-process stages are loaded at start.

## Benchmarks

See [`docs/bench.md`](docs/bench.md). Roughly: a 2-hour video takes about 25–30 minutes for Vietnamese only, and about 1 hour for EN + VI fully local, on an RTX 3050 6 GB.
