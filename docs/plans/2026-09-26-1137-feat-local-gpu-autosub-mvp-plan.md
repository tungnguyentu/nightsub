---
title: Local GPU Autosub MVP - Plan
type: feat
date: 2026-09-26
topic: local-gpu-autosub-mvp
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-brainstorm
execution: code
---

# Local GPU Autosub MVP - Plan

## Goal Capsule

- **Objective:** On a single RTX 3050 6GB machine, the user can queue adult videos (JA/ZH/KO audio) in a local browser page and wake up to English or Vietnamese `.srt` files that follow the dialogue without hallucinated moan lines or refused lines.
- **Means:** Non-speech gate before ASR → stage-major batching (ASR for the whole queue, then one LLM load) → context-aware translation with a selective polish pass → deterministic retiming.
- **Product authority:** The user (sole operator now; repo intended to be shareable on GitHub later). Glossary/learning, disagreement review, and media-server integration are not active scope.
- **Open blockers:** None. Speed target and Vietnamese quality are verified by U9 before tuning.
- **Stop conditions:** Stop and ask if a 2h benchmark exceeds ~60 min after U9 tuning, or if no candidate model produces usable Vietnamese.
- **Execution profile:** Single developer, greenfield repo, sequential units U1→U9.

## Product Contract

### Summary

A local-only web app that subtitles and translates adult videos overnight on a 6GB GPU. Speed is the primary success bar; quality comes from keeping non-speech out of ASR and polishing only the lines that need it.

### Problem Frame

Generic Whisper tools hallucinate text over moans, breathing and music, and the closest existing tool (WhisperJAV) fixes this after the fact and is fragile (model.bin errors, silent empty SRTs). Aligned LLMs refuse explicit content. A 6GB card cannot hold an ASR model (~4.7GB) and an 8B LLM (~5GB) at once, so naive per-file pipelines spend their time swapping models.

### Actors

- A1. Operator — the user on their own Linux machine, dropping files in and collecting subtitles the next morning.

### Requirements

**Queue and web UI**

- R1. The operator adds one or more local video files to a queue from a local browser page and picks a target language per job: English or Vietnamese.
- R2. The page shows per-job stage and progress (gate, ASR, translate, polish, retime) and an estimated time remaining.
- R3. Output is an `.srt` written next to the source video, named with the language code (e.g. `.en.srt`, `.vi.srt`).
- R4. The app runs and serves only on the local machine; no video, audio or text leaves it.

**Speech capture**

- R5. Non-speech audio (moaning, breathing, music, silence) is classified before transcription, and only speech spans are transcribed.
- R6. Non-speech spans can optionally appear as short bracketed tags (e.g. `[moans]`) in the target language; the default is off.
- R7. Long files do not accumulate timestamp drift; subtitle timing stays aligned with speech through the whole video.

**GPU scheduling**

- R8. The ASR stage runs for every queued job before the translation model loads, and the translation model loads once for all queued transcripts.
- R9. GPU memory is confirmed released before the translation stage, and the translation model is confirmed fully GPU-resident; otherwise the affected jobs fail with a reported reason instead of falling back to CPU silently or running out of memory.
- R10. Work is checkpointed per stage, so a crash or restart resumes from the last completed stage instead of from zero.

**Translation**

- R11. Each line is translated with nearby lines as context, using a model that does not refuse explicit content.
- R12. Lines flagged as low-confidence or unnatural get a second rewrite pass for natural register; other lines skip it.
- R13. A refused or deflected translation is detected and retried with a fallback model; a line that still fails is marked in the output and counted in the job report, never dropped silently.
- R14. After translation, lines are retimed and split to readable length and reading speed.

**Reliability**

- R15. Before a job starts, the app checks that the GPU, ffmpeg, ollama and required models are present and reports anything missing in plain language.
- R16. A job that produces an empty or near-empty subtitle file is reported as failed with a reason, never as success.

### Key Flows

- F1. Overnight batch. **Trigger:** Operator drops 5 videos, picks EN for 4 and VI for 1, clicks start. **Steps:** preflight check → gate + ASR runs across all 5 → GPU freed → translation model loads once, translates all 5 → polish pass on flagged lines → retime → `.srt` files written. **Outcome:** In the morning the page shows 5 completed jobs with line counts and any failed-line counts. **Covers R1–R3, R5, R8, R11–R15.**
- F2. Crash recovery. **Trigger:** Machine reboots mid-translation. **Steps:** Operator reopens the app → the queue shows completed stages → resume. **Outcome:** ASR is not re-run. **Covers R10.**

### Acceptance Examples

- AE1. A 2-hour JA video with long non-speech sections produces no subtitle lines inside those sections unless tags are enabled. **Covers R5, R6.**
- AE2. With another GPU process holding memory, the job reports the conflict instead of crashing or running on CPU unannounced. **Covers R9.**
- AE3. A line the primary model refuses shows up translated by the fallback model, or as a marked failed line; the job report counts it. **Covers R13.**
- AE4. Missing ollama model → job does not start; the page names the missing model. **Covers R15.**

### Success Criteria

- A 2-hour video completes end-to-end (gate → retime) in about 30 minutes or less on the RTX 3050 6GB. *(Assumption — unbenchmarked.)*
- Across a sample of 3 real videos, zero hallucinated lines over non-speech and zero silently refused lines.
- The operator can follow the dialogue in both EN and VI outputs without rewinding for meaning. *(VI quality is an assumption; see Outstanding Questions.)*

### Scope Boundaries

**Out of scope for this MVP**
- Per-performer/studio glossary and learning from corrections.
- Two-model disagreement review queue.
- Watch-folder daemon or Jellyfin/Plex/Bazarr integration.
- Editing subtitles inside the UI; burning subtitles into video; dubbing.
- Target languages other than English and Vietnamese.
- Installer polish or GUI packaging for other users (repo stays reproducible, not packaged).

### Key Decisions

- Speed is the primary success bar over fansub-level quality. (session-settled: user-directed — chosen over "watchable without rewind" and "near-fansub quality": overnight throughput on a 3050 matters most.)
- Second translation pass only on flagged lines. (session-settled: user-directed — chosen over one pass with an opt-in polish toggle, and over always two passes: keeps most of the speed while fixing the worst lines.)
- Local web UI is the interface. (session-settled: user-directed — chosen over a CLI and a watch-folder daemon.)
- Targets are English and Vietnamese. (session-settled: user-directed — chosen over English-only and any-language.)
- Personal tool first, shareable on GitHub later. (session-settled: user-directed — chosen over personal-only and public product from day one.)
- Non-speech is gated before ASR rather than filtered after it; the gate uses SenseVoice audio-event tags plus VAD.
- Stage-major ordering (all ASR, then all translation) instead of per-file model swapping.

### Outstanding Questions

**Resolve Before Planning**
- None.

**Deferred to Planning**
- Which uncensored model(s) to use per language. dolphin3:8b is Llama-based and likely weak for Vietnamese; an uncensored Qwen-family model may serve both. Needs a quick VI quality check.
- Benchmark one real 2-hour file to confirm the ~30 min target and find the slowest stage.
- Check that SenseVoice event tags line up with the spans where Whisper hallucinates, on 3 sample clips.
- What signal flags a line for the polish pass (ASR confidence, translation heuristics, or both).
- Whether the first SenseVoice pass should run on CPU in parallel with GPU translation.

**Product Contract preservation:** Product Contract unchanged.

## Planning Contract

### Key Technical Decisions

- KTD1. Python 3.12 pinned via `uv` (system Python is 3.14, which the CUDA ML wheels may not support yet). Governs all units.
- KTD2. Each GPU stage (gate, ASR) runs as a separate subprocess over the whole queue. The process exits, so the driver reclaims all VRAM. Translation talks to ollama over HTTP with `keep_alive: 0` on the last request and confirms unload via `/api/ps`. Governs R8, R9.
- KTD3. Job and stage state lives in one SQLite file (stdlib `sqlite3`). Every stage writes its output as a JSON file under the job's work dir and marks itself done, so a restart resumes at the first stage that isn't done. Governs R10.
- KTD4. Gate: FunASR `SenseVoiceSmall` with `fsmn-vad`. Spans whose tags are events (BGM, laughter, crying, cough/sneeze and similar) or that have no text are non-speech. Everything else is a speech span. Governs R5, R6.
- KTD5. ASR: faster-whisper `large-v3`, `compute_type=int8_float16`, run per speech span with absolute offsets (no drift, R7). Settings: `condition_on_previous_text=False`, `vad_filter=False` (the gate already did VAD), and each segment's `avg_logprob` is kept as the confidence signal. Swapping in anime-whisper/kotoba is a config change, not in MVP.
- KTD6. Translation: ollama `/api/chat` with windows of ~12 lines, prompting for a numbered JSON array out. Primary model per language comes from config. Default is a Qwen3 abliterated 8B Q4 for both EN and VI, with `dolphin3:8b` as fallback. *(Assumption: tags unverified; U9 confirms.)* Governs R11, R13.
- KTD7. Refusal detection is a heuristic, not a model: the line count doesn't match, or the output matches refusal phrases (EN and VI), or the output is in the source script. Governs R13.
- KTD8. Polish flag = ASR `avg_logprob` below a threshold, OR translation length ratio outside bounds, OR the output repeats itself. Flagged lines get one rewrite call with nearby context. Governs R12.
- KTD9. Web UI: FastAPI serving one static HTML page. The page polls a JSON status endpoint every 2s. Bound to `127.0.0.1` only. No SSE, no frontend build. Governs R1, R2, R4.
- KTD10. The scheduler is a single in-process worker thread that walks the queue stage-major: all gates → all ASR → all translate → all polish → all retime. Jobs added mid-run join the next pass. Governs R8.

### High-Level Technical Design

*Directional, not a spec.*

```
browser ──HTTP──> FastAPI (127.0.0.1)
                    │ enqueue / status
                    ▼
               SQLite jobs.db  <──── worker thread (stage-major)
                                        │
   preflight ─> [gate subprocess: all jobs] ─> [asr subprocess: all jobs]
            ─> VRAM check ─> translate (ollama HTTP) ─> polish flagged
            ─> retime ─> write <video>.<lang>.srt ─> empty-check ─> done|failed
```

Job stage states: `queued → gated → transcribed → translated → polished → retimed → done`, plus `failed(reason)` from any stage. A restart resumes from the stored stage.

### Assumptions

- Ollama model tags for Qwen3-abliterated and dolphin3 exist and fit about 5GB at Q4. Verified in U9.
- SenseVoice event tags cover moaning/breathing well enough. If not, U3 falls back to VAD plus an ASR no-speech/low-logprob filter.
- Throughput reaches ~30 min per 2h video. Measured in U9.

## Implementation Units

### U1. Project scaffold and preflight check

**Goal:** Runnable `uv` project with config and a `doctor` check.
**Requirements:** R15, AE4.
**Dependencies:** none.
**Files:** `pyproject.toml`, `autosub/__init__.py`, `autosub/config.py`, `autosub/preflight.py`, `tests/test_preflight.py`.
**Approach:**
1. `config.py` holds a plain dict of defaults (models per language, thresholds, work dir), overridable by an optional `autosub.toml` (stdlib `tomllib`).
2. `preflight.py` returns a list of human-readable problems covering: `nvidia-smi` present and GPU free VRAM ≥ 5GB, `ffmpeg` on PATH, ollama reachable, and the configured models present in `/api/tags`.
3. `python -m autosub doctor` prints the problems.

**Test scenarios:**
- Missing ollama model → the problem text names the model. Covers AE4.
- Ollama unreachable → one clear problem, no traceback.
- All present → empty list.

**Verification:** `doctor` on this machine reports the missing translation models (only minicpm/bge-m3 are installed today).

### U2. Job store and stage-major scheduler

**Goal:** Persistent queue that runs stages across all jobs and resumes after a crash.
**Requirements:** R8, R9, R10, F2, AE2.
**Dependencies:** U1.
**Files:** `autosub/jobs.py`, `autosub/scheduler.py`, `autosub/gpu.py`, `tests/test_jobs.py`, `tests/test_scheduler.py`.
**Approach:**
1. `jobs.py` is a `sqlite3` table (id, video path, lang, stage, progress, eta, error, counts) with small helper functions.
2. `scheduler.py` loops through the stage list. For each stage it collects jobs at the previous state and runs the stage once for all of them (KTD10). It runs preflight at the start of each pass.
3. `gpu.py` reads free VRAM from `nvidia-smi --query-gpu=memory.free` and the holders from `--query-compute-apps=pid,process_name,used_memory`. Before the LLM stage it waits up to 30s for VRAM to free. If it doesn't, the affected jobs fail with a "GPU busy: <process>" reason (AE2).
4. Subprocess stages are called as `python -m autosub.stages.<name> <job ids>`. A non-zero exit marks those jobs failed with stderr's last line.

**Test scenarios:**
- Three jobs at `gated`: the ASR stage is invoked once with all three ids.
- Restart with one job at `transcribed`: resumes at translate, and ASR is not called. Covers F2.
- VRAM never frees (stubbed): the job fails with a GPU-busy reason. Covers AE2.
- A stage subprocess exits 1: only its jobs fail, and others continue.

**Verification:** Killing the app mid-run and restarting continues from the stored stage.

### U3. Non-speech gate stage

**Goal:** Speech spans plus optional non-speech tags per job.
**Requirements:** R5, R6, AE1.
**Dependencies:** U2.
**Files:** `autosub/stages/gate.py`, `autosub/tags.py`, `tests/test_tags.py`, `tests/test_gate.py`.
**Approach:**
1. `ffmpeg` extracts 16kHz mono wav into the job dir.
2. SenseVoiceSmall + fsmn-vad (KTD4) run on it.
3. Each VAD span is classified as speech or non-speech by its tags. Output is `spans.json`: `[{start, end, kind, tag}]`.
4. `tags.py` maps event tags to localized labels (`[moans]`/`[rên]`, `[music]`/`[nhạc]`). They're used only when the job's tags option is on.

**Test scenarios:**
- A span with only an event tag and no text → non-speech.
- A span with text and an emotion tag → speech.
- Tags option off → no tag lines produced. Covers AE1.

**Verification:** On 3 sample clips, listening to the non-speech spans confirms they contain no dialogue.

### U4. ASR stage

**Goal:** Timed source-language segments with confidence.
**Requirements:** R7.
**Dependencies:** U3.
**Files:** `autosub/stages/asr.py`, `tests/test_asr_offsets.py`.
**Approach:** Load faster-whisper once per subprocess (KTD5). Transcribe each speech span sliced from the wav and add the span's start to every timestamp. Write `segments.json` with `[{start, end, text, logprob}]`. Report progress as spans done / total.
**Test scenarios:**
- Offset math: a segment at 1.2s inside a span starting at 3600s → 3601.2s.
- An empty span result is skipped, not written as an empty line.

**Verification:** On a 2h file, the last line's timing matches the audio by ear (no drift).

### U5. Translation with refusal fallback

**Goal:** Every segment translated to the job language, with refusals retried.
**Requirements:** R11, R13, AE3.
**Dependencies:** U4.
**Files:** `autosub/stages/translate.py`, `autosub/refusal.py`, `autosub/ollama.py`, `tests/test_refusal.py`, `tests/test_translate_windows.py`.
**Approach:**
1. Windows of ~12 lines, with the previous 3 translated lines as context (KTD6).
2. Parse the JSON array. A parse failure (truncated, fenced or stray text) counts as a refusal. After the first request, check `/api/ps`: if `size_vram < size` the stage fails with "model did not fit in GPU memory" (R9). If `refusal.py` flags the window (KTD7), retry that window with the fallback model.
3. If the fallback also fails, split the window into single lines and retry once more.
4. Still failing → the line text becomes `[untranslated]` and the job's failed-line count goes up.
5. On the last request, send `keep_alive: 0`.

**Test scenarios:**
- Refusal phrase in EN or VI output → flagged.
- 12 in, 11 out → flagged.
- Output still in Japanese kana → flagged.
- Primary refuses and fallback succeeds → line translated, count 0. Covers AE3.
- Both refuse → `[untranslated]` marker, count 1. Covers AE3.
- Output is invalid JSON → treated as a refusal and retried.
- `/api/ps` shows partial CPU offload → stage fails with the GPU-fit reason.

**Verification:** A clip known to trigger refusals on an aligned model completes with a zero or reported count.

### U6. Selective polish pass

**Goal:** Rewrite only the lines flagged for natural register.
**Requirements:** R12.
**Dependencies:** U5.
**Files:** `autosub/stages/polish.py`, `autosub/flags.py`, `tests/test_flags.py`.
**Approach:** `flags.py` implements KTD8. Flagged lines are batched with ±2 lines of context for one rewrite call. The pass reuses U5's refusal handling. It records the flagged share per job for the report.
**Test scenarios:**
- Low logprob → flagged.
- Normal line → not flagged.
- Repeated-token output ("あああ…" or "ha ha ha ha") → flagged.

**Verification:** The flagged share on sample clips is roughly 10–30%. If it's far outside that, tune the thresholds in config.

### U7. Retime and SRT output

**Goal:** Readable, well-timed `.srt` files next to the video, plus an empty-output guard.
**Requirements:** R3, R14, R16.
**Dependencies:** U6.
**Files:** `autosub/retime.py`, `autosub/srt.py`, `tests/test_retime.py`, `tests/test_srt.py`.
**Approach:**
1. `retime.py` splits lines over 42 characters into 2 lines, and splits over-long cues at punctuation.
2. It extends a cue's duration up to a reading speed of about 17 CPS without overlapping the next cue, and enforces a minimum of 0.8s.
3. `srt.py` writes `<stem>.<lang>.srt`. Fewer than 5 cues per 10 minutes of speech → the job fails with an "empty/near-empty output" reason (R16).

**Test scenarios:**
- A 60-character line is split into 2 lines.
- A cue at 30 CPS is extended, but not past the next cue's start.
- Zero cues → job failed, and no file claims success.
- SRT timestamp format `00:01:02,345`.

**Verification:** The generated file loads in mpv with no overlapping cues.

### U8. Local web UI

**Goal:** Add videos, pick a language, and watch progress.
**Requirements:** R1, R2, R4, F1.
**Dependencies:** U2 (and U3–U7 for real runs).
**Files:** `autosub/web.py`, `autosub/static/index.html`, `tests/test_web.py`.
**Approach:**
1. FastAPI bound to `127.0.0.1` (KTD9) with three endpoints:
   - `POST /jobs`, taking local paths or a folder, plus lang and tags options.
   - `GET /jobs` for status.
   - `POST /jobs/{id}/retry`.
2. One HTML page with a path input (the browser can't read local paths from a drag-drop, so the user pastes a path or a folder), an EN/VI select, and a table. It polls every 2s.
3. ETA = measured seconds per audio-minute per stage from past jobs × remaining audio minutes. Show "—" until at least one job has finished that stage.
4. `python -m autosub serve` starts the worker thread and the server together.

**Test scenarios:**
- POST with a folder → one job per video file.
- A non-existent path → 400 with the reason.
- The server refuses to bind to `0.0.0.0` unless explicitly overridden. Covers R4.
- A failed job shows its reason in the status JSON.

**Verification:** Run F1 end-to-end on 2 short clips through the browser.

### U9. Benchmark and model check

**Goal:** Measure the success criteria and confirm the default models.
**Requirements:** Success Criteria, Assumptions.
**Dependencies:** U1–U8.
**Files:** `scripts/bench.py`.
**Approach:** Run the full pipeline on one 2h file. Print per-stage wall time, the flagged share and the failed-line count. Separately, translate a fixed 30-line sample with each candidate model into EN and VI so the user can judge by eye.
**Test expectation:** none -- this is a measurement script, and a human reads its output.
**Verification:** The numbers are recorded in `docs/bench.md`, and the default models in config are updated to the winners.

## Verification Contract

- `uv run pytest` passes (unit tests U1–U8; no GPU needed thanks to stubs).
- `uv run python -m autosub doctor` reports no problems on the dev machine after the models are pulled.
- Smoke: a 2-min sample clip goes through the UI to `.en.srt` and `.vi.srt` on a real GPU.
- Target: `scripts/bench.py` on a 2h file ≤ ~30 min total. Otherwise record the gap and the slowest stage.

## Definition of Done

- All of R1–R16 have a unit or test tracing to them. AE1–AE4 pass.
- The benchmark has been run and `docs/bench.md` records it.
- Nothing is sent off-machine: no network calls except to `127.0.0.1` ollama and model downloads.
- Abandoned experimental code (alternate ASR models, unused prompts) is removed.
