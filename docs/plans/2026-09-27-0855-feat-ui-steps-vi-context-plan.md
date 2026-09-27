---
title: UI Pipeline Steps and Vietnamese Context - Plan
type: feat
date: 2026-09-27
topic: ui-steps-vi-context
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
origin: docs/plans/2026-09-26-1137-feat-local-gpu-autosub-mvp-plan.md
---

# UI Pipeline Steps and Vietnamese Context - Plan

## Goal Capsule

- **Objective:** The operator can see in the web UI which step each job is on and what the steps are, and Vietnamese subtitles follow the scene: who is talking to whom, their relationship, and what an elided subject refers to.
- **Means:** A new language-independent "scene brief" stage (KTD1), two-sided source context in translation prompts (KTD2), a step strip plus a side-by-side subtitle viewer in the UI (KTD4, KTD5).
- **Product authority:** The operator (single user). Subtitle editing, bigger models, and real speaker diarization are not active scope.
- **Open blockers:** None.
- **Stop conditions:** Stop and ask if the brief stage pushes a 2 h video past ~75 min total for EN+VI, or if `num_ctx 4096` no longer fits Gemma 3 4B fully on the GPU and 3072 doesn't either.
- **Execution profile:** Single developer, sequential U1 → U7. Mostly Python stdlib + existing FastAPI; no new dependencies.

## Product Contract

### Summary

Show the pipeline as named steps per job and let the operator read source and translation side by side. Give the translator a per-video scene brief and both-sided source context so Vietnamese pronouns and meaning match the scene.

### Problem Frame

Each translation call sees 12 Japanese lines plus only the 6 previous *Vietnamese* lines: no Japanese source for them, no following lines, no idea who the characters are. Japanese drops subjects constantly, so meaning and pronouns get guessed per window. The Vietnamese address rule is a fixed male-anh/female-em for every video, which is wrong when the relationship is e.g. boss/employee or older/younger. In the UI, a job shows one stage word and a bar, with ETA stuck at "—" on first use (the step strip's per-step elapsed time replaces the ETA as the progress signal), so the operator can't tell what's left or judge quality without opening files.

### Requirements

**Pipeline visibility (UI)**

- R1. Each job shows every step in order (Lọc âm thanh, Nhận giọng nói, Tóm tắt bối cảnh, Dịch, Chỉnh câu, Căn thời gian), each marked done / running / pending / failed.
- R2. Done steps show how long they took; the running step shows its progress and elapsed time.
- R3. The add-video card explains in one line per step what it does.
- R4. A finished (or partly finished) job has a "Xem phụ đề" view listing each line: time, speaker tag (nam/nữ/?), Japanese source, translation, and whether it was polished or failed. Read-only.
- R5. The scene brief a job used is visible in the viewer, so the operator can see why pronouns were chosen.

**Vietnamese context**

- R6. Before translating, each video gets a scene brief from its full transcript: characters, their relationship, setting, and the Vietnamese address pair to use. It's computed once per video and shared by every language job on that video.
- R7. Every translation and polish prompt includes the scene brief.
- R8. Every translation window sees the Japanese source of the previous lines with their translations, and the Japanese source of the following lines, marked as context only.
- R9. If the brief can't be produced (model error, refusal, unparseable), translation continues with the static `address` rule and the viewer shows the brief was skipped.
- R10. English jobs also get the brief and two-sided context. The Vietnamese address pair is ignored for English.

### Acceptance Examples

- AE1. A job mid-translation shows the first two steps done with durations, "Tóm tắt bối cảnh" done, "Dịch" running with a bar, and the last two pending. **Covers R1, R2.**
- AE2. Opening "Xem phụ đề" on the 5-min clip lists about 80 rows with JA and VI side by side, and the brief at the top. **Covers R4, R5.**
- AE3. Adding EN+VI for one video runs the brief once; the second job reuses it. **Covers R6.**
- AE4. With the brief model stubbed to fail, the job still finishes; the viewer shows "Không có tóm tắt bối cảnh" and the static rule was used. **Covers R9.**

### Success Criteria

- Operator review on the 5-min clip: pronouns match the brief's relationship, and lines the operator flagged as wrong-context in the current output read correctly in the new one. *(Human judgment; no automatic metric exists for this.)*
- Full 2 h video, EN+VI, stays under ~75 min total on the RTX 3050 (currently ~50 min without the brief).
- Failed VI lines stay at or below the current 10 on the full video.

### Scope Boundaries

- Out: editing lines in the UI, re-translating single lines on demand, bigger translation models, real diarization (pyannote), changing the ASR model.

## Planning Contract

### Key Technical Decisions

- KTD1. New scheduler stage `brief` between `asr` and `translate` (`transcribed → briefed`), in LLM_STAGES so the VRAM wait applies. The output `brief.json` is cached per video like the ASR transcript (key: video path + ASR model + brief prompt version), so a second language job copies it. Governs R6, AE3.
- KTD2. Brief = map-reduce over the Japanese transcript. Chunks of ~150 lines (fits `num_ctx` 4096) are each summarized to ≤5 bullet facts. One final call merges the facts into a JSON object: `{"summary", "characters": [{"name_or_role", "gender", "age_hint"}], "relationship", "setting", "vi_address": {"male_self", "male_to_female", "female_self", "female_to_male"}}`. The model is a fixed `brief_model` config key (default: the `vi` model, Gemma), so the shared brief doesn't depend on which language job runs first. It uses the same refusal/fallback path as translation. The cache key includes `brief_model`. Governs R6, R9.
- KTD3. Prompt context becomes: brief (compact, ≤600 chars), then the previous `context_lines` pairs as `JA => VI`, then the window to translate, then the next `lookahead_lines` (default 4) Japanese lines under "Following lines (context only, do not translate)". The address rule comes from `brief.vi_address` when present, else the static `address` config. Governs R7, R8, R9, R10.
- KTD4. `num_ctx` becomes a config key `llm_ctx`, default 4096. It's verified in U3 to keep the models actually configured on this machine (`autosub.toml`: `gemma3:4b` for VI, `huihui_ai/qwen3-abliterated:4b` for EN) 100% on the GPU with `num_gpu 99`; fall back to 3072 if not. The code default for EN is still the 8B model, which doesn't fit this GPU at any context size. Governs the stop condition.
- KTD5. The UI builds the step strip from `scheduler.STAGES` exposed by the API (no hard-coded list in JS). Durations come from the existing `stage_times`. The running step's elapsed time comes from a new `stage_started_at` column set when a stage begins. Governs R1, R2.
- KTD6. The viewer reads the job's newest artifact (`polished.json` → `translated.json` → `segments.json`) through a new `GET /jobs/{id}/lines`. It returns rows plus the brief; no new storage. Governs R4, R5.

### High-Level Technical Design

*Directional, not a spec.*

```
queued → gate → gated → asr → transcribed → brief → briefed → translate → translated → polish → polished → retime → done
                                   │                 │
                     segments.json (cached)   brief.json (cached per video)
                                                     │
translate window prompt:  [brief] + prev 6 (JA => VI) + WINDOW (12 JA, tagged M/F) + next 4 JA (context only)
```

### Assumptions

- A 4B model can extract relationship/setting facts from noisy ASR text well enough to beat a fixed rule. U7's review checks this. If it can't, the fallback rule still applies.
- 150-line chunks × ~16 for a 2 h video, at ~3–5 s per call, adds ~1–2 min per video.
- Longer prompts (brief + pairs + lookahead, roughly 2× input tokens) may slow translate by ~30–50%, from ~20 to ~26–30 min for EN+VI. That keeps the total inside the ~75 min budget. Measured in U7.

## Implementation Units

### U1. Brief stage and cache

**Goal:** Produce and cache `brief.json` per video.
**Requirements:** R6, R9, AE3, AE4.
**Dependencies:** none.
**Files:** `autosub/stages/brief.py`, `autosub/scheduler.py`, `autosub/web.py` (STAGE_FROM/ETA pick it up), `tests/test_brief.py`, `tests/test_scheduler.py`.
**Approach:**
1. Add `("brief", "transcribed", "briefed")` to STAGES, and change translate to start from `"briefed"`. Add `"brief"` to LLM_STAGES and run it in-process like translate.
2. `brief.run`: for each job, check the cache (hash of video + asr_model + `BRIEF_VERSION`). On a miss, chunk the segment texts, call the model per chunk for facts, then do one merge call to JSON.
3. Validate the JSON shape. On any failure write `{"skipped": "<reason>"}` rather than failing the job (R9). Unload the model at the end, as translate does.
4. Store `brief.json` in the job dir and the cache.

**Patterns to follow:** `stages/asr.py` cache_path, `stages/translate.py` with_fallback/unload_all.
**Test scenarios:**
- Two jobs on the same video → the model is called for the first only; the second copies the cache. Covers AE3.
- The merge returns invalid JSON on both models → `brief.json` has `skipped`, the job reaches `briefed`. Covers AE4.
- A transcript of 400 lines → 3 chunk calls + 1 merge.
- Stage order: a job at `transcribed` goes through brief before translate; an old job already at `translated` isn't touched.

**Verification:** On the 5-min clip, `brief.json` has a relationship and a `vi_address` that make sense to the operator.

### U2. Two-sided context and brief in prompts

**Goal:** The translator sees the scene brief, previous source+translation pairs, and following source lines.
**Requirements:** R7, R8, R9, R10.
**Dependencies:** U1.
**Files:** `autosub/stages/translate.py`, `autosub/stages/polish.py`, `autosub/config.py`, `tests/test_translate_windows.py`.
**Approach:**
1. `translate_texts` takes the source list and keeps pairs, passing `prev=[(ja, vi)…]` and `next=[ja…]` per window (`lookahead_lines`, default 4).
2. `prompt()` renders the order in KTD3. Pick the address from `brief.vi_address` (vi only) else `address_rule`.
3. The polish prompt gets the brief line and the same address source.
4. Keep the pivot path working. The pivot legs get the brief only: their source isn't Japanese, so two-sided JA context applies to the direct path.

**Test scenarios:**
- Prompt contains the brief summary, the `JA => VI` previous pairs, and the "do not translate" following lines, in that order.
- The last window has no following lines, and no empty section header is emitted.
- VI with `brief.vi_address` → it replaces the static rule; EN → no address rule.
- Brief skipped → the static rule is used.
- The window output count is still validated against the window only (following lines aren't counted).

**Verification:** Unit tests pass; manual look at one rendered prompt.

### U3. Larger LLM context, verified on the GPU

**Goal:** Room for brief + two-sided context without spilling to CPU.
**Requirements:** supports R7, R8; stop condition.
**Dependencies:** none.
**Files:** `autosub/ollama.py`, `autosub/config.py`, `tests/test_translate_windows.py`.
**Approach:** Replace the hard-coded `num_ctx` with `cfg["llm_ctx"]` (default 4096), passing cfg into `ollama.chat`. Check the two configured 4B models, not the 8B code default. The existing fit check (`min_gpu_share`) catches spills.
**Execution note:** Runtime check on the real GPU is the proof: load Gemma 3 4B and Qwen3 4B at 4096 and read `/api/ps` `size_vram == size`.
**Test scenarios:**
- `chat` sends `num_ctx` from cfg.

**Verification:** Both models are 100% on the GPU at the chosen value; the value is recorded in the config comment.

### U4. Step data in the API

**Goal:** The API tells the UI the steps and their timing.
**Requirements:** R1, R2.
**Dependencies:** U1.
**Files:** `autosub/jobs.py`, `autosub/scheduler.py`, `autosub/web.py`, `tests/test_web.py`, `tests/test_jobs.py`.
**Approach:**
1. Add a `stage_started_at` column: `ALTER TABLE` if missing, so existing dbs keep working.
2. The scheduler sets `stage_started_at` for the batch when a stage starts.
3. `GET /steps` returns `[{name, label_vi, help_vi}]` from one list next to STAGES.
4. `GET /jobs` adds `steps: [{name, state, secs}]`, where state is done / running / pending / failed.

**Test scenarios:**
- A job at `gated` with no error → gate done, asr running, the rest pending. Covers AE1.
- A job with an error at `transcribed` → brief failed, later steps pending.
- An old db without the new column opens fine and gets the column.

**Verification:** `/jobs` JSON shows correct states during a real clip run.

### U5. Subtitle viewer API

**Goal:** Rows of source/translation/speaker for a job, plus its brief.
**Requirements:** R4, R5.
**Dependencies:** U1.
**Files:** `autosub/web.py`, `autosub/stages/polish.py`, `tests/test_web.py`, `tests/test_flags.py`.
**Approach:** `polish.polish()` sets `seg["polished"] = True` only when a rewrite actually replaced the text. A refused rewrite keeps the old line and isn't marked. `GET /jobs/{id}/lines` reads the newest of polished / translated / segments. It returns `{brief, rows: [{start, end, gender, src, text, polished, failed}]}`. It returns 404 before ASR finishes.
**Test scenarios:**
- A job with only `segments.json` → rows have src, no text.
- A job with `polished.json` → rows have both; `[untranslated]` rows are marked failed.
- A flagged line whose rewrite was refused → `polished` is false (the text is unchanged).
- The brief is skipped → `brief.skipped` is set. Covers AE4.

**Verification:** Endpoint returns the clip's ~80 rows.

### U6. UI: step strip, step legend, viewer

**Goal:** Show the steps and let the operator read subtitles side by side.
**Requirements:** R1–R5, AE1, AE2.
**Dependencies:** U4, U5.
**Files:** `autosub/static/index.html`.
**Approach:**
1. Replace the stage/progress columns with a step strip per job: small labeled pills, with a tick and duration for done steps, a bar and elapsed time for the running step, grey for pending, red for failed.
2. Add a collapsible "Các bước xử lý" legend in the add-video card, built from `/steps`.
3. Add a "Xem phụ đề" button that opens a panel with the brief summary and the address pair at the top. The table below has time · nam/nữ · 日本語 · Tiếng Việt, and polished/failed badges. Add a filter for "chỉ dòng lỗi/đã chỉnh".
4. Keep the existing retry/mux/remove buttons.

**Test expectation:** none -- static page; verified by the manual run below.
**Verification:** Headless screenshot during a clip run matches AE1; the viewer on the finished clip matches AE2.

### U7. Before/after review and timing

**Goal:** Prove the context change helps, and stays inside the time budget.
**Requirements:** Success Criteria.
**Dependencies:** U1–U6.
**Files:** `docs/bench.md`.
**Approach:**
1. Run the 5-min clip before (current `master` outputs kept in `~/Downloads/autosub-test/vi-gemma/`) and after.
2. The operator compares them in the viewer.
3. Run the full 2 h video EN+VI and record per-stage times, failed lines, and the brief time.

**Test expectation:** none -- measurement.
**Verification:** `docs/bench.md` has both runs; the operator confirms the context got better or names remaining bad lines.

## Verification Contract

- `uv run pytest` passes.
- `uv run python -m autosub serve`: add the 5-min clip EN+VI and watch the step strip move through all 6 steps (AE1). Open "Xem phụ đề" (AE2).
- The full 2 h video EN+VI finishes in about 75 min or less, with 10 or fewer failed VI lines.

## Definition of Done

- R1–R10 are implemented and traced by the units above. AE1–AE4 are covered by tests or the manual run.
- `docs/bench.md` records the before/after clip review and the full-video timing.
- No leftover experimental prompts or dead config keys.
