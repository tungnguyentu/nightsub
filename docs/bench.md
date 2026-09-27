# Benchmarks

Machine: RTX 3050 6 GB (~4.5 GB free with the desktop running), 32 GB RAM.
Local config: ASR `kotoba-whisper-v2.0-faster`, `ja` fixed; EN `huihui_ai/qwen3-abliterated:4b`,
VI `gemma3:4b`; `llm_ctx` 4096 with `num_gpu 99`. Both 4B models load 100% on GPU
at 4096 (Gemma 2742 MiB, Qwen 3030 MiB). Subtitle text is not reproduced here.

## 5-min clip, EN+VI in one pass (2026-09-27)

| Stage | Minutes |
|---|---|
| gate | 0.6 |
| asr | 0.5 |
| brief | 0.2 |
| translate | 1.0 |
| polish | 0.2 |
| **total** | **2.6** |

| | EN | VI |
|---|---|---|
| cues | 80 | 80 |
| failed lines | 0 | 0 |
| flagged for polish | 6.3% | 18.8% |

Vietnamese pronoun counts (automatic word count):

| Pronoun | Before brief (`vi-gemma`) | After |
|---|---|---|
| mày / tao | 4 / 3 | 1 / 0 |
| cậu | 0 | 1 |
| anh / em | 0 / 0 | 5 / 4 |

The scene brief's `vi_address` was inconsistent on every clip run
(e.g. tôi/em/tôi/cháu), so the consistency check fell back to anh/em.
A 4B model does not pick a coherent pronoun pair reliably; the fallback plus
the off-register polish flag is what moves the numbers.

## Full 121-min video, EN+VI (2026-09-27, before the brief stage)

| Stage | Minutes |
|---|---|
| gate | 2.9 |
| asr (kotoba) | 15.1 |
| translate | 34.6 |
| polish | 5.6 |
| **total** | **58.2** |

EN 2423 cues, 0 failed. VI 2425 cues; 111 failed in that run from a refusal
false positive on "xin lỗi", fixed since (re-run of VI alone: 10 failed,
13.3 min reusing the transcript). Not yet re-measured with the brief stage.
