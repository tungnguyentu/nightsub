"""U9 benchmark. Human reads the output; record results in docs/bench.md.

  uv run python scripts/bench.py pipeline /path/video.mp4 [--lang en,vi]
  uv run python scripts/bench.py models sample.txt [model ...]   # sample.txt: ~30 source lines, one per line
"""
import argparse
import tempfile
import time

from autosub import config, jobs, scheduler
from autosub.stages import translate


def pipeline(video, langs, work_dir=None):
    # reuse a work dir to hit its ASR cache (skips re-transcribing the same video)
    cfg = {**config.load(), "work_dir": work_dir or tempfile.mkdtemp(prefix="autosub-bench-")}
    db = config.db_path(cfg)
    ids = [jobs.add(db, video, lang) for lang in langs.split(",")]  # one pass: ASR once, reused via the cache
    times = {}

    def timed(name, fn):
        def run(cfg, db, batch):
            t0 = time.monotonic()
            fn(cfg, db, batch)
            times[name] = time.monotonic() - t0
        return run
    t0 = time.monotonic()
    scheduler.run_pass(cfg, db, {n: timed(n, f) for n, f in scheduler.RUNNERS.items()})
    total = time.monotonic() - t0
    for n, t in times.items():
        print(f"{n:10} {t / 60:6.1f} min")
    print(f"{'total':10} {total / 60:6.1f} min for {jobs.get(db, ids[0])['audio_min'] or 0:.0f} min of audio")
    for i in ids:
        j = jobs.get(db, i)
        print(f"[{j['lang']}] stage={j['stage']} error={j['error']} cues={j['cues']} "
              f"failed_lines={j['failed_lines']} flagged_share={j['flagged_share']}")
    print("work dir:", cfg["work_dir"])


def models(sample, names):
    cfg = config.load()
    lines = [l.strip() for l in open(sample, encoding="utf-8") if l.strip()]
    for model in names or sorted({*cfg["models"].values(), cfg["fallback_model"]}):
        for lang in ("en", "vi"):
            c = {**cfg, "models": {lang: model}, "fallback_model": model}
            used = set()
            t0 = time.monotonic()
            out, failed = translate.translate_texts(c, lines, lang, used)
            translate.unload_all(c, used)
            print(f"\n=== {model} -> {lang}: {time.monotonic() - t0:.0f}s, {failed} failed ===")
            for s, t in zip(lines, out):
                print(f"{s}\n    {t}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pipeline")
    a.add_argument("video")
    a.add_argument("--lang", default="en", help="comma list, e.g. en,vi")
    a.add_argument("--work-dir", help="reuse this dir (its ASR cache) instead of a fresh temp dir")
    b = sub.add_parser("models")
    b.add_argument("sample")
    b.add_argument("names", nargs="*")
    args = p.parse_args()
    pipeline(args.video, args.lang, args.work_dir) if args.cmd == "pipeline" else models(args.sample, args.names)
