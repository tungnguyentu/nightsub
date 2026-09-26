"""Embed finished <stem>.<lang>.srt files as soft subtitle tracks -> <stem>.subs.mkv (no re-encode)."""
import subprocess
from pathlib import Path

LANG3 = {"en": "eng", "vi": "vie"}


def srts_for(video):
    v = Path(video)
    return sorted(p for p in v.parent.glob(f"{v.stem}.*.srt") if p.suffixes[-2][1:] in LANG3)


def mux_cmd(video, srts, out):
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video)]
    for s in srts:
        cmd += ["-i", str(s)]
    cmd += ["-map", "0:v", "-map", "0:a?"]
    for i, s in enumerate(srts):
        lang = s.suffixes[-2][1:]
        cmd += ["-map", f"{i + 1}:0", f"-metadata:s:s:{i}", f"language={LANG3[lang]}"]
    return cmd + ["-c", "copy", "-c:s", "srt", "-disposition:s:0", "default", str(out)]


def mux(video):
    srts = srts_for(video)
    if not srts:
        raise FileNotFoundError(f"no .en.srt/.vi.srt next to {video}")
    out = Path(video).with_suffix(".subs.mkv")
    subprocess.run(mux_cmd(video, srts, out), check=True)
    return out
