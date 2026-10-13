#!/usr/bin/env python3
"""Decode the audio track of every test video to 16 kHz mono PCM16 wav: data/audio16k/<dataset>/<video_id>.wav.
CPU only (ffmpeg). Idempotent: existing files with the expected duration are kept. Writes a manifest line per video."""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_manifest  # noqa: E402

OUT = ROOT / "data/audio16k"
rows = load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", ("HateMM", "HateClipSeg"))
log = []
for k, r in enumerate(rows, 1):
    dst = OUT / r["dataset"] / (r["video_id"] + ".wav"); dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(r["video_path"])
    if not src.exists():  # the manifest's HateClipSeg paths are stale: use the file the 0.5 fps frame cache was decoded from
        meta = ROOT / "data/temporal_retrieved_kv" / r["dataset"] / r["video_id"] / "metadata.json"
        if meta.exists():
            src = Path(json.loads(meta.read_text())["source"]["input_video"])
    if not src.exists():
        log.append(dict(dataset=r["dataset"], video_id=r["video_id"], error="video missing")); continue
    if not dst.exists():
        tmp = dst.with_suffix(".tmp.wav")
        p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                            "-acodec", "pcm_s16le", str(tmp)], capture_output=True, text=True)
        if p.returncode != 0 or not tmp.exists():
            log.append(dict(dataset=r["dataset"], video_id=r["video_id"], error=p.stderr[-300:])); continue
        tmp.replace(dst)
    n = (dst.stat().st_size - 44) // 2
    log.append(dict(dataset=r["dataset"], video_id=r["video_id"], seconds=n / 16000, manifest_duration=r["duration"]))
    if k % 50 == 0:
        print(k, len(rows), flush=True)
(OUT / "manifest_test.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
err = [x for x in log if "error" in x]
print("DONE", len(log), "errors", len(err), flush=True)
for x in err[:10]:
    print(x)
