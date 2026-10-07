#!/usr/bin/env python
"""LAVAD stage 01 (BLIP-2 captions at 1 fps), ported 2026-10-08 from Retrieval-hate
`scripts/repro_campaign/blip2_caption.py` (uoa-lab2, read only) for `experiments/20261008_baselines/lavad/`.

Port changes (paths and id lists only; captioner, decoding, frame extraction and batch size unchanged):
  * ROOT is this repository; frames are read from / written to `data/frames_1fps/<DS>/<vid>/%06d.jpg`
    (hard-linked copies of the campaign's frames, see that directory's PROVENANCE.md), captions go to
    `data/blip2_captions_1fps/<DS>/<vid>.json`.
  * The video list comes from `--ids-file` (one id per line) instead of the campaign's GT npz; durations and
    video paths come from the manifests used by `exact_cohort.py` (HateClipSeg videos: the campaign's video
    directory on uoa-lab2, read only, needed only if a frame directory is missing).

Original docstring follows.

LAVAD's `00_extract_frames.sh` writes **every native frame** of every video as a
JPEG and `01_caption.sh` then captions all of them (`frame_interval=1`).
`MODEL_ASSETS_STATUS §2` measured 283 MB of JPEG for one 222 s video; the four
corpora are 84 h of video, i.e. ~700-800 GB, which does not fit beside the other
campaign jobs.

**Declared adaptation (REPRO_CAMPAIGN_RESULTS §K).** Frames are extracted at
`fps=1` instead of the native rate, so `frames/<DS>/<vid>/000123.jpg` is the video
content at `t = 123 s`, and one caption exists per second of video.  LAVAD's
center grid `frame_interval=16` at its assumed 30 fps is 0.533 s; ours is 1.0 s.
Everything downstream (10 s clip window, 10 uniform samples per window) then
lands on exactly this grid, which is why the same JPEGs serve stages 01, 03 and
06 without a second decode.

Everything about the captioner itself is LAVAD's, from
`lavad/src/models/image_captioner.py`: `Salesforce/blip2-opt-6.7b-coco`,
float16, **unconditional** captioning (no text prompt), `model.generate(**inputs)`
with the checkpoint's own generation config, `batch_decode(skip_special_tokens)`,
`.strip()`.  Frames are read from the JPEGs with PIL exactly as it does.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HOME = Path.home()
sys.path.insert(0, str(ROOT / "experiments/20261008_baselines"))
import exact_cohort as ec  # noqa: E402

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "HateClipSeg": HOME / "Retrieval-hate/data/video/HateClipSeg/All",   # read only
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}  # freeze §12 D2
MODEL = "Salesforce/blip2-opt-6.7b-coco"

FRAME_ROOT = ROOT / "data/frames_1fps"
CAP_ROOT = ROOT / "data/blip2_captions_1fps"


def find_video(ds: str, vid: str):
    if ds == "DeHate":
        p = Path(ec.manifest_rows(ds)[vid]["video_path"])
        return p if p.exists() else None
    for ext in EXTS:
        p = VIDEO_DIR[ds] / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def dataset_rows(ds: str, ids_file: str):
    """(video id, duration) for the ids in `ids_file` (port: replaces the campaign's GT-npz id list)."""
    dur = ec.durations(ds)
    ids = [l.strip() for l in open(ids_file) if l.strip()]
    return sorted((v, dur[v]) for v in ids)


def extract_1fps(path: Path, out_dir: Path, nmax: int) -> list[Path]:
    """ffmpeg `fps=1` -> `%06d.jpg` starting at 0, quality matching cv2.imwrite's
    default (95).  Returns the frame files in index order."""
    if out_dir.exists():
        have = sorted(out_dir.glob("*.jpg"))
        if have:
            return have
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-vf", "fps=1", "-q:v", "2", "-start_number", "0",
           "-frames:v", str(nmax), str(tmp / "%06d.jpg")]
    subprocess.run(cmd, check=True, timeout=3600,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = sorted(tmp.glob("*.jpg"))
    if not files:
        shutil.rmtree(tmp, ignore_errors=True)
        return []
    os.replace(tmp, out_dir)
    return sorted(out_dir.glob("*.jpg"))


def prefetch(jobs, fn, depth=2):
    """Decode video N+1 on a worker thread while the GPU captions video N."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        for j in jobs:
            try:
                q.put((j, fn(*j), None))
            except Exception as e:
                q.put((j, None, f"{type(e).__name__}: {e}"))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        got = q.get()
        if got is None:
            return
        yield got


@torch.no_grad()
def caption_batch(model, processor, dtype, paths):
    from PIL import Image

    images = []
    for p in paths:
        with open(p, "rb") as fh:
            images.append(Image.open(fh).convert("RGB"))
    inputs = processor(images=images, return_tensors="pt").to("cuda", dtype=dtype)
    ids = model.generate(**inputs)
    return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--ids-file", required=True, help="one video id per line (port)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--keep-frames", action="store_true", default=True)
    args = ap.parse_args()

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    jobs = []
    for ds in args.datasets.split(","):
        (CAP_ROOT / ds).mkdir(parents=True, exist_ok=True)
        (FRAME_ROOT / ds).mkdir(parents=True, exist_ok=True)
        for vid, dur in dataset_rows(ds, args.ids_file):
            if vid in NO_VIDEO_STREAM:
                continue
            if (CAP_ROOT / ds / f"{vid}.json").exists():
                continue
            p = find_video(ds, vid)
            # port (2026-10-08, campus run): the video file is needed only to extract frames; a video whose
            # frame directory already holds JPEGs is captioned from them (extract_1fps returns them unchanged)
            if p is None and not any((FRAME_ROOT / ds / vid).glob("*.jpg")):
                print(f"[fail] {ds}/{vid} missing_file", flush=True)
                continue
            jobs.append((ds, vid, p, min(int(dur) + 2, args.max_frames), dur))
    if args.limit:
        jobs = jobs[: args.limit]
    total_sec = sum(j[4] for j in jobs)
    print(f"PROGRESS plan videos={len(jobs)} video_hours={total_sec/3600:.2f}", flush=True)
    if not jobs:
        print("[done] videos=0 failed=0 frames=0 nothing to caption", flush=True)
        return 0

    dtype = torch.float16
    processor = Blip2Processor.from_pretrained(MODEL)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype)
    model.to("cuda").eval()
    print(f"PROGRESS model_loaded vram={torch.cuda.memory_allocated()/2**30:.2f}GiB",
          flush=True)

    def decode(ds, vid, path, nmax, dur):
        return extract_1fps(path, FRAME_ROOT / ds / vid, nmax)

    t0 = time.time()
    n_done = n_frames = n_fail = 0
    sec_done = 0.0
    for (ds, vid, path, nmax, dur), files, err in prefetch(jobs, decode):
        if err is not None or not files:
            print(f"[fail] {ds}/{vid} {err or 'no_frames_decoded'}", flush=True)
            n_fail += 1
            continue
        caps: dict[str, str] = {}
        try:
            for s in range(0, len(files), args.batch_size):
                chunk = files[s: s + args.batch_size]
                txt = caption_batch(model, processor, dtype, chunk)
                for p, t in zip(chunk, txt):
                    caps[str(int(p.stem))] = t
        except Exception as e:
            print(f"[fail] {ds}/{vid} caption {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            torch.cuda.empty_cache()
            continue
        out_p = CAP_ROOT / ds / f"{vid}.json"
        tmp = out_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(caps, indent=1))
        os.replace(tmp, out_p)
        n_done += 1
        n_frames += len(caps)
        sec_done += dur
        if n_done % 5 == 0 or n_done == len(jobs):
            el = time.time() - t0
            eta = (total_sec - sec_done) * el / max(sec_done, 1e-9)
            print(f"PROGRESS {n_done}/{len(jobs)} vids frames={n_frames} "
                  f"{n_frames/max(el,1e-9):.1f} img/s elapsed={el/60:.1f}min "
                  f"eta={eta/60:.1f}min fail={n_fail}", flush=True)
    el = time.time() - t0
    print(f"[done] videos={n_done} failed={n_fail} frames={n_frames} "
          f"wall={el/60:.1f}min {n_frames/max(el,1e-9):.1f} img/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
