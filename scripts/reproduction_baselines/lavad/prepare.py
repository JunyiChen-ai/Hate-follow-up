#!/usr/bin/env python
"""Prepare the frozen test cohort in the file layout expected by LAVAD.

The upstream code consumes directories of numbered JPEG frames plus a four
column annotation file.  This adapter decodes exactly one image for every
second in the frozen gold array.  Consequently upstream ``frame_interval`` is
1 and a LAVAD score key is already a gold-frame index; no repeat/crop heuristic
is permitted downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
sys.path.insert(0, str(BASELINES))
from hate_common import data as hdata  # noqa: E402

VIDEO_DIRS = {
    "hatemm": Path("/home/jehc223/data/HateMM/video"),
    "mhclip_en": Path("/home/jehc223/data/Multihateclip/English/video_mp4"),
    "mhclip_zh": Path("/home/jehc223/data/Multihateclip/Chinese/video"),
    "hateclipseg": Path("/home/jehc223/data/HateClipSeg/video"),
}


def cohort(corpus):
    gt = hdata.gt_arrays(corpus, "test")
    ids = [v for v in hdata.load_split(corpus, "test") if v in gt]
    return ids, gt


def extract(video, out_dir, n_frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    if len(existing) == n_frames:
        return "resume"
    if existing:
        raise RuntimeError("partial frame directory %s (%d/%d); remove it explicitly"
                           % (out_dir, len(existing), n_frames))
    # This is the same ffmpeg 1 fps fallback used by extract_clip_features.py.
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
           "-vf", "fps=1", "-vsync", "0", "-q:v", "2",
           str(out_dir / "%06d.jpg")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    made = sorted(out_dir.glob("*.jpg"))
    if res.returncode and not made:
        raise RuntimeError("ffmpeg failed for %s: %s" % (video, res.stderr[:300]))
    if not made:
        raise RuntimeError("ffmpeg produced no frames for %s" % video)
    # Audio can outlive the video.  The feature extractor clamps such gold
    # timestamps to the last visual frame; reproduce that rule with hardlinks.
    if len(made) < n_frames:
        last = made[-1]
        for i in range(len(made) + 1, n_frames + 1):
            os.link(last, out_dir / ("%06d.jpg" % i))
    elif len(made) > n_frames:
        for path in made[n_frames:]:
            path.unlink()
    return "decoded"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    ap.add_argument("--out-root", default="runs/legacy_1fps/lab1/reproduction/lavad_inputs")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke only; prepare the first N cohort videos")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.out_root).resolve() / args.corpus
    frames = root / "frames"
    ids, gt = cohort(args.corpus)
    if args.limit is not None:
        ids = ids[:args.limit]
    rows, meta = [], []
    labels = hdata.load_labels(args.corpus)
    for vid in ids:
        video = VIDEO_DIRS[args.corpus] / (vid + ".mp4")
        if not video.is_file():
            raise FileNotFoundError(video)
        status = "manifest_only" if args.manifest_only else extract(
            video, frames / vid, len(gt[vid]))
        # Upstream VideoRecord uses inclusive, zero-based start/end fields.
        rows.append("%s 0 %d %d" % (vid, len(gt[vid]) - 1,
                                     int(labels[vid])))
        meta.append({"video_id": vid, "n_gold_frames": len(gt[vid]),
                     "video": str(video), "status": status})
    root.mkdir(parents=True, exist_ok=True)
    (root / "test.txt").write_text("\n".join(rows) + "\n")
    (root / "prepare_meta.json").write_text(json.dumps({
        "corpus": args.corpus, "fps": 1, "frame_interval": 1,
        "image_template": "%06d.jpg", "n_videos": len(ids),
        "records": meta}, indent=2))
    print("prepared %d videos under %s" % (len(ids), root))


if __name__ == "__main__":
    main()
