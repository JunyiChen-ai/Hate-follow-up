#!/usr/bin/env python3
"""ZS-ImageBind: the zero-shot baseline LAVAD (CVPR 2024) defines, rerun under this repo's evaluator.

Rule 8 requires every new label-free comparator to be rerun with `src/eval/evaluate.py` before it can
enter the results table; the numbers in `docs/protocol_1fps_legacy/` came from another repository's
evaluator and cannot be merged.

Per frame of the 4 fps grid: ImageBind-huge vision embedding, cosine similarity against the two cached
text anchors (normal, hateful), score = sim_hateful - sim_normal. No training, no labels, no threshold,
no per-corpus constant. The text anchors are the ones cached in data/assets/imagebind/ and used by the
2026-08 pipeline; their provenance is in that directory.

Grid: ceil(duration * 4) frames, duration from the manifest -- the same construction the method's own
predictions use, so the evaluator's truncation to the GT length behaves identically for both.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE_PATH = "scripts/reproduction_baselines/zs_imagebind.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/lavad/libs/ImageBind"))
from src.video_inputs import FPS, load_manifest  # noqa: E402

ANCHORS = ROOT / "data/assets/imagebind/text_embeddings_normal_hateful.npy"
WEIGHTS = ROOT / "data/assets/imagebind/imagebind_huge.pth"
VIDEO_DIRS = [Path.home() / "data", Path("/data/jehc223")]


def video_path(row):
    """The manifest carries the authoritative media path; the search below is only a fallback."""
    mp = row.get("video_path")
    if mp and Path(mp).is_file():
        return Path(mp)
    # HateClipSeg's `video/` entries are symlinks into a 2026-09-archived path and are dead; the real
    # files sit next to them in `videos/`. Search both, and require is_file so a dead link never matches.
    dataset, vid = row["dataset"], row["video_id"]
    for base in VIDEO_DIRS:
        for sub in ("video", "videos"):
            d = base / dataset / sub
            if d.is_dir():
                for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
                    p = d / f"{vid}{ext}"
                    if p.is_file():
                        return p
    for base in VIDEO_DIRS:
        d = base / dataset / "video"
        if not d.is_dir():
            continue
        for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            p = d / f"{vid}{ext}"
            if p.is_file():
                return p
        hits = [h for h in d.glob(f"{vid}.*") if h.is_file()]
        if hits:
            return hits[0]
    return None


def build_model(device):
    from imagebind.models import imagebind_model
    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def frame_tensor(frames_uint8, device):
    """(n,H,W,3) uint8 -> ImageBind's own preprocessing (bicubic 224, centre crop, CLIP normalisation)."""
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    x = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    return tf(x)


@torch.no_grad()
def score_video(model, anchors, path, duration, device, batch=64):
    import decord
    from imagebind.models.imagebind_model import ModalityType
    vr = decord.VideoReader(str(path), num_threads=4)
    n = int(math.ceil(duration * FPS))
    native = float(vr.get_avg_fps()) or 30.0
    idx = np.minimum((np.arange(n) / FPS * native).round().astype(int), len(vr) - 1)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch):
        chunk = idx[s:s + batch]
        arr = vr.get_batch(chunk.tolist()).asnumpy()
        emb = model({ModalityType.VISION: frame_tensor(arr, device)})[ModalityType.VISION]
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        sims = emb @ anchors.T                       # (b, 2) = (normal, hateful)
        out[s:s + len(chunk)] = (sims[:, 1] - sims[:, 0]).cpu().numpy()
    return out, len(vr), native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="zs_imagebind")
    ap.add_argument("--exp-id", default="20260912_baselines")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"),
         "host": socket.gethostname(), "anchors": str(ANCHORS), "weights": str(WEIGHTS),
         "score": "cos(frame, hateful) - cos(frame, normal)", "grid": "ceil(duration*4)"}, indent=2))

    rows = load_manifest(args.manifest, args.datasets)
    rows.sort(key=lambda r: (r["dataset"], r["video_id"]))
    if args.limit:
        rows = rows[:args.limit]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    anchors = torch.from_numpy(np.load(ANCHORS)).float().to(device)
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    model = build_model(device)
    logging.info("%d videos, anchors %s, device %s", len(rows), tuple(anchors.shape), device)

    fh = open(out_dir / "predictions.jsonl", "w")
    t0, n_ok, missing = time.time(), 0, []
    for k, r in enumerate(rows):
        ds, vid, dur = r["dataset"], r["video_id"], float(r["duration"])
        p = video_path(r)
        if p is None:
            missing.append(f"{ds}/{vid}")
            fh.write(json.dumps({"dataset": ds, "video_id": vid, "error": "video not found",
                                 "method": "zs_imagebind"}) + "\n")
            continue
        try:
            curve, n_dec, native = score_video(model, anchors, p, dur, device, args.batch)
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s/%s failed: %s", ds, vid, exc)
            fh.write(json.dumps({"dataset": ds, "video_id": vid, "error": str(exc),
                                 "method": "zs_imagebind"}) + "\n")
            continue
        fh.write(json.dumps({"schema_version": 1, "method": "zs_imagebind", "dataset": ds,
                             "video_id": vid, "duration": dur, "native_rate": FPS,
                             "score_curve": [float(x) for x in curve], "intervals": [], "error": None,
                             "calls": 0, "seed": 0, "code_path": CODE_PATH,
                             "extra": {"decoded_frames": n_dec, "native_fps": native}}) + "\n")
        fh.flush()
        n_ok += 1
        if k % 20 == 0:
            logging.info("%d/%d %s %.0fs", k, len(rows), vid, time.time() - t0)
    fh.close()
    logging.info("DONE %d/%d videos in %.0fs; missing %d %s", n_ok, len(rows), time.time() - t0,
                 len(missing), missing[:5])


if __name__ == "__main__":
    main()
