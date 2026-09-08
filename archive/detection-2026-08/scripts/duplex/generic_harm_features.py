#!/usr/bin/env python3
"""Extract frozen SigLIP generic-harm nuisance features from frames_16."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "our_method"))
from data_utils import load_clean_split_ids  # noqa: E402

MODEL_ID = "google/siglip-base-patch16-224"
TEXTS = [
    "a video showing physical violence, fighting, weapons, riots, injury, or threatening behavior",
    "a peaceful video with no physical conflict, weapons, injury, or threatening behavior",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HateMM", "ImpliHateVid"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    out = args.output or ROOT / "results/generic_harm_nuisance" / f"{args.dataset}_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["video_id"] for x in out.open() if x.strip()}

    ids = [v for v in load_clean_split_ids(args.dataset, args.split) if v not in done]
    frame_root = Path(os.environ.get("HVD_DATA_ROOT", "/home/jehc223/data")) / args.dataset / "frames_16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    text_inputs = processor(text=TEXTS, padding="max_length", return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    for vi, vid in enumerate(ids, 1):
        paths = sorted((frame_root / vid).glob("*.jpg"))
        if not paths:
            paths = sorted((frame_root / vid).glob("*.png"))
        probs = []
        for start in range(0, len(paths), args.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + args.batch_size]]
            inp = processor(images=images, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.inference_mode():
                image_features = model.get_image_features(**inp)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                # SigLIP's learned temperature is load-bearing for the frozen
                # two-concept probability. The shared scalar bias cancels in
                # the two-way softmax, but the exp(logit_scale) does not.
                logits = (image_features @ text_features.T) * model.logit_scale.exp()
                p = torch.softmax(logits.float(), dim=-1)[:, 0]
            probs.extend(p.cpu().tolist())
        arr = np.asarray(probs, dtype=float)
        topk = np.sort(arr)[-min(4, len(arr)):] if len(arr) else np.asarray([])
        rec = {
            "video_id": vid, "n_frames": len(arr),
            "g_max": float(np.max(arr)) if len(arr) else None,
            "g_mean": float(np.mean(arr)) if len(arr) else None,
            "g_top4": float(np.mean(topk)) if len(topk) else None,
        }
        with out.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if vi == 1 or vi % 50 == 0:
            print(f"[{vi}/{len(ids)}] {args.dataset} {vid} frames={len(arr)} g={rec['g_max']}", flush=True)


if __name__ == "__main__":
    main()
