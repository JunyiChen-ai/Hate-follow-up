#!/usr/bin/env python3
"""Extract the 128-bin CLIP-L/14 features expected by Vid-Group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .run_numpro import sample_numbered_frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reverse", action="store_true",
                        help="Process the manifest in reverse order for a second decoder worker")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Rotate processing order to this manifest index for another worker")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if args.reverse:
        rows = list(reversed(rows))
    if args.start_index:
        index = args.start_index % len(rows)
        rows = rows[index:] + rows[:index]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    rows = rows[args.shard_index::args.shard_count]
    import clip
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    model.eval()
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        destination = root / row["dataset"] / f"{row['video_id']}.npy"
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            frames, _ = sample_numbered_frames(row["video_path"], args.bins)
            batches = []
            for start in range(0, len(frames), args.batch_size):
                images = torch.stack([preprocess(image) for image in
                                      frames[start:start + args.batch_size]]).cuda()
                with torch.inference_mode():
                    feature = model.encode_image(images).float()
                    feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                batches.append(feature.cpu().numpy())
            values = np.concatenate(batches)
            if len(values) != args.bins:
                positions = np.linspace(0, len(values) - 1, args.bins)
                values = np.stack([np.interp(positions, np.arange(len(values)), values[:, i])
                                   for i in range(values.shape[1])], axis=1)
            np.save(destination, values.astype(np.float32))
            print(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                              "shape": list(values.shape)}), flush=True)
        except Exception as exc:
            print(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
