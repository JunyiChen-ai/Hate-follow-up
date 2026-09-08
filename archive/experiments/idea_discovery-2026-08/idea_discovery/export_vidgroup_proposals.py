#!/usr/bin/env python3
"""Export a frozen Vid-Group pre-readout proposal bank for ceiling audits."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.label_free_adapt.mechanisms import GENERIC_QUERY
from scripts.label_free_adapt.run_vidgroup import configure, decode_candidates


def interval_iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[1], b[1]))
    union = max(a[2], b[2]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def greedy_diverse(candidates, limit, nms_tiou):
    selected = []
    for candidate in candidates:
        if all(interval_iou(candidate, old) <= nms_tiou for old in selected):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("third_party/label_free_adapt/Vid-Group"))
    ap.add_argument("--checkpoint", default="ckpt/charades/zero_shot.ckpt")
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--nms-tiou", type=float, default=None)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    repo = args.repo.resolve()
    config = json.loads((repo / "experiment/charades/recorrect_eval_configs_on_ZeroShot+Unsup+Full.json").read_text())
    config, anchors = configure(config)
    sys.path.insert(0, str(repo / "lib"))
    # The released inference code imports IPython only for an unused debug
    # breakpoint.  Keep the runtime minimal when that optional package is absent.
    try:
        import IPython  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("IPython")
        shim.embed = lambda *unused_args, **unused_kwargs: None
        sys.modules["IPython"] = shim
    from models.simbase import SimBase

    model = SimBase(config["model"]["config"]).cuda().float().eval()
    model.load_non_clip_parameters(str(repo / args.checkpoint))
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    with args.out.open("x", encoding="utf-8") as out:
        for row in rows:
            duration = float(row["duration"])
            try:
                feat = torch.from_numpy(np.load(args.features / row["dataset"] / f"{row['video_id']}.npy")).unsqueeze(0).cuda().float()
                with torch.inference_mode():
                    outputs = model([GENERIC_QUERY], feat, None)
                candidates = decode_candidates(
                    (outputs["pred"]["pred_overlap"], outputs["pred"]["pred_reg"]),
                    anchors, duration, 128, config,
                )
                if args.nms_tiou is not None:
                    candidates = greedy_diverse(candidates, args.top_m, args.nms_tiou)
                else:
                    candidates = candidates[: args.top_m]
                proposals = [
                    {"start": float(max(0.0, l / 128 * duration)),
                     "end": float(min(duration, r / 128 * duration)),
                     "logit": float(score), "rank": rank + 1}
                    for rank, (score, l, r) in enumerate(candidates) if r > l
                ]
                record = {"dataset": row["dataset"], "video_id": row["video_id"],
                          "duration": duration, "proposals": proposals, "error": None}
            except Exception as exc:
                record = {"dataset": row["dataset"], "video_id": row["video_id"],
                          "duration": duration, "proposals": [],
                          "error": f"{type(exc).__name__}: {exc}"}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({"dataset": record["dataset"], "video_id": record["video_id"],
                              "n": len(record["proposals"]), "error": record["error"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
