#!/usr/bin/env python3
"""Frozen single-encoder visual hate axis on cached CoCa frame features."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Prediction


POSITIVE = "hateful content targeting a person or group"
NEGATIVE = "normal, non-toxic content"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k")
    model = model.eval().cuda()
    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
    with torch.no_grad():
        text = model.encode_text(tokenizer([POSITIVE, NEGATIVE]).cuda()).float()
        text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = model.logit_scale.exp().detach().float()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    missing, written = [], 0
    with args.out.open("w") as handle:
        for row in map(json.loads, args.manifest.open()):
            dataset, video_id = str(row["dataset"]), str(row["video_id"])
            path = args.feature_root / dataset / "coca_vitL14_4fps" / f"{video_id}.npy"
            if not path.exists():
                missing.append(f"{dataset}:{video_id}")
                continue
            features = torch.from_numpy(np.load(path)).float().cuda()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            with torch.no_grad():
                logits = scale * features @ text.T
                score = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
            duration = float(row.get("duration", len(score) / 4.0))
            prediction = Prediction(
                "coca_contrastive_hate_axis_v1", dataset, video_id, duration,
                native_rate=4.0, score_curve=score.tolist(), intervals=[], calls=0,
                modality_evidence={"positive_text": POSITIVE, "negative_text": NEGATIVE},
                raw={"gt_access": False, "module": "frozen_contrastive_visual_hate_axis",
                     "backbone": "coca_ViT-L-14_mscoco_finetuned_laion2B-s13B-b90k",
                     "single_visual_encoder": True, "tta": False,
                     "pretrained_logit_scale": float(scale.cpu()),
                     "dataset_parameters": 0, "label_selected_parameters": 0,
                     "code_sha256": code_hash})
            handle.write(json.dumps(prediction.to_dict(), separators=(",", ":")) + "\n")
            written += 1
    audit = {"manifest_rows": sum(1 for _ in args.manifest.open()), "written": written,
             "missing": missing, "model_queries": [POSITIVE, NEGATIVE]}
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
