#!/usr/bin/env python3
"""Build and audit LESS-3V temporal evidence without ground truth.

The audit projects a visual family, timestamped language chunks, and raw-audio
ImageBind evidence onto the common 4 fps grid.  It reports coverage,
cross-view dependence, effective rank, and deterministic hash-fold stability;
it does not fit or evaluate a localizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
IB_DIR = ROOT / "third_party/lavad/libs/ImageBind"
IB_CKPT = ROOT / "results/label_free_adapt/assets/imagebind/imagebind_huge.pth"


def load_predictions(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def hash_fold(key: tuple[str, str]) -> int:
    return int(hashlib.sha256("::".join(key).encode()).hexdigest(), 16) % 2


def resize(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.astype(float)
    indices = np.minimum((np.arange(length) * len(values) / length).astype(int), len(values) - 1)
    return values[indices].astype(float)


def text_curve(rows: list[dict], length: int) -> np.ndarray:
    sums = np.zeros(length, dtype=float)
    counts = np.zeros(length, dtype=float)
    for row in rows:
        lo = max(0, int(np.floor(float(row["start"]) * 4)))
        hi = min(length, max(lo + 1, int(np.ceil(float(row["end"]) * 4))))
        sums[lo:hi] += float(row["log_odds"])
        counts[lo:hi] += 1
    output = np.full(length, -12.0)
    valid = counts > 0
    output[valid] = sums[valid] / counts[valid]
    return output


def ternary_text(values: np.ndarray) -> np.ndarray:
    # Two explicit anchors: Yes/No indifference at 0 and the empty-transcript
    # null at -12.  Values between them are abstentions.
    return np.where(values > 0, 1, np.where(values < -12, -1, 0)).astype(np.int8)


def effective_rank(correlation: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0)
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    entropy = -(probabilities[probabilities > 0] *
                np.log(probabilities[probabilities > 0])).sum()
    return float(np.exp(entropy))


def imagebind_text_embeddings() -> np.ndarray:
    sys.path.insert(0, str(IB_DIR))
    previous = Path.cwd()
    os.chdir(ROOT / "third_party/lavad")
    try:
        from imagebind import data as ibdata
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType

        model = imagebind_model.imagebind_huge(pretrained=False)
        model.load_state_dict(torch.load(IB_CKPT, map_location="cpu"))
        model = model.eval()
        with torch.no_grad():
            tokens = ibdata.load_and_transform_text(["normal", "hateful"], "cpu")
            return model({ModalityType.TEXT: tokens})[ModalityType.TEXT].float().numpy()
    finally:
        os.chdir(previous)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_predictions(args.manifest)
    sources = [load_predictions(path) for path in (args.a08, args.a10, args.a12)]
    chunks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in map(json.loads, args.text.open()):
        chunks[(row["dataset"], row["video_id"])].append(row)
    common = sorted(set(manifest).intersection(*(set(source) for source in sources)))
    text_embeddings = imagebind_text_embeddings()
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    per_video = []
    pooled = []
    missing_audio = []
    for key in common:
        row = manifest[key]
        length = max(1, int(np.ceil(float(row["duration"]) * 4)))
        visual_bits = []
        for source in sources:
            curve = resize(np.asarray(source[key]["score_curve"]), length)
            visual_bits.append(curve > 0)
        visual = np.where(np.stack(visual_bits).sum(0) >= 2, 1, -1).astype(np.int8)
        language = ternary_text(text_curve(chunks.get(key, []), length))
        audio_path = args.audio_dir / key[0] / f"{key[1]}.npy"
        if not audio_path.exists():
            missing_audio.append(list(key))
            continue
        audio_embeddings = np.load(audio_path).astype(np.float32)
        audio_embeddings /= np.maximum(np.linalg.norm(audio_embeddings, axis=1, keepdims=True), 1e-12)
        similarity = audio_embeddings @ text_embeddings.T
        audio_margin = similarity[:, 1] - similarity[:, 0]
        audio = np.where(resize(audio_margin, length) >= 0, 1, -1).astype(np.int8)
        matrix = np.stack([visual, language, audio], axis=1)
        pooled.append(matrix)
        per_video.append({
            "dataset": key[0], "video_id": key[1], "fold": hash_fold(key),
            "frames": length,
            "coverage": {"visual": float(np.mean(visual != 0)),
                         "language": float(np.mean(language != 0)),
                         "audio": float(np.mean(audio != 0))},
            "support": {"visual": float(np.mean(visual > 0)),
                        "language": float(np.mean(language > 0)),
                        "audio": float(np.mean(audio > 0))},
            "pair_agreement": {"visual_language": float(np.mean(visual == language)),
                               "visual_audio": float(np.mean(visual == audio)),
                               "language_audio": float(np.mean(language == audio))}})

    all_evidence = np.concatenate(pooled, axis=0).astype(float)
    correlation = np.corrcoef(all_evidence, rowvar=False)
    folds = {}
    for fold in (0, 1):
        subset = np.concatenate([matrix for matrix, row in zip(pooled, per_video)
                                 if row["fold"] == fold], axis=0).astype(float)
        corr = np.corrcoef(subset, rowvar=False)
        folds[str(fold)] = {"n_frames": len(subset), "correlation": corr.tolist(),
                            "effective_rank": effective_rank(corr),
                            "support_rates": dict(zip(
                                ["visual", "language", "audio"],
                                np.mean(subset > 0, axis=0).tolist()))}
    report = {
        "schema_version": 1, "method": "less_3v_identifiability_audit_v1",
        "gt_access": False, "views": ["visual_family", "timestamp_language", "raw_audio"],
        "n_common": len(common), "n_audited": len(per_video),
        "missing_audio": missing_audio, "correlation": correlation.tolist(),
        "effective_rank": effective_rank(correlation), "folds": folds,
        "fold_effective_rank_gap": abs(folds["0"]["effective_rank"] -
                                       folds["1"]["effective_rank"]),
        "per_video": per_video}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in
                      ("n_common", "n_audited", "missing_audio", "correlation",
                       "effective_rank", "folds", "fold_effective_rank_gap")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
