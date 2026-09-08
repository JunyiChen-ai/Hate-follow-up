"""I/O helpers for OMSL-v6: prediction loading, 4 fps resampling, timestamped
text curve, and the ImageBind text anchors ("normal" / "hateful").

Migrated 2026-09-09 from scripts/idea_discovery/audit_less_views.py (the four
helpers OMSL-v6 uses; the audit main() and its hash-based fold split were not
carried over).  Paths follow the repository layout in CLAUDE.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IB_DIR = ROOT / "third_party/lavad/libs/ImageBind"
IB_CKPT = ROOT / "data/assets/imagebind/imagebind_huge.pth"
IB_TEXT_CACHE = ROOT / "data/assets/imagebind/text_embeddings_normal_hateful.npy"


def load_predictions(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def resize(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.astype(float)
    indices = np.minimum((np.arange(length) * len(values) / length).astype(int), len(values) - 1)
    return values[indices].astype(float)


def text_curve(rows: list[dict], length: int) -> np.ndarray:
    """Average chunk log-odds onto the 4 fps grid; -12 where no chunk covers."""
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


def imagebind_text_embeddings() -> np.ndarray:
    """ImageBind text embeddings for ["normal", "hateful"], shape (2, 1024).

    Cached at data/assets/imagebind/text_embeddings_normal_hateful.npy after the
    first call so OMSL-v6 inference needs neither torch nor the 4.5 GB checkpoint.
    """
    if IB_TEXT_CACHE.exists():
        return np.load(IB_TEXT_CACHE)
    import torch
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
            output = model({ModalityType.TEXT: tokens})[ModalityType.TEXT].float().numpy()
    finally:
        os.chdir(previous)
    IB_TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(IB_TEXT_CACHE, output)
    return output
