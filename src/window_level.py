"""Window-level pre-check helpers shared by the M1 candidates since 2026-10-17 (first users: candidates 45 and 46).

Reads the 4 fps test GT, so every function here is rule-10 territory: analysis only, never inside a scoring path.
Conventions (candidate 44 README §2, candidate 45 README §2): a window [t1, t2) maps to frames
[round(4 t1), round(4 t2)); a window is positive when its hateful-frame share is >= .5, negative when it is 0, and
dropped otherwise; "window-level within" = one ROC-AUC per video over that video's counted windows (videos that have both
classes), averaged over videos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FPS = 4.0


def gold(dataset):
    """video_id -> 4 fps frame labels (int array) for the test split."""
    g = np.load(ROOT / f"data/gt_4fps/{dataset}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, int) for v, y, s in zip(g["video_ids"], g["y4"], g["split"]) if str(s) == "test"}


def within_eligible(y):
    """True when the video's GT has both classes (the within-video macro ROC counts it)."""
    return 0 < int(y.sum()) < len(y)


def frame_span(t1, t2, length):
    a, b = int(round(FPS * t1)), int(round(FPS * t2))
    return max(0, min(a, length)), max(0, min(b, length))


def window_label(y, t1, t2):
    """1 / 0 / None (dropped) for window [t1, t2) under the >= .5 / == 0 rule; None also when the span is empty."""
    a, b = frame_span(t1, t2, len(y))
    if b <= a:
        return None, None
    frac = float(y[a:b].mean())
    if frac >= .5:
        return 1, frac
    if frac == 0:
        return 0, frac
    return None, frac


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, np.asarray(s, float))) if 0 < y.sum() < len(y) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def within_by_video(rows, key, video_key="video_id", label_key="label"):
    """rows: dicts with a video id, a label and the score under `key`; returns video -> AUC (None when one class)."""
    per = {}
    for vid in sorted({r[video_key] for r in rows}):
        rr = [r for r in rows if r[video_key] == vid and r.get(key) is not None]
        per[vid] = auc([r[label_key] for r in rr], [r[key] for r in rr]) if rr else None
    return per


def window_within(rows, key):
    """Macro window-level within and the number of videos it is defined on."""
    per = within_by_video(rows, key)
    return mean(per.values()), sum(v is not None for v in per.values())
