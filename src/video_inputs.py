"""Shared inputs for MLLM-based localization: manifest, Whisper segments, cached frames, fixed windows.

Promoted from experiments/20260910_spvl/spvl.py (2026-09-11) so later experiments can import it
(experiment directories must not import each other). Behaviour identical to the SPVL version.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FPS = 4.0


def load_manifest(path, datasets):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r["dataset"] in datasets]


def load_asr(dataset):
    path = ROOT / f"data/asr_whisper_large_v3/{dataset}/timestamped_chunks.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        segs = [(float(c["start"]), float(c["end"]), (c.get("text") or ""))
                for c in (r.get("chunks") or []) if c.get("end") is not None and c.get("start") is not None]
        out[r["video_id"]] = [s for s in segs if s[1] > s[0]]
    return out


def frame_paths(dataset, vid, k, source="k20"):
    """source k20: K of the 20 uniform frames; source w8: every window-centre frame (k ignored)."""
    d = ROOT / f"data/frames_{source}/{dataset}/{vid}"
    files = sorted(d.glob("f*_t*.jpg"))
    if not files:
        return []
    if source == "k20" and k < len(files):
        idx = np.linspace(0, len(files) - 1, k).round().astype(int)
        files = [files[i] for i in idx]
    return [(float(f.stem.split("_t")[1]), f) for f in files]


def fixed_windows(duration, seconds):
    n = max(1, int(math.ceil(duration / seconds - 1e-9)))
    return [(i * seconds, min((i + 1) * seconds, duration)) for i in range(n)]


def window_text(segments, t1, t2):
    """Transcript inside [t1,t2]: segments are sliced proportionally at word boundaries."""
    parts = []
    for s, e, text in segments:
        lo, hi = max(s, t1), min(e, t2)
        if hi <= lo or not text:
            continue
        if s >= t1 and e <= t2:
            parts.append(text)
            continue
        words = text.split()
        if not words:
            continue
        a = int(round((lo - s) / (e - s) * len(words)))
        b = int(round((hi - s) / (e - s) * len(words)))
        piece = " ".join(words[a:b]).strip()
        if piece:
            parts.append(piece)
    return " ".join(parts)


def transcript_block(segments):
    if not segments:
        return "(no speech detected)"
    return "\n".join(f"[{s:.1f}s-{e:.1f}s] {t.strip()}" for s, e, t in segments if t.strip())


def within_defined_ids(datasets):
    """Video ids whose 4 fps GT has both classes: used ONLY to pick pilot subsets (rule 10)."""
    out = set()
    for ds in datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        for vid, y in zip(g["video_ids"], g["y4"]):
            y = np.asarray(y)
            if len(y) and y.min() != y.max():
                out.add((ds, str(vid)))
    return out
