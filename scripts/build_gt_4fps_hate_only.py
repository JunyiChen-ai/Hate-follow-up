#!/usr/bin/env python3
"""Secondary evaluation GT for HateClipSeg: only the 'Hateful' category, not the offensive union.

The main-table GT (data/gt_4fps/HateClipSeg.npz) marks a segment positive when ANY of the five offensive
categories is set (Hateful, Insulting, Sexual, Violence, Self-Harm), while the method's prompt asks about
hate rules only. That mismatch is recorded as a limitation; this script builds a parallel GT restricted to
the Hateful category so the two can be told apart (user ruling 2026-09-12: main table unchanged, secondary
subset reported separately).

Category order [Normal, Hateful, Insulting, Sexual, Violence, Self-Harm] is from the HateClipSeg paper
(arXiv 2508.01712). Grid, split, video set and rasterisation are identical to data/gt_4fps.

Usage: python scripts/build_gt_4fps_hate_only.py [--out data/gt_4fps_hate_only]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD = Path("/home/jehc223/Retrieval-hate/data/gt/HateClipSeg/gold_segments.json")
MAIN = ROOT / "data/gt_4fps/HateClipSeg.npz"
FPS = 4.0
HATEFUL_DIM = 1  # [Normal, Hateful, Insulting, Sexual, Violence, Self-Harm]


def rasterize(spans, duration, n=None):
    n = int(math.ceil(duration * FPS)) if n is None else int(n)
    y = np.zeros(n, dtype=np.int8)
    out = []
    for a, b in spans:
        i, j = int(round(a * FPS)), min(int(round(b * FPS)), n)
        if j > i:
            y[i:j] = 1
            out.append([float(a), float(b)])
    return y, np.asarray(out, dtype=float).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/gt_4fps_hate_only"))
    args = ap.parse_args()
    gold = json.loads(GOLD.read_text())
    main_gt = np.load(MAIN, allow_pickle=True)
    ids = [str(v) for v in main_gt["video_ids"]]

    labels, spans_out, kept, durs = [], [], [], []
    n_pos_main = n_pos_hate = 0
    for vid, dur, y_main in zip(ids, main_gt["duration"], main_gt["y4"]):
        entry = gold.get(vid)
        if entry is None:
            raise SystemExit(f"{vid} missing from gold_segments.json")
        spans = [[a, b] for a, b, dims in entry["segments"] if dims[HATEFUL_DIM]]
        # the grid is taken from the main GT array so the two are frame-aligned by construction
        y, sp = rasterize(spans, float(dur), n=len(np.asarray(y_main)))
        labels.append(y); spans_out.append(sp); kept.append(vid); durs.append(float(dur))
        n_pos_main += int(np.asarray(y_main).sum()); n_pos_hate += int(y.sum())

    obj = lambda vals: np.asarray(vals + [None], dtype=object)[:-1]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "HateClipSeg.npz",
                        video_ids=np.asarray(kept), split=np.asarray(["test"] * len(kept)),
                        duration=np.asarray(durs), y4=obj(labels), spans=obj(spans_out),
                        n_spans=np.asarray([len(s) for s in spans_out], dtype=np.int16))
    both = sum(1 for y in labels if y.min() != y.max())
    n_frames = sum(len(y) for y in labels)
    report = {"dataset": "HateClipSeg", "category": "Hateful only (dim 1)", "n_videos": len(kept),
              "n_frames": n_frames, "n_positive_hate_only": n_pos_hate, "n_positive_main": n_pos_main,
              "base_rate_hate_only": n_pos_hate / n_frames, "base_rate_main": n_pos_main / n_frames,
              "n_videos_with_both_classes": both,
              "source": str(GOLD), "grid_fps": FPS}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
