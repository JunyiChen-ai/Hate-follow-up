#!/usr/bin/env python3
"""Identity check of the read sets used for the window-evidence contribution (README §2, task 5). Reads predictions
only (no GT, no labels). For each corpus set it compares, read by read, the four read sets old / frames / words / both:
- the video verdict z_video must be the same in all four (the prefix and the verdict turn are unchanged);
- frames only changes the visual read: frames.z_visual == both.z_visual, frames.z_speech == old.z_speech;
- words only changes the speech read: words.z_speech == both.z_speech, words.z_visual == old.z_visual;
- the gate verdict sets (frames-only prefix, transcript-only prefix) cover every video of the adopted reads.
Prints the max absolute difference per comparison and the share of windows whose read differs (|d| > 1e-6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(d):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(ROOT / d / "predictions.jsonl"))}


def cmp(a, b, field, keys):
    diffs, n_missing_a, n_missing_b, n = [], 0, 0, 0
    for k in keys:
        if field == "z_video":
            diffs.append(abs(a[k]["extra"]["z_video"] - b[k]["extra"]["z_video"])); n += 1
            continue
        wa, wb = a[k]["extra"]["windows"], b[k]["extra"]["windows"]
        assert len(wa) == len(wb), k
        for x, y in zip(wa, wb):
            assert (x["start"], x["end"]) == (y["start"], y["end"]), k
            if (field in x) != (field in y):
                n_missing_a += field not in x; n_missing_b += field not in y; continue
            if field in x:
                diffs.append(abs(x[field] - y[field])); n += 1
    d = np.array(diffs) if diffs else np.zeros(1)
    return f"n {n}  max|d| {d.max():.3g}  share |d|>1e-6 {np.mean(d > 1e-6):.3f}  present only in 2nd {n_missing_a}  only in 1st {n_missing_b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--both", required=True)
    ap.add_argument("--gate", nargs="*", default=[])
    a = ap.parse_args()
    S = {n: load(getattr(a, n)) for n in ("old", "frames", "words", "both")}
    keys = sorted(S["both"])
    for n, s in S.items():
        print(f"{n:6s} {getattr(a, n)}: {len(s)} videos, same keys as both: {sorted(s) == keys}")
    for x, y, f in [("old", "both", "z_video"), ("frames", "both", "z_video"), ("words", "both", "z_video"),
                    ("frames", "both", "z_visual"), ("frames", "old", "z_speech"),
                    ("words", "both", "z_speech"), ("words", "old", "z_visual"),
                    ("old", "both", "z_visual"), ("old", "both", "z_speech")]:
        print(f"{x:6s} vs {y:6s} {f:9s} {cmp(S[x], S[y], f, keys)}")
    for g in a.gate:
        s = load(g)
        miss = [k for k in keys if k not in s or s[k].get("error") or "z_video" not in s[k]["extra"]]
        print(f"gate verdicts {g}: {len(s)} videos, missing for {len(miss)} of {len(keys)} adopted videos")


if __name__ == "__main__":
    main()
