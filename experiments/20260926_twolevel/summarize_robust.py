#!/usr/bin/env python3
"""Table of README §12 from the evaluator outputs (metrics.json only; no GT read here)."""
import argparse
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "runs/20260926_twolevel/robust"
MODELS = ["q3vl-2b", "q3vl-4b", "q3vl-8b", "q3vl-32b", "q25vl-7b", "internvl35-8b", "llava-ov-7b", "gemma3-12b"]
ARMS = ["full", "nostance", "noctx", "noframes", "joint", "winonly"]
KEYS = ["frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC"]
FLOOR = [.005, .005, .01]


def load(tag):
    d = json.load(open(R / tag / "metrics.json"))
    return {p["dataset"]: [p[k] for k in KEYS] for p in d["per_dataset"]}


def fmt(v):
    return " / ".join(f"{x:.4f}" for x in v)


ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="new", help="method suffix of the tags: new (§12) or r3 (§14)")
SUF = ap.parse_args().suffix
lines = [f"## other MLLMs (pooled ROC / PR / within; {SUF} - current)"]
ok = {ds: [0, 0, 0] for ds in ["HateMM", "HateClipSeg"]}
mean_w = {}
for m in MODELS:
    c, n = load(f"{m}_cur"), load(f"{m}_{SUF}")
    for ds in c:
        d = [b - a for a, b in zip(c[ds], n[ds])]
        for i in range(3):
            ok[ds][i] += d[i] >= -FLOOR[i]
        lines.append(f"  {m:14s} {ds:11s} current {fmt(c[ds])}  {SUF} {fmt(n[ds])}  diff " + " / ".join(f"{x:+.4f}" for x in d))
        mean_w.setdefault(ds, []).append(d[2])
for ds, v in ok.items():
    lines.append(f"  {ds}: models where {SUF} is not below current beyond the noise floor: ROC {v[0]}/8, PR {v[1]}/8, "
                 f"within {v[2]}/8; mean within change {sum(mean_w[ds]) / len(mean_w[ds]):+.4f}")
lines.append("## reading-module components (Qwen3-VL-8B cache path; each arm minus full)")
base = {meth: load(f"full_{meth}") for meth in ("cur", SUF)}
for arm in ARMS:
    for meth in ("cur", SUF):
        x = load(f"{arm}_{meth}")
        for ds in x:
            d = [b - a for a, b in zip(base[meth][ds], x[ds])]
            lines.append(f"  {arm:9s} {meth} {ds:11s} {fmt(x[ds])}  minus full " + " / ".join(f"{v:+.4f}" for v in d))
(R / ("table.txt" if SUF == "new" else f"table_{SUF}.txt")).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
