#!/usr/bin/env python3
"""Reads averaged over several MLLMs (README.md §"Is it worth trying"). Label-free: each model's window reads (per
modality) and whole-video verdicts are replaced by normal scores of their rank within the corpus, then averaged over
the models that have the read. Output has the format of a reading run (`predictions.jsonl` with `extra.windows`), so
`experiments/20260926_twolevel/twolevel_r2.py --run <out>` can use it. No gold is read here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[2]
MODS = ("z_visual", "z_speech")


def nscore(x):
    x = np.asarray(x, float)
    return norm.ppf((rankdata(x) - .5) / len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="models averaged for the whole-video verdict")
    ap.add_argument("--window-models", nargs="*", default=None, help="models averaged for the window reads (default: --models)")
    ap.add_argument("--mllm-root", default=str(ROOT / "runs/20260910_spvl/mllm"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wmodels = a.window_models or a.models
    runs = {m: [json.loads(l) for l in open(Path(a.mllm_root) / m / "full" / "predictions.jsonl")]
            for m in dict.fromkeys(a.models + wmodels)}
    acc_w = defaultdict(lambda: defaultdict(list))      # (ds, vid, start, end) -> modality -> normal scores
    acc_v = defaultdict(list)                           # (ds, vid) -> verdict normal scores
    base = {}
    for m, recs in runs.items():
        recs = [r for r in recs if not r.get("error")]
        for ds in sorted({r["dataset"] for r in recs}):
            rs = [r for r in recs if r["dataset"] == ds]
            if m in a.models:
                for r, z in zip(rs, nscore([r["extra"]["z_video"] for r in rs])):
                    acc_v[(ds, r["video_id"])].append(float(z)); base.setdefault((ds, r["video_id"]), r)
            if m not in wmodels:
                continue
            for mod in MODS:
                refs = [(r["video_id"], w) for r in rs for w in r["extra"]["windows"] if w.get(mod) is not None]
                for (vid, w), z in zip(refs, nscore([w[mod] for _, w in refs])):
                    acc_w[(ds, vid, round(w["start"], 3), round(w["end"], 3))][mod].append(float(z))
    wins = defaultdict(list)
    for (ds, vid, s, e), d in acc_w.items():
        w = {"start": s, "end": e, **{mod: float(np.mean(v)) for mod, v in d.items()}}
        w["z"] = max(w[mod] for mod in MODS if mod in w)
        wins[(ds, vid)].append(w)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w") as fh:
        for key, r in sorted(base.items()):
            ws = sorted(wins[key], key=lambda w: w["start"])
            for i, w in enumerate(ws):
                w["i"] = i
            rec = {k: v for k, v in r.items() if k != "extra"}   # score_curve kept: its length is the frame count
            rec["extra"] = {"z_video": float(np.mean(acc_v[key])), "windows": ws, "n_models": len(acc_v[key])}
            fh.write(json.dumps(rec) + "\n")
    (out / "config.json").write_text(json.dumps({**vars(a), "code_path": "experiments/20260928_headroom/ensemble_reads.py"},
                                                indent=2))
    print(f"wrote {len(base)} videos to {out}")


if __name__ == "__main__":
    main()
