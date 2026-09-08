#!/usr/bin/env python3
"""Responsibility-aware fusion of relation-conditioned modality fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_melt import dense_bins
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def active_from(scores):
    scores = np.asarray(scores, float)
    return scores[:, 0] > scores[:, 1:].max(1)


def responsibility_active(joint, visual, transcript, mode):
    j, v, t = map(active_from, (joint, visual, transcript))
    if mode == "majority": return (j.astype(int) + v + t) >= 2
    if mode == "joint_or_agree": return j | (v & t)
    if mode == "joint_and_any": return j & (v | t)
    if mode == "union": return j | v | t
    if mode == "joint_anchored_union":
        union = j | v | t
        out = np.zeros_like(j)
        bounds = np.flatnonzero(np.diff(np.r_[False, union, False])).reshape(-1, 2)
        for a, b in bounds:
            if j[a:b].any(): out[a:b] = True
        return out
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("joint", "visual", "transcript"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--mode", choices=("majority", "joint_or_agree", "joint_and_any",
                                       "union", "joint_anchored_union"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    def load(path):
        return {(r["dataset"],r["video_id"]):r
                for r in (json.loads(x) for x in path.open(encoding="utf-8"))}
    arms = {name: load(getattr(args,name)) for name in ("joint","visual","transcript")}
    if not (set(arms["joint"]) == set(arms["visual"]) == set(arms["transcript"])):
        raise RuntimeError("modality cohorts differ")
    for key in sorted(arms["joint"]):
        rows = [arms[x][key] for x in ("joint","visual","transcript")]
        method = f"role_responsibility_{args.mode}"
        error = next((r.get("error") for r in rows if r.get("error")), None)
        if error:
            append_jsonl(args.out, Prediction(method,key[0],key[1],rows[0]["duration"],error=error)); continue
        scores = [np.asarray(r["modality_evidence"]["binding_scores"],float) for r in rows]
        active = responsibility_active(*scores,args.mode)
        bounds=np.flatnonzero(np.diff(np.r_[False,active,False])).reshape(-1,2)
        duration=float(rows[0]["duration"])
        # Equal-weight geometric mean prevents one modality from setting dense ranking.
        full=np.cbrt(np.clip(scores[0][:,0]*scores[1][:,0]*scores[2][:,0],0,1))
        intervals=[Interval(a/16*duration,b/16*duration,float(full[a:b].mean())) for a,b in bounds]
        append_jsonl(args.out,Prediction(method,key[0],key[1],duration,
            score_curve=dense_bins(full,len(rows[0]["score_curve"])).tolist(),intervals=intervals,
            calls=0,modality_evidence={"fusion":args.mode,"equal_modality_weight":True,
             "paths":[[int(a),int(b)] for a,b in bounds]},raw={"arms":{x:str(getattr(args,x)) for x in arms}}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
