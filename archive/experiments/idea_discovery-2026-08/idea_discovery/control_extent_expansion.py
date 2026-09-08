#!/usr/bin/env python3
"""Matched controls for RAVEL's modality-conditioned extent expansion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.fuse_role_modalities import active_from
from scripts.idea_discovery.run_melt import dense_bins
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def comps(active):
    return np.flatnonzero(np.diff(np.r_[False,np.asarray(active,bool),False])).reshape(-1,2)


def anchored_union(j, v, t):
    union=j|v|t; out=np.zeros_like(j)
    for a,b in comps(union):
        if j[a:b].any(): out[a:b]=True
    return out


def controlled(j, v, t, mode, key):
    if mode.startswith("dilate"):
        radius=int(mode[-1]); out=np.zeros_like(j)
        for a,b in comps(j): out[max(0,a-radius):min(len(j),b+radius)]=True
        return out
    if mode=="shift8": return anchored_union(j,np.roll(v,8),np.roll(t,8))
    if mode=="random_matched":
        seed=int(hashlib.sha256(f"{key[0]}/{key[1]}".encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed); arms=[]
        for x in (v,t):
            y=np.zeros_like(x); n=int(x.sum())
            if n: y[rng.choice(len(x),n,replace=False)]=True
            arms.append(y)
        return anchored_union(j,*arms)
    raise ValueError(mode)


def main():
    ap=argparse.ArgumentParser()
    for x in ("joint","visual","transcript"): ap.add_argument(f"--{x}",type=Path,required=True)
    ap.add_argument("--mode",choices=("dilate1","dilate2","shift8","random_matched"),required=True)
    ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise RuntimeError(f"refusing existing output: {a.out}")
    def load(p): return {(r["dataset"],r["video_id"]):r for r in (json.loads(x) for x in p.open())}
    arms={x:load(getattr(a,x)) for x in ("joint","visual","transcript")}
    for key in sorted(set.intersection(*(set(x) for x in arms.values()))):
        rows=[arms[x][key] for x in ("joint","visual","transcript")]
        method=f"extent_control_{a.mode}"; error=next((r.get("error") for r in rows if r.get("error")),None)
        if error: append_jsonl(a.out,Prediction(method,key[0],key[1],rows[0]["duration"],error=error));continue
        scores=[np.asarray(r["modality_evidence"]["binding_scores"],float) for r in rows]
        active=controlled(*(active_from(x) for x in scores),a.mode,key); bounds=comps(active)
        d=float(rows[0]["duration"]); full=scores[0][:,0]
        intervals=[Interval(x/16*d,y/16*d,float(full[x:y].mean())) for x,y in bounds]
        append_jsonl(a.out,Prediction(method,key[0],key[1],d,score_curve=dense_bins(full,len(rows[0]["score_curve"])).tolist(),
         intervals=intervals,calls=0,modality_evidence={"control":a.mode,"paths":[[int(x),int(y)] for x,y in bounds]}))


if __name__=="__main__": main()
