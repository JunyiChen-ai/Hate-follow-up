#!/usr/bin/env python3
"""Evaluate RTG (real-target-gate) rule on gemma-27B band output.

One judge call, two outputs: real_target_identified (Yes/No) + verdict (hateful/normal).
Rule: flip to hateful only if verdict=hateful AND real_target_identified=Yes.
"""
import json, re
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP2/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
V1 = {"MHClip_EN":(0.7826,0.6958),"MHClip_ZH":(0.8255,0.8023),
      "HateMM":(0.8465,0.8362),"ImpliHateVid":(0.8204,0.8199)}
N = {"MHClip_EN":161,"MHClip_ZH":149,"HateMM":215,"ImpliHateVid":401}
LABMAP = {"MHClip_EN":{"Hateful":1,"Offensive":1,"Normal":0},
          "MHClip_ZH":{"Hateful":1,"Offensive":1,"Normal":0},
          "HateMM":{"Hate":1,"Non Hate":0},
          "ImpliHateVid":{"Hateful":1,"Normal":0}}

def f1m(y,yh):
    cls=sorted(set(y)); s=0.0
    for c in cls:
        tp=sum(1 for a,b in zip(y,yh) if a==c and b==c)
        fp=sum(1 for a,b in zip(y,yh) if a!=c and b==c)
        fn=sum(1 for a,b in zip(y,yh) if a==c and b!=c)
        p=tp/(tp+fp) if tp+fp else 0
        r=tp/(tp+fn) if tp+fn else 0
        s += 2*p*r/(p+r) if p+r else 0
    return s/len(cls) if cls else 0
def ld(p):
    p=Path(p); return [json.loads(l) for l in open(p)] if p.exists() else []

def parse_rtg(rec):
    """Parse real_target_identified and verdict from RTG output."""
    text = rec.get('raw_response','') or ''
    # verdict field
    pred = None
    vm = re.search(r"verdict\s*:\s*(\w+)", text, re.IGNORECASE)
    if vm:
        v = vm.group(1).lower()
        if "hateful" in v: pred = 1
        elif "normal" in v: pred = 0
    # real_target
    rt = None
    rm = re.search(r"real_target(?:_identified)?\s*:\s*(\w+)", text, re.IGNORECASE)
    if rm:
        v = rm.group(1).lower()
        if v.startswith("yes"): rt = True
        elif v.startswith("no"): rt = False
    return pred, rt

def run(band_fname, rtg_fname, rule):
    row = f"{band_fname[:30]:30s} | {rule:22s}"
    passes = 0
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
        base = {r['video_id']: r['pred_baseline'] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band_rows = ld(ROOT/ds/band_fname)
        band = set(r['video_id'] for r in band_rows)
        post_map = {r['video_id']: r.get('posterior_hi',0.5) for r in band_rows}
        rtg = {r['video_id']: parse_rtg(r) for r in ld(ROOT/ds/rtg_fname)}
        cov = sum(1 for v in band if v in rtg)
        if len(band)==0 or cov < 0.9*len(band):
            row += f" [cov {cov}/{len(band)}]"; continue
        valid = [v for v in base if lab.get(v,-1)>=0]
        y = [lab[v] for v in valid]; yh = [base[v] for v in valid]
        for i,v in enumerate(valid):
            if v in band and v in rtg:
                pr, rt = rtg[v]
                if pr not in (0,1): continue
                if rule == "raw":
                    yh[i] = pr
                elif rule == "rtg_flip1_gate":
                    # flip-to-hateful ONLY if real_target=Yes; flip-to-0 freely
                    if pr == 1 and rt is True:
                        yh[i] = 1
                    elif pr == 0:
                        yh[i] = 0
                elif rule == "rtg_both_gate":
                    # both flip-to-1 needs rt=Yes, flip-to-0 needs rt=No
                    if pr == 1 and rt is True:
                        yh[i] = 1
                    elif pr == 0 and rt is False:
                        yh[i] = 0
                elif rule == "verdict_AND_target":
                    # pred=1 only if verdict=hateful AND rt=Yes; pred=0 else
                    if pr == 1 and rt is True:
                        yh[i] = 1
                    else:
                        yh[i] = 0
        c = sum(1 for a,b in zip(y,yh) if a==b); m = f1m(y,yh)
        v1a, v1m_ = V1[ds]
        ok = (c/N[ds] >= v1a - 1e-9) and (m >= v1m_ - 1e-9)
        if ok: passes += 1
        tag = '✓' if ok else '✗'
        row += f" {ds[:2]}:{c}/{m:.3f}{tag}"
    row += f"  => {passes}/4"
    if passes == 4: row += " 🎯🎯🎯"
    print(row)
    return passes

def main():
    bands = ["candidates_bayes_band_rate.jsonl",
             "candidates_bayes_band_mass0.75.jsonl"]
    rtg_file = "offline_test_band_rtg_gemma-3-27b-it.jsonl"
    for b in bands:
        for r in ["raw", "rtg_flip1_gate", "rtg_both_gate", "verdict_AND_target"]:
            run(b, rtg_file, r)

if __name__ == "__main__":
    main()
