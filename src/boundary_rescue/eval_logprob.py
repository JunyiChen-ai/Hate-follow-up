#!/usr/bin/env python3
"""Evaluate confidence-gated rules using judge verdict_logprobs.

Rule ideas:
- conf_gate(tau): flip only if p(chosen_class) > tau
- bayes_native: Bayes fusion where sens=spec=p(chosen_class) (per-sample)
- bayes_averaged: Bayes fusion with S = dataset-avg p(chosen) on band
"""
import json, re, argparse
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

def parse_lp(rec):
    """Parse logprobs: return (pred, log_ratio, p_h, p_n).
    log_ratio = log(p_chosen) - log(p_other). Larger = more confident."""
    import math
    pred = rec.get('pred')
    try: pred = int(pred)
    except: pred = None
    vl = rec.get('verdict_logprobs', {})
    if isinstance(vl, str):
        try: vl = eval(vl)
        except: vl = {}
    p_h = max(vl.get('p_hateful', 1e-40), 1e-40)
    p_n = max(vl.get('p_normal', 1e-40), 1e-40)
    chosen = p_h if pred == 1 else (p_n if pred == 0 else None)
    other = p_n if pred == 1 else (p_h if pred == 0 else None)
    log_ratio = (math.log(chosen) - math.log(other)) if chosen and other else None
    return pred, log_ratio, p_h, p_n

def bayes_update(P1, j, S):
    P1 = max(min(P1, 1-1e-9), 1e-9)
    if j == 1:
        num = S * P1; den = num + (1-S) * (1-P1)
    else:
        num = (1-S) * P1; den = num + S * (1-P1)
    return num/den

def run_one(band_fname, judge_fname, rule, param=None):
    row = f"{band_fname[:34]:34s} | {judge_fname[:32]:32s} | {rule:25s}"
    total_pass = 0
    per_ds_s = []
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
        base = {r['video_id']: r['pred_baseline'] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band_rows = ld(ROOT/ds/band_fname)
        band = set(r['video_id'] for r in band_rows)
        post_map = {r['video_id']: r.get('posterior_hi',0.5) for r in band_rows}
        jrec = {r['video_id']: r for r in ld(ROOT/ds/judge_fname)}
        cov = sum(1 for v in band if v in jrec)
        if len(band) == 0 or cov < 0.9*len(band):
            per_ds_s.append(f"{ds[:2]}:[cov {cov}/{len(band)}]")
            continue
        # dataset-median log_ratio on band
        lr_list = []
        for v in band:
            if v in jrec:
                _, lr, _, _ = parse_lp(jrec[v])
                if lr is not None: lr_list.append(lr)
        lr_list.sort()
        lr_med = lr_list[len(lr_list)//2] if lr_list else 20.0
        S_ds = 0.85
        valid = [v for v in base if lab.get(v,-1)>=0]
        y = [lab[v] for v in valid]; yh = [base[v] for v in valid]
        for i,v in enumerate(valid):
            if v in band and v in jrec:
                pr, lr, ph, pn = parse_lp(jrec[v])
                if pr not in (0,1): continue
                if rule == "raw":
                    yh[i] = pr
                elif rule == "lr_gate":
                    if lr is not None and lr >= param:
                        yh[i] = pr
                elif rule == "lr_gate_med":
                    # gate at dataset-local median (adaptive)
                    if lr is not None and lr >= lr_med:
                        yh[i] = pr
                elif rule == "lr_gate_quart":
                    # gate at dataset-local 75th percentile (stricter adaptive)
                    if lr is not None and lr >= lr_list[3*len(lr_list)//4]:
                        yh[i] = pr
                elif rule == "lr_asym_flip1":
                    # flip-to-1 needs high lr; flip-to-0 allowed anywhere
                    if pr == 1 and lr is not None and lr >= param:
                        yh[i] = 1
                    elif pr == 0:
                        yh[i] = 0
                elif rule == "lr_asym_flip0":
                    # flip-to-0 needs high lr; flip-to-1 allowed anywhere
                    if pr == 0 and lr is not None and lr >= param:
                        yh[i] = 0
                    elif pr == 1:
                        yh[i] = 1
                elif rule == "bayes_lr_norm":
                    # turn lr into S via sigmoid, use bayes update
                    import math
                    S = 1.0 / (1.0 + math.exp(-lr/10)) if lr is not None else 0.5
                    S = max(min(S, 0.999), 0.501)
                    P1 = post_map.get(v, 0.5)
                    P1n = bayes_update(P1, pr, S)
                    yh[i] = 1 if P1n > 0.5 else 0
        c = sum(1 for a,b in zip(y,yh) if a==b); m = f1m(y,yh)
        v1a, v1m_ = V1[ds]
        ok = (c/N[ds] >= v1a - 1e-9) and (m >= v1m_ - 1e-9)
        if ok: total_pass += 1
        tag = '✓' if ok else '✗'
        per_ds_s.append(f"{ds[:2]}:{c}/{m:.3f}{tag}")
    row += f" | {' '.join(per_ds_s)}  => {total_pass}/4"
    if total_pass == 4: row += " 🎯🎯🎯"
    print(row)
    return total_pass

def main():
    bands = [
        "candidates_bayes_band_rate.jsonl",
        "candidates_bayes_band_mass0.75.jsonl",
    ]
    judges = [
        "offline_test_lp_qwen3-vl-8b.jsonl",   # produced by new run
        "offline_test_lp_gemma-3-27b-it.jsonl", # produced by new run
    ]
    for b in bands:
        for j in judges:
            if not any((ROOT/ds/j).exists() for ds in DATASETS):
                print(f"skip {j} (not yet produced)"); continue
            run_one(b, j, "raw")
            for tau in [5, 10, 15, 20, 25, 28, 30]:
                run_one(b, j, "lr_gate", param=tau)
            run_one(b, j, "lr_gate_med")
            run_one(b, j, "lr_gate_quart")
            for tau in [15, 20, 25, 28, 30]:
                run_one(b, j, "lr_asym_flip1", param=tau)
            for tau in [15, 20, 25, 28, 30]:
                run_one(b, j, "lr_asym_flip0", param=tau)
            run_one(b, j, "bayes_lr_norm")

if __name__ == "__main__":
    main()
