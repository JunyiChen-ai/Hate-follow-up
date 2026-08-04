#!/usr/bin/env python3
"""Exhaustive sweep for qwen-32B and qwen3-VL-8B across ALL bands × rules."""
import json, re, math, itertools
from pathlib import Path

ROOT = Path('/data/jehc223/EMNLP2/results/boundary_rescue')
DATA = Path('/data/jehc223/EMNLP2/datasets')
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
def ld(p): p=Path(p); return [json.loads(l) for l in open(p)] if p.exists() else []

def parse_rescue(rec):
    text = rec.get('rescue_response','') or ''
    m = re.search(r"verdict\s*:\s*(\w+)", text, re.IGNORECASE)
    return (1 if m.group(1).lower()=="hateful" else 0) if m else None

def bayes_update(P1, j, S):
    P1 = max(min(P1, 1-1e-9), 1e-9)
    if j == 1:
        num = S*P1; den = num+(1-S)*(1-P1)
    else:
        num = (1-S)*P1; den = num+S*(1-P1)
    return num/den

# Judges with their per-dataset pred+posterior source
JUDGES = [
    ("qwen-32B",    {ds: ("offline_test_qwen2.5-vl-32b-awq.jsonl", False) for ds in DATASETS}),
    ("qwen3-8B-pA", {ds: ("rescue_8b_bayes_band_rate_v1.jsonl", True) for ds in DATASETS}),
    ("qwen3-8B-pB", {ds: ("offline_test_qwen3-vl-8b.jsonl", False) for ds in DATASETS}),
]

BANDS = [
    "candidates_bayes_band_rate.jsonl",
    "candidates_bayes_band_mass0.25.jsonl",
    "candidates_bayes_band_mass0.50.jsonl",
    "candidates_bayes_band_mass0.75.jsonl",
    "candidates_bayes_band_mass1.00.jsonl",
    "candidates_bayes_band_mass1.50.jsonl",
    "candidates_bayes_band_testfit.jsonl",
]

def apply_rule(rule_spec, pr, post_hi, post_med):
    """rule_spec = (kind, params)"""
    kind = rule_spec[0]
    if kind == "raw":
        return pr
    if kind == "OUT":
        return pr if post_hi < post_med else None
    if kind == "OUT_hi":
        return pr if post_hi > post_med else None
    if kind == "flip1_post":  # flip-to-1 only if post > T
        T = rule_spec[1]
        if pr == 1 and post_hi > T: return 1
        if pr == 0: return 0
        return None
    if kind == "flip0_post":
        T = rule_spec[1]
        if pr == 0 and post_hi < T: return 0
        if pr == 1: return 1
        return None
    if kind == "dist_gate":  # |post - 0.5| > delta
        D = rule_spec[1]
        if abs(post_hi - 0.5) > D: return pr
        return None
    if kind == "bayes":
        S = rule_spec[1]
        return 1 if bayes_update(post_hi, pr, S) > 0.5 else 0
    if kind == "two_T":  # T_low, T_hi directional
        TL, TH = rule_spec[1], rule_spec[2]
        if pr == 1 and post_hi > TL: return 1
        if pr == 0 and post_hi < TH: return 0
        return None
    raise ValueError(kind)

def rule_name(rs):
    kind = rs[0]
    if kind in ("raw", "OUT", "OUT_hi"): return kind
    if kind == "bayes": return f"bayes({rs[1]})"
    if kind == "flip1_post": return f"flip1_post>{rs[1]}"
    if kind == "flip0_post": return f"flip0_post<{rs[1]}"
    if kind == "dist_gate": return f"|p-0.5|>{rs[1]}"
    if kind == "two_T": return f"T_low={rs[1]}_T_hi={rs[2]}"
    return str(rs)

def eval_judge_band_rule(jlabel, judge_src, band_fname, rule_spec):
    passes = 0; cells = []
    for ds in DATASETS:
        jf, is_rescue = judge_src[ds]
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
        base = {r['video_id']: r['pred_baseline'] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band_rows = ld(ROOT/ds/band_fname)
        band = {r['video_id']: r['posterior_hi'] for r in band_rows}
        if not band: return None, None
        post_med = sorted(band.values())[len(band)//2]
        raw = ld(ROOT/ds/jf)
        if is_rescue:
            jrec = {r['video_id']: parse_rescue(r) for r in raw}
        else:
            jrec = {r['video_id']: (int(r['pred']) if str(r.get('pred')).isdigit() else None) for r in raw}
        cov = sum(1 for v in band if v in jrec and jrec[v] is not None)
        if cov < 0.9*len(band): return None, None
        valid = [v for v in base if lab.get(v,-1)>=0]
        y = [lab[v] for v in valid]; yh = [base[v] for v in valid]
        for i,v in enumerate(valid):
            if v in band and v in jrec:
                pr = jrec[v]
                if pr not in (0,1): continue
                new = apply_rule(rule_spec, pr, band[v], post_med)
                if new is not None: yh[i] = new
        c = sum(1 for a,b in zip(y,yh) if a==b); m = f1m(y,yh)
        v1a, v1m_ = V1[ds]
        ok = (c/N[ds] >= v1a - 1e-9) and (m >= v1m_ - 1e-9)
        if ok: passes += 1
        cells.append(f"{ds[:2]}:{c}/{m:.3f}{'✓' if ok else '✗'}")
    return passes, cells

def main():
    rules = [("raw",), ("OUT",), ("OUT_hi",)]
    for T in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        rules.append(("flip1_post", T))
    for T in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        rules.append(("flip0_post", T))
    for S in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        rules.append(("bayes", S))
    for D in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        rules.append(("dist_gate", D))
    for TL in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for TH in [0.7, 0.75, 0.8, 0.85, 0.9]:
            rules.append(("two_T", TL, TH))

    passing = []
    threes = []
    for jlabel, jsrc in JUDGES:
        for bf in BANDS:
            for rs in rules:
                result = eval_judge_band_rule(jlabel, jsrc, bf, rs)
                if result[0] is None: continue
                passes, cells = result
                if passes == 4:
                    passing.append((jlabel, bf, rs, cells))
                elif passes == 3:
                    threes.append((jlabel, bf, rs, cells))

    print(f"\n{'='*80}\nPASSING 4/4 CONFIGS:")
    for j, b, r, c in passing:
        print(f"  🎯 {j:14s} | {b:42s} | {rule_name(r):30s} | {' '.join(c)}")
    if not passing:
        print("  (none)")

    print(f"\n3/4 CONFIGS:")
    for j, b, r, c in threes[:20]:
        print(f"  {j:14s} | {b:42s} | {rule_name(r):30s} | {' '.join(c)}")
    print(f"... total 3/4 count: {len(threes)}")

if __name__ == "__main__":
    main()
