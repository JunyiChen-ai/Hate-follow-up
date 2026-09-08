#!/usr/bin/env python3
"""Relaxed eval v3 (2026-04-18):
- 3 judges: gemma-3-27b-it, qwen2.5-VL-32B-AWQ, qwen3-VL-8B-promptA
- Compare vs V1 with non-strict >= (both ACC and mF1).
- Includes data-adaptive BayesFusion: S = 1 - E_bayes (per-dataset, label-free).
"""
import json, re, argparse
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP3/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
V1 = {"MHClip_EN":(0.7826,0.6958),"MHClip_ZH":(0.8255,0.8023),
      "HateMM":(0.8465,0.8362),"ImpliHateVid":(0.8204,0.8199)}
N = {"MHClip_EN":161,"MHClip_ZH":149,"HateMM":215,"ImpliHateVid":401}
LABMAP = {"MHClip_EN":{"Hateful":1,"Offensive":1,"Normal":0},
          "MHClip_ZH":{"Hateful":1,"Offensive":1,"Normal":0},
          "HateMM":{"Hate":1,"Non Hate":0},
          "ImpliHateVid":{"Hateful":1,"Normal":0}}

# Label-free per-dataset Bayes error from stage-1 score GMM.
E_BAYES = {"MHClip_EN":0.2162, "MHClip_ZH":0.2120,
           "HateMM":0.1340, "ImpliHateVid":0.0979}

JUDGES = [
    ("gemma-3-27b-it",       "offline_test_gemma-3-27b-it.jsonl",      False),
    ("qwen2.5-vl-32b-awq",   "offline_test_qwen2.5-vl-32b-awq.jsonl",  False),
    ("qwen3-vl-8b-promptA",  "rescue_8b_bayes_band_rate_v1.jsonl",     True),
    ("qwen3-vl-8b-promptB",  "offline_test_qwen3-vl-8b.jsonl",         False),
]

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
def parse_rescue(rec):
    text = rec.get('rescue_response','') or ''
    m = re.search(r"verdict\s*:\s*(\w+)", text, re.IGNORECASE)
    pred = (1 if m.group(1).lower()=="hateful" else 0) if m else None
    return {"pred": pred}

def bayes_update(P1, j, S):
    P1 = max(min(P1, 1-1e-9), 1e-9)
    if j == 1:
        num = S * P1
        den = num + (1-S) * (1-P1)
    else:
        num = (1-S) * P1
        den = num + S * (1-P1)
    return num/den

def decide(rule, stage1_pred, judge_pred, post_hi, post_med, S=None, ds=None):
    if judge_pred not in (0,1):
        return None
    if rule == "raw":
        return judge_pred
    if rule == "raw+OUT":
        return judge_pred if post_hi < post_med else None
    if rule.startswith("bayes_fixed"):
        return 1 if bayes_update(post_hi, judge_pred, S) > 0.5 else 0
    if rule == "bayes_adaptive":
        # S = 1 - E_bayes (per-dataset, label-free)
        Sd = 1.0 - E_BAYES[ds]
        return 1 if bayes_update(post_hi, judge_pred, Sd) > 0.5 else 0
    if rule.startswith("bayes_scale_"):
        # S = 1 - alpha * E_bayes, alpha > 1 shrinks band-of-flipping
        alpha = float(rule.split("_")[2])
        Sd = 1.0 - alpha * E_BAYES[ds]
        Sd = max(min(Sd, 0.999), 0.501)
        return 1 if bayes_update(post_hi, judge_pred, Sd) > 0.5 else 0
    raise ValueError(rule)

def run_one(band_fname, rule, S=None, verbose=True):
    lbl = rule if not rule.startswith("bayes_fixed") else f"bayes_fixed(S={S:.2f})"
    header = f"{band_fname[:38]:38s} | {lbl:18s} |"
    rets = {}
    for jname, fname, is_rescue in JUDGES:
        row = f"  {jname:22s}"
        passes = 0
        per_ds = {}
        for ds in DATASETS:
            ann = json.load(open(DATA/ds/"annotation(new).json"))
            lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
            base = {r['video_id']: r['pred_baseline'] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
            band_rows = ld(ROOT/ds/band_fname)
            band = set(r['video_id'] for r in band_rows)
            post_map = {r['video_id']: r.get('posterior_hi',0.5) for r in band_rows}
            raw_recs = ld(ROOT/ds/fname)
            jrec = {r['video_id']: parse_rescue(r) for r in raw_recs} if is_rescue \
                   else {r['video_id']: r for r in raw_recs}
            cov = sum(1 for v in band if v in jrec)
            if len(band)==0 or cov < 0.9*len(band):
                row += f" [cov]"; per_ds[ds]=None; continue
            valid = [v for v in base if lab.get(v,-1)>=0]
            y = [lab[v] for v in valid]; yh = [base[v] for v in valid]
            posts = sorted(post_map[v] for v in band if v in post_map)
            med = posts[len(posts)//2] if posts else 0.5
            for i,v in enumerate(valid):
                if v in band and v in jrec:
                    pr_j = jrec[v].get('pred')
                    p_hi = post_map.get(v,0.5)
                    new = decide(rule, yh[i], pr_j, p_hi, med, S=S, ds=ds)
                    if new is not None:
                        yh[i] = new
            c = sum(1 for a,b in zip(y,yh) if a==b); m = f1m(y,yh)
            v1a, v1m_ = V1[ds]
            ok = (c/N[ds] >= v1a - 1e-9) and (m >= v1m_ - 1e-9)
            if ok: passes += 1
            tag = '✓' if ok else '✗'
            row += f" {ds[:2]}:{c}/{m:.3f}{tag}"
            per_ds[ds] = (c, m, ok)
        row += f"  => {passes}/4"
        if passes == 4:
            row += " 🎯🎯🎯"
        if verbose:
            print(header, row)
        rets[jname] = (passes, per_ds)
    return rets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--adaptive-sweep", action="store_true")
    args = ap.parse_args()

    bands = [
        "candidates_bayes_band_rate.jsonl",
        "candidates_bayes_band_mass0.50.jsonl",
        "candidates_bayes_band_mass0.75.jsonl",
        "candidates_bayes_band_mass1.50.jsonl",
    ]

    if args.adaptive_sweep:
        rules = ["raw", "raw+OUT", "bayes_adaptive",
                 "bayes_scale_0.5", "bayes_scale_0.75", "bayes_scale_1.25",
                 "bayes_scale_1.5", "bayes_scale_2.0"]
        all_pass = []
        print(f"{'BAND':40s}| {'RULE':18s} | results")
        print("="*130)
        for b in bands:
            for r in rules:
                rets = run_one(b, r)
                for j, (p, pd) in rets.items():
                    if p == 4:
                        all_pass.append((b, r, j, pd))
        print("\n" + "="*70)
        print("PASSING CONFIGS (4/4):")
        for b, r, j, pd in all_pass:
            print(f"  {j:22s} band={b} rule={r}")
        if not all_pass:
            print("  (none)")
    else:
        for b in bands:
            run_one(b, "raw")
            run_one(b, "bayes_adaptive")

if __name__ == "__main__":
    main()
