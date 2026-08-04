#!/usr/bin/env python3
"""MADP-based relaxed eval (>=V1).
- Judges with MADP: qwen2.5-VL-32B-AWQ, qwen3-VL-8B
  (gemma-27B only has EAA v1/v3, no MADP; try v1 separately)
- Band options: rate, mass subsets
- Rules: many Boolean combinations on {target, hostile, protected, answer}
"""
import json, re, argparse
from pathlib import Path
from itertools import product

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

MADP_FILES = {
    "qwen-32B":       "offline_test_band_eaa_qwen2.5-vl-32b-awq.jsonl",
    "qwen3-8B-madp":  "offline_test_band_eaa_qwen3-vl-8b_madp.jsonl",
    "gemma-27B-madp": "offline_test_band_eaa_gemma-3-27b-it.jsonl",
}

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

def get_field(text, name):
    if not text: return None
    m = re.search(rf"^\s*{name}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not m: return None
    v = m.group(1).strip().lower().rstrip(".").split()[0] if m.group(1).strip() else ""
    if v.startswith("yes"): return True
    if v.startswith("no"):  return False
    return None

def parse_madp(rec):
    raw = rec.get('raw_response','') or ''
    return {
        'target':    get_field(raw, 'target_real_group'),
        'hostile':   get_field(raw, 'hostile_framing'),
        'protected': get_field(raw, 'protected_context'),
        'answer':    get_field(raw, 'answer'),
    }

def rule_apply(rule, m):
    T, H, P, A = m['target'], m['hostile'], m['protected'], m['answer']
    if rule == "answer":
        if A is True: return 1
        if A is False: return 0
        return None
    if rule == "bool_THP":
        if T is True and H is True and P is False: return 1
        if T is False or H is False or P is True: return 0
        return None
    if rule == "bool_TH":
        if T is True and H is True: return 1
        if T is False or H is False: return 0
        return None
    if rule == "answer_AND_not_P":
        # Keep stage-1 if protected=Yes
        if A is True and P is False: return 1
        if A is False: return 0
        if P is True: return None  # Veto, keep stage-1
        return None
    if rule == "hostile_only":
        if H is True: return 1
        if H is False: return 0
        return None
    if rule == "answer_AND_TH":
        if A is True and T is True and H is True: return 1
        if A is False: return 0
        if T is False or H is False: return 0
        return None
    if rule == "answer_OR_THnP":
        pos = (A is True) or (T is True and H is True and P is False)
        neg = (A is False) and not (T is True and H is True and P is False)
        if pos: return 1
        if neg: return 0
        return None
    if rule == "2of3_THnP":
        flags = [T is True, H is True, P is False]
        if sum(flags) >= 2: return 1
        if sum(f is False or f is None for f in [T,H]) >= 2 and P is True: return 0
        if A is False: return 0
        return None
    if rule == "answer_OR_bool_veto_P":
        # Flip to 1 if answer=Yes OR bool_THP; veto any flip if protected=Yes
        if P is True: return None
        if A is True: return 1
        if T is True and H is True: return 1
        if A is False: return 0
        return None
    raise ValueError(rule)

def run(band_fname, jkey, rule):
    passes = 0
    per_ds_s = []
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
        base = {r['video_id']: r['pred_baseline'] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band_rows = ld(ROOT/ds/band_fname)
        band = set(r['video_id'] for r in band_rows)
        madp_recs = ld(ROOT/ds/MADP_FILES[jkey])
        jrec = {r['video_id']: parse_madp(r) for r in madp_recs}
        cov = sum(1 for v in band if v in jrec)
        if len(band)==0 or cov < 0.9*len(band):
            per_ds_s.append(f"{ds[:2]}:[cov {cov}/{len(band)}]"); continue
        valid = [v for v in base if lab.get(v,-1)>=0]
        y = [lab[v] for v in valid]; yh = [base[v] for v in valid]
        for i,v in enumerate(valid):
            if v in band and v in jrec:
                new = rule_apply(rule, jrec[v])
                if new is not None:
                    yh[i] = new
        c = sum(1 for a,b in zip(y,yh) if a==b); m = f1m(y,yh)
        v1a, v1m_ = V1[ds]
        ok = (c/N[ds] >= v1a - 1e-9) and (m >= v1m_ - 1e-9)
        if ok: passes += 1
        tag = '✓' if ok else '✗'
        per_ds_s.append(f"{ds[:2]}:{c}/{m:.3f}{tag}")
    tag = " 🎯🎯🎯" if passes == 4 else ""
    print(f"{band_fname[:35]:35s} | {jkey:15s} | {rule:22s} | {' '.join(per_ds_s)}  => {passes}/4{tag}")
    return passes

def main():
    bands = [
        "candidates_bayes_band_rate.jsonl",
        "candidates_bayes_band_mass0.50.jsonl",
        "candidates_bayes_band_mass0.75.jsonl",
    ]
    rules = ["answer", "bool_THP", "bool_TH", "answer_AND_not_P",
             "hostile_only", "answer_AND_TH", "answer_OR_THnP",
             "2of3_THnP", "answer_OR_bool_veto_P"]
    best = []
    for b in bands:
        for jkey in MADP_FILES:
            for r in rules:
                p = run(b, jkey, r)
                if p == 4:
                    best.append((b, jkey, r))
    print("\nPASSING CONFIGS (4/4 >=V1):")
    for b, j, r in best:
        print(f"  {j} band={b} rule={r}")
    if not best:
        print("  (none)")

if __name__ == "__main__":
    main()
