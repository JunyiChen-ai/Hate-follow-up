#!/usr/bin/env python3
"""
Probe Dual-Polarity Prompting (DPP) on Qwen3-VL-8B using existing data.

For each band sample, we have TWO independent predictions from Qwen3-VL-8B:
  (A) verdict: hateful/normal            (from V1 rescue_8b_*_v1.jsonl)
  (B) answer:  Yes/No                    (from offline_test_qwen3-vl-8b.jsonl)

Semantically (A)=="hateful" is equivalent to (B)=="Yes". If they disagree,
the judge is inconsistent under prompt-polarity and we ABSTAIN (keep stage-1).
If they agree, we commit to that verdict.

Rules tested:
  DPP_agree_flip: flip band only when both framings agree.
  raw_A, raw_B:   baselines using either framing alone.
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path("/data/jehc223/EMNLP3/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP3/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
V1 = {
    "MHClip_EN":   (0.7826, 0.6958),
    "MHClip_ZH":   (0.8255, 0.8023),
    "HateMM":      (0.8465, 0.8362),
    "ImpliHateVid":(0.8204, 0.8199),
}
LABELMAP = {
    "MHClip_EN":   {"Hateful":1,"Offensive":1,"Normal":0},
    "MHClip_ZH":   {"Hateful":1,"Offensive":1,"Normal":0},
    "HateMM":      {"Hate":1,"Non Hate":0},
    "ImpliHateVid":{"Hateful":1,"Normal":0},
}

def acc(y,yh):
    return sum(1 for a,b in zip(y,yh) if a==b)/len(y) if y else 0.0
def f1m(y,yh):
    cls = sorted(set(y)); s=0.0
    for c in cls:
        tp=sum(1 for a,b in zip(y,yh) if a==c and b==c)
        fp=sum(1 for a,b in zip(y,yh) if a!=c and b==c)
        fn=sum(1 for a,b in zip(y,yh) if a==c and b!=c)
        p=tp/(tp+fp) if tp+fp else 0
        r=tp/(tp+fn) if tp+fn else 0
        s += 2*p*r/(p+r) if p+r else 0
    return s/len(cls) if cls else 0

def ld(p):
    p=Path(p)
    if not p.exists(): return []
    return [json.loads(l) for l in open(p)]

def extract_rescue_verdict(rec):
    """Parse rescue_8b raw text to find 'verdict: hateful'/'normal'."""
    t = rec.get("rescue_response","") or ""
    t = t.lower()
    # find 'verdict:' after rationale
    for line in t.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("verdict:"):
            payload = line[len("verdict:"):].strip()
            if "hateful" in payload[:20]:
                return 1
            if "normal"  in payload[:20]:
                return 0
            return None
    # fallback scan
    if "verdict: hateful" in t: return 1
    if "verdict: normal" in t: return 0
    return None

def main():
    import sys
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
        base = {r["video_id"]: r for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band = {r["video_id"]: r for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl")}
        # Prompt A (V1 verdict:hateful/normal)
        rescue_path = ROOT/ds/"rescue_8b_bayes_band_rate_v1.jsonl"
        promptA = {}
        for r in ld(rescue_path):
            v = extract_rescue_verdict(r)
            if v is not None:
                promptA[r["video_id"]] = v
        # Prompt B (offline answer:Yes/No)
        off_path = ROOT/ds/"offline_test_qwen3-vl-8b.jsonl"
        promptB = {r["video_id"]: int(r["pred"]) for r in ld(off_path) if "pred" in r}

        all_ids = sorted(base.keys())
        y = [lab[v] for v in all_ids if lab.get(v,-1)>=0]
        y0 = [base[v]["pred_baseline"] for v in all_ids if lab.get(v,-1)>=0]
        valid_ids = [v for v in all_ids if lab.get(v,-1)>=0]

        band_ids = [v for v in valid_ids if v in band]
        covA = [v for v in band_ids if v in promptA]
        covB = [v for v in band_ids if v in promptB]
        both = [v for v in band_ids if v in promptA and v in promptB]
        print(f"\n=== {ds} ===  band={len(band_ids)}  cov(A)={len(covA)}  cov(B)={len(covB)}  both={len(both)}")
        agree = [v for v in both if promptA[v]==promptB[v]]
        disagree = [v for v in both if promptA[v]!=promptB[v]]
        print(f"  A==B on band: {len(agree)}/{len(both)}   A!=B: {len(disagree)}/{len(both)}")

        # Accuracy of each framing on band alone
        def band_acc(pmap):
            cov = [v for v in band_ids if v in pmap]
            return sum(1 for v in cov if pmap[v]==lab[v])/len(cov) if cov else 0.0, len(cov)
        aA, nA = band_acc(promptA)
        aB, nB = band_acc(promptB)
        print(f"  band-acc(A) = {aA:.4f} ({nA}), band-acc(B) = {aB:.4f} ({nB})")
        # Consistent subset accuracy
        aC_correct = sum(1 for v in agree if promptA[v]==lab[v])
        print(f"  band-acc(consistent subset) = {aC_correct/len(agree):.4f} ({aC_correct}/{len(agree)})")
        # Inconsistent subset: keep stage-1
        aI_correct = sum(1 for v in disagree if base[v]["pred_baseline"]==lab[v])
        print(f"  stage1-acc(inconsistent subset) = {aI_correct/max(1,len(disagree)):.4f} ({aI_correct}/{len(disagree)})")

        def apply_rule(rule_name):
            yh = list(y0)
            for i, v in enumerate(valid_ids):
                if v not in band:
                    continue
                if rule_name == "raw_A":
                    if v in promptA: yh[i] = promptA[v]
                elif rule_name == "raw_B":
                    if v in promptB: yh[i] = promptB[v]
                elif rule_name == "DPP_agree_flip":
                    if v in promptA and v in promptB and promptA[v]==promptB[v]:
                        yh[i] = promptA[v]
                elif rule_name == "DPP_asym_both_hateful":
                    if v in promptA and v in promptB:
                        if promptA[v]==1 and promptB[v]==1: yh[i] = 1
                        elif promptA[v]==0 and promptB[v]==0: yh[i] = 0
                elif rule_name == "asym_A_flip_to_1_only":
                    if v in promptA and promptA[v]==1:
                        yh[i] = 1
                elif rule_name == "asym_A_and_B_flip_to_1":
                    if v in promptA and v in promptB and promptA[v]==1 and promptB[v]==1:
                        yh[i] = 1
                elif rule_name == "DPP_either_hateful":
                    if v in promptA and v in promptB:
                        if promptA[v]==1 or promptB[v]==1: yh[i] = 1
                        else:
                            yh[i] = 0
                elif rule_name == "DPP_A_when_disagree_else_agree":
                    if v in promptA and v in promptB:
                        yh[i] = promptA[v] if promptA[v]==promptB[v] else promptA[v]
            return yh

        for rn in ["raw_A","raw_B","DPP_agree_flip",
                   "DPP_asym_both_hateful","asym_A_flip_to_1_only",
                   "asym_A_and_B_flip_to_1","DPP_either_hateful",
                   "DPP_A_when_disagree_else_agree"]:
            yh = apply_rule(rn)
            a = acc(y,yh); m = f1m(y,yh)
            bar_a, bar_m = V1[ds]
            flag = "*" if (a > bar_a and m > bar_m) else " "
            print(f"   {flag} {rn:30s}  ACC={a:.4f}  mF1={m:.4f}")

if __name__ == "__main__":
    main()
