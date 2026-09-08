#!/usr/bin/env python3
"""
Evaluate the Evidence-Anchored Abstention (EAA) rule on band samples.

Rule (universal, tuning-free, judge-agnostic):
  For each band video, stage-2 judge outputs one of {Yes, No, Unsure}.
  - Yes   → flip to hateful
  - No    → flip to normal
  - Unsure → keep stage-1

Also probes an "evidence-gate" variant:
  - If rationale's 'evidence:' field contains "insufficient" (case-insensitive)
    OR is missing, treat as Unsure.

Inputs: offline_test_band_eaa_<judge>.jsonl  (pred ∈ {0,1,-2}).
"""
import json, re
from pathlib import Path
from collections import Counter, defaultdict

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
    return [json.loads(l) for l in open(p)] if p.exists() else []

EVIDENCE_INSUFF = re.compile(r"evidence\s*:\s*insufficient\b", re.IGNORECASE)

def main():
    found_any = False
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
        base = {r["video_id"]: r for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band = {r["video_id"]: r for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl")}
        valid_ids = [v for v in base if lab.get(v, -1) >= 0]
        y  = [lab[v] for v in valid_ids]
        y0 = [base[v]["pred_baseline"] for v in valid_ids]

        eaa_files = sorted((ROOT/ds).glob("offline_test_band_eaa_*.jsonl"))
        if not eaa_files:
            print(f"[{ds}] no EAA files yet — skipping")
            continue
        found_any = True
        print(f"\n=== {ds}  (V1 bar {V1[ds][0]:.4f}/{V1[ds][1]:.4f}, stage-1 "
              f"{acc(y,y0):.4f}/{f1m(y,y0):.4f}) ===")
        for fp in eaa_files:
            jname = fp.name[len("offline_test_band_eaa_"):-len(".jsonl")]
            jrec = {r["video_id"]: r for r in ld(fp)}
            cov_band = [v for v in band if v in jrec]
            if len(cov_band) < 0.9 * len(band):
                print(f"  {jname:35s} band coverage {len(cov_band)}/{len(band)} — skip")
                continue
            preds_dist = Counter(r.get("pred") for r in jrec.values())
            print(f"  --- {jname} (cov {len(cov_band)}/{len(band)}, verdict dist {dict(preds_dist)}) ---")

            # Rule 1: EAA core (Unsure → keep stage-1)
            yh = list(y0)
            abst = 0; flipped = 0
            for i, v in enumerate(valid_ids):
                if v in band and v in jrec:
                    p = jrec[v].get("pred")
                    if p == -2:  # abstain
                        abst += 1
                    elif p in (0, 1):
                        if p != y0[i]:
                            flipped += 1
                        yh[i] = p
            a = acc(y, yh); m = f1m(y, yh)
            bar_a, bar_m = V1[ds]
            flag = "*" if (a > bar_a and m > bar_m) else " "
            print(f"   {flag} rule_EAA              acc={a:.4f} mF1={m:.4f}  (abst={abst}, flip={flipped})")

            # Rule 2: EAA + evidence-gate (insufficient → abstain)
            yh = list(y0)
            abst = 0; flipped = 0
            for i, v in enumerate(valid_ids):
                if v in band and v in jrec:
                    p = jrec[v].get("pred")
                    raw = jrec[v].get("raw_response", "") or ""
                    evid_insuff = bool(EVIDENCE_INSUFF.search(raw))
                    if p == -2 or evid_insuff:
                        abst += 1
                    elif p in (0, 1):
                        if p != y0[i]:
                            flipped += 1
                        yh[i] = p
            a = acc(y, yh); m = f1m(y, yh)
            flag = "*" if (a > bar_a and m > bar_m) else " "
            print(f"   {flag} rule_EAA_evidGate     acc={a:.4f} mF1={m:.4f}  (abst={abst}, flip={flipped})")

            # Rule 3: raw-flip (ignore abstention, treat Unsure as stage-1 flip attempt?)
            # Actually with EAA, judge has (0,1,-2). raw-flip = take judge verdict where available,
            # abstain -> keep stage-1 (same as EAA). So identical to Rule 1.
            # Variant: trust judge including abstain as 'keep stage-1' but also force binary via
            # looking into rationale — skip for now.

    if not found_any:
        print("No EAA output files found. Waiting for job to complete.")

if __name__ == "__main__":
    main()
