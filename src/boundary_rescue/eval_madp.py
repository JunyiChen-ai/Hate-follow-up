#!/usr/bin/env python3
"""
Evaluate Multi-Aspect Definition Probe (MADP) outputs.

Each judge response has 4 fields:
  target_real_group: Yes/No
  hostile_framing: Yes/No
  protected_context: Yes/No
  answer: Yes/No

Universal Boolean rule (no tuning):
  hateful = (target_real_group == Yes) AND (hostile_framing == Yes)
            AND (protected_context == No)

We compare:
  R0: stage-1 only (no rescue)
  R1: raw answer (use the judge's answer field)
  R2: Boolean(target & hostile & !protected) — universal
  R3: answer AND Boolean (only flip if both agree)
  R4: answer OR Boolean (flip if either agrees)
"""
import json, re
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP3/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
V1 = {
    "MHClip_EN":   (126, 0.6958),
    "MHClip_ZH":   (123, 0.8023),
    "HateMM":      (182, 0.8362),
    "ImpliHateVid":(329, 0.8199),
}
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

# Field extractors
def get_field(text, name):
    if not text: return None
    m = re.search(rf"^\s*{name}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not m: return None
    v = m.group(1).strip().lower().rstrip(".").split()[0] if m.group(1).strip() else ""
    if v.startswith("yes"): return True
    if v.startswith("no"):  return False
    return None

def main():
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r["Video_ID"]: LABMAP[ds].get(r["Label"], -1) for r in ann}
        base = {r["video_id"]: r["pred_baseline"] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band = set(r["video_id"] for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl"))
        v1c, v1m = V1[ds]

        files = sorted((ROOT/ds).glob("offline_test_band_eaa_*.jsonl"))
        files = [f for f in files if "_v1" not in f.name and "_v2" not in f.name and "_v3" not in f.name and "_saved" not in f.name]
        if not files:
            print(f"\n=== {ds} === no MADP file yet"); continue
        valid = [v for v in base if lab.get(v,-1) >= 0]
        y  = [lab[v] for v in valid]
        y0 = [base[v] for v in valid]
        # Stage-1 baseline
        c0 = sum(1 for a,b in zip(y,y0) if a==b)
        print(f"\n=== {ds} (V1 {v1c}/{N[ds]}={v1c/N[ds]:.4f}/{v1m:.4f}, stage-1 {c0}/{N[ds]}={c0/N[ds]:.4f}) ===")

        for fp in files:
            jname = fp.name[len("offline_test_band_eaa_"):-len(".jsonl")]
            jrec = {r["video_id"]: r for r in ld(fp)}
            cov = sum(1 for v in band if v in jrec)
            if cov < 0.9 * len(band):
                print(f"  [{jname}] coverage {cov}/{len(band)} — skip")
                continue
            # Parse fields
            parsed = {}
            for v, r in jrec.items():
                raw = r.get('raw_response','') or ''
                parsed[v] = {
                    'target': get_field(raw, 'target_real_group'),
                    'hostile': get_field(raw, 'hostile_framing'),
                    'protected': get_field(raw, 'protected_context'),
                    'answer': get_field(raw, 'answer'),
                }

            from collections import Counter
            field_dist = {}
            for f in ['target','hostile','protected','answer']:
                field_dist[f] = Counter(parsed[v][f] for v in band if v in parsed)
            print(f"  --- {jname} (cov={cov}/{len(band)}) ---")
            for fname, d in field_dist.items():
                print(f"     {fname:12s}: {dict(d)}")

            def run(name, decide):
                yh = list(y0); flips=0
                for i, v in enumerate(valid):
                    if v in band and v in parsed:
                        p = decide(parsed[v])
                        if p in (0,1):
                            if p != yh[i]: flips += 1
                            yh[i] = p
                c = sum(1 for a,b in zip(y,yh) if a==b)
                m = f1m(y, yh)
                strict = (c > v1c) and (m > v1m + 1e-9)
                flag = '✓' if strict else '✗'
                print(f"     {flag} {name:30s} {c:>4d}/{N[ds]}={c/N[ds]:.4f} mF1={m:.4f} flips={flips}")
                return strict

            # R1: raw answer
            run("R1 raw_answer",
                lambda p: (1 if p['answer'] is True else (0 if p['answer'] is False else None)))
            # R2: Boolean of 3 sub-questions
            run("R2 Boolean(T&H&~P)",
                lambda p: (1 if (p['target'] is True and p['hostile'] is True and p['protected'] is False) else
                           (0 if (p['target'] is False or p['hostile'] is False or p['protected'] is True) else None)))
            # R3: AND
            run("R3 answer AND Boolean",
                lambda p: (1 if (p['answer'] is True and p['target'] is True and p['hostile'] is True and p['protected'] is False) else
                           (0 if (p['answer'] is False or p['protected'] is True or p['target'] is False) else None)))
            # R4: OR (flip to hateful if either says yes)
            run("R4 answer OR Boolean",
                lambda p: (1 if (p['answer'] is True or (p['target'] is True and p['hostile'] is True and p['protected'] is False)) else
                           (0 if (p['answer'] is False and p['target'] is False) else None)))

if __name__ == "__main__":
    main()
