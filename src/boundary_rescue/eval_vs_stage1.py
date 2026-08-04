#!/usr/bin/env python3
"""Compare each judge × rule against STAGE-1 baseline (no rescue), not V1."""
import json, re
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP3/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
N = {"MHClip_EN":161,"MHClip_ZH":149,"HateMM":215,"ImpliHateVid":401}
LABMAP = {"MHClip_EN":{"Hateful":1,"Offensive":1,"Normal":0},
          "MHClip_ZH":{"Hateful":1,"Offensive":1,"Normal":0},
          "HateMM":{"Hate":1,"Non Hate":0},
          "ImpliHateVid":{"Hateful":1,"Normal":0}}

SCG_PHRASES = [
    r"\bnot directed at\b", r"\bnot target(?:ed|ing)?\b",
    r"\bno (?:specific|clear|identifiable) (?:target|group|person)\b",
    r"\bdoes not (?:target|attack|mock)\b", r"\bdoesn't (?:target|attack|mock)\b",
    r"\bnot (?:present|shown) in the (?:video|content) (?:itself|directly)\b",
    r"\bnot in the video content\b", r"\bnot directly related\b",
    r"\bpresented in a (?:medical|educational|documentary|commentary|critical) context\b",
    r"\bparody\b", r"\bsatire\b",
    r"\bnews footage\b", r"\breports? on\b", r"\bcoverage of\b", r"\bpresents news\b",
    r"\b(?:mocks?|mocking|misrepresents?) (?:[\w\s]+? )?(?:brand|company|product|celebrity|fictional)\b",
]
SCG_RE = re.compile("|".join(SCG_PHRASES), re.IGNORECASE)

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
    rat_m = re.search(r"rationale\s*:\s*(.+?)(?:\n\s*verdict|$)", text, re.IGNORECASE | re.DOTALL)
    rat = rat_m.group(1).strip() if rat_m else ""
    return {"pred": pred, "rationale": rat}

JUDGES = [
    ("gemma-3-12b-it", "offline_test_gemma-3-12b-it.jsonl", False),
    ("gemma-3-27b-it", "offline_test_gemma-3-27b-it.jsonl", False),
    ("internvl35-8b",  "offline_test_internvl35-8b.jsonl", False),
    ("llava-OV-7b", "offline_test_llava-onevision-qwen2-7b-ov-hf.jsonl", False),
    ("minicpm-v-26",   "offline_test_minicpm-v-26.jsonl", False),
    ("qwen2.5-vl-32b-awq", "offline_test_qwen2.5-vl-32b-awq.jsonl", False),
    ("qwen2.5-vl-72b-awq", "offline_test_qwen2.5-vl-72b-awq.jsonl", False),
    ("qwen3-vl-8b-pA", "rescue_8b_bayes_band_rate_v1.jsonl", True),
    ("qwen3-vl-8b-pB", "offline_test_qwen3-vl-8b.jsonl", False),
]

def main():
    # Load stage-1 baseline metrics
    stage1 = {}
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
        base = {r['video_id']: r['pred_baseline'] for r in (json.loads(l) for l in open(ROOT/ds/"baseline_preds_v2.jsonl"))}
        valid = [v for v in base if lab.get(v,-1)>=0]
        y  = [lab[v] for v in valid]
        y0 = [base[v] for v in valid]
        c0 = sum(1 for a,b in zip(y,y0) if a==b)
        m0 = f1m(y, y0)
        stage1[ds] = (c0, m0)

    print("Stage-1 baseline:")
    for ds in DATASETS:
        c, m = stage1[ds]
        print(f"  {ds}: {c}/{N[ds]}={c/N[ds]:.4f} mF1={m:.4f}")

    # Pre-compute band posterior median per dataset
    band_post = {}
    for ds in DATASETS:
        rows = ld(ROOT/ds/"candidates_bayes_band_rate.jsonl")
        posts = sorted(r['posterior_hi'] for r in rows)
        n = len(posts)
        med = posts[n//2] if n % 2 else 0.5*(posts[n//2-1]+posts[n//2])
        band_post[ds] = ({r['video_id']: r['posterior_hi'] for r in rows}, med)

    print("\n" + "="*82)
    print(f"{'Judge':22s} {'Rule':18s} EN     ZH     HM     IH    strict-vs-stage1")
    print("="*82)

    for jname, fname, is_rescue in JUDGES:
        for rule_name in ["raw-flip", "raw+SCG", "raw+OUT", "raw+SCG+OUT"]:
            row = f"{jname:22s} {rule_name:18s}"
            count_strict = 0
            for ds in DATASETS:
                ann = json.load(open(DATA/ds/"annotation(new).json"))
                lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
                base = {r['video_id']: r['pred_baseline'] for r in (json.loads(l) for l in open(ROOT/ds/"baseline_preds_v2.jsonl"))}
                band = set(r['video_id'] for r in (json.loads(l) for l in open(ROOT/ds/"candidates_bayes_band_rate.jsonl")))
                raw_recs = ld(ROOT/ds/fname)
                if is_rescue:
                    jrec = {r['video_id']: parse_rescue(r) for r in raw_recs}
                else:
                    jrec = {r['video_id']: r for r in raw_recs}
                post_map, med = band_post[ds]
                valid = [v for v in base if lab.get(v,-1)>=0]
                y  = [lab[v] for v in valid]
                yh = [base[v] for v in valid]
                for i, v in enumerate(valid):
                    if v in band and v in jrec:
                        r = jrec[v]; pr = r.get('pred')
                        if pr not in (0,1): continue
                        rat = r.get('rationale','') or ''
                        # SCG check (asymmetric, only blocks flip-to-hateful)
                        if "SCG" in rule_name and pr == 1 and pr != yh[i] and SCG_RE.search(rat):
                            continue
                        # OUTER band check (only flip if posterior_hi < median = less ambiguous)
                        if "OUT" in rule_name:
                            p = post_map.get(v, 0)
                            if p >= med:
                                continue
                        yh[i] = pr
                c = sum(1 for a,b in zip(y,yh) if a==b)
                m = f1m(y, yh)
                c0, m0 = stage1[ds]
                strict = (c > c0) and (m > m0 + 1e-9)
                tag = '✓' if strict else ('=' if c==c0 and abs(m-m0)<1e-6 else '✗')
                row += f"{c:>3d}{tag} "
                if strict: count_strict += 1
            row += f"   {count_strict}/4"
            if count_strict == 4: row += " 🎯"
            print(row)

if __name__ == "__main__":
    main()
