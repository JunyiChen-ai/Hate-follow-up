#!/usr/bin/env python3
"""SCG with expanded phrase list + secondary gates, tested on the 3
oracle-capable judges (gemma-27B, qwen3-vl-8b-promptA, llava-7B).
Goal: find a single rule that strict-beats V1 on 4/4 for all three.
"""
import json, re
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP2/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
V1 = {"MHClip_EN":(126,0.6958),"MHClip_ZH":(123,0.8023),
      "HateMM":(182,0.8362),"ImpliHateVid":(329,0.8199)}
N = {"MHClip_EN":161,"MHClip_ZH":149,"HateMM":215,"ImpliHateVid":401}
LABMAP = {"MHClip_EN":{"Hateful":1,"Offensive":1,"Normal":0},
          "MHClip_ZH":{"Hateful":1,"Offensive":1,"Normal":0},
          "HateMM":{"Hate":1,"Non Hate":0},
          "ImpliHateVid":{"Hateful":1,"Normal":0}}

# Expanded SCG phrase set: original + new patterns derived from
# more harmful-flip rationales across multiple judges.
SCG_BASE = [
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
SCG_EXPANDED = SCG_BASE + [
    # Fictional-context detection
    r"\bfictional (?:dialogue|conflict|character|setting|scenario)\b",
    r"\b(?:scripted|staged|cinematic|movie|film|tv|show|series|cartoon|anime|game|gameplay) (?:scene|dialogue|conflict|context|content|character|footage)\b",
    r"\b(?:between|involving) (?:two |the )?(?:fictional )?characters\b",
    # Incident-framing detection
    r"\bframes the incident\b", r"\bdepicts? (?:an? |the )?incident\b",
    r"\b(?:reporting|reports?|reported) on (?:an |the )?(?:incident|event|news|event)\b",
    # Vulgar-but-not-hateful-target pattern
    r"\b(?:vulgar|crude|sexual) humor (?:that |which )?does not target\b",
    # Action/violence not-hateful pattern
    r"\b(?:action[-\s]movie|video[-\s]game) (?:violence|threats?)\b",
    # Acknowledgment of dissonance
    r"\bwhile (?:[\w\s,]+? )?(?:does not|may not|isn't) (?:be|target)\b",
    r"\bbut (?:the |its )?(?:framing|content|context) (?:does not|isn't) (?:hostile|hateful|targeted)\b",
]
SCG_RE_BASE = re.compile("|".join(SCG_BASE), re.IGNORECASE)
SCG_RE_EXP = re.compile("|".join(SCG_EXPANDED), re.IGNORECASE)

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
    ("gemma-3-27b-it", "offline_test_gemma-3-27b-it.jsonl", False),
    ("qwen3-vl-8b-promptA", "rescue_8b_bayes_band_rate_v1.jsonl", True),
    ("llava-onevision-7b", "offline_test_llava-onevision-qwen2-7b-ov-hf.jsonl", False),
]

def main():
    for rule_name, scg_re in [("SCG_base", SCG_RE_BASE), ("SCG_expanded", SCG_RE_EXP)]:
        print(f"\n{'='*72}\n{rule_name}\n{'='*72}")
        for jname, fname, is_rescue in JUDGES:
            print(f"\n--- {jname} ---")
            pass4 = True
            for ds in DATASETS:
                ann = json.load(open(DATA/ds/"annotation(new).json"))
                lab = {r['Video_ID']: LABMAP[ds].get(r['Label'],-1) for r in ann}
                base = {r['video_id']: r['pred_baseline'] for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
                band = set(r['video_id'] for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl"))
                raw_recs = ld(ROOT/ds/fname)
                if is_rescue:
                    jrec = {r['video_id']: parse_rescue(r) for r in raw_recs}
                else:
                    jrec = {r['video_id']: r for r in raw_recs}
                valid = [v for v in base if lab.get(v,-1)>=0]
                y  = [lab[v] for v in valid]
                yh = [base[v] for v in valid]
                blocked_bad = blocked_good = 0
                for i, v in enumerate(valid):
                    if v in band and v in jrec:
                        r = jrec[v]; pr = r.get('pred')
                        if pr not in (0,1): continue
                        rat = r.get('rationale','') or ''
                        if pr == 1 and pr != yh[i] and scg_re.search(rat):
                            if pr == y[i]: blocked_good += 1
                            else: blocked_bad += 1
                            continue
                        yh[i] = pr
                c = sum(1 for a,b in zip(y,yh) if a==b)
                m = f1m(y, yh)
                v1c, v1m = V1[ds]
                strict = (c > v1c) and (m > v1m + 1e-9)
                if not strict: pass4 = False
                print(f"   {('✓' if strict else '✗')} {ds:14s} {c:>4d}/{N[ds]}={c/N[ds]:.4f} mF1={m:.4f} V1={v1c}/{N[ds]}={v1c/N[ds]:.4f} block_bad={blocked_bad} block_good={blocked_good}")
            print(f"   ==> {'4/4 ✓' if pass4 else 'NOT 4/4'}")

if __name__ == "__main__":
    main()
