#!/usr/bin/env python3
"""
Self-Contradiction Gate (SCG): universal, judge-agnostic flip gate.

Observation: when a judge flips a sample to "hateful" but then hedges with
phrases like "not directed at", "not specifically targeting", "not in the
video itself", "presented in a medical context", the flip is unreliable.
We treat any such self-negation as an abstention → keep stage-1.

Rule is parameter-free (fixed English phrase list), judge-agnostic.
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

# Universal self-negation phrases. If judge says "hateful" but rationale
# contains any of these, the claim undercuts itself → keep stage-1.
SELF_NEG = [
    # Target-denial (most reliable signal)
    r"\bnot directed at\b", r"\bnot target(?:ed|ing)?\b",
    r"\bno (?:specific|clear|identifiable) (?:target|group|person)\b",
    r"\bdoes not (?:target|attack|mock)\b", r"\bdoesn't (?:target|attack|mock)\b",
    # Content-not-in-video (judge relying on title/transcript only)
    r"\bnot (?:present|shown) in the (?:video|content) (?:itself|directly)\b",
    r"\bnot in the video content\b", r"\bnot directly related\b",
    # Context-based immunity (per MHClip / HateMM definition's "NOT hateful" list)
    r"\bpresented in a (?:medical|educational|documentary|commentary|critical) context\b",
    r"\bparody\b", r"\bsatire\b",
    r"\bnews footage\b", r"\breports? on\b", r"\bcoverage of\b", r"\bpresents news\b",
    # Non-protected target (brand/celebrity/fictional — not group identity)
    r"\b(?:mocks?|mocking|misrepresents) (?:[\w\s]+? )?(?:brand|company|product|celebrity|fictional)\b",
]
SELF_NEG_RE = re.compile("|".join(SELF_NEG), re.IGNORECASE)

def self_contradicts(rat):
    if not rat: return False
    return bool(SELF_NEG_RE.search(rat))

def acc(y,yh):
    return sum(1 for a,b in zip(y,yh) if a==b)/len(y) if y else 0.0
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

def main():
    JUDGES = [
        ("gemma-3-12b-it", "offline_test_gemma-3-12b-it.jsonl"),
        ("gemma-3-27b-it", "offline_test_gemma-3-27b-it.jsonl"),
        ("internvl35-8b",  "offline_test_internvl35-8b.jsonl"),
        ("llava-onevision-qwen2-7b-ov-hf", "offline_test_llava-onevision-qwen2-7b-ov-hf.jsonl"),
        ("minicpm-v-26",   "offline_test_minicpm-v-26.jsonl"),
        ("qwen2.5-vl-32b-awq", "offline_test_qwen2.5-vl-32b-awq.jsonl"),
        ("qwen2.5-vl-72b-awq", "offline_test_qwen2.5-vl-72b-awq.jsonl"),
        ("qwen3-vl-8b-promptA", "rescue_8b_bayes_band_rate_v1.jsonl"),
        ("qwen3-vl-8b-promptB", "offline_test_qwen3-vl-8b.jsonl"),
    ]

    results = {}
    for ds in DATASETS:
        bar_a, bar_m = V1[ds]
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
        base = {r["video_id"]: r for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band = set(r["video_id"] for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl"))
        valid = [v for v in base if lab.get(v,-1) >= 0]
        y = [lab[v] for v in valid]
        y0 = [base[v]["pred_baseline"] for v in valid]

        print(f"\n=== {ds}  (V1 {bar_a:.4f}/{bar_m:.4f}, stage-1 {acc(y,y0):.4f}/{f1m(y,y0):.4f}) ===")
        for jname, fname in JUDGES:
            jrec_raw = {r["video_id"]: r for r in ld(ROOT/ds/fname)}
            # Check band coverage (some files are band-only, others full test)
            band_cov = sum(1 for v in band if v in jrec_raw)
            if band_cov < 0.9 * len(band):
                continue
            # If this is the rescue file (promptA for qwen3-vl-8b), parse
            # 'verdict: hateful/normal' from rescue_response and convert.
            jrec = {}
            for v, r in jrec_raw.items():
                if 'rescue_response' in r:
                    text = r.get('rescue_response','') or ''
                    m = re.search(r"verdict\s*:\s*(\w+)", text, re.IGNORECASE)
                    pred = None
                    if m:
                        w = m.group(1).lower()
                        pred = 1 if w == "hateful" else (0 if w == "normal" else None)
                    rat_m = re.search(r"rationale\s*:\s*(.+?)(?:\n\s*verdict|$)", text, re.IGNORECASE | re.DOTALL)
                    rat = rat_m.group(1).strip() if rat_m else ""
                    jrec[v] = {"pred": pred, "rationale": rat}
                else:
                    jrec[v] = r

            for rule_name, apply_scg in [("raw-flip", False), ("raw+SCG", True)]:
                yh = list(y0); flipped = 0; blocked = 0
                for i, v in enumerate(valid):
                    if v in band and v in jrec:
                        r = jrec[v]; pr = r.get("pred")
                        if pr not in (0,1): continue
                        rat = r.get("rationale", "") or ""
                        # Asymmetric: SCG only blocks flip-to-1 (hateful)
                        # since self-negation is about undercutting hate claim.
                        if apply_scg and pr == 1 and pr != y0[i] and self_contradicts(rat):
                            blocked += 1
                            continue  # keep stage-1
                        if pr != y0[i]: flipped += 1
                        yh[i] = pr
                a = acc(y, yh); m = f1m(y, yh)
                # Only strict > V1
                strict = (a > bar_a + 1e-6) and (m > bar_m + 1e-6)
                flag = "*" if strict else " "
                print(f"  {flag} {jname:22s} {rule_name:10s} acc={a:.4f} mF1={m:.4f} flip={flipped} blocked={blocked}")
                results.setdefault(rule_name, {}).setdefault(jname, {})[ds] = (a, m, strict)

    print("\n=== Strict-beat count per (rule, judge) across 4 datasets ===")
    for rule in results:
        for j in results[rule]:
            beats = sum(1 for d in DATASETS if results[rule][j].get(d, (0,0,False))[2])
            tag = "🎯" if beats == 4 else ""
            print(f"  rule={rule:10s} judge={j:22s} strict_beats={beats}/4  {tag}")

if __name__ == "__main__":
    main()
