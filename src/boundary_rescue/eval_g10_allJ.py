#!/usr/bin/env python3
"""Test G10 gate (hedge≤1 AND (c∈{1,3} OR (c=2 AND L≤90))) on ALL judges'
offline_test data. G10 is based on English-text features — potentially
judge-agnostic as long as rationales are in English.
"""
import json, re
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP2/datasets")
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

HEDGES = [
    "might","possibly","perhaps","unclear","ambiguous","could be","may be",
    "appears to","seems to","not entirely","borderline","hard to tell","unsure",
    "uncertain","arguably","potentially","to some extent",
]
CONCRETE = [
    "transcript","audio","image","frame","shows","depicts",
    "displays","uses","says","states","speaks","speaker",
    "title","caption","subtitle",
]

def hedge_count(rat):
    t=(rat or "").lower()
    return sum(1 for h in HEDGES if h in t)
def concrete_count(rat):
    t=(rat or "").lower()
    return sum(1 for c in CONCRETE if c in t)

def g10_pass(rat):
    h = hedge_count(rat); c = concrete_count(rat); L = len((rat or "").split())
    return h <= 1 and (c in (1,3) or (c == 2 and L <= 90))

def g9_pass(rat):
    h = hedge_count(rat); c = concrete_count(rat)
    return h <= 1 and c in (1, 3)

def g7_pass(rat):
    h = hedge_count(rat); c = concrete_count(rat)
    return h <= 1 and 1 <= c <= 3

def g1_pass(rat):
    return hedge_count(rat) == 0

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

def run_gate(ds, judge, rec_path, gate_fn, gate_name):
    ann = json.load(open(DATA/ds/"annotation(new).json"))
    lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
    base = {r["video_id"]: r for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
    band = set(r["video_id"] for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl"))
    jrec = {r["video_id"]: r for r in ld(rec_path)}
    valid = [v for v in base if lab.get(v,-1) >= 0]
    y = [lab[v] for v in valid]
    y0 = [base[v]["pred_baseline"] for v in valid]
    yh = list(y0); flipped = 0; passed_gate = 0; rejected = 0
    for i, v in enumerate(valid):
        if v in band and v in jrec:
            r = jrec[v]; pr = r.get('pred')
            if pr not in (0,1): continue  # abstain or missing
            if pr == y0[i]: continue
            rat = r.get("rationale", "") or ""
            if gate_fn(rat):
                passed_gate += 1
                if pr != y0[i]: flipped += 1; yh[i] = pr
            else:
                rejected += 1
    return acc(y,yh), f1m(y,yh), flipped, passed_gate, rejected

def main():
    JUDGES = [
        ("gemma-3-12b-it", "offline_test_gemma-3-12b-it.jsonl"),
        ("gemma-3-27b-it", "offline_test_gemma-3-27b-it.jsonl"),
        ("internvl35-8b",  "offline_test_internvl35-8b.jsonl"),
        ("minicpm-v-26",   "offline_test_minicpm-v-26.jsonl"),
        ("qwen2.5-vl-32b-awq", "offline_test_qwen2.5-vl-32b-awq.jsonl"),
        ("qwen2.5-vl-72b-awq", "offline_test_qwen2.5-vl-72b-awq.jsonl"),
        ("qwen3-vl-8b-promptB", "offline_test_qwen3-vl-8b.jsonl"),
        ("qwen3-vl-8b-promptA", None),  # use rescue file
    ]

    GATES = [("none", lambda r: True), ("G1", g1_pass), ("G7", g7_pass), ("G9", g9_pass), ("G10", g10_pass)]

    for ds in DATASETS:
        bar_a, bar_m = V1[ds]
        print(f"\n=== {ds}  (V1 {bar_a:.4f}/{bar_m:.4f}) ===")
        for jname, fname in JUDGES:
            if fname is None:
                # promptA = rescue file
                path = ROOT/ds/"rescue_8b_bayes_band_rate_v1.jsonl"
                # that file has different structure — need to convert
                # Skip for now; or parse verdict from rescue_response
                recs = []
                for r in ld(path):
                    text = r.get('rescue_response','') or ''
                    m = re.search(r"verdict\s*:\s*(\w+)", text, re.IGNORECASE)
                    if m:
                        w = m.group(1).lower()
                        pr = 1 if w == "hateful" else (0 if w == "normal" else -1)
                    else:
                        pr = -1
                    rat_m = re.search(r"rationale\s*:\s*(.+?)(?:\n\s*verdict|$)", text, re.IGNORECASE | re.DOTALL)
                    rat = rat_m.group(1).strip() if rat_m else ""
                    recs.append({"video_id": r["video_id"], "pred": pr, "rationale": rat})
                # Write temp in-memory: use run_gate with dict
                # Actually easiest: temp dict-based rec lookup
                jrec = {r["video_id"]: r for r in recs}
                ann = json.load(open(DATA/ds/"annotation(new).json"))
                lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
                base = {r["video_id"]: r for r in (json.loads(l) for l in open(ROOT/ds/"baseline_preds_v2.jsonl"))}
                band = set(r["video_id"] for r in (json.loads(l) for l in open(ROOT/ds/"candidates_bayes_band_rate.jsonl")))
                valid = [v for v in base if lab.get(v,-1) >= 0]
                y = [lab[v] for v in valid]
                y0 = [base[v]["pred_baseline"] for v in valid]
                for gname, gfn in GATES:
                    yh = list(y0); flipped = 0
                    for i, v in enumerate(valid):
                        if v in band and v in jrec:
                            r = jrec[v]; pr = r.get('pred')
                            if pr not in (0,1): continue
                            if pr == y0[i]: continue
                            rat = r.get("rationale","") or ""
                            if gfn(rat):
                                yh[i] = pr; flipped += 1
                    a = acc(y, yh); m = f1m(y, yh)
                    flag = "*" if (a > bar_a + 1e-9 and m > bar_m + 1e-9) else " "
                    if gname == "none":
                        print(f"  {flag} {jname:22s} gate={gname:5s} acc={a:.4f} mF1={m:.4f} flip={flipped}")
                    else:
                        print(f"  {flag} {jname:22s} gate={gname:5s} acc={a:.4f} mF1={m:.4f} flip={flipped}")
            else:
                path = ROOT/ds/fname
                if not path.exists():
                    continue
                for gname, gfn in GATES:
                    a, m, fl, _, _ = run_gate(ds, jname, path, gfn, gname)
                    flag = "*" if (a > bar_a + 1e-9 and m > bar_m + 1e-9) else " "
                    print(f"  {flag} {jname:22s} gate={gname:5s} acc={a:.4f} mF1={m:.4f} flip={fl}")

if __name__ == "__main__":
    main()
