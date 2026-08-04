#!/usr/bin/env python3
"""Try more rule variants on existing EAA data to understand HM collapse."""
import json, re
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
    p=Path(p); return [json.loads(l) for l in open(p)] if p.exists() else []

EVID_RE = re.compile(r"evidence\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

def parse_evidence(raw):
    if not raw: return ""
    m = EVID_RE.search(raw)
    return (m.group(1).strip() if m else "").lower()

def is_insufficient(evid):
    return evid.startswith("insufficient") or evid == "" or "no clear" in evid

def main():
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
        base = {r["video_id"]: r for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band = {r["video_id"]: r for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl")}
        valid_ids = [v for v in base if lab.get(v, -1) >= 0]
        y  = [lab[v] for v in valid_ids]
        y0 = [base[v]["pred_baseline"] for v in valid_ids]

        fp = ROOT/ds/"offline_test_band_eaa_qwen3-vl-8b.jsonl"
        jrec = {r["video_id"]: r for r in ld(fp)}
        if not jrec: print(f"[{ds}] no EAA data"); continue
        bar_a, bar_m = V1[ds]
        print(f"\n=== {ds}  (V1 {bar_a:.4f}/{bar_m:.4f}, stage-1 "
              f"{acc(y,y0):.4f}/{f1m(y,y0):.4f}) ===")

        def run(name, decide):
            yh = list(y0); flipped=0; abst=0
            for i, v in enumerate(valid_ids):
                if v in band and v in jrec:
                    p = decide(jrec[v], y0[i])
                    if p == -99: abst += 1
                    elif p != y0[i]: flipped += 1; yh[i] = p
                    else: yh[i] = p
            a = acc(y, yh); m = f1m(y, yh)
            flag = "*" if (a > bar_a + 1e-9 and m > bar_m + 1e-9) else " "
            print(f"  {flag} {name:50s} acc={a:.4f} mF1={m:.4f}  abst={abst} flip={flipped}")

        # Rule A: raw-flip (ignore abstain, use answer when 0/1, else keep)
        def A(r, s1):
            p = r.get("pred")
            return p if p in (0,1) else s1
        run("A raw-flip (abstain=keep s1)", A)

        # Rule B: EAA core (pred=-2 → abstain=keep s1)
        def B(r, s1):
            p = r.get("pred")
            return p if p in (0,1) else s1
        run("B EAA core", B)  # same as A

        # Rule C: flip only if pred==1 AND specific evidence (not insufficient)
        def C(r, s1):
            p = r.get("pred")
            evid = parse_evidence(r.get("raw_response",""))
            if p == 1 and not is_insufficient(evid):
                return 1
            if p == 0:
                return 0
            return s1  # abstain or pred=1 without evidence
        run("C flip-to-1 needs evidence", C)

        # Rule D: flip ONLY if pred commits AND evidence backs it
        def D(r, s1):
            p = r.get("pred")
            evid = parse_evidence(r.get("raw_response",""))
            if p in (0,1) and not is_insufficient(evid):
                return p
            return s1
        run("D commit+evidence required", D)

        # Rule E: ASYMMETRIC hate-bias caution:
        # - flip to hateful (1) only if evidence is specific
        # - flip to normal (0) always
        def E(r, s1):
            p = r.get("pred")
            evid = parse_evidence(r.get("raw_response",""))
            if p == 0: return 0
            if p == 1 and not is_insufficient(evid): return 1
            return s1
        run("E asymmetric (hate needs evid)", E)

        # Rule F: ASYMMETRIC the OTHER way:
        # - flip to normal only if evidence is specific
        # - flip to hateful always
        def F(r, s1):
            p = r.get("pred")
            evid = parse_evidence(r.get("raw_response",""))
            if p == 1: return 1
            if p == 0 and not is_insufficient(evid): return 0
            return s1
        run("F asymmetric (normal needs evid)", F)

        # Rule G: stage-1 preserves when judge abstains OR when judge direction
        # matches stage-1 (no-op). Only act on disagreement with non-abstain.
        def G(r, s1):
            p = r.get("pred")
            if p in (0,1) and p != s1:  # disagreement — flip
                return p
            return s1
        run("G disagreement-only flip", G)

        # Rule H: within-band posterior gating — flip only if posterior_hi high
        # (keep threshold 0.3 fixed as universal Bayes-rate intuition)
        band_post = {v: band[v]["posterior_hi"] for v in band}
        def H(r, s1, v=None):
            p = r.get("pred")
            return p if p in (0,1) else s1
        # make H read posterior per-video:
        def H_fn(r, s1):
            vid = r.get("video_id")
            p = r.get("pred")
            post = band_post.get(vid, 0)
            if p in (0,1) and post >= 0.3:
                return p
            return s1
        run("H post>=0.3 gate + EAA", H_fn)

if __name__ == "__main__":
    main()
