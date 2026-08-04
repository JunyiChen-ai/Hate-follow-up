#!/usr/bin/env python3
"""
Bayesian fusion of stage-1 posterior and judge binary verdict.

Rule (parameter-free, judge-agnostic):
    Given band sample with stage-1 posterior q = P(hateful | score) from the
    GMM fit, and judge verdict v ∈ {0, 1}:
        p = (q * LR_pos) / (q * LR_pos + (1 - q) * LR_neg)      if v == 1
        p = (q * LR_neg) / (q * LR_neg + (1 - q) * LR_pos)      if v == 0
    where LR_pos = sens / (1 - spec), LR_neg = spec / (1 - sens).
    Assume sens = spec = S for any strong MLLM. Final: 1 if p > 0.5 else 0.

Equivalent flip threshold (for v = 1): q > (1 - S) / (2S - 1)  [with S=0.8 → q > 0.333].

We sweep a SMALL fixed set of S ∈ {0.7, 0.75, 0.8, 0.85, 0.9} to show the rule
is not S-sensitive, not as tuning. (S is chosen a priori from literature on
strong-VLM moderation error rates: typical accuracy 70-90%.)

Run against every (judge, dataset) pair and report strict-beat counts.
"""
import json
from pathlib import Path
from collections import defaultdict

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

S_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90]

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
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []

def bayes_fuse(q, v, S):
    """Fuse stage-1 posterior q with binary verdict v under sens=spec=S."""
    if S <= 0.5 or S >= 1.0:
        return v  # degenerate
    lr_pos = S / (1 - S)    # LR if judge says positive
    lr_neg = (1 - S) / S    # LR if judge says positive but actually negative? No.
    # Actually under binary Bayes:
    # p(H|v=1) = q*P(v=1|H) / (q*P(v=1|H) + (1-q)*P(v=1|¬H))
    #         = q*S / (q*S + (1-q)*(1-S))
    # p(H|v=0) = q*(1-S) / (q*(1-S) + (1-q)*S)
    if v == 1:
        num = q * S
        den = num + (1 - q) * (1 - S)
    else:
        num = q * (1 - S)
        den = num + (1 - q) * S
    return num / den if den > 0 else 0.5

def main():
    by_rule = {}
    for S in S_VALUES:
        rule_name = f"bayes_fuse_S{S:.2f}"
        by_rule[rule_name] = defaultdict(dict)

    # Also: include baseline raw_flip for reference
    by_rule["_raw_flip"] = defaultdict(dict)
    by_rule["_stage1"] = defaultdict(dict)

    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        lab = {r["Video_ID"]: LABELMAP[ds].get(r["Label"], -1) for r in ann}
        base = {r["video_id"]: r for r in ld(ROOT/ds/"baseline_preds_v2.jsonl")}
        band = {r["video_id"]: r for r in ld(ROOT/ds/"candidates_bayes_band_rate.jsonl")}

        # Posterior q comes from candidates file: `posterior_hi = err_contribution = min(q, 1-q)`
        # But we need signed q (which side of threshold). We have `side` field.
        def get_q(vid):
            r = band[vid]
            # If side == 'below' then score<threshold, q (hateful prob) ~ posterior_hi? NO — posterior_hi IS min(q, 1-q).
            # We need q = P(hateful) itself. Reconstruct from side:
            #   if side == 'below' (predicted normal), then P(hateful) = posterior_hi
            #   if side == 'above' (predicted hateful), then P(hateful) = 1 - posterior_hi
            post_hi = r["posterior_hi"]
            if r.get("side") == "below":
                return post_hi       # small q (close to err rate) since stage-1 says normal
            else:
                return 1 - post_hi   # q > 0.5 since stage-1 says hateful

        # Iterate judges
        valid_ids = [v for v in base if lab.get(v, -1) >= 0]
        y  = [lab[v] for v in valid_ids]
        y0 = [base[v]["pred_baseline"] for v in valid_ids]

        # Baseline: stage-1 alone
        by_rule["_stage1"][ds]["(stage-1 only)"] = (acc(y, y0), f1m(y, y0))

        for jpath in sorted((ROOT/ds).glob("offline_test_*.jsonl")):
            jname = jpath.name[len("offline_test_"):-len(".jsonl")]
            if jname.startswith("band_lp"): continue
            jrec = {r["video_id"]: int(r["pred"]) for r in ld(jpath) if "pred" in r}
            band_cov = [v for v in band if v in jrec]
            if len(band_cov) < 0.9 * len(band):
                continue

            # raw_flip
            yh = list(y0)
            for i, v in enumerate(valid_ids):
                if v in band and v in jrec:
                    yh[i] = jrec[v]
            by_rule["_raw_flip"][ds][jname] = (acc(y, yh), f1m(y, yh))

            # Bayesian fuse, per S
            for S in S_VALUES:
                rule_name = f"bayes_fuse_S{S:.2f}"
                yh = list(y0)
                for i, v in enumerate(valid_ids):
                    if v in band and v in jrec:
                        q = get_q(v)
                        p = bayes_fuse(q, jrec[v], S)
                        yh[i] = 1 if p > 0.5 else 0
                by_rule[rule_name][ds][jname] = (acc(y, yh), f1m(y, yh))

    # Report
    for rule_name in ["_stage1", "_raw_flip"] + [f"bayes_fuse_S{S:.2f}" for S in S_VALUES]:
        print(f"\n=== {rule_name} ===")
        print(f"  V1 bars: EN 0.7826/0.6958  ZH 0.8255/0.8023  HM 0.8465/0.8362  ImplHV 0.8204/0.8199")
        for ds in DATASETS:
            bar_a, bar_m = V1[ds]
            print(f"  --- {ds} (bar {bar_a:.4f}/{bar_m:.4f}) ---")
            for j, (a, m) in sorted(by_rule[rule_name][ds].items()):
                flag = "*" if (a > bar_a and m > bar_m) else " "
                print(f"   {flag} {j:35s} acc={a:.4f} mF1={m:.4f}")
        # per-rule summary
        judges = set()
        for ds in DATASETS:
            judges.update(by_rule[rule_name][ds].keys())
        hit = defaultdict(int)
        for j in judges:
            for ds in DATASETS:
                ab = by_rule[rule_name][ds].get(j)
                if ab is None: continue
                a, m = ab
                bar_a, bar_m = V1[ds]
                if a > bar_a and m > bar_m:
                    hit[j] += 1
        print(f"  [judges with strict-beat 4/4: {sorted([j for j,c in hit.items() if c==4]) or '—'}]")
        print(f"  [judges with 3/4: {sorted([j for j,c in hit.items() if c==3])}]")

if __name__ == "__main__":
    main()
