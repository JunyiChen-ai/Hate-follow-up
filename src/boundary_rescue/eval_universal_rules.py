#!/usr/bin/env python3
"""
Judge-agnostic rule audit (CPU-only).

For every (judge, dataset) pair with offline_test_<judge>.jsonl present,
compute ACC / mF1 of multiple band-only decision rules. All rules are:
- band-only (judge only consulted on Bayes-rate band samples)
- tuning-free (no train-set / full-test / judge-specific knobs)
- judge-agnostic (one rule applied to every judge)

V1 baseline bars (strict-beat target):
  EN 0.7826/0.6958, ZH 0.8255/0.8023, HM 0.8465/0.8362, ImplHV 0.8204/0.8199.
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path("/data/jehc223/EMNLP2/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP2/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

def load_annot_labels():
    maps = {
        "MHClip_EN":   {"Hateful":1,"Offensive":1,"Normal":0},
        "MHClip_ZH":   {"Hateful":1,"Offensive":1,"Normal":0},
        "HateMM":      {"Hate":1,"Non Hate":0},
        "ImpliHateVid":{"Hateful":1,"Normal":0,"Non-hateful":0,"Non_hateful":0},
    }
    out = {}
    for ds in DATASETS:
        ann = json.load(open(DATA/ds/"annotation(new).json"))
        m = maps[ds]
        out[ds] = {r["Video_ID"]: m.get(r["Label"], -1) for r in ann}
    return out

LABELS = load_annot_labels()
V1 = {
    "MHClip_EN":   (0.7826, 0.6958),
    "MHClip_ZH":   (0.8255, 0.8023),
    "HateMM":      (0.8465, 0.8362),
    "ImpliHateVid":(0.8204, 0.8199),
}

# Fixed universal hedge vocabulary. Picked once and frozen; not tuned per judge.
HEDGES = (
    r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b",
    r"\bseems?\b", r"\bappears?\b", r"\bunclear\b", r"\bambiguous\b",
    r"\bsuggests?\b", r"\buncertain\b", r"\blikely\b", r"\bprobably\b",
    r"\bpotentially\b", r"\bsomewhat\b",
)
HEDGE_RE = re.compile("|".join(HEDGES), re.IGNORECASE)

def load_jsonl(p):
    out = []
    if not Path(p).exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def f1_macro(y, yh):
    cls = sorted(set(y))
    s = 0.0
    for c in cls:
        tp = sum(1 for a,b in zip(y,yh) if a==c and b==c)
        fp = sum(1 for a,b in zip(y,yh) if a!=c and b==c)
        fn = sum(1 for a,b in zip(y,yh) if a==c and b!=c)
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        s += 2*p*r/(p+r) if p+r else 0.0
    return s/len(cls) if cls else 0.0

def acc(y, yh):
    return sum(1 for a,b in zip(y,yh) if a==b)/len(y) if y else 0.0

def hedge_count(text:str) -> int:
    if not text: return 0
    return len(HEDGE_RE.findall(text))

def rule_raw_flip(ctx) -> int:
    """Always trust the judge on band."""
    return ctx["judge_pred"]

def rule_no_hedge(ctx) -> int:
    """Flip only if judge rationale contains 0 universal hedge words; else keep stage-1."""
    return ctx["judge_pred"] if hedge_count(ctx["rationale"]) == 0 else ctx["base_pred"]

def rule_low_hedge(ctx) -> int:
    """Flip only if hedge_count <= 1."""
    return ctx["judge_pred"] if hedge_count(ctx["rationale"]) <= 1 else ctx["base_pred"]

def rule_long_rationale(ctx) -> int:
    """Flip only if rationale length >= within-band median (geometric, within-band split)."""
    med = ctx["band_len_median"]
    return ctx["judge_pred"] if len(ctx["rationale"] or "") >= med else ctx["base_pred"]

def rule_short_rationale(ctx) -> int:
    """Symmetric test: flip only if rationale is shorter than median."""
    med = ctx["band_len_median"]
    return ctx["judge_pred"] if len(ctx["rationale"] or "") < med else ctx["base_pred"]

def rule_inner_band(ctx) -> int:
    """Flip only if the sample lies in the inner half of the band
    (posterior_hi >= within-band median) — geometric within-band split."""
    med = ctx["band_post_median"]
    if ctx["post"] is None:
        return ctx["base_pred"]
    return ctx["judge_pred"] if ctx["post"] >= med else ctx["base_pred"]

def rule_outer_band(ctx) -> int:
    """Symmetric: flip only if in outer half (less ambiguous in stage-1)."""
    med = ctx["band_post_median"]
    if ctx["post"] is None:
        return ctx["base_pred"]
    return ctx["judge_pred"] if ctx["post"] < med else ctx["base_pred"]

def rule_agree_confident_keep(ctx) -> int:
    """Only apply the judge if its verdict *disagrees* with stage-1 AND the rationale
    carries no hedge words. Equivalent to flipping only on confident disagreement."""
    if ctx["judge_pred"] == ctx["base_pred"]:
        return ctx["base_pred"]
    return ctx["judge_pred"] if hedge_count(ctx["rationale"]) == 0 else ctx["base_pred"]

RULES = {
    "raw_flip":              rule_raw_flip,
    "no_hedge_flip":         rule_no_hedge,
    "low_hedge_flip":        rule_low_hedge,
    "long_rationale_flip":   rule_long_rationale,
    "short_rationale_flip":  rule_short_rationale,
    "inner_band_flip":       rule_inner_band,
    "outer_band_flip":       rule_outer_band,
    "confident_disagree_flip": rule_agree_confident_keep,
}

def collect_judges(ds_dir: Path):
    js = []
    for p in sorted(ds_dir.glob("offline_test_*.jsonl")):
        name = p.name[len("offline_test_"):-len(".jsonl")]
        if name.startswith("band_lp"):  # skip band-lp probing artefacts
            continue
        js.append((name, p))
    return js

def evaluate_dataset(ds: str):
    ds_dir = ROOT / ds
    base   = {r["video_id"]: r for r in load_jsonl(ds_dir/"baseline_preds_v2.jsonl")}
    band   = {r["video_id"]: r for r in load_jsonl(ds_dir/"candidates_bayes_band_rate.jsonl")}

    rows = []
    judges = collect_judges(ds_dir)
    for jname, jpath in judges:
        jrec = {r["video_id"]: r for r in load_jsonl(jpath)}
        if not jrec:
            continue
        # Only keep judges that actually cover the band (otherwise unfair).
        band_ids = set(band.keys())
        covered = band_ids & set(jrec.keys())
        if len(covered) < 0.9 * len(band_ids):
            rows.append({"judge": jname, "note": f"band coverage {len(covered)}/{len(band_ids)} — skipped"})
            continue

        # Precompute within-band medians used by some rules.
        band_lens  = [len(jrec[vid].get("rationale") or "") for vid in covered]
        band_posts = [band[vid]["posterior_hi"] for vid in covered]
        band_lens.sort(); band_posts.sort()
        def med(xs):
            n=len(xs); return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])
        band_len_med  = med(band_lens)  if band_lens  else 0
        band_post_med = med(band_posts) if band_posts else 0.5

        # Labels always come from annotation(new).json (offline_test files
        # have wrong labels on HateMM).
        all_ids = [vid for vid in base.keys() if LABELS[ds].get(vid, -1) >= 0]
        y  = [LABELS[ds][vid]                for vid in all_ids]
        y0 = [base[vid]["pred_baseline"]    for vid in all_ids]

        per_rule = {}
        for rname, rfn in RULES.items():
            yh = list(y0)
            for i, vid in enumerate(all_ids):
                if vid in band and vid in jrec:
                    jr = jrec[vid]
                    if "pred" not in jr:
                        continue
                    ctx = {
                        "judge_pred": int(jr["pred"]),
                        "base_pred":  int(base[vid]["pred_baseline"]),
                        "rationale":  jr.get("rationale") or "",
                        "post":       band[vid]["posterior_hi"],
                        "band_len_median":  band_len_med,
                        "band_post_median": band_post_med,
                    }
                    yh[i] = int(rfn(ctx))
            per_rule[rname] = (acc(y, yh), f1_macro(y, yh))

        rows.append({"judge": jname, "rules": per_rule, "n": len(all_ids)})
    return rows

def strict_beat(a_mf1, b_mf1, a_acc, b_acc, bar_acc, bar_mf1):
    return a_acc > bar_acc and a_mf1 > bar_mf1

def main():
    agg = {r: {ds: {} for ds in DATASETS} for r in RULES}
    per_ds_judge_scores = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        print(f"\n=== {ds} (V1 bar: acc>{V1[ds][0]:.4f}  mF1>{V1[ds][1]:.4f}) ===")
        rows = evaluate_dataset(ds)
        for row in rows:
            if "rules" not in row:
                print(f"  [skip] {row['judge']}: {row['note']}")
                continue
            judge = row["judge"]
            print(f"  --- {judge} (n={row['n']}) ---")
            for rn, (a, m) in row["rules"].items():
                flag = "*" if strict_beat(m, 0, a, 0, *V1[ds]) else " "
                print(f"   {flag} {rn:24s} acc={a:.4f} mF1={m:.4f}")
                agg[rn][ds][judge] = (a, m)

    # Summarize: for each rule, count judges × datasets that strict-beat, and
    # report judges that strict-beat on ALL 4 datasets.
    print("\n=========== SUMMARY ===========")
    for rn in RULES:
        hit_any = 0
        hits_4 = []
        judge_hits = defaultdict(int)
        judges_seen = set()
        for ds in DATASETS:
            for j, (a, m) in agg[rn][ds].items():
                judges_seen.add(j)
                if strict_beat(m, 0, a, 0, *V1[ds]):
                    hit_any += 1
                    judge_hits[j] += 1
        for j in judges_seen:
            if judge_hits[j] == 4:
                hits_4.append(j)
        print(f"  rule {rn:24s} strict-beats/(judge,ds) = {hit_any}, judges_4/4 = {hits_4 or '—'}")

    print("\nPer-judge strict-beat count across rules (max across rules):")
    best = defaultdict(lambda: defaultdict(int))
    for rn in RULES:
        for ds in DATASETS:
            for j, (a,m) in agg[rn][ds].items():
                if strict_beat(m,0,a,0,*V1[ds]):
                    best[j][ds] = max(best[j][ds], 1)
    for j in sorted(best):
        n = sum(best[j].values())
        dss = [ds for ds in DATASETS if best[j].get(ds)]
        print(f"  {j:40s} {n}/4  ({','.join(dss)})")

if __name__ == "__main__":
    main()
