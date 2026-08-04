#!/usr/bin/env python3
"""
Diagnostic: for each (judge × ZH band sample), which prediction is correct?
What is the per-sample agreement pattern? What is the achievable ceiling if
we had an oracle for the band flip decision?
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/data/jehc223/EMNLP3/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP3/datasets")
DS = "MHClip_ZH"

ann = json.load(open(DATA/DS/"annotation(new).json"))
m = {"Hateful":1,"Offensive":1,"Normal":0}
lab = {r["Video_ID"]: m[r["Label"]] for r in ann if r["Label"] in m}

def ld(p):
    return [json.loads(l) for l in open(p)]

base = {r["video_id"]: r for r in ld(ROOT/DS/"baseline_preds_v2.jsonl")}
band = {r["video_id"]: r for r in ld(ROOT/DS/"candidates_bayes_band_rate.jsonl")}

judges = ["gemma-3-12b-it","gemma-3-27b-it","internvl35-8b",
          "llava-onevision-qwen2-7b-ov-hf","minicpm-v-26",
          "qwen2.5-vl-32b-awq","qwen2.5-vl-72b-awq","qwen3-vl-8b"]
jdata = {}
for j in judges:
    p = ROOT/DS/f"offline_test_{j}.jsonl"
    if p.exists():
        jdata[j] = {r["video_id"]: r for r in ld(p)}

# For every band sample, compute: (stage1 pred, true label, per-judge pred,
# whether stage1 is right, whether majority of judges agree with label).
band_ids = sorted(band.keys())
print(f"ZH band size = {len(band_ids)}; test size = {len(base)}")
print(f"V1 bar: ACC > 0.8255 (>= 124 correct)  mF1 > 0.8023")
print(f"Stage-1 on band (how many base predictions are correct on band alone?):")

s1_correct_band = sum(1 for v in band_ids if base[v]["pred_baseline"] == lab[v])
print(f"  stage-1 correct on band: {s1_correct_band}/{len(band_ids)} = {s1_correct_band/len(band_ids):.3f}")

# Full test correct counts
all_ids = list(base.keys())
s1_correct_full = sum(1 for v in all_ids if base[v]["pred_baseline"] == lab[v])
print(f"  stage-1 correct on full test: {s1_correct_full}/{len(all_ids)} (= ACC {s1_correct_full/len(all_ids):.4f})")

# Per-judge: what would raw-flip do on band? How many correct?
print("\nPer-judge raw-flip on band:")
for j, d in jdata.items():
    cov = [v for v in band_ids if v in d and "pred" in d[v]]
    # raw-flip: replace stage1 with judge on band
    judge_correct = sum(1 for v in cov if int(d[v]["pred"]) == lab[v])
    # final ACC: stage-1 outside band + judge on band
    non_band = [v for v in all_ids if v not in band]
    non_band_correct = sum(1 for v in non_band if base[v]["pred_baseline"] == lab[v])
    # for band members not covered, fall back to stage-1
    uncov = [v for v in band_ids if v not in cov]
    uncov_correct = sum(1 for v in uncov if base[v]["pred_baseline"] == lab[v])
    tot_correct = non_band_correct + judge_correct + uncov_correct
    print(f"  {j:30s} band_cov={len(cov)}/{len(band_ids)} judge_band_correct={judge_correct}/{len(cov)} "
          f"final_ACC={tot_correct/len(all_ids):.4f} ({tot_correct}/{len(all_ids)})")

# Oracle band-ceiling per judge (rule = "flip only if judge would be correct"):
print("\nOracle-selective flip per judge (flip iff judge==label, else keep stage-1):")
for j, d in jdata.items():
    cov = [v for v in band_ids if v in d and "pred" in d[v]]
    # oracle keeps stage-1 whenever judge is wrong
    band_correct = 0
    for v in band_ids:
        if v in d and "pred" in d[v]:
            if int(d[v]["pred"]) == lab[v]:
                band_correct += 1  # oracle flips, correct
            else:
                if base[v]["pred_baseline"] == lab[v]:
                    band_correct += 1  # stage-1 was correct, keep
        else:
            if base[v]["pred_baseline"] == lab[v]:
                band_correct += 1
    non_band_correct = sum(1 for v in all_ids if v not in band and base[v]["pred_baseline"] == lab[v])
    tot = band_correct + non_band_correct
    print(f"  {j:30s} oracle_ACC = {tot/len(all_ids):.4f}  (band_correct {band_correct}/{len(band_ids)})")

# Disagreement-only view: on band samples where stage-1 is WRONG, which
# judge correctly provides the fix?
wrong_s1 = [v for v in band_ids if base[v]["pred_baseline"] != lab[v]]
print(f"\nBand samples where stage-1 is WRONG: {len(wrong_s1)}")
for j, d in jdata.items():
    n_covered = sum(1 for v in wrong_s1 if v in d and "pred" in d[v])
    n_correctly_flipped = sum(1 for v in wrong_s1 if v in d and "pred" in d[v] and int(d[v]["pred"]) == lab[v])
    # also: on band samples where stage-1 is CORRECT, how often does judge wrongly flip?
    right_s1 = [v for v in band_ids if base[v]["pred_baseline"] == lab[v]]
    n_wrong_flip = sum(1 for v in right_s1 if v in d and "pred" in d[v] and int(d[v]["pred"]) != lab[v])
    print(f"  {j:30s} recoverable={n_correctly_flipped}/{n_covered}  harmful_flip={n_wrong_flip}/{len(right_s1)}")

# What's the intersection of recoverable samples across judges?
rec_sets = {}
for j, d in jdata.items():
    rec_sets[j] = set(v for v in wrong_s1 if v in d and "pred" in d[v] and int(d[v]["pred"]) == lab[v])

all_rec_union = set().union(*rec_sets.values())
print(f"\nUnion of recoverable samples across all judges: {len(all_rec_union)}")
for j,s in rec_sets.items():
    print(f"  {j:30s} recovered_uniquely = {len(s - set().union(*(rec_sets[k] for k in rec_sets if k!=j)))}  recovered = {len(s)}")

# Count how many judges recover each sample
recovery_count = Counter()
for v in all_rec_union:
    for j,s in rec_sets.items():
        if v in s:
            recovery_count[v] += 1
dist = Counter(recovery_count.values())
print(f"\nRecovery count distribution (how many judges get each recoverable sample right):")
for k in sorted(dist):
    print(f"  {k} judges → {dist[k]} samples")

# Majority-rule simulation (for reference only; actual method must be single-judge)
print("\n[reference only] Majority-of-all-judges on band:")
vids_in_all = [v for v in band_ids if all(v in d and "pred" in d[v] for j,d in jdata.items())]
maj_correct = 0
for v in vids_in_all:
    votes = [int(d[v]["pred"]) for j,d in jdata.items()]
    maj = 1 if sum(votes) > len(votes)/2 else 0
    if maj == lab[v]:
        maj_correct += 1
non_band_correct = sum(1 for v in all_ids if v not in band and base[v]["pred_baseline"] == lab[v])
# fill in non-covered band with stage-1
rest = [v for v in band_ids if v not in vids_in_all]
rest_correct = sum(1 for v in rest if base[v]["pred_baseline"] == lab[v])
tot = maj_correct + non_band_correct + rest_correct
print(f"  majority final ACC = {tot/len(all_ids):.4f}  ({tot}/{len(all_ids)})")
