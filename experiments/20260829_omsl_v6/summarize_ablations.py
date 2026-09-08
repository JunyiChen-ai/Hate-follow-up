#!/usr/bin/env python3
"""Print the ablation table from runs/20260829_omsl_v6/ablation_*/metrics.json."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["full", "drop_language", "drop_audio", "visual_only", "mains_only", "mobius_raw",
         "perm_99", "perm_255", "order_sum", "intercept_mllm", "intercept_occ", "intercept_none"]
DATASETS = sys.argv[1:] or ["HateMM", "HateClipSeg"]

def cell(r):
    w = r.get("within_video_macro_ROC_AUC")
    return "%.4f / %.4f / %s" % (r["frame_ROC_AUC"], r["frame_PR_AUC"], "%.4f" % w if w is not None else "-")

rows = []
for tag in ORDER:
    f = ROOT / f"runs/20260829_omsl_v6/ablation_{tag}/metrics.json"
    if not f.exists():
        continue
    d = json.load(open(f)); per = {r["dataset"]: r for r in d["per_dataset"]}
    rows.append((tag, [cell(per[ds]) for ds in DATASETS]))
print("| ablation | " + " | ".join(f"{ds} ROC / PR / within" for ds in DATASETS) + " |")
print("|---|" + "---|" * len(DATASETS))
for tag, cells in rows:
    print(f"| {tag} | " + " | ".join(cells) + " |")
