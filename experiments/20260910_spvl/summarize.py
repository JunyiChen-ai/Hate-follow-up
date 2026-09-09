#!/usr/bin/env python3
"""Ablation table from runs/20260910_spvl/*/metrics_*.json plus OMSL-v6 and comparators."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260910_spvl"
ROWS = [  # (label, metrics file)
    ("OMSL-v6 (current method)", ROOT / "runs/20260829_omsl_v6/v6_migrated_seed0_20260909/metrics.json"),
    ("T3AL rerun (label-free comparator, 611 videos)", ROOT / "runs/20260829_omsl_v6/t3al_anchor_s20250819_metrics.json"),
    ("MultiHateLoc rerun (weakly supervised, video labels)", ROOT / "runs/20260829_omsl_v6/multihateloc_frozen_current4fps_v1_metrics.json"),
    ("SPVL full (K=20, context, fixed 8 s, block mask)", R / "full/metrics_ispvl_rrank.json"),
    ("  − transcript context", R / "abl_noctx/metrics_ispvl_rrank.json"),
    ("  − frames", R / "abl_noframes/metrics_ispvl_rrank.json"),
    ("  − context − frames (windows only)", R / "abl_noctx_noframes/metrics_ispvl_rrank.json"),
    ("  causal mask (branches see each other)", R / "abl_causal/metrics_ispvl_rrank.json"),
    ("  ASR segments instead of fixed windows", R / "abl_asr_windows/metrics_ispvl_rrank.json"),
    ("  S = 4 s", R / "abl_s4/metrics_ispvl_rrank.json"),
    ("  S = 16 s", R / "abl_s16/metrics_ispvl_rrank.json"),
    ("  K = 8 frames", R / "abl_k8/metrics_ispvl_rrank.json"),
    ("SPVL + M3: intercept = z_video + mean window z", R / "full/metrics_izv_plus_mean_rrank.json"),
    ("**SPVL-r2**: dual visual/speech branches + evidence question + stance conditioning + M3", R / "full2_dual_evid_stance/metrics_izv_plus_mean_rrank.json"),
    ("  r2 − stance conditioning", R / "abl2_nostance/metrics_izv_plus_mean_rrank.json"),
    ("  r2 rules question instead of evidence question", R / "abl2_rulesq/metrics_izv_plus_mean_rrank.json"),
    ("  r2 joint branch instead of dual", R / "abl2_joint/metrics_izv_plus_mean_rrank.json"),
    ("  r2 triple (joint + visual + speech)", R / "abl2_triple/metrics_izv_plus_mean_rrank.json"),
    ("  r3: r2 with one frame per window (frames_w8)", R / "full3_dual_evid_stance_w8/metrics_izv_plus_mean_rrank.json"),
    ("  r3 joint + evidence + stance, frames_w8", R / "abl3_joint_evid_stance_w8/metrics_izv_plus_mean_rrank.json"),
    ("  r3 joint + rules, no stance, frames_w8", R / "abl3_joint_rules_none_w8/metrics_izv_plus_mean_rrank.json"),
    ("  M3 variant: z_video + median window z", R / "full/metrics_izv_plus_median_rrank.json"),
    ("  M3 variant: z_video + logit(mean window prob.)", R / "full/metrics_izv_plus_extent_rrank.json"),
    ("  M3 variant: mean window z only", R / "full/metrics_imean_win_rrank.json"),
    ("  intercept = 2026-08 z (old judge)", R / "full/metrics_ilegacy_rrank.json"),
    ("  intercept only (no residual)", R / "full/metrics_ispvl_rnone.json"),
    ("  residual only (no intercept)", R / "full/metrics_inone_rrank.json"),
    ("  + Vid-Group visual primary order", R / "full/metrics_ispvl_rrank_vis.json"),
    ("legacy chunk replica (2026-08 text scorer, old z)", R / "legacy_chunk_replica/metrics_ilegacy_rrank.json"),
]
DS = sys.argv[1:] or ["HateMM", "HateClipSeg"]
print("| variant | " + " | ".join(f"{d} ROC / PR / within (n)" for d in DS) + " |")
print("|---|" + "---|" * len(DS))
for label, f in ROWS:
    if not f.exists():
        print(f"| {label} | " + " | ".join("(missing)" for _ in DS) + " |"); continue
    per = {p["dataset"]: p for p in json.load(open(f))["per_dataset"]}
    cells = []
    for d in DS:
        p = per.get(d)
        cells.append("—" if p is None else f"{p['frame_ROC_AUC']:.4f} / {p['frame_PR_AUC']:.4f} / {p['within_video_macro_ROC_AUC']:.4f} ({p['n_videos_predicted']})")
    print(f"| {label} | " + " | ".join(cells) + " |")
