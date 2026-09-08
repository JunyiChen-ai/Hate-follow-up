# OMSL-v6 — Orthogonal Möbius Semantic Localization (label-free frame scoring)

Status: current method (as of 2026-09-09). Numbers below are development-selected
on this cohort (exploratory, not confirmatory); see `research-wiki/STATUS.md`.

## Mechanism (three modules)
1. **Frozen whole-video MLLM logit as intercept.** The Qwen3-VL-8B whole-video
   Yes/No logit `z` is pooled (equal-mass logmeanexp) with a Jeffreys-smoothed
   visual-occupancy logit to form a per-video semantic intercept.
2. **Coalition Möbius decomposition of three local streams** (visual proposal
   curve, timestamped-transcript log-odds curve, ImageBind audio-vs-text margin),
   each centred-ranked on the 4 fps grid. Interaction terms (VL, VA, LA, VLA) are
   calibrated against a joint-max null from 31 non-wrapping block permutations of
   the language and audio streams (block length = first joint ACF crossing).
3. **Lexicographic visual dominance + orthogonal residual.** Language, audio and
   calibrated interaction evidence only break ties of the visual curve; the
   resulting average-rank residual is mean-centred so the global logit cannot
   change temporal ordering. `score = intercept + residual`. No learned or
   dataset parameters, zero MLLM calls at this stage, empty interval output.

## Inputs (all under `data/omsl_v6_inputs/`, provenance in `PROVENANCE.md` there)
manifest `manifests/all_test.jsonl`; visual `visual_A10_vidgroup_zero_shot_full643_v1.jsonl`;
text `text_unified_qwen3vl8b_chunk_scores_b1_fullcoverage.jsonl`; audio `audio_embeddings/<dataset>/<video_id>.npy`;
whole-video logits `holistic_consistent/<dataset>/scores.jsonl` (plus `crossbench_judge_8b/` for MHC, historical).
ImageBind text anchors: `data/assets/imagebind/text_embeddings_normal_hateful.npy` (cached from `imagebind_huge.pth`).
Ground truth: `data/gt_4fps/<dataset>.npz` (evaluator only).

## How to run
```
conda activate HateVideo
bash experiments/20260829_omsl_v6/launch/run_omsl_v6.sh <run_name>
# -> runs/20260829_omsl_v6/<run_name>/{predictions.jsonl,predictions.audit.json,metrics.json,run.log}
```
CPU only, under a minute. Evaluator: `src/eval/evaluate_four_datasets.py` (the single shared evaluator).

## Results (test split, 4 fps; pooled ROC / pooled PR / within-video macro ROC)
Source: `runs/20260829_omsl_v6/v6_migrated_seed0_20260909/metrics.json` (current code) and
`runs/20260829_omsl_v6/orthogonal_mobius_semantic_localizer_full643_v6_metrics.json` (2026-08-29 original).

| dataset | current (seed 0) | 2026-08-29 original |
|---|---|---|
| HateMM | .8507 / .5781 / .6494 | .8507 / .5781 / .6497 |
| HateClipSeg | .6692 / .6622 / .5473 | .6692 / .6622 / .5473 |
| MHC (historical) | .7458 / .4970 / .7011 | .7458 / .4970 / .7005 |
| MHC_zh (historical) | .7522 / .5354 / .6837 | .7522 / .5354 / .6835 |

Pooled metrics are unchanged by the migration; within differs in the 4th decimal
because the permutation null is now seeded with a constant (`PERMUTATION_SEED = 0`)
instead of a data-derived digest (hash ban). Comparators on the same protocol:
`runs/20260829_omsl_v6/multihateloc_frozen_current4fps_v1_metrics.json`,
`runs/20260829_omsl_v6/t3al_anchor_s20250819_metrics.json`.

## History
- 2026-08-29: v6 selected after v1–v5 iterations (`archive/experiments/idea_discovery-2026-08/`).
  Audit and claim gate: `archive/root-2026-09/EXPERIMENT_AUDIT.md`, `CLAIMS_FROM_RESULTS.md`.
- 2026-09-09: migrated here from `scripts/idea_discovery/project_orthogonal_mobius_localizer.py`;
  verified identical metrics under conda HateVideo before the seed change.

## Ablations (2026-09-09, development-selected; test split, 4 fps; ROC / PR / within)

Source: `runs/20260829_omsl_v6/ablation_<tag>/metrics.json`; launcher `launch/run_ablations.sh`; table by `summarize_ablations.py`. `full` is bit-identical to `v6_migrated_seed0_20260909`.

| ablation | HateMM ROC / PR / within | HateClipSeg ROC / PR / within |
|---|---|---|
| full | 0.8507 / 0.5781 / 0.6494 | 0.6692 / 0.6622 / 0.5473 |
| drop_language | 0.8501 / 0.5761 / 0.6337 | 0.6687 / 0.6626 / 0.5411 |
| drop_audio | 0.8508 / 0.5802 / 0.6457 | 0.6690 / 0.6621 / 0.5450 |
| visual_only | 0.8498 / 0.5771 / 0.6180 | 0.6683 / 0.6630 / 0.5321 |
| mains_only | 0.8507 / 0.5781 / 0.6494 | 0.6692 / 0.6622 / 0.5475 |
| mobius_raw | 0.8496 / 0.5748 / 0.6039 | 0.6683 / 0.6615 / 0.5340 |
| perm_99 | 0.8507 / 0.5781 / 0.6494 | 0.6692 / 0.6622 / 0.5473 |
| perm_255 | 0.8507 / 0.5781 / 0.6494 | 0.6692 / 0.6622 / 0.5473 |
| order_sum | 0.8510 / 0.5783 / 0.6172 | 0.6704 / 0.6604 / 0.5479 |
| intercept_mllm | 0.8506 / 0.5780 / 0.6494 | 0.6711 / 0.6630 / 0.5473 |
| intercept_occ | 0.5358 / 0.2540 / 0.6494 | 0.4972 / 0.4714 / 0.5473 |
| intercept_none | 0.5227 / 0.2534 / 0.6494 | 0.5145 / 0.4815 / 0.5473 |

Tags: `drop_*` zero one stream; `visual_only` no coalition field (visual order, ties averaged);
`mains_only` language+audio main effects without interaction terms; `mobius_raw` uncalibrated
Möbius interactions; `perm_N` N block permutations; `order_sum` equal-weight sum instead of
visual-primary lexicographic order; `intercept_*` video intercept = z only / occupancy only / 0.

Reading (HateMM, HateClipSeg):
- Pooled ROC/PR are carried by the whole-video MLLM logit z: removing it (`intercept_occ`, `intercept_none`) drops pooled ROC to .54/.52 and .50/.51. Occupancy adds nothing (`intercept_mllm` ≈ `full`). No module-1/2 change moves pooled by more than .002.
- Modules 1–2 act only on within-video order: `visual_only` .618/.532 → `full` .649/.547. Language carries most of it (`drop_language` .634/.541), audio less (`drop_audio` .646/.545).
- Calibrated interactions contribute nothing measurable: `mains_only` equals `full` (.6494 vs .6494; .5475 vs .5473). Uncalibrated interactions hurt (`mobius_raw` .604/.534), so calibration works by suppressing them. Permutation count (31/99/255) changes nothing.
- Lexicographic order matters on HateMM within (.649 vs `order_sum` .617), not on HateClipSeg (.547 vs .548).
