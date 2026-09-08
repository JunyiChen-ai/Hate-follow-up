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
