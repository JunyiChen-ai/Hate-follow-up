#!/bin/bash
# OMSL-v6 inference + evaluation on the 4 fps protocol. CPU only (numpy/scipy);
# the ImageBind text anchors are read from data/assets/imagebind/ (cached).
# Usage: cd <repo> && bash experiments/20260829_omsl_v6/launch/run_omsl_v6.sh <run_name>
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN_NAME="${1:?run_name}"
OUT="runs/20260829_omsl_v6/${RUN_NAME}"
mkdir -p "$OUT"
{
  echo "host: $(hostname)  date: $(date -Iseconds)  commit: $(git rev-parse --short HEAD)  dirty: $(git status --short | grep -v '^??' | wc -l)"
  IN=data/omsl_v6_inputs
  PYTHONPATH=. python experiments/20260829_omsl_v6/omsl_v6.py \
    --manifest  "$IN/manifests/all_test.jsonl" \
    --visual    "$IN/visual_A10_vidgroup_zero_shot_full643_v1.jsonl" \
    --text      "$IN/text_unified_qwen3vl8b_chunk_scores_b1_fullcoverage.jsonl" \
    --audio-dir "$IN/audio_embeddings" \
    --scores HateMM      "$IN/holistic_consistent/HateMM/scores.jsonl" \
    --scores HateClipSeg "$IN/holistic_consistent/HateClipSeg/scores.jsonl" \
    --scores MHC    "$IN/crossbench_judge_8b/mhclip_en/scores.jsonl" --scores MHC    "$IN/holistic_consistent/MHC/scores.jsonl" \
    --scores MHC_zh "$IN/crossbench_judge_8b/mhclip_zh/scores.jsonl" --scores MHC_zh "$IN/holistic_consistent/MHC_zh/scores.jsonl" \
    --out "$OUT/predictions.jsonl"
  PYTHONPATH=. python src/eval/evaluate_four_datasets.py \
    --predictions "$OUT/predictions.jsonl" --gt-dir data/gt_4fps --out "$OUT/metrics.json"
  echo DONE
} > "$OUT/run.log" 2>&1
tail -3 "$OUT/run.log"
