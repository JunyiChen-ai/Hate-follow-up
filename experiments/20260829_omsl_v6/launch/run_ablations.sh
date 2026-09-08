#!/bin/bash
# OMSL-v6 ablations (2026-09-09). CPU only. One run per configuration; each writes
# runs/20260829_omsl_v6/ablation_<tag>/{predictions.jsonl,metrics.json,run.log}.
# Usage: cd <repo> && bash experiments/20260829_omsl_v6/launch/run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
IN=data/omsl_v6_inputs
run() {  # $1 = tag, rest = extra flags
  tag="$1"; shift
  OUT="runs/20260829_omsl_v6/ablation_${tag}"; mkdir -p "$OUT"
  [ -e "$OUT/predictions.jsonl" ] && rm -f "$OUT/predictions.jsonl" "$OUT/predictions.audit.json"
  {
    echo "host: $(hostname)  date: $(date -Iseconds)  commit: $(git rev-parse --short HEAD)  flags: $*"
    PYTHONPATH=. python experiments/20260829_omsl_v6/omsl_v6.py \
      --manifest "$IN/manifests/all_test.jsonl" \
      --visual "$IN/visual_A10_vidgroup_zero_shot_full643_v1.jsonl" \
      --text "$IN/text_unified_qwen3vl8b_chunk_scores_b1_fullcoverage.jsonl" \
      --audio-dir "$IN/audio_embeddings" \
      --scores HateMM "$IN/holistic_consistent/HateMM/scores.jsonl" \
      --scores HateClipSeg "$IN/holistic_consistent/HateClipSeg/scores.jsonl" \
      --scores MHC "$IN/crossbench_judge_8b/mhclip_en/scores.jsonl" --scores MHC "$IN/holistic_consistent/MHC/scores.jsonl" \
      --scores MHC_zh "$IN/crossbench_judge_8b/mhclip_zh/scores.jsonl" --scores MHC_zh "$IN/holistic_consistent/MHC_zh/scores.jsonl" \
      --tag "$tag" "$@" --out "$OUT/predictions.jsonl" > /dev/null
    PYTHONPATH=. python src/eval/evaluate_four_datasets.py \
      --predictions "$OUT/predictions.jsonl" --gt-dir data/gt_4fps --out "$OUT/metrics.json" > /dev/null
    echo DONE
  } > "$OUT/run.log" 2>&1
  echo "$tag: $(tail -1 "$OUT/run.log")"
}
run full
# modality removal
run drop_language     --drop language
run drop_audio        --drop audio
run visual_only       --fusion none
# module 1/2: fusion form
run mains_only        --fusion mains_only
run mobius_raw        --fusion mobius_raw
run perm_99           --permutations 99
run perm_255          --permutations 255
# module 2: ordering rule
run order_sum         --order sum
# module 3: intercept
run intercept_mllm    --intercept mllm
run intercept_occ     --intercept occupancy
run intercept_none    --intercept none
