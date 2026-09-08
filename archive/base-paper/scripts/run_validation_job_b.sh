#!/bin/bash
#SBATCH -J val_mllm
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH -o /data/jehc223/EMNLP3/logs/validation_job_b_%j.out
#SBATCH -e /data/jehc223/EMNLP3/logs/validation_job_b_%j.err

set -euo pipefail
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3
mkdir -p logs results/validation_runs

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export CUDA_LAUNCH_BLOCKING=1

DATASETS=(MHClip_EN MHClip_ZH HateMM ImpliHateVid)

audit_jsonl() {
  local ds="$1"
  local path="$2"
  local skip_file="$3"
  python scripts/validation_audit.py jsonl \
    --dataset "$ds" --split validation --path "$path" --skip-file "$skip_file"
}

run_supervised() {
  local ds="$1"
  local out="$2"
  local skip="$3"
  local logdir="$4"
  shift 4
  python scripts/validation_supervisor.py \
    --dataset "$ds" \
    --split validation \
    --output "$out" \
    --skip-file "$skip" \
    --log-dir "$logdir" \
    -- "$@"
  audit_jsonl "$ds" "$out" "$skip"
}

run_supervised_adaptive() {
  local ds="$1"
  local out="$2"
  local skip="$3"
  local logdir="$4"
  local frame_fallbacks="$5"
  shift 5
  python scripts/validation_supervisor.py \
    --dataset "$ds" \
    --split validation \
    --output "$out" \
    --skip-file "$skip" \
    --log-dir "$logdir" \
    --adaptive-frames \
    --frame-fallbacks "$frame_fallbacks" \
    -- "$@"
  audit_jsonl "$ds" "$out" "$skip"
}

run_judge() {
  local ds="$1"
  local model="$2"
  local tag="$3"
  local batch="$4"
  local gpu_mem="$5"
  local max_len="$6"
  local no_video="$7"
  local frame_fallbacks="${8:-}"
  local ih_arg=()
  local no_video_arg=()
  local max_len_arg=()
  local stem="offline_validation_${tag}"
  if [[ "$ds" == "ImpliHateVid" ]]; then
    ih_arg=(--ih-prompt)
    stem="offline_validation_ih_${tag}"
  fi
  if [[ "$no_video" == "yes" ]]; then
    no_video_arg=(--no-video)
  fi
  if [[ "$max_len" != "default" ]]; then
    max_len_arg=(--max-model-len "$max_len")
  fi
  local out="results/boundary_rescue/${ds}/${stem}.jsonl"
  local skip="results/validation_runs/crash_skip/judge_${tag}_${ds}.txt"
  local logdir="results/validation_runs/logs/judge_${tag}_${ds}"
  local cmd=(
    python src/boundary_rescue/judge_offline.py
      --model "$model"
      --dataset "$ds"
      --split validation
      --batch-size "$batch"
      --gpu-mem "$gpu_mem"
      "${max_len_arg[@]}"
      "${no_video_arg[@]}"
      "${ih_arg[@]}"
  )
  if [[ -n "$frame_fallbacks" ]]; then
    run_supervised_adaptive "$ds" "$out" "$skip" "$logdir" "$frame_fallbacks" "${cmd[@]}"
  else
    run_supervised "$ds" "$out" "$skip" "$logdir" "${cmd[@]}"
  fi
}

echo "=== validation frames ==="
bash scripts/ensure_validation_frames.sh

echo "=== offline judge validation ==="
echo "Skipping gated Gemma models until HF auth/access is available:"
echo "  - google/gemma-3-12b-it"
echo "  - google/gemma-3-27b-it"
echo "Skipping gated/unavailable MiniCPM until HF auth/cache is available:"
echo "  - openbmb/MiniCPM-V-2_6"
for ds in "${DATASETS[@]}"; do
  if [[ "$ds" != "MHClip_EN" ]]; then
    run_judge "$ds" Qwen/Qwen3-VL-8B-Instruct qwen3-vl-8b 4 0.88 default yes 16,8,4,2,1
    run_judge "$ds" Qwen/Qwen2.5-VL-32B-Instruct-AWQ qwen2.5-vl-32b-awq 2 0.88 default no 16,8,4,2,1
    run_judge "$ds" Qwen/Qwen2.5-VL-72B-Instruct-AWQ qwen2.5-vl-72b-awq 1 0.88 16384 no 16,8,4,2,1
    run_judge "$ds" OpenGVLab/InternVL3_5-8B internvl35-8b 4 0.90 default yes 8,4,2,1
  else
    echo "Skipping completed MHClip_EN Qwen/InternVL judge outputs; resuming at LLaVA-OV."
  fi
  run_judge "$ds" llava-hf/llava-onevision-qwen2-7b-ov-hf llava-onevision-qwen2-7b-ov-hf 2 0.88 default yes 4,2,1
done

echo "=== MARS 32B-AWQ validation ==="
for ds in "${DATASETS[@]}"; do
  run_supervised "$ds" \
    "results/mars_32b_awq/${ds}/validation_mars.jsonl" \
    "results/validation_runs/crash_skip/mars_32b_awq_${ds}.txt" \
    "results/validation_runs/logs/mars_32b_awq_${ds}" \
    python src/mars_repro/reproduce_mars_32b_awq.py \
      --dataset "$ds" --split validation
done

echo "=== LoReHM validation retrieval ==="
for ds in "${DATASETS[@]}"; do
  python src/lorehm_repro/retrieval.py --dataset "$ds" --eval-split validation
done

echo "=== LoReHM validation inference ==="
for ds in "${DATASETS[@]}"; do
  run_supervised "$ds" \
    "results/lorehm/${ds}/validation_lorehm.jsonl" \
    "results/validation_runs/crash_skip/lorehm_32b_awq_${ds}.txt" \
    "results/validation_runs/logs/lorehm_32b_awq_${ds}" \
    python src/lorehm_repro/reproduce_lorehm.py \
      --dataset "$ds" --eval-split validation
done

python scripts/validation_eval_summary.py || true
echo "DONE validation job B"
