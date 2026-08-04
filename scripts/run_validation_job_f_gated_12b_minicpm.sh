#!/bin/bash
#SBATCH -J val_gated_a
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 72:00:00
#SBATCH -o /data/jehc223/EMNLP2/logs/validation_job_f_%j.out
#SBATCH -e /data/jehc223/EMNLP2/logs/validation_job_f_%j.err

set -euo pipefail
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
mkdir -p logs results/validation_runs

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export CUDA_LAUNCH_BLOCKING=1

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && ! -s "$HOME/.cache/huggingface/token" ]]; then
  echo "No HF token found. Run scripts/hf_login_prompt.sh first or export HF_TOKEN before sbatch." >&2
  exit 2
fi

DATASETS=(MHClip_EN MHClip_ZH HateMM ImpliHateVid)

audit_jsonl() {
  local ds="$1"
  local path="$2"
  local skip_file="$3"
  python scripts/validation_audit.py jsonl \
    --dataset "$ds" --split validation --path "$path" --skip-file "$skip_file"
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
  local frame_fallbacks="$7"
  local ih_arg=()
  local max_len_arg=()
  local stem="offline_validation_${tag}"
  if [[ "$ds" == "ImpliHateVid" ]]; then
    ih_arg=(--ih-prompt)
    stem="offline_validation_ih_${tag}"
  fi
  if [[ "$max_len" != "default" ]]; then
    max_len_arg=(--max-model-len "$max_len")
  fi
  run_supervised_adaptive "$ds" \
    "results/boundary_rescue/${ds}/${stem}.jsonl" \
    "results/validation_runs/crash_skip/judge_${tag}_${ds}.txt" \
    "results/validation_runs/logs/judge_${tag}_${ds}" \
    "$frame_fallbacks" \
    python src/boundary_rescue/judge_offline.py \
      --model "$model" \
      --dataset "$ds" \
      --split validation \
      --no-video \
      --batch-size "$batch" \
      --gpu-mem "$gpu_mem" \
      "${max_len_arg[@]}" \
      "${ih_arg[@]}"
}

echo "=== gated MLLM validation: Gemma-3-12B + MiniCPM-V-2.6 ==="
for ds in "${DATASETS[@]}"; do
  run_judge "$ds" google/gemma-3-12b-it gemma-3-12b-it 2 0.90 default 16,8,4,2,1
  run_judge "$ds" openbmb/MiniCPM-V-2_6 minicpm-v-26 1 0.90 default 16,8,4,2,1
done

python scripts/validation_eval_summary.py || true
echo "DONE validation job F"
