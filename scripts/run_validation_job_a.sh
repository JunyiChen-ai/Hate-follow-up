#!/bin/bash
#SBATCH -J val_ext
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 72:00:00
#SBATCH -o /data/jehc223/EMNLP2/logs/validation_job_a_%j.out
#SBATCH -e /data/jehc223/EMNLP2/logs/validation_job_a_%j.err

set -euo pipefail
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
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

echo "=== validation frames ==="
bash scripts/ensure_validation_frames.sh

echo "=== Pro-Cap V3 validation captions ==="
python src/procap_v3_repro/generate_captions_qwen2vl.py --all --split validation
for ds in "${DATASETS[@]}"; do
  python scripts/validation_audit.py captions \
    --dataset "$ds" --split validation \
    --path "results/procap_v3/${ds}/captions_validation.pkl"
done

echo "=== Mod-HATE validation ==="
for ds in "${DATASETS[@]}"; do
  for k in 4 8; do
    run_supervised "$ds" \
      "results/mod_hate/${ds}/validation_mod_hate_${k}shot.jsonl" \
      "results/validation_runs/crash_skip/mod_hate_${k}shot_${ds}.txt" \
      "results/validation_runs/logs/mod_hate_${k}shot_${ds}" \
      python src/mod_hate_repro/reproduce_mod_hate.py \
        --dataset "$ds" --eval-split validation --shots "$k" --no-load-8bit
  done
done

echo "=== ALARM validation train/reference phase ==="
for ds in "${DATASETS[@]}"; do
  python src/alarm_repro/reproduce_alarm.py \
    --dataset "$ds" \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --results-subdir alarm_validation_7b \
    --eval-split validation \
    --no-do-inpredict
done

echo "=== ALARM validation InPredict phase ==="
for ds in "${DATASETS[@]}"; do
  run_supervised "$ds" \
    "results/alarm_validation_7b/${ds}/validation_alarm.jsonl" \
    "results/validation_runs/crash_skip/alarm_7b_${ds}.txt" \
    "results/validation_runs/logs/alarm_7b_${ds}" \
    python src/alarm_repro/reproduce_alarm.py \
      --dataset "$ds" \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --results-subdir alarm_validation_7b \
      --eval-split validation \
      --no-do-label --no-do-embed --no-do-retrieve --no-do-experience --no-do-reference
done

python scripts/validation_eval_summary.py || true
echo "DONE validation job A"
