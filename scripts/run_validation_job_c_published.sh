#!/bin/bash
#SBATCH -J val_pub
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 72:00:00
#SBATCH -o /data/jehc223/EMNLP3/logs/validation_job_c_%j.out
#SBATCH -e /data/jehc223/EMNLP3/logs/validation_job_c_%j.err

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
echo "DONE validation job C"
