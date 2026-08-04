#!/bin/bash
# Rebuttal E4b — SAGE train + eval on OUR fixed splits (1 GPU).
# Usage:
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=48G scripts/rebuttal_e4_sage_train.sh
#     -> runs all 3 datasets x seeds {0,1,2} sequentially in one job, then aggregates.
#   sbatch --gres=gpu:1 ... scripts/rebuttal_e4_sage_train.sh hatemm "0 1 2"
#     -> single dataset, chosen seeds (lets the director shard around the E3 GPU job).
# Datasets: hatemm (HateMM), mhclip_yt (MHClip_EN), mhclip_bl (MHClip_ZH).
cd /data/jehc223/EMNLP3
source ~/.bashrc                    # /etc/bashrc reads unset vars — must precede `set -u`
conda activate SafetyContradiction
set -euo pipefail                  # strict flags for the actual work
export HF_HUB_OFFLINE=1   # backbones pre-cached; avoid network stalls on compute node

DATASETS="${1:-hatemm mhclip_yt mhclip_bl}"
SEEDS="${2:-0 1 2}"
CFG=external_repos/SAGE/config/config_ours.yaml
OUT=results/rebuttal/E4_sage

echo "=== E4b TRAIN start $(date) host=$(hostname) datasets=[$DATASETS] seeds=[$SEEDS] ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for ds in $DATASETS; do
  for s in $SEEDS; do
    echo "--- $ds seed $s @ $(date) ---"
    python scripts/rebuttal_e4_sage_run.py --mode train --config "$CFG" \
        --dataset "$ds" --seed "$s" --out_dir "$OUT"
  done
  python scripts/rebuttal_e4_sage_run.py --mode aggregate --dataset "$ds" --out_dir "$OUT"
done
echo "=== E4b TRAIN done $(date) ==="
