#!/usr/bin/env bash
#SBATCH --job-name=scope-read
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261114_exp_gates/scope_reads/slurm_%j.out
# Scope reads on uoa-lab1 (experiments/20261114_exp_gates/README.md §3). One Yes/No branch per video and prefix variant.
#   sbatch experiments/20261114_exp_gates/launch/scope_lab1.sbatch                       (VARIANTS default: the two ablated prefixes)
#   VARIANTS="full" LIMIT=4 sbatch experiments/20261114_exp_gates/launch/scope_lab1.sbatch   (smoke: reproduces stored reads on 4 videos)
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
VLM=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin/python
SFX=${LIMIT:+_smoke}
if [ "${QUESTION:-protected}" = hcs_taxonomy ]; then   # README §6: HateClipSeg only, full prefix
  for v in full no_transcript no_prefix_frames; do
    $VLM -u experiments/20261114_exp_gates/scope_read.py --run-name hcs_taxonomy_${v}$SFX --variant $v --question hcs_taxonomy \
        --datasets HateClipSeg ${LIMIT:+--limit $LIMIT} || { echo "FAILED hcs_taxonomy $v"; exit 1; }
  done
  echo SCOPE_READ_DONE; exit 0
fi
for v in ${VARIANTS:-no_transcript no_prefix_frames}; do
  $VLM -u experiments/20261114_exp_gates/scope_read.py --run-name ${v}_main$SFX --variant $v ${LIMIT:+--limit $LIMIT} || { echo "FAILED $v main"; exit 1; }
  $VLM -u experiments/20261114_exp_gates/scope_read.py --run-name ${v}_dehate$SFX --variant $v --datasets DeHate \
      --manifest data/manifests/DeHate_test.jsonl ${LIMIT:+--limit $LIMIT} || { echo "FAILED $v dehate"; exit 1; }
done
echo SCOPE_READ_DONE
