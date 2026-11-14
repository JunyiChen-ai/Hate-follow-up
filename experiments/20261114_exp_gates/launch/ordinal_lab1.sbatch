#!/usr/bin/env bash
#SBATCH --job-name=ord-read
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261114_exp_gates/ordinal_reads/slurm_%j.out
# Video-level ordinal read on uoa-lab1 (experiments/20261114_exp_gates/README.md §10). LIMIT=<n> for a smoke run.
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
$VLM -u experiments/20261114_exp_gates/ordinal_read.py --run-name ord_full_main$SFX ${LIMIT:+--limit $LIMIT} \
    --stored runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl || { echo "FAILED main"; exit 1; }
$VLM -u experiments/20261114_exp_gates/ordinal_read.py --run-name ord_full_dehate$SFX --datasets DeHate \
    --manifest data/manifests/DeHate_test.jsonl ${LIMIT:+--limit $LIMIT} \
    --stored runs/20261031_window_evidence/DeHate/adopted/predictions.jsonl || { echo "FAILED dehate"; exit 1; }
echo ORDINAL_READ_DONE
