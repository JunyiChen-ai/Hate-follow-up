#!/usr/bin/env bash
#SBATCH --job-name=reader-abl
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261101_reader_ablation/slurm_%j.out
# Reader ablation arms on uoa-lab1 (experiments/20261101_reader_ablation/README.md):
#   SCOPE=smoke sbatch experiments/20261101_reader_ablation/launch/lab1.sbatch                 (2 + 2 + 2 videos, every arm, checks)
#   SCOPE=HateMM ARMS="no_transcript joint" sbatch experiments/20261101_reader_ablation/launch/lab1.sbatch   (one corpus, the listed arms)
# ARMS empty = all six arms. Output runs/20261101_reader_ablation/<corpus or NAME>/ (resumable).
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
VLM=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin/python
case ${SCOPE:-smoke} in
  smoke) $VLM -u experiments/20261101_reader_ablation/read_arms.py --smoke;;
  HateMM|HateClipSeg|DeHate) $VLM -u experiments/20261101_reader_ablation/read_arms.py --datasets $SCOPE ${ARMS:+--arms $ARMS} ${NAME:+--name $NAME};;
  *) echo FAILED_invalid_scope; exit 2;;
esac
echo READER_ABLATION_DONE
