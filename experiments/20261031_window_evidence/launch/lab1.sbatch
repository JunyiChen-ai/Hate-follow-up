#!/usr/bin/env bash
#SBATCH --job-name=win-evidence
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261031_window_evidence/slurm_%j.out
# Adopted window-evidence reader on uoa-lab1:
#   SCOPE=smoke  sbatch experiments/20261031_window_evidence/launch/lab1.sbatch   (4 HateMM/HateClipSeg videos, checks against stored reads)
#   SCOPE=dehate sbatch experiments/20261031_window_evidence/launch/lab1.sbatch   (forced alignment, then the full DeHate test split)
# DeHate needs prep_inputs.py (CPU) to have finished first.
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
  smoke)
    $VLM -u experiments/20261031_window_evidence/read.py --smoke;;
  dehate)
    if [ ! -s data/asr_whisper_aligned/DeHate/words.jsonl ]; then
      (source /home/jehc223/miniconda3/bin/activate HateVideo && python -u experiments/20261014_m1_audio_bound/align_words.py --datasets DeHate)
    fi
    $VLM -u experiments/20261031_window_evidence/read.py --datasets DeHate;;
  *) echo FAILED_invalid_scope; exit 2;;
esac
echo WINDOW_EVIDENCE_DONE
