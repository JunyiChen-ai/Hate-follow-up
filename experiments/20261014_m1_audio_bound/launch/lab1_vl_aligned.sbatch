#!/usr/bin/env bash
#SBATCH --job-name=m1-c43-vl
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261014_m1_audio_bound/slurm_vl_%j.out
# Aligned word slice under the Qwen3-VL reader (measure_vl_aligned.py), both corpora in one job on uoa-lab1:
#   sbatch experiments/20261014_m1_audio_bound/launch/lab1_vl_aligned.sbatch     (SMOKE=1 for the five smoke videos)
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u experiments/20261014_m1_audio_bound/measure_vl_aligned.py --datasets HateMM HateClipSeg ${SMOKE:+--smoke --name vl_aligned_smoke}
echo C43_VL_DONE
