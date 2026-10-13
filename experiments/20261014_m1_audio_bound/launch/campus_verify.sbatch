#!/usr/bin/env bash
#SBATCH --job-name=m1-c43-verify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261014_m1_audio_bound/slurm_%j.out
# Cache-vs-standalone numerics check (verify_cache.py) on uoa-campus2 or uoa-campus3: sbatch experiments/20261014_m1_audio_bound/launch/campus_verify.sbatch
set -euo pipefail
cd /data/jehc223/Hate-follow-up
case $(hostname) in foscsmlprd03*) CONDA=/data/jehc223/home/miniconda3;; *) CONDA=/data/jehc223/miniconda3;; esac
source $CONDA/bin/activate HateVLM
export HF_HOME=/data/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/data/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/data/jehc223/Hate-follow-up/.cache/cuda
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p runs/20261014_m1_audio_bound
python -u experiments/20261014_m1_audio_bound/verify_cache.py
echo C43_VERIFY_DONE
