#!/usr/bin/env bash
#SBATCH --job-name=m1-c46
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261017_m1_topic_removal/slurm_%j.out
# Candidate 46 stage 2: topic-removed speech reads (measure.py), both corpora in one job on uoa-lab1:
#   sbatch experiments/20261017_m1_topic_removal/launch/lab1_measure.sbatch     (SMOKE=1 for the five smoke videos)
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u experiments/20261017_m1_topic_removal/measure.py --datasets HateMM HateClipSeg ${SMOKE:+--smoke --name main_smoke}
echo C46_MEASURE_DONE
