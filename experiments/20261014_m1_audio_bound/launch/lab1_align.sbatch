#!/usr/bin/env bash
#SBATCH --job-name=m1-c43-align
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261014_m1_audio_bound/slurm_align_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVideo
export OMP_NUM_THREADS=4
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u experiments/20261014_m1_audio_bound/align_words.py --datasets HateMM HateClipSeg
echo C43_ALIGN_DONE
