#!/bin/bash
#SBATCH --job-name=gemma27madp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemma27madp_%j.out
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
python src/boundary_rescue/judge_offline.py \
  --model google/gemma-3-27b-it \
  --all --batch-size 2 --gpu-mem 0.88 --eaa-mode --band-only --no-video
