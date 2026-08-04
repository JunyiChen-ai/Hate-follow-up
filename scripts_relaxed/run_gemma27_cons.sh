#!/bin/bash
#SBATCH --job-name=gem27cons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/gem27cons_%j.out
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3
python src/boundary_rescue/judge_offline.py --model google/gemma-3-27b-it --all --batch-size 2 --gpu-mem 0.85 --cons-mode --band-only --no-video
