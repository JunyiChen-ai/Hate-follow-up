#!/bin/bash
#SBATCH --job-name=qw3rtg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/qw3rtg_%j.out
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3
python src/boundary_rescue/judge_offline.py --model Qwen/Qwen3-VL-8B-Instruct --all --batch-size 8 --gpu-mem 0.85 --rtg-mode --band-only
