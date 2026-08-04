#!/bin/bash
#SBATCH --job-name=qw32rtg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/qw32rtg_%j.out
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
python src/boundary_rescue/judge_offline.py --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ --all --batch-size 4 --gpu-mem 0.85 --rtg-mode --band-only
