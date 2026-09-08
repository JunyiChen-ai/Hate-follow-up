#!/bin/bash
#SBATCH --job-name=qwen3lp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen3lp_%j.out
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3
python src/boundary_rescue/judge_offline.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --all --batch-size 8 --gpu-mem 0.88 --logprobs 20
