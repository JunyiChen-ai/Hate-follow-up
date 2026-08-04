#!/bin/bash
#SBATCH --job-name=s1_min_policy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage1_minimal_missing_%j.out

set -euo pipefail

cd /data/jehc223/EMNLP2
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction

MODEL="Qwen/Qwen3-VL-2B-Instruct"
BS=32

python src/our_method/score_holistic_2b.py \
  --dataset HateMM \
  --split test \
  --mode binary \
  --prompt-style minimal \
  --model "$MODEL" \
  --batch-size "$BS"

python src/our_method/score_holistic_2b.py \
  --dataset ImpliHateVid \
  --split test \
  --mode binary \
  --prompt-style minimal \
  --model "$MODEL" \
  --batch-size "$BS"
