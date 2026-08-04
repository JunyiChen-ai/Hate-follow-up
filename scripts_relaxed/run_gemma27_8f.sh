#!/bin/bash
#SBATCH --job-name=gem27_8f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/gem27_8f_%j.out
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
export NUM_FRAMES=8
# Use default prompt (rationale + answer Yes/No since judge_offline.py default); 
# But we want hateful/normal format to compare with original. Add --rtg-mode would have rt field.
# Actually just use default — if ZH matches original 122, then 8 frames is irrelevant.
python src/boundary_rescue/judge_offline.py --model google/gemma-3-27b-it --all --batch-size 2 --gpu-mem 0.85 --band-only --no-video
