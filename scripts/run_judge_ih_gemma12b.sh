#!/bin/bash
#SBATCH -J ih_gem12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

python src/boundary_rescue/judge_offline.py \
  --model google/gemma-3-12b-it \
  --dataset ImpliHateVid \
  --ih-prompt \
  --no-video \
  --batch-size 2 --gpu-mem 0.90

wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_gemma-3-12b-it.jsonl
