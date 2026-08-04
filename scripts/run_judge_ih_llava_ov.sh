#!/bin/bash
#SBATCH -J ih_llava
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o /data/jehc223/EMNLP3/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3

python src/boundary_rescue/judge_offline.py \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --dataset ImpliHateVid \
  --ih-prompt \
  --no-video \
  --batch-size 4 --gpu-mem 0.90

wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_llava-onevision-qwen2-7b-ov-hf.jsonl
