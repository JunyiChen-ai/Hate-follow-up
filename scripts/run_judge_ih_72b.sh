#!/bin/bash
#SBATCH -J ih_72b
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 10:00:00
#SBATCH -o /data/jehc223/EMNLP3/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3

python src/boundary_rescue/judge_offline.py \
  --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --dataset ImpliHateVid \
  --ih-prompt \
  --batch-size 1 --gpu-mem 0.92

wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_qwen2.5-vl-72b-awq.jsonl
