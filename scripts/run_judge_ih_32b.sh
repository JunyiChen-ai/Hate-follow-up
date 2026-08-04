#!/bin/bash
#SBATCH -J ih_32b
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -o /data/jehc223/EMNLP3/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3

echo "=== qwen2.5-VL-32B-AWQ on ImpliHateVid with IH-tailored prompt ==="
python src/boundary_rescue/judge_offline.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --dataset ImpliHateVid \
  --ih-prompt \
  --batch-size 2 --gpu-mem 0.90

echo "=== output ==="
ls -la results/boundary_rescue/ImpliHateVid/offline_test_ih_qwen2.5-vl-32b-awq.jsonl
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_qwen2.5-vl-32b-awq.jsonl
echo "DONE"
