#!/bin/bash
#SBATCH -J ih_8b
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /data/jehc223/EMNLP3/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3

echo "=== qwen3-vl-8b on ImpliHateVid with IH-tailored prompt ==="
python src/boundary_rescue/judge_offline.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset ImpliHateVid \
  --ih-prompt \
  --batch-size 4 --gpu-mem 0.88

echo "=== output ==="
ls -la results/boundary_rescue/ImpliHateVid/offline_test_ih_qwen3-vl-8b.jsonl
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_qwen3-vl-8b.jsonl
echo "DONE"
