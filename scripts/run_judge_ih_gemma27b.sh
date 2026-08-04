#!/bin/bash
#SBATCH -J ih_gem27
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -o /data/jehc223/EMNLP3/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3

echo "=== gemma-3-27b-it on ImpliHateVid with IH-tailored prompt ==="
python src/boundary_rescue/judge_offline.py \
  --model google/gemma-3-27b-it \
  --dataset ImpliHateVid \
  --ih-prompt \
  --no-video \
  --batch-size 2 --gpu-mem 0.90

echo "=== output ==="
ls -la results/boundary_rescue/ImpliHateVid/offline_test_ih_gemma-3-27b-it.jsonl
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_gemma-3-27b-it.jsonl
echo "DONE"
