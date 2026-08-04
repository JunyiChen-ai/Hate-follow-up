#!/bin/bash
#SBATCH -J ih_bundle
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

echo "=== 1/3: internvl3_5-8b ==="
python src/boundary_rescue/judge_offline.py \
  --model OpenGVLab/InternVL3_5-8B \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 4 --gpu-mem 0.90 || echo "internvl failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_internvl3_5-8b.jsonl 2>/dev/null || true

echo "=== 2/3: llava-onevision-qwen2-7b-ov-hf ==="
python src/boundary_rescue/judge_offline.py \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 4 --gpu-mem 0.90 || echo "llava failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_llava-onevision-qwen2-7b-ov-hf.jsonl 2>/dev/null || true

echo "=== 3/3: minicpm-v-26 ==="
python src/boundary_rescue/judge_offline.py \
  --model openbmb/MiniCPM-V-2_6 \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 4 --gpu-mem 0.90 || echo "minicpm failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_minicpm-v-26.jsonl 2>/dev/null || true

echo "=== BUNDLE DONE ==="
