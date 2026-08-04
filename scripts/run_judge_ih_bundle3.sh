#!/bin/bash
#SBATCH -J ih_bund3
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

# Final IH resume bundle:
# - llava-onevision at NUM_FRAMES=4 (8 frames still blew max_model_len).
# - qwen2.5-vl-72b-awq resume (AWQ hits CUDA asserts on a few more videos;
#   new SKIP_VIDEOS updated to catch these too).
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

echo "=== 1/2: llava-onevision (NUM_FRAMES=4) ==="
NUM_FRAMES=4 python src/boundary_rescue/judge_offline.py \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 2 --gpu-mem 0.88 || echo "llava failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_llava-onevision-qwen2-7b-ov-hf.jsonl 2>/dev/null || true

echo "=== 2/2: qwen2.5-vl-72b-awq (resume) ==="
python src/boundary_rescue/judge_offline.py \
  --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --dataset ImpliHateVid --ih-prompt \
  --batch-size 1 --gpu-mem 0.92 || echo "72B failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_qwen2.5-vl-72b-awq.jsonl 2>/dev/null || true

echo "=== BUNDLE3 DONE ==="
