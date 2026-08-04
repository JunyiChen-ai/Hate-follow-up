#!/bin/bash
#SBATCH -J ih_bund2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

# Resume + retry for the 3 small IH-prompt judges. Fixes from bundle1:
# - internvl3_5-8b: NUM_FRAMES=8 (16 frames → 37-53K tokens, over 32K cap)
# - llava-onevision: NUM_FRAMES=8 (72K tokens at 16 frames)
# - minicpm-v-26: batch-size=1 + NUM_FRAMES=8 to sidestep vLLM 0.11.0
#   shape-mismatch bug on multi-image batches.
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export NUM_FRAMES=8

echo "=== 1/3: internvl3_5-8b (resume, NUM_FRAMES=8) ==="
python src/boundary_rescue/judge_offline.py \
  --model OpenGVLab/InternVL3_5-8B \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 4 --gpu-mem 0.90 || echo "internvl failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_internvl35-8b.jsonl 2>/dev/null || true

echo "=== 2/3: llava-onevision (resume, NUM_FRAMES=8) ==="
python src/boundary_rescue/judge_offline.py \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 4 --gpu-mem 0.90 || echo "llava failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_llava-onevision-qwen2-7b-ov-hf.jsonl 2>/dev/null || true

echo "=== 3/3: minicpm-v-26 (resume, NUM_FRAMES=8, batch=1) ==="
python src/boundary_rescue/judge_offline.py \
  --model openbmb/MiniCPM-V-2_6 \
  --dataset ImpliHateVid --ih-prompt --no-video \
  --batch-size 1 --gpu-mem 0.90 || echo "minicpm failed, continuing"
wc -l results/boundary_rescue/ImpliHateVid/offline_test_ih_minicpm-v-26.jsonl 2>/dev/null || true

echo "=== BUNDLE2 DONE ==="
