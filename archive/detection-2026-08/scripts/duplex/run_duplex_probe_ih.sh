#!/bin/bash
#SBATCH -J duplex_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o /data/jehc223/EMNLP3/logs/slurm-%j-duplex-probe.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
source /data/jehc223/hf_cache_env.sh
cd /data/jehc223/EMNLP3

echo "=== duplex probe: 2B, ImpliHateVid train, 5 readers === $(date)"
python src/duplex/score_duplex_probe.py \
  --dataset ImpliHateVid --split train \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 16 --gpu-mem 0.90

echo "=== duplex probe: 8B, ImpliHateVid train, 5 readers === $(date)"
python src/duplex/score_duplex_probe.py \
  --dataset ImpliHateVid --split train \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 8 --gpu-mem 0.90

echo "=== output inventory ==="
wc -l results/duplex_probe/ImpliHateVid/*.jsonl
echo "=== DONE === $(date)"
