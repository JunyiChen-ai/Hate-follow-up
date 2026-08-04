#!/usr/bin/env bash
#SBATCH -J meme_base_a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /data/jehc223/EMNLP3/logs/meme_baselines/slurm/%x_%j.out
#SBATCH -e /data/jehc223/EMNLP3/logs/meme_baselines/slurm/%x_%j.err

set -euo pipefail
cd /data/jehc223/EMNLP3
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
mkdir -p logs/meme_baselines/slurm results/meme_baselines

python -m src.meme_baselines.validate_outputs --stage processed --dataset all

echo "=== Naive MLLM smoke ==="
python -m src.meme_baselines.score_naive --dataset all --limit 5 --batch-size 4
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline naive_2b --filename test_naive.jsonl --allow-incomplete
done

echo "=== Naive MLLM full ==="
python -m src.meme_baselines.score_naive --dataset all --batch-size 8
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline naive_2b --filename test_naive.jsonl
done

echo "=== MARS 32B-AWQ smoke ==="
python -m src.meme_baselines.score_mars --dataset all --limit 2
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline mars_32b_awq --filename test_mars.jsonl --allow-incomplete
done

echo "=== MARS 32B-AWQ full ==="
python -m src.meme_baselines.score_mars --dataset all
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline mars_32b_awq --filename test_mars.jsonl
done

python -m src.meme_baselines.eval_metrics || true
echo "DONE meme baseline job A"
