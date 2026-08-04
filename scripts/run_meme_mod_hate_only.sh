#!/usr/bin/env bash
#SBATCH -J meme_modhate
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

echo "=== Mod-HATE full ==="
python -m src.meme_baselines.mod_hate --dataset all --shots 4 8 --no-load-8bit
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline mod_hate --filename test_mod_hate_4shot.jsonl
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline mod_hate --filename test_mod_hate_8shot.jsonl
done

python -m src.meme_baselines.eval_metrics || true
echo "DONE meme Mod-HATE job"
