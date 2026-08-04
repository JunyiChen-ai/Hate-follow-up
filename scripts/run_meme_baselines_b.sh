#!/usr/bin/env bash
#SBATCH -J meme_base_b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /data/jehc223/EMNLP2/logs/meme_baselines/slurm/%x_%j.out
#SBATCH -e /data/jehc223/EMNLP2/logs/meme_baselines/slurm/%x_%j.err

set -euo pipefail
cd /data/jehc223/EMNLP2
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
mkdir -p logs/meme_baselines/slurm results/meme_baselines

python -m src.meme_baselines.validate_outputs --stage processed --dataset all

echo "=== Caption smoke ==="
python -m src.meme_baselines.generate_captions --dataset all --split all --limit 5 --batch-size 4

echo "=== Caption full ==="
python -m src.meme_baselines.generate_captions --dataset all --split all --batch-size 8

echo "=== ALARM 7B full ==="
python -m src.meme_baselines.alarm --dataset all
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline alarm_7b --filename test_alarm.jsonl
done

echo "=== LoReHM smoke ==="
python -m src.meme_baselines.lorehm --dataset all --limit 2
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline lorehm_32b_awq --filename test_lorehm.jsonl --allow-incomplete
done

echo "=== LoReHM full ==="
python -m src.meme_baselines.lorehm --dataset all
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline lorehm_32b_awq --filename test_lorehm.jsonl
done

echo "=== Mod-HATE full ==="
python -m src.meme_baselines.mod_hate --dataset all --shots 4 8
for ds in FHM MAMI ToxiCN_MM; do
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline mod_hate --filename test_mod_hate_4shot.jsonl
  python -m src.meme_baselines.validate_outputs --stage result --dataset "$ds" \
    --baseline mod_hate --filename test_mod_hate_8shot.jsonl
done

python -m src.meme_baselines.eval_metrics || true
echo "DONE meme baseline job B"
