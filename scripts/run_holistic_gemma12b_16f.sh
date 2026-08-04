#!/bin/bash
#SBATCH -J h1_g12_16f
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH -o /data/jehc223/EMNLP3/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP3

MODEL="google/gemma-3-12b-it"
SLUG="gemma-3-12b-it-16f"
FLAGS="--no-video --no-mm-kwargs --num-frames 16"

for DS in MHClip_EN MHClip_ZH ImpliHateVid; do
  for SP in train test; do
    python src/our_method/score_holistic_2b.py \
      --dataset $DS --split $SP --mode binary \
      --model $MODEL --model-slug $SLUG $FLAGS --batch-size 2
  done
done
python src/our_method/score_holistic_2b.py \
  --dataset HateMM --split test --mode binary \
  --model $MODEL --model-slug $SLUG $FLAGS --batch-size 2

echo "DONE $SLUG"
for ds in MHClip_EN MHClip_ZH HateMM ImpliHateVid; do
  for sp in train test; do
    f=results/holistic_$SLUG/$ds/${sp}_binary.jsonl
    [ -f "$f" ] && echo "$(wc -l <$f) $f"
  done
done
