#!/bin/bash
#SBATCH -J h1_mcpm
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

MODEL="openbmb/MiniCPM-V-2_6"
SLUG="minicpm-v-26"
FLAGS="--no-video --no-mm-kwargs"

for DS in MHClip_EN MHClip_ZH ImpliHateVid; do
  for SP in train test; do
    python src/our_method/score_holistic_2b.py \
      --dataset $DS --split $SP --mode binary \
      --model $MODEL --model-slug $SLUG $FLAGS --batch-size 4
  done
done
python src/our_method/score_holistic_2b.py \
  --dataset HateMM --split test --mode binary \
  --model $MODEL --model-slug $SLUG $FLAGS --batch-size 4

echo "DONE $SLUG"
for ds in MHClip_EN MHClip_ZH HateMM ImpliHateVid; do
  for sp in train test; do
    f=results/holistic_$SLUG/$ds/${sp}_binary.jsonl
    [ -f "$f" ] && echo "$(wc -l <$f) $f"
  done
done
