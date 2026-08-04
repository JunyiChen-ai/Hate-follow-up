#!/bin/bash
#SBATCH -J tsar_32b
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

STEM="qwen2.5-vl-32b-awq"
MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
ROOT="results/boundary_rescue"

echo "=== Stage 1: rerun band with full logprobs ==="
python src/boundary_rescue/judge_offline.py \
  --model "$MODEL" \
  --all --band-only \
  --batch-size 4 --gpu-mem 0.88 \
  --logprobs 1 --save-full-logprobs

echo "=== Stage 2: TokenSAR scoring (GPU cross-encoder) ==="
for DS in MHClip_EN MHClip_ZH HateMM ImpliHateVid; do
  IN="$ROOT/$DS/offline_test_band_lp_tsar_${STEM}.jsonl"
  OUT="$ROOT/$DS/offline_test_band_lp_tsar_${STEM}_scored.jsonl"
  if [ ! -f "$IN" ]; then
    echo "MISSING $IN, skip"
    continue
  fi
  echo "-- $DS"
  python src/boundary_rescue/tokensar_score.py \
    --in "$IN" --out "$OUT" \
    --encoder cross-encoder/stsb-roberta-large \
    --device cuda --batch-size 128
done

echo "=== Stage 3: gated eval ==="
python src/boundary_rescue/eval_tokensar.py --judge-stem "$STEM"

echo "DONE"
