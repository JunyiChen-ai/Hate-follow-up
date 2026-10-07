#!/bin/bash
# PRISM (Qwen3-VL-Embedding-2B) on uoa-lab3. From the repo root on uoa-lab3:
#   sbatch experiments/20261008_baselines/launch/prism_lab3.sbatch                      (HateMM HateClipSeg DeHate)
#   sbatch --export=ALL,DATASETS=HateMM,LIMIT=3 experiments/20261008_baselines/launch/prism_lab3.sbatch   (smoke)
#   sbatch --export=ALL,STAGE=desc experiments/20261008_baselines/launch/prism_lab3.sbatch   (write the description pools)
# Extraction is resumable per video; scoring runs after extraction of each corpus.
#SBATCH --job-name=prism
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=runs/20261008_baselines/prism/slurm_%j.out
set -uo pipefail
REPO=$HOME/Hate-follow-up
cd "$REPO"
PY=$REPO/.cache/envs/prism/bin/python              # torch 2.8.0+cu128, transformers 4.57.1, qwen-vl-utils 0.0.14
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host $(hostname) job ${SLURM_JOB_ID:-} stage ${STAGE:-run} start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
if [ "${STAGE:-run}" = desc ]; then
  GEN=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/* | head -1)
  $PY -u experiments/20261008_baselines/prism/make_descriptions.py --model "$GEN" || { echo "FAILED desc"; exit 1; }
  echo "PRISM_JOB_DONE desc"; exit 0
fi
MODEL=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/* | head -1)
DATASETS=${DATASETS:-HateMM HateClipSeg DeHate}
EXTRA=${LIMIT:+--limit $LIMIT}
for DS in ${DATASETS//,/ }; do
  $PY -u experiments/20261008_baselines/prism/prism_hate.py extract --dataset "$DS" --model "$MODEL" $EXTRA || { echo "FAILED extract $DS"; exit 1; }
  [ -n "${LIMIT:-}" ] && continue
  $PY -u experiments/20261008_baselines/prism/prism_hate.py score --dataset "$DS" --model "$MODEL" || { echo "FAILED score $DS"; exit 1; }
done
echo "PRISM_JOB_DONE $(date '+%F %T')"
