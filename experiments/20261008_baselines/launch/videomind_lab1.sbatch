#!/bin/bash
# VideoMind-7B (grounder + verifier) on uoa-lab1. From the repo root on uoa-lab1:
#   sbatch experiments/20261008_baselines/launch/videomind_lab1.sbatch                 (HateMM HateClipSeg DeHate)
#   sbatch --export=ALL,DATASETS=HateMM,LIMIT=3 experiments/20261008_baselines/launch/videomind_lab1.sbatch   (smoke)
# Resumable: videos already in raw.jsonl without an error are skipped.
#SBATCH --job-name=vmind
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=runs/20261008_baselines/videomind/slurm_%j.out
set -uo pipefail
REPO=$HOME/Hate-follow-up
cd "$REPO"
PY=$REPO/.cache/envs/videomind/bin/python          # torch 2.8.0+cu128, transformers 4.45.2 (pinned by VideoMind)
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$REPO/third_party/VideoMind
MODEL=$(ls -d $HF_HOME/hub/models--yeliudev--VideoMind-7B/snapshots/* | head -1)
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/* | head -1)
DATASETS=${DATASETS:-HateMM HateClipSeg DeHate}
EXTRA=${LIMIT:+--limit $LIMIT}
echo "host $(hostname) job ${SLURM_JOB_ID:-} datasets ${DATASETS//,/ } start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
for DS in ${DATASETS//,/ }; do
  $PY -u experiments/20261008_baselines/videomind/videomind_hate.py infer --dataset "$DS" --model "$MODEL" --base-model "$BASE" $EXTRA || { echo "FAILED $DS"; exit 1; }
done
echo "VIDEOMIND_JOB_DONE $(date '+%F %T')"
