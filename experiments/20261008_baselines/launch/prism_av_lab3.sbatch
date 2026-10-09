#!/bin/bash
# PRISM transcript (audio-visual) variant (prism/README.md, "Transcript (audio-visual) variant, 2026-10-09") on
# uoa-lab3, which holds the PRISM env and model files. Same stages as campus_prism_av.sbatch: per corpus, the
# processor-input check, window embeddings of frames + window transcript in one input item, then the 4 fps curves.
# From the repo root on uoa-lab3:
#   sbatch --export=ALL,DATASETS=DeHate experiments/20261008_baselines/launch/prism_av_lab3.sbatch
# Extraction is resumable per video. Evaluation runs on uoa-lab1 after rsync (prism_hate.py eval --av).
#SBATCH --job-name=prism-av
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=runs/20261008_baselines/prism_av/slurm_%j.out
set -uo pipefail
REPO=$HOME/Hate-follow-up
cd "$REPO"
PY=$REPO/.cache/envs/prism/bin/python              # torch 2.8.0+cu128, transformers 4.57.1, qwen-vl-utils 0.0.14
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host $(hostname) job ${SLURM_JOB_ID:-} start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
MODEL=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/* | head -1)
DATASETS=${DATASETS:-HateMM HateClipSeg DeHate}
EXTRA=${LIMIT:+--limit $LIMIT}
for DS in ${DATASETS//,/ }; do
  $PY -u experiments/20261008_baselines/prism/prism_hate.py check --av --dataset "$DS" --model "$MODEL" --limit 1 || { echo "FAILED check $DS"; exit 1; }
  $PY -u experiments/20261008_baselines/prism/prism_hate.py extract --av --dataset "$DS" --model "$MODEL" $EXTRA || { echo "FAILED extract $DS"; exit 1; }
  [ -n "${LIMIT:-}" ] && continue
  $PY -u experiments/20261008_baselines/prism/prism_hate.py score --av --dataset "$DS" --model "$MODEL" || { echo "FAILED score $DS"; exit 1; }
  echo "CORPUS_DONE $DS $(date '+%F %T')"
done
echo "PRISM_AV_JOB_DONE $(date '+%F %T')"
