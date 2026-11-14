#!/bin/bash
#SBATCH --job-name=rh-prism-av
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=runs/20261115_rh_hcs_split/slurm_%x_%j.out
# PRISM transcript variant (prism_av) on the 79 test videos of the Retrieval-hate HateClipSeg split, uoa-lab3
# (README.md). Same stages as experiments/20261008_baselines/launch/prism_av_lab3.sbatch plus the eval stage (which
# here stops after predictions.jsonl). The window embeddings of the test videos shared with the existing run are
# carried over (reuse_existing.py prism, copied from uoa-lab1); extraction is resumable per video.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
REPO=$PWD
source experiments/20261115_rh_hcs_split/env.sh
PY=$REPO/.cache/envs/prism/bin/python              # torch 2.8.0+cu128, transformers 4.57.1, qwen-vl-utils 0.0.14
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host $(hostname) job ${SLURM_JOB_ID:-} start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
MODEL=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/* | head -1)
DS=HateClipSeg
X=experiments/20261008_baselines/prism/prism_hate.py
$PY -u $X check --av --dataset $DS --model "$MODEL" --limit 1 || { echo "FAILED check $DS"; exit 1; }
$PY -u $X extract --av --dataset $DS --model "$MODEL" || { echo "FAILED extract $DS"; exit 1; }
$PY -u $X score --av --dataset $DS --model "$MODEL" || { echo "FAILED score $DS"; exit 1; }
$HOME/miniconda3/envs/HateVideo/bin/python $X eval --av --dataset $DS || { echo "FAILED eval $DS"; exit 1; }
echo "PRISM_AV_JOB_DONE $(date '+%F %T')"
