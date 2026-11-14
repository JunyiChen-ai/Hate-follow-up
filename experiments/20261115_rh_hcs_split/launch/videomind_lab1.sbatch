#!/bin/bash
#SBATCH --job-name=rh-vmind-av
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=runs/20261115_rh_hcs_split/slurm_%x_%j.out
# VideoMind-7B transcript variant (videomind_av) on the 79 test videos of the Retrieval-hate HateClipSeg split, uoa-lab1
# (README.md). Same commands as experiments/20261008_baselines/launch/videomind_av_lab2.sbatch plus the raster stage.
# Weights: VideoMind-7B copied from uoa-lab2; Qwen2-VL-7B-Instruct downloaded again (runs/_setup_uoa-lab1/).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
REPO=$PWD
source experiments/20261115_rh_hcs_split/env.sh
PY=$REPO/.cache/envs/videomind/bin/python          # torch 2.8.0+cu128, transformers 4.45.2 (pinned by VideoMind)
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$REPO/third_party/VideoMind
MODEL=$(ls -d $HF_HOME/hub/models--yeliudev--VideoMind-7B/snapshots/* | head -1)
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/* | head -1)
DS=HateClipSeg
echo "host $(hostname) job ${SLURM_JOB_ID:-} start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -u experiments/20261008_baselines/videomind/videomind_hate.py infer --av --dataset $DS --model "$MODEL" --base-model "$BASE" || { echo "FAILED infer $DS"; exit 1; }
$HOME/miniconda3/envs/HateVideo/bin/python experiments/20261008_baselines/videomind/videomind_hate.py raster --av --dataset $DS || { echo "FAILED raster $DS"; exit 1; }
echo "VIDEOMIND_AV_JOB_DONE $(date '+%F %T')"
