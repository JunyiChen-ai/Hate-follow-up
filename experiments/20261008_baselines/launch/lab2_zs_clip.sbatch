#!/usr/bin/env bash
#SBATCH --job-name=bl-zsclip-dehate
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/zs_clip/slurm_%j.out
# ZS-CLIP, DeHate: reproduction check on 5 HateMM videos, 4 fps CLIP-L/336 features for the 1341 test videos, score.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
S=experiments/20261008_baselines/zs_clip/zs_clip.py
$PY_VIDEO $S verify || { echo "FAILED verify"; exit 1; }
$PY_VIDEO $S extract || { echo "FAILED extract"; exit 1; }
$PY_VIDEO $S score || echo "FAILED score (see run.log)"
echo "JOB_DONE $(date -Is)"
