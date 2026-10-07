#!/usr/bin/env bash
#SBATCH --job-name=bl-qwen-text-winonly
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/qwen3_8b_text/slurm_%j.out
# Two short jobs in one allocation (the user account may run only 2 GPU jobs at a time across the lab cluster):
# Qwen3-8B transcript segments (lab2_qwen3_text.sbatch), then Qwen2.5-VL winonly on DeHate (lab2_winonly_dehate.sbatch).
set -uo pipefail
cd /home/jehc223/Hate-follow-up
bash experiments/20261008_baselines/launch/lab2_qwen3_text.sbatch
bash experiments/20261008_baselines/launch/lab2_winonly_dehate.sbatch
echo "JOB_DONE_ALL $(date -Is)"
