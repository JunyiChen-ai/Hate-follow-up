#!/bin/bash
#SBATCH --job-name=naive-prompt
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=runs/20261113_lf_baselines/slurm_%j.out
# Naive-prompt baselines on uoa-lab2 (README §1). Usage: sbatch experiments/20261113_lf_baselines/launch/naive_lab2.sbatch <arm>:<dataset> ...
bash experiments/20261113_lf_baselines/launch/naive_run.sh "$@"
