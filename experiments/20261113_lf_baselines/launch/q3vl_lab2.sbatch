#!/bin/bash
#SBATCH --job-name=lf-q3vl
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=runs/20261113_lf_baselines/slurm_%j.out
# Published methods on Qwen3-VL-8B on uoa-lab2 (README §3). Usage: sbatch experiments/20261113_lf_baselines/launch/q3vl_lab2.sbatch <method> <dataset> ...
bash experiments/20261113_lf_baselines/launch/q3vl_run.sh "$@"
