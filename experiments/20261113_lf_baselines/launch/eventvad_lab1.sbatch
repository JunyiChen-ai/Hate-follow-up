#!/bin/bash
#SBATCH --job-name=eventvad-q3vl
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=runs/20261113_lf_baselines/slurm_%j.out
# EventVAD with Qwen3-VL-8B on uoa-lab1 (README §2). Usage: sbatch experiments/20261113_lf_baselines/launch/eventvad_lab1.sbatch <dataset> ...
bash experiments/20261113_lf_baselines/launch/eventvad_run.sh "$@"
