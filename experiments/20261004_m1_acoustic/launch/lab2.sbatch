#!/usr/bin/env bash
#SBATCH --job-name=m1-acoustic
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261004_m1_acoustic/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
bash experiments/20261004_m1_acoustic/launch/run_lab.sh "$@"
