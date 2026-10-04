#!/usr/bin/env bash
#SBATCH --job-name=m1-program
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261005_m1_program/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
bash experiments/20261005_m1_program/launch/run_lab.sh "$@"
