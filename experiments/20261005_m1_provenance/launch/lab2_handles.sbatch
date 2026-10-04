#!/usr/bin/env bash
#SBATCH --job-name=m1-provenance-Bgraph
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261005_m1_provenance/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
bash experiments/20261005_m1_provenance/launch/run_lab_handles.sh "$@"
