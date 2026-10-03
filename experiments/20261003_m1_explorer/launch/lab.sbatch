#!/usr/bin/env bash
#SBATCH --job-name=m1_explorer
#SBATCH --partition=local-sc474399
#SBATCH --nodelist=sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261003_m1_explorer/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
bash experiments/20261003_m1_explorer/launch/run_lab.sh "${1:-smoke}"
