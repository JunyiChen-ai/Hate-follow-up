#!/usr/bin/env bash
#SBATCH --job-name=m1_preserver
#SBATCH --partition=local-sc448960
#SBATCH --nodelist=sc448960
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261003_m1_preserver/slurm_%j.out
set -euo pipefail
cd /home/junyi/Hate-follow-up
echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
bash archive/experiments/20261003_m1_preserver/launch/run_lab.sh "${1:-smoke}"
