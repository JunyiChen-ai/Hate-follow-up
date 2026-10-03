#!/usr/bin/env bash
#SBATCH --job-name=m1_explorer_control
#SBATCH --partition=local-sc474399
#SBATCH --nodelist=sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261003_m1_explorer/control_slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PWD/.cache/hf"
exec /home/jehc223/miniconda3/envs/HateVLM/bin/python -u experiments/20261003_m1_explorer/controls.py "$@"
