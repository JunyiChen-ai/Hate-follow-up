#!/usr/bin/env bash
#SBATCH --job-name=m1-program-cap
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261005_m1_program/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
python -u experiments/20261005_m1_program/handle_extract.py --capacity-check
echo CAPACITY_DONE
