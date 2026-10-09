#!/usr/bin/env bash
#SBATCH --job-name=m1-c41-grounded
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261010_m1_grounded_adjacent/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton
export CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname
nvidia-smi
script=experiments/20261010_m1_grounded_adjacent/grounded.py
case ${SCOPE:-smoke} in
  smoke) python -u $script --smoke;;
  main) python -u $script;;
  both) python -u $script --smoke; python -u experiments/20261010_m1_grounded_adjacent/analyze.py --stage prepare --smoke; python -u $script;;
  *) echo FAILED_invalid_scope; exit 2;;
esac
echo C41_DONE
