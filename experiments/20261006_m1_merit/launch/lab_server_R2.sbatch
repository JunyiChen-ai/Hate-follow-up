#!/usr/bin/env bash
#SBATCH --job-name=m1-merit-R2
#SBATCH --partition=local-sc448960
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261006_m1_merit/slurm_%j.out
set -euo pipefail
cd /home/junyi/Hate-follow-up
source /home/junyi/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/junyi/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/junyi/Hate-follow-up/.cache/triton
export CUDA_CACHE_PATH=/home/junyi/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi
export READER_REVISION=2
args=()
case ${SCOPE:-smoke} in smoke) args+=(--smoke);; main) ;; *) echo FAILED_invalid_scope; exit 2;; esac
python -u experiments/20261006_m1_merit/measure.py "${args[@]}"
