#!/usr/bin/env bash
#SBATCH --job-name=m1-ordered-slots-R3
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261006_m1_ordered_slots/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source .cache/envs/HateVLM/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton
export CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SOURCE_INTERFACE=B READER_REVISION=3
nvidia-smi
args=()
case ${SCOPE:-smoke} in smoke) args+=(--smoke);; main) ;; *) exit 2;; esac
# Same complete333 R1 B sources are read-only and their original costs charged.
python -u experiments/20261006_m1_ordered_slots/measure.py "${args[@]}"
