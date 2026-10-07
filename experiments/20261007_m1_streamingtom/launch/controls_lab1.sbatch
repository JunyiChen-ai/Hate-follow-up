#!/usr/bin/env bash
#SBATCH --job-name=m1-stom-ctl-unif
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261007_m1_streamingtom/slurm_controls_%j.out
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
args=(--job uniform)
case ${SCOPE:-smoke} in
  smoke) args+=(--smoke);;
  main) ;;
  *) echo FAILED_invalid_scope; exit 2;;
esac
python -u experiments/20261007_m1_streamingtom/controls.py "${args[@]}"
echo STREAMINGTOM_CONTROLS_DONE
