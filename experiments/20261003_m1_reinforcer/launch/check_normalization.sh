#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PWD/.cache/hf"
exec /home/junyi/miniconda3/envs/HateVLM/bin/python -u experiments/20261003_m1_reinforcer/check_normalization.py
