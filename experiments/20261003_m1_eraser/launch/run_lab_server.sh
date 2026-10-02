#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/.cache/hf"
phase="${1:-main}"
args=()
if [ "$phase" = smoke ]; then args+=(--smoke); elif [ "$phase" != main ]; then exit 2; fi
exec /home/junyi/miniconda3/envs/HateVLM/bin/python -u experiments/20261003_m1_eraser/measure.py --run-name "r1_$phase" "${args[@]}"
