#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
phase="${1:-main_fp32_mem}"
args=()
if [[ "$phase" = smoke* ]]; then args+=(--smoke); elif [[ "$phase" = main_fp32* ]]; then args+=(--fp32-query --checkpoint-query); elif [ "$phase" != main ]; then exit 2; fi
exec /home/jehc223/miniconda3/envs/HateVLM/bin/python -u archive/experiments/20261002_m1_attributor/measure.py --run-name "r1_$phase" "${args[@]}"
