#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
phase="${1:-main}"
if [ "$phase" = smoke ]; then
    args=(--smoke --arms base select all_heads permuted_heads shifted_support)
elif [ "$phase" = main ]; then
    args=(--arms base select)
elif [ "$phase" = controls ]; then
    args=(--arms all_heads permuted_heads shifted_support --replay-run runs/20261002_m1_selector/r1_main/select)
else
    echo "Unknown phase: $phase" >&2; exit 2
fi
exec /home/jehc223/miniconda3/envs/HateVLM/bin/python -u archive/experiments/20261002_m1_selector/measure.py --run-name "r1_$phase" "${args[@]}"
