#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
video="${1:-non_hate_video_82}"
run="${2:-numeric_fp32}"
exec /home/jehc223/miniconda3/envs/HateVLM/bin/python -u archive/experiments/20261002_m1_attributor/measure.py --run-name "$run" --smoke --numeric-video "$video" --fp32-query --checkpoint-query
