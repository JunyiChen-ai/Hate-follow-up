#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /home/jehc223/miniconda3/envs/HateVLM/bin/python -u experiments/20261002_m1_attributor/measure.py --run-name numeric_adaptive --smoke --numeric-video non_hate_video_82 --adaptive
