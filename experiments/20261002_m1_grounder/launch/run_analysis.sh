#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/jehc223/miniconda3/envs/HateVideo/bin/python -u experiments/20261002_m1_grounder/analyze.py "$@"
