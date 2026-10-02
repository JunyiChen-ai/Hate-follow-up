#!/bin/bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/jehc223/miniconda3/envs/HateVideo/bin/python -u archive/experiments/20261002_revisable_prior/adaptive_run.py
