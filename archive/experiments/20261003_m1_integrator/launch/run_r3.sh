#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/20261003_m1_integrator/r3_main
mkdir -p "$out"
exec > "$out/run.log" 2>&1
exec /home/jehc223/miniconda3/envs/HateVideo/bin/python -u archive/experiments/20261003_m1_integrator/branch_revision.py
