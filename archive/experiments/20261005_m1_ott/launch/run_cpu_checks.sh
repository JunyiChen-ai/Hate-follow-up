#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
out=runs/20261005_m1_ott/cpu_checks
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.cache/envs/HateVLM/bin/python -u experiments/20261005_m1_ott/selfcheck.py
