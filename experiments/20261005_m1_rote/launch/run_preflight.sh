#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/20261005_m1_rote/full_input_preflight
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
.cache/envs/HateVLM/bin/python -u experiments/20261005_m1_rote/preflight.py
