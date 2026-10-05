#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVideo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
out=runs/20261005_m1_ott/full_input_preflight
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
python -u experiments/20261005_m1_ott/preflight.py
