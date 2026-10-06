#!/usr/bin/env bash
set -euo pipefail
export SOURCE_INTERFACE=C
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
out=runs/20261006_m1_vtimecot/r1_full_main_C_analysis
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
.cache/envs/HateVLM/bin/python -u experiments/20261006_m1_vtimecot/analyze.py --stage prepare
source /home/jehc223/miniconda3/bin/activate HateVideo
python -u experiments/20261006_m1_vtimecot/analyze.py --stage evaluate --name base
python -u experiments/20261006_m1_vtimecot/analyze.py --stage evaluate --name optimized
python -u experiments/20261006_m1_vtimecot/analyze.py --stage report
