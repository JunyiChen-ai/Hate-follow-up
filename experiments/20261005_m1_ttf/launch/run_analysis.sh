#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVideo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
out=runs/20261005_m1_ttf/r1_full_main_analysis
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
# Source-selection float replay uses the actual GPU run's Torch/HF backend on CPU.
.cache/envs/HateVLM/bin/python -u experiments/20261005_m1_ttf/analyze.py --stage prepare
python -u experiments/20261005_m1_ttf/analyze.py --stage evaluate --name base
python -u experiments/20261005_m1_ttf/analyze.py --stage evaluate --name optimized
python -u experiments/20261005_m1_ttf/analyze.py --stage report
