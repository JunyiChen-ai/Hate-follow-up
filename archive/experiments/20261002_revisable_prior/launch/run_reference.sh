#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'host %s\n' "$(hostname)"
/home/jehc223/miniconda3/envs/HateVideo/bin/python -u experiments/20260926_twolevel/twolevel_r2.py \
    --run runs/20260910_spvl/mllm/q3vl-8b/full --datasets HateMM HateClipSeg \
    --noleak --transform nscore --key calib --duration bma --bma-prior length \
    --min-windows 2 --bma-grid 6 --arm m2 --selftest \
    --out-root runs/20261002_revisable_prior --tag baseline_full
