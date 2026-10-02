#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
out=runs/20261002_verdict_analysis
printf 'host %s\n' "$(hostname)"
/home/jehc223/miniconda3/envs/HateVideo/bin/python -u experiments/20261002_verdict_analysis/shift_control.py
/home/jehc223/miniconda3/envs/HateVideo/bin/python -u experiments/20260926_twolevel/twolevel_r2.py \
    --run "$out/shift_reads" --datasets HateMM HateClipSeg \
    --noleak --transform nscore --key calib --duration bma --bma-prior length \
    --min-windows 2 --bma-grid 6 --arm m2 --out-root "$out" --tag main_shift_only
echo RUN_DONE
