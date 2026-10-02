#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
out=runs/20261002_verdict_analysis
mkdir -p "$out"
printf 'host %s\n' "$(hostname)"
for arm in full nostance; do
    /home/jehc223/miniconda3/envs/HateVideo/bin/python experiments/20260926_twolevel/twolevel_r2.py \
        --run "runs/20260910_spvl/mllm/q3vl-8b/$arm" \
        --datasets HateMM HateClipSeg --noleak --transform nscore --key calib \
        --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2 \
        --out-root "$out" --tag "main_$arm"
done
echo RUN_DONE
