#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/20261002_revisable_prior
printf 'host %s\n' "$(hostname)"
printf '%s\n' "$$" > "$out/launch.pid"
for arm in full independent no_global; do
    /home/jehc223/miniconda3/envs/HateVideo/bin/python -u archive/experiments/20261002_revisable_prior/model.py \
        --arm "$arm" --tag "r1_$arm"
done
echo RUN_DONE
