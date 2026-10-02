#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/20261002_m1_attributor/r1_controls_analysis
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=experiments/20261002_m1_attributor/controls.py
"$py" -u "$runner" --stage generate
for arm in endpoint absolute rotated density shift; do
    "$py" -u "$runner" --stage evaluate --arm "$arm" > "$out/$arm.out" 2>&1
done
"$py" -u "$runner" --stage report
