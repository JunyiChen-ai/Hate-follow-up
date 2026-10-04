#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
arm="${1:-full}"
out="runs/20261004_m1_latents/r1_${arm}_main_analysis"
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=experiments/20261004_m1_latents/analyze.py
"$py" -u "$runner" --stage prepare --arm "$arm"
for name in base optimized; do
    "$py" -u "$runner" --stage evaluate --arm "$arm" --name "$name"
done
"$py" -u "$runner" --stage report --arm "$arm"
