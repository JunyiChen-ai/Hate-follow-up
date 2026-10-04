#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
revision=${1:-r1}
case "$revision" in r1|r2|r3|r4) ;; *) exit 2;; esac
out=runs/20261004_m1_tree/${revision}_full_main_analysis
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=experiments/20261004_m1_tree/analyze.py
"$py" -u "$runner" --stage prepare --revision "$revision"
for name in base optimized; do
  "$py" -u "$runner" --stage evaluate --name "$name" --revision "$revision"
done
"$py" -u "$runner" --stage report --revision "$revision"
