#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
revision=${1:-r3}
case "$revision" in r3|r4) ;; *) exit 2;; esac
out=runs/20261004_m1_tree/${revision}_controls_main_analysis
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=experiments/20261004_m1_tree/control_analyze.py
"$py" -u "$runner" --stage prepare --revision "$revision"
for name in base main flat wrong_links no_depth no_added_pixels temporal temporal_fresh_priority; do
  "$py" -u "$runner" --stage evaluate --name "$name" --revision "$revision"
done
"$py" -u "$runner" --stage report --revision "$revision"
