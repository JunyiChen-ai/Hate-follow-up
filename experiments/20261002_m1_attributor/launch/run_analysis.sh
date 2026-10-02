#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/20261002_m1_attributor/r1_main_fp32_analysis
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
runner=experiments/20261002_m1_attributor/analyze.py
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
"$py" -u "$runner" --stage prepare
pids=()
for arm in base attribute; do
    "$py" -u "$runner" --stage evaluate --arm "$arm" > "$out/$arm.out" 2>&1 &
    pids+=("$!")
done
failed=0
for task_pid in "${pids[@]}"; do if ! wait "$task_pid"; then failed=1; fi; done
if (( failed )); then echo ANALYSIS_FAILED; exit 1; fi
"$py" -u "$runner" --stage report
