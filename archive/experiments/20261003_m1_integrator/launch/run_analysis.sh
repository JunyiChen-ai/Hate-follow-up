#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
revision="${1:-r1}"
if [[ "$revision" != r1 && "$revision" != r2 ]]; then exit 2; fi
out=runs/20261003_m1_integrator/${revision}_main_analysis
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=archive/experiments/20261003_m1_integrator/analyze.py
"$py" -u "$runner" --run-name "${revision}_main" --stage prepare
pids=()
for arm in base causal future; do
    "$py" -u "$runner" --run-name "${revision}_main" --stage evaluate --arm "$arm" > "$out/$arm.out" 2>&1 &
    pids+=("$!")
done
failed=0
for task_pid in "${pids[@]}"; do if ! wait "$task_pid"; then failed=1; fi; done
if ((failed)); then echo ANALYSIS_FAILED; exit 1; fi
"$py" -u "$runner" --run-name "${revision}_main" --stage report
