#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
version="${1:-r1}"
if [[ "$version" != r1 && "$version" != r2 && "$version" != r3 ]]; then exit 2; fi
out="runs/20261003_m1_explorer/${version}_main_analysis"
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=experiments/20261003_m1_explorer/analyze.py
"$py" -u "$runner" --stage prepare --version "$version"
pids=()
for arm in base explore; do
    "$py" -u "$runner" --stage evaluate --arm "$arm" --version "$version" > "$out/$arm.out" 2>&1 &
    pids+=("$!")
done
failed=0
for task_pid in "${pids[@]}"; do if ! wait "$task_pid"; then failed=1; fi; done
if ((failed)); then echo ANALYSIS_FAILED; exit 1; fi
"$py" -u "$runner" --stage report --version "$version"
