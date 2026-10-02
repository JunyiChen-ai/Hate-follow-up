#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=runs/20261002_m1_grounder/r1_full_analysis
mkdir -p "$out"
exec > "$out/run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/run.pid"
runner=experiments/20261002_m1_grounder/launch/run_analysis.sh
bash "$runner" --stage prepare
pids=()
for arm in base late verdict_only all_local shifted early shift_only; do
    bash "$runner" --stage evaluate --evaluate-arm "$arm" > "$out/$arm.out" 2>&1 &
    pids+=("$!")
done
failed=0
for task_pid in "${pids[@]}"; do
    if ! wait "$task_pid"; then failed=1; fi
done
if (( failed )); then echo ANALYSIS_FAILED; exit 1; fi
bash "$runner" --stage report
