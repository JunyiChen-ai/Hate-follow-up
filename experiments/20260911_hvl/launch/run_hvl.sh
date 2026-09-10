#!/usr/bin/env bash
# Usage: bash experiments/20260911_hvl/launch/run_hvl.sh <run_name> [hvl.py args...]
# One video at a time on the local GPU; then compose (revised verdict; first verdict; + window mean) + shared evaluator.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="$1"; shift
OUT="runs/20260911_hvl/$RUN"; mkdir -p "$OUT"
PY="${HVL_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN args: $*" | tee -a "$OUT/launch.log"
"$PY" experiments/20260911_hvl/hvl.py --run-name "$RUN" "$@" 2>&1 | tee -a "$OUT/launch.log" | grep -E "VERIFY|progress|DONE|FAILED|OOM|Traceback|Error|GATE|hypothesis" || true
if ! grep -q "DONE" "$OUT/launch.log"; then echo "RUN_FAILED $RUN" | tee -a "$OUT/launch.log"; exit 1; fi
if grep -q "DONE verify-only" "$OUT/launch.log"; then echo "RUN_DONE $RUN (verify-only)" | tee -a "$OUT/launch.log"; exit 0; fi
for I in zv_rev spvl zv_rev_plus_mean zv_plus_mean; do
  "$PY" experiments/20260911_hvl/compose.py --run-dir "$OUT" --intercept $I --residual rank 2>&1 | tee -a "$OUT/launch.log" | tail -3
done
"$PY" experiments/20260911_hvl/compose.py --run-dir "$OUT" --intercept zv_rev --residual none 2>&1 | tee -a "$OUT/launch.log" | tail -3
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
