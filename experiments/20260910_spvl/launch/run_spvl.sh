#!/usr/bin/env bash
# Usage: bash experiments/20260910_spvl/launch/run_spvl.sh <run_name> [spvl.py args...]
# One video per forward on the local GPU; then compose + shared evaluator. Runs from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="$1"; shift
OUT="runs/20260910_spvl/$RUN"; mkdir -p "$OUT"
PY="${SPVL_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN args: $*" | tee -a "$OUT/launch.log"
"$PY" experiments/20260910_spvl/spvl.py --run-name "$RUN" "$@" 2>&1 | tee -a "$OUT/launch.log"
INTERCEPT=spvl; case " $* " in *" --legacy-chunk-arm "*) INTERCEPT=legacy;; esac
"$PY" experiments/20260910_spvl/compose.py --run-dir "$OUT" --intercept "$INTERCEPT" --residual rank 2>&1 | tee -a "$OUT/launch.log"
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
