#!/usr/bin/env bash
# GLR measurement pass (glr_measure.py); both corpora. Extra arguments are passed through (e.g. --limit 3 --run-name smoke).
# Usage (lab): cd ~/Hate-follow-up && setsid nohup bash experiments/20260926_glr/launch/run_glr.sh > runs/20260926_glr/launch_glr.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${GLR_RUN:-glr_pilot}"; OUT="runs/20260926_glr/$RUN"; mkdir -p "$OUT"
PY="${GLR_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN" | tee -a "$OUT/launch.log"
set +e
"$PY" experiments/20260926_glr/glr_measure.py --run-name "$RUN" "$@" >> "$OUT/launch.log" 2>&1
RC=$?
set -e
grep -E "VERIFY|progress|DONE|FAILED|OOM|Traceback|GATE|MISMATCH" "$OUT/launch.log" | tail -5 || true
if [ $RC -ne 0 ] || ! grep -q "DONE videos" "$OUT/launch.log"; then echo "RUN_FAILED $RUN rc=$RC" | tee -a "$OUT/launch.log"; exit 1; fi
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
