#!/usr/bin/env bash
# SPVL-r2 grid-A measurements (til_measure.py, unchanged) with the fixed ASR loader of 2026-09-26; both corpora.
# Usage (lab): cd ~/Hate-follow-up && setsid nohup bash experiments/20260926_glr/launch/run_base.sh > runs/20260926_glr/launch_base.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=base_gridA; OUT="runs/20260926_glr/$RUN"; mkdir -p "$OUT"
PY="${GLR_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN" | tee -a "$OUT/launch.log"
set +e
"$PY" experiments/20260922_til/til_measure.py --exp-id 20260926_glr --run-name "$RUN" --window-offset 0 "$@" >> "$OUT/launch.log" 2>&1
RC=$?
set -e
grep -E "VERIFY|progress|DONE|FAILED|OOM|Traceback|GATE" "$OUT/launch.log" | tail -5 || true
if [ $RC -ne 0 ] || ! grep -q "DONE videos" "$OUT/launch.log"; then echo "RUN_FAILED $RUN rc=$RC" | tee -a "$OUT/launch.log"; exit 1; fi
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
