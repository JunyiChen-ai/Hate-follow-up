#!/usr/bin/env bash
# Usage: bash experiments/20260922_til/launch/run_measure.sh <run_name> <window_offset_seconds> [datasets...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="$1"; OFF="$2"; shift 2
DS=("$@"); [ ${#DS[@]} -eq 0 ] && DS=(HateMM HateClipSeg)
OUT="runs/20260922_til/$RUN"; mkdir -p "$OUT"
PY="${TIL_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN offset $OFF datasets ${DS[*]}" | tee -a "$OUT/launch.log"
set +e
"$PY" experiments/20260922_til/til_measure.py --run-name "$RUN" --window-offset "$OFF" --datasets "${DS[@]}" >> "$OUT/launch.log" 2>&1
RC=$?
set -e
grep -E "VERIFY|progress|DONE|FAILED|OOM|Traceback|GATE" "$OUT/launch.log" | tail -5
if [ $RC -ne 0 ] || ! grep -q "DONE videos" "$OUT/launch.log"; then echo "RUN_FAILED $RUN rc=$RC" | tee -a "$OUT/launch.log"; exit 1; fi
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
