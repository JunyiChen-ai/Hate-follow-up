#!/usr/bin/env bash
# SPVL-r2 grid-A reads (experiments/20260922_til/til_measure.py, unchanged; same settings as runs/20260926_glr/base_gridA)
# on the DeHate test split. Needs the HateVLM env, data/manifests/DeHate_test.jsonl, data/frames_k20/DeHate and
# data/asr_whisper_large_v3/DeHate.
# Usage (lab): cd ~/Hate-follow-up && setsid nohup bash experiments/20260927_dehate_external/launch/run_reads.sh > runs/20260927_dehate_external/launch_reads.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=reads_gridA; OUT="runs/20260927_dehate_external/$RUN"; mkdir -p "$OUT"
PY="${READS_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN" | tee -a "$OUT/launch.log"
set +e
"$PY" experiments/20260922_til/til_measure.py --exp-id 20260927_dehate_external --run-name "$RUN" --window-offset 0 \
  --datasets DeHate --manifest data/manifests/DeHate_test.jsonl "$@" >> "$OUT/launch.log" 2>&1
RC=$?
set -e
grep -E "VERIFY|progress|DONE|FAILED|OOM|Traceback" "$OUT/launch.log" | tail -5 || true
if [ $RC -ne 0 ] || ! grep -q "DONE videos" "$OUT/launch.log"; then echo "RUN_FAILED $RUN rc=$RC" | tee -a "$OUT/launch.log"; exit 1; fi
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
