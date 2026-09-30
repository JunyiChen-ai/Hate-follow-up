#!/usr/bin/env bash
# README §15: current-family read sets on the main corpora that DeHate already has: joint branch, frames-only prefix
# (no transcript), transcript-only prefix (no frames). GPU on one lab machine (not uoa-lab1).
set -euo pipefail
cd "$(dirname "$0")/../../.."
RPY="${READS_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
OUT=runs/20260928_infer/main
mkdir -p $OUT
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD)" | tee -a $OUT/launch_reads.log
run_reads() {  # name, flags...
  local name=$1; shift
  local d=$OUT/reads_$name; mkdir -p $d
  if grep -q "DONE videos" $d/launch.log 2>/dev/null; then echo "READS_SKIP $name" | tee -a $OUT/launch_reads.log; return; fi
  "$RPY" experiments/20260922_til/til_measure.py --exp-id 20260928_infer --run-name main/reads_$name --window-offset 0 "$@" >> $d/launch.log 2>&1
  grep -q "DONE videos" $d/launch.log || { echo "RUN_FAILED reads_$name" | tee -a $OUT/launch_reads.log; exit 1; }
  echo "READS_DONE $name $(date -Is)" | tee -a $OUT/launch_reads.log
}
run_reads joint --branches joint
run_reads noctx --no-transcript-context
run_reads noframes --frames 0
echo ALL_DONE | tee -a $OUT/launch_reads.log
