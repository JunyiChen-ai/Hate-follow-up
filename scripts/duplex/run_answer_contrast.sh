#!/usr/bin/env bash
# Stage B of the answer-contrast pilot: GPU extraction of the two answer arms.
#
# Pre-registration: docs/duplex/PREREG_answer_contrast_pilot.md.
#
# GPU discipline. This script waits until Pilot 1 (spec displacement) has
# written results/spec_displacement/DONE and nvidia-smi reports no compute
# process, polling every five minutes. It never kills anything.
#
# Split order is the preregistered fallback order: HateMM train first, then
# every eval split, then the larger ImpliHateVid train split. If the run is cut
# short, what survives on disk is exactly the declared fallback (fit on HateMM
# train only) plus a complete eval set.
#
# Writes results/answer_contrast/STATUS and, on success,
# results/answer_contrast/DONE.

set -u
ROOT=/home/jehc223/Hate-follow-up
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
OUTROOT="$ROOT/results/answer_contrast"
STATUS="$OUTROOT/STATUS"
LOG="$OUTROOT/extract.log"
export HVD_DATA_ROOT=/home/jehc223/data

mkdir -p "$OUTROOT"
echo "waiting for GPU: $(date -Is)" > "$STATUS"

while true; do
  gpu_busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ -f "$ROOT/results/spec_displacement/DONE" ] && [ "$gpu_busy" -eq 0 ]; then
    break
  fi
  echo "waiting: pilot1_done=$([ -f "$ROOT/results/spec_displacement/DONE" ] && echo yes || echo no) gpu_procs=$gpu_busy $(date -Is)" >> "$STATUS"
  sleep 300
done

echo "starting extraction: $(date -Is)" >> "$STATUS"

run_split () {
  ds="$1"; sp="$2"; extra="${3:-}"
  echo "[$(date -Is)] $ds/$sp start" >> "$STATUS"
  $PY "$ROOT/src/duplex/extract_answer_contrast.py" \
      --dataset "$ds" --split "$sp" \
      --out-dir "$OUTROOT/${ds}_${sp}" $extra >> "$LOG" 2>&1
  rc=$?
  echo "[$(date -Is)] $ds/$sp exit=$rc" >> "$STATUS"
  if [ $rc -ne 0 ]; then
    echo "FAILED at $ds/$sp" >> "$STATUS"
    exit $rc
  fi
}

# The first split also runs the cache-versus-full-forward equivalence check on
# its first three videos.
run_split HateMM train "--verify-full 3"
run_split HateMM test
run_split HateClipSeg test
run_split MHClip_EN test
run_split ImpliHateVid test
run_split ImpliHateVid train

echo "extraction complete: $(date -Is)" >> "$STATUS"
touch "$OUTROOT/DONE"
