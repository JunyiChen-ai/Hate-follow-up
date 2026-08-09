#!/usr/bin/env bash
# Re-extraction of the fit corpus under the frozen judge's real input
# configuration. Waits for the eval refresh to finish, then re-runs
# ImpliHateVid train_clean with the full-corpus override map and no character
# cap -- the same two flags the c2 full-corpus judge run used.
#
# HateMM train is not re-run: no fresh Whisper transcript exists for that
# split anywhere in the repository, so the frozen judge's input for it cannot
# be reconstructed and it leaves the fit set.

set -u
ROOT=/home/jehc223/Hate-follow-up
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
OUTROOT="$ROOT/results/answer_contrast"
STATUS="$OUTROOT/STATUS"
LOG="$OUTROOT/extract.log"
export HVD_DATA_ROOT=/home/jehc223/data

while [ ! -f "$OUTROOT/EVAL_REFRESH_DONE" ]; do
  if grep -q "FAILED" "$STATUS" 2>/dev/null; then exit 1; fi
  sleep 60
done

d="$OUTROOT/ImpliHateVid_train"
[ -d "$d" ] && mv "$d" "${d}.capped300"
echo "[$(date -Is)] ImpliHateVid/train refresh start" >> "$STATUS"
$PY "$ROOT/src/duplex/extract_answer_contrast.py" \
    --dataset ImpliHateVid --split train \
    --transcript-limit 0 \
    --transcript-override-json "$ROOT/results/c2_fullcorpus/c2_overrides.json" \
    --out-dir "$d" >> "$LOG" 2>&1
rc=$?
echo "[$(date -Is)] ImpliHateVid/train refresh exit=$rc" >> "$STATUS"
[ $rc -ne 0 ] && { echo "FAILED at fit refresh" >> "$STATUS"; exit $rc; }

rm -f "$OUTROOT/ANALYSIS_DONE"
echo "[$(date -Is)] analysis start" >> "$STATUS"
$PY "$ROOT/scripts/duplex/answer_contrast_analyze.py" \
    >> "$OUTROOT/analyze.log" 2>&1
rc=$?
echo "[$(date -Is)] analysis exit=$rc" >> "$STATUS"
[ $rc -eq 0 ] && touch "$OUTROOT/ANALYSIS_DONE"
exit $rc
