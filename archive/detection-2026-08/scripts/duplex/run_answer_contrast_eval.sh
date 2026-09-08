#!/usr/bin/env bash
# Re-extraction of the four eval corpora under the frozen judge's real input
# configuration: fresh Whisper transcripts through the c2 override map and no
# character cap, exactly the two flags run_testrun.sh and run_hateclipseg.sh
# pass to src/duplex/extract_duplex_readout.py.
#
# The first pass of this pilot used the extractor's defaults (dataset
# transcript, 300-character cap) and the preregistered pre-flight caught it:
# HateMM test reproduced the frozen z at Spearman 0.827 with a median absolute
# difference of 1.75, far outside the 0.999 / 0.05 gate. The old arms are
# parked rather than deleted.

set -u
ROOT=/home/jehc223/Hate-follow-up
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
OUTROOT="$ROOT/results/answer_contrast"
STATUS="$OUTROOT/STATUS"
LOG="$OUTROOT/extract.log"
export HVD_DATA_ROOT=/home/jehc223/data

while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; do
  echo "waiting for GPU $(date -Is)" >> "$STATUS"
  sleep 60
done

run_eval () {
  ds="$1"; ov="$2"
  d="$OUTROOT/${ds}_test"
  if [ -d "$d" ]; then mv "$d" "${d}.capped300"; fi
  echo "[$(date -Is)] $ds/test refresh start" >> "$STATUS"
  $PY "$ROOT/src/duplex/extract_answer_contrast.py" \
      --dataset "$ds" --split test \
      --transcript-limit 0 \
      --transcript-override-json "$ov" \
      --out-dir "$d" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date -Is)] $ds/test refresh exit=$rc" >> "$STATUS"
  [ $rc -ne 0 ] && { echo "FAILED at $ds" >> "$STATUS"; exit $rc; }
}

run_eval HateMM       "$ROOT/results/testruns/hatemm/c2_overrides.json"
run_eval HateClipSeg  "$ROOT/results/hateclipseg/c2_overrides.json"
run_eval MHClip_EN    "$ROOT/results/testruns/mhclip_en/c2_overrides.json"
run_eval ImpliHateVid "$ROOT/results/testruns/implihatevid/c2_overrides.json"

echo "eval refresh complete: $(date -Is)" >> "$STATUS"
touch "$OUTROOT/EVAL_REFRESH_DONE"
