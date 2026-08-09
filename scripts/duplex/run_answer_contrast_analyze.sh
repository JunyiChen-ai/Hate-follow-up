#!/usr/bin/env bash
# Waits for the answer-contrast extraction to finish, then runs the frozen
# Stage-B analysis. Detached so that the pilot completes even if the session
# that launched it ends.
#
#   setsid nohup bash scripts/duplex/run_answer_contrast_analyze.sh \
#     > results/answer_contrast/analyze.log 2>&1 < /dev/null &

set -u
ROOT=/home/jehc223/Hate-follow-up
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
OUTROOT="$ROOT/results/answer_contrast"
export HVD_DATA_ROOT=/home/jehc223/data

while [ ! -f "$OUTROOT/DONE" ]; do
  if grep -q "FAILED" "$OUTROOT/STATUS" 2>/dev/null; then
    echo "extraction failed; analysis not run" >> "$OUTROOT/STATUS"
    exit 1
  fi
  sleep 120
done

echo "[$(date -Is)] analysis start" >> "$OUTROOT/STATUS"
$PY "$ROOT/scripts/duplex/answer_contrast_analyze.py"
rc=$?
echo "[$(date -Is)] analysis exit=$rc" >> "$OUTROOT/STATUS"
[ $rc -eq 0 ] && touch "$OUTROOT/ANALYSIS_DONE"
exit $rc
