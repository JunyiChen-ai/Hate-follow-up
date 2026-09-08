#!/usr/bin/env bash
# Interleaved-timeline kill test, stage A driver: Whisper chunk-timestamp
# recovery over the four corpora, one at a time on the single GPU.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
export HVD_DATA_ROOT=/home/jehc223/data
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
OUT=results/interleaved_timeline
LOG=$OUT/logs
mkdir -p "$LOG"
echo "stage_A_running" > "$OUT/STATUS"
for C in mhclip_zh mhclip_en implihatevid hateclipseg; do
  echo "=== $C $(date -Is)"
  echo "stage_A:$C" > "$OUT/STATUS"
  for attempt in 1 2 3; do
    $PY -u scripts/duplex/interleaved_timeline_asr.py --corpus "$C" \
      2>&1 | tee -a "$LOG/asr_$C.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && break
    echo "attempt $attempt rc=$rc; resuming in 20s"; sleep 20
  done
done
echo "stage_A_done" > "$OUT/STATUS"
touch "$OUT/ASR_DONE"
echo "=== all done $(date -Is)"
