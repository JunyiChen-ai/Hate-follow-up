#!/usr/bin/env bash
# Interleaved-timeline kill test, stages B-E. Stage A (Whisper chunk-timestamp
# recovery) runs separately; this driver waits for its DONE marker, builds the
# frame-slot segmentation, replicates the frozen baseline on a 20-video sample
# per corpus, scores the two new arms, and analyses.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
export HVD_DATA_ROOT=/home/jehc223/data
PY=/home/jehc223/venvs/SafetyContradiction/bin/python
OUT=results/interleaved_timeline
LOG=$OUT/logs
mkdir -p "$LOG"
CORPORA="mhclip_zh mhclip_en implihatevid hateclipseg"

say() { echo "[$(date -Is)] $*"; }
fail() { say "FAIL: $*"; echo "failed:$*" > "$OUT/STATUS"; exit 1; }

while [ ! -f "$OUT/ASR_DONE" ]; do say "waiting for stage A"; sleep 60; done

say "stage B: segmentation"
echo "stage_B" > "$OUT/STATUS"
$PY -u scripts/duplex/interleaved_timeline_build.py --corpus all \
  2>&1 | tee "$LOG/build.log" || fail "build"

say "stage C: baseline replication check, 20 videos per corpus"
echo "stage_C" > "$OUT/STATUS"
for C in $CORPORA; do
  $PY -u scripts/duplex/interleaved_timeline_score.py --corpus "$C" \
    --arm baseline --limit 20 --check-baseline \
    2>&1 | tee -a "$LOG/replication.log" || fail "replication $C"
done

say "stage D: the two new arms"
for ARM in interleaved misaligned; do
  for C in $CORPORA; do
    echo "stage_D:$ARM:$C" > "$OUT/STATUS"
    for attempt in 1 2 3; do
      $PY -u scripts/duplex/interleaved_timeline_score.py --corpus "$C" \
        --arm "$ARM" 2>&1 | tee -a "$LOG/score_${ARM}_${C}.log"
      rc=${PIPESTATUS[0]}
      [ "$rc" -eq 0 ] && break
      say "score $ARM $C attempt $attempt rc=$rc; resuming in 20s"; sleep 20
    done
    N=$(wc -l < "$OUT/$C/$ARM/scores.jsonl")
    say "$ARM/$C coverage: $N"
  done
done

say "stage E: analysis"
echo "stage_E" > "$OUT/STATUS"
$PY -u scripts/duplex/interleaved_timeline_analyze.py \
  2>&1 | tee "$LOG/analyze.log" || fail "analyze"

echo "done" > "$OUT/STATUS"
touch "$OUT/DONE"
say "all done"
