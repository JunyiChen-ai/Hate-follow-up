#!/usr/bin/env bash
# Reproduction study, Phase 2 task 1: Whisper large-v3 timestamped ASR over the
# splits the frozen testruns never covered -- the three train splits plus the
# test videos whose media only arrived with the Phase 0 pull.
#
# One corpus at a time on the single GPU, resumable (each corpus appends to its
# own timestamped_chunks.jsonl and skips ids already present). The Whisper
# configuration is the frozen one imported by interleaved_timeline_asr.py:
# openai/whisper-large-v3, fp16 on cuda, chunk_length_s=30, batch_size=8,
# automatic language detection.
#
#   setsid nohup bash scripts/duplex/run_reproduction_asr.sh \
#     > results/reproduction/asr/run.log 2>&1 &
set -uo pipefail
cd /home/jehc223/Hate-follow-up || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-/home/jehc223/miniconda3/envs/HateVideo/bin/python}
OUT=results/reproduction/asr
mkdir -p "$OUT"
rm -f "$OUT/DONE"

fail=0
for C in hatemm_all mhclip_en_all mhclip_zh_all hateclipseg_all; do
  echo "=== $C $(date -Is)"
  echo "asr:$C started $(date -Is)" > "$OUT/STATUS"
  rc=1
  for attempt in 1 2 3; do
    $PY -u scripts/duplex/interleaved_timeline_asr.py \
      --corpus "$C" --out-root "$OUT"
    rc=$?
    [ "$rc" -eq 0 ] && break
    echo "!!! $C attempt $attempt exited rc=$rc; resuming in 20s"
    sleep 20
  done
  if [ "$rc" -ne 0 ]; then
    echo "!!! $C FAILED after 3 attempts (rc=$rc)"
    echo "asr:$C FAILED rc=$rc $(date -Is)" > "$OUT/STATUS"
    fail=1
  fi
  echo "--- $C rows: $(wc -l < "$OUT/$C/timestamped_chunks.jsonl" 2>/dev/null || echo 0)"
done

if [ "$fail" -eq 0 ]; then
  echo "asr:all done $(date -Is)" > "$OUT/STATUS"
  touch "$OUT/DONE"
else
  echo "asr:finished WITH FAILURES $(date -Is)" > "$OUT/STATUS"
  touch "$OUT/DONE_WITH_FAILURES"
fi
echo "=== all done $(date -Is)"
