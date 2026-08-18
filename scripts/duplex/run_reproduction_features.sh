#!/usr/bin/env bash
# Reproduction study, Phase 2 tasks 2 and 3: the shared frozen features.
#
#   bash scripts/duplex/run_reproduction_features.sh clip
#   bash scripts/duplex/run_reproduction_features.sh vggish
#
# One corpus at a time on the single GPU, resumable (a video with its .npy
# already on disk is skipped). Detached use:
#
#   setsid nohup bash scripts/duplex/run_reproduction_features.sh clip \
#     > results/reproduction/features/clip_b16_1fps/run.log 2>&1 &
set -uo pipefail
cd /home/jehc223/Hate-follow-up || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
PY=/home/jehc223/venvs/SafetyContradiction/bin/python

STAGE=${1:?usage: run_reproduction_features.sh clip OR vggish}
EXTRA=()
case "$STAGE" in
  clip)   SCRIPT=scripts/duplex/extract_clip_features.py
          OUT=results/reproduction/features/clip_b16_1fps
          # On disk, not the 31 GB tmpfs /tmp: the ffmpeg fallback writes one
          # PNG per second of video before encoding it.
          TMP=results/reproduction/features/.ffmpeg_scratch
          mkdir -p "$TMP"
          EXTRA=(--tmp-dir "$TMP") ;;
  vggish) SCRIPT=scripts/duplex/extract_vggish_features.py
          OUT=results/reproduction/features/vggish_1s ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac

mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/DONE_WITH_FAILURES"
fail=0
for C in hatemm mhclip_en mhclip_zh; do
  echo "=== $STAGE $C $(date -Is)"
  echo "$STAGE:$C started $(date -Is)" > "$OUT/STATUS"
  $PY -u "$SCRIPT" --corpus "$C" ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
  # rc=1 means "finished, but some videos failed"; the failures are listed in
  # $OUT/$C/failures.json and are not retried silently.
  if [ "$rc" -gt 1 ]; then
    echo "!!! $STAGE $C ABORTED rc=$rc"
    echo "$STAGE:$C ABORTED rc=$rc $(date -Is)" > "$OUT/STATUS"
    fail=1
  elif [ "$rc" -eq 1 ]; then
    echo "!!! $STAGE $C finished with per-video failures"
    fail=1
  fi
  echo "--- $C npy files: $(ls "$OUT/$C" 2>/dev/null | grep -c '\.npy$')"
done

if [ "$fail" -eq 0 ]; then
  echo "$STAGE:all done $(date -Is)" > "$OUT/STATUS"
  touch "$OUT/DONE"
else
  echo "$STAGE:finished WITH FAILURES $(date -Is)" > "$OUT/STATUS"
  touch "$OUT/DONE_WITH_FAILURES"
fi
echo "=== $STAGE all done $(date -Is)"
