#!/bin/bash
# Detached driver for the HateClipSeg measurement of the frozen c2 pipeline.
#
#   setsid nohup bash scripts/duplex/run_hateclipseg.sh \
#     > results/hateclipseg/logs/run.log 2>&1 < /dev/null &
#
# Same stages, same modules and same order as scripts/duplex/run_testrun.sh:
# audio extraction with silero voice activity, fresh Whisper large-v3
# transcription on cuda, the frozen degeneracy gate, then one Qwen3-VL-8B judge
# call per video over the uniform-16 frame grid. Only the 8B arm runs: the
# anchored operating point is defined on 8B scores and the 2B judge showed no
# saturation anchor, so a 2B pass would buy nothing this pre-registration uses.
#
# Every stage is resume-safe. The two GPU stages are serialised behind one
# flock, and the second never starts before the first finishes.
#
# Progress: results/hateclipseg/logs/run.log
# Stage:    results/hateclipseg/STATUS  (final value DONE, or FAILED: <stage>)

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
DS=HateClipSeg
OUT=$ROOT/results/hateclipseg
MEDIA=/home/jehc223/data/HateClipSeg/video
STATUS=$OUT/STATUS
LOGDIR=$OUT/logs
GPULOCK=$ROOT/results/testruns/gpu.lock
mkdir -p "$OUT" "$LOGDIR"
: > "$GPULOCK" 2>/dev/null || true
rm -f "$OUT/DONE"

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
export PATH=$HOME/.local/bin:$PATH
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] [$DS] $*"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE: $1 ==="; }
fail() { stage "FAILED: $1"; say "ABORTING at $1"; exit 1; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

N_IDS=$(python -c "
import os,sys; sys.path.insert(0,'src/our_method')
from data_utils import load_clean_split_ids
print(len(load_clean_split_ids('HateClipSeg','test')))")
say "test_clean ids: $N_IDS"

# Per-video STATUS: a watcher rewrites the stage line with the live per-video
# count every 20 s, so an interrupted run's progress is visible from the file
# alone and not only from the log.
watch_counts() {
  local label="$1" path="$2"
  while :; do
    echo "$label $(nlines "$path")/$N_IDS $(date '+%F %T')" > "$STATUS"
    sleep 20
  done
}
WATCH_PID=

start_watch() { watch_counts "$1" "$2" & WATCH_PID=$!; }
stop_watch() { [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null; WATCH_PID=; }

say "run_hateclipseg.sh starting; pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ')"
say "gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>&1 | head -1)"

# ------------------------------------------------------------ 0: frame check
# Decode-check every frame before any GPU time is spent. A truncated JPEG once
# killed a scoring run partway while the pipeline still reported success.
stage "frame_precheck"
python -u scripts/duplex/hateclipseg_prep.py verify 2>&1 \
  | tee "$LOGDIR/frame_precheck.log" || fail "frame_precheck"

# -------------------------------------------------------------- A: audio (CPU)
stage "audio"
start_watch audio "$OUT/audio_meta.jsonl"
python -u scripts/duplex/crossbench_audio.py \
  --dataset "$DS" --split test --out-dir "$OUT" --mp4-dir "$MEDIA" 2>&1 \
  | tee -a "$LOGDIR/audio.log"
RC=${PIPESTATUS[0]}
stop_watch
[ "$RC" = 0 ] || fail "audio"

NWAV=$(python - "$OUT/audio_meta.jsonl" <<'PY'
import json, sys
n = 0
try:
    for line in open(sys.argv[1]):
        line = line.strip()
        if line and json.loads(line).get("wav_ok"):
            n += 1
except FileNotFoundError:
    pass
print(n)
PY
)
say "$NWAV videos with usable audio"

# --------------------------------------------------------------- B: ASR (GPU)
stage "asr"
if [ "$NWAV" -gt 0 ]; then
  for attempt in 1 2 3; do
    HAVE=$(nlines "$OUT/fresh_transcripts.jsonl")
    [ "$HAVE" -ge "$NWAV" ] && break
    say "asr attempt $attempt (have $HAVE/$NWAV)"
    start_watch asr "$OUT/fresh_transcripts.jsonl"
    flock "$GPULOCK" python -u scripts/duplex/crossbench_asr.py \
      --dataset "$DS" --split test --out-dir "$OUT" 2>&1 \
      | tee -a "$LOGDIR/asr.log"
    stop_watch
    NOW=$(nlines "$OUT/fresh_transcripts.jsonl")
    if [ "$NOW" -le "$HAVE" ]; then
      say "asr attempt $attempt added nothing ($NOW); stopping retries"
      break
    fi
  done
else
  say "no usable audio; every video takes the gate's no_fresh_pass branch"
fi
say "asr coverage: $(nlines "$OUT/fresh_transcripts.jsonl")/$NWAV"

# -------------------------------------------------------------- C: gate (CPU)
stage "gate"
python -u scripts/duplex/crossbench_gate.py \
  --dataset "$DS" --split test --out-dir "$OUT" 2>&1 \
  | tee "$LOGDIR/gate.log" || fail "gate"

# ------------------------------------------------------------- D: judge (GPU)
stage "judge_8b"
JDIR=$OUT/judge_8b
mkdir -p "$JDIR"
for attempt in 1 2 3; do
  HAVE=$(nlines "$JDIR/scores.jsonl")
  [ "$HAVE" -ge "$N_IDS" ] && break
  say "judge attempt $attempt (have $HAVE/$N_IDS)"
  start_watch judge_8b "$JDIR/scores.jsonl"
  flock "$GPULOCK" python -u src/duplex/extract_duplex_readout.py \
    --dataset "$DS" --split test \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --transcript-limit 0 \
    --transcript-override-json "$OUT/c2_overrides.json" \
    --out-dir "$JDIR" 2>&1 | tee -a "$LOGDIR/judge_8b.log"
  stop_watch
  NOW=$(nlines "$JDIR/scores.jsonl")
  [ "$NOW" -gt "$HAVE" ] || { say "judge added nothing; stopping retries"; break; }
done
NSCORED=$(nlines "$JDIR/scores.jsonl")
say "judge 8b final coverage: $NSCORED/$N_IDS"
[ "$NSCORED" -eq "$N_IDS" ] || fail "judge_8b coverage $NSCORED != $N_IDS"

stage "DONE"
date > "$OUT/DONE"
say "=== DONE ==="
