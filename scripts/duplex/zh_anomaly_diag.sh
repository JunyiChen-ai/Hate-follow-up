#!/bin/bash
# MHClip_ZH anomaly diagnostic: complete the aborted test run and add a control.
#
# The MHClip_ZH test_clean C2 run reported AUC 0.675 for the 8B and 0.8125 for
# the 2B, the only scale inversion across the four benchmarks. The cause was not
# the method: frames_16/BV1sg411i7Ft/frame_012.jpg was a truncated JPEG, PIL
# raised OSError, and the judge loop died at video 14 of 149. Both arms scored
# 13 videos and the reported AUCs rest on 5 positives and 8 negatives.
#
# The frame has been re-decoded from the local mp4 at the extractor's own index
# (validated against the intact neighbour frame at 37.6 dB PSNR). This script
# then runs three arms over the full 149:
#
#   1. judge_8b     resume, gated fresh Whisper transcript  (the C2 condition)
#   2. judge_2b     resume, gated fresh Whisper transcript  (the contrast arm)
#   3. judge_8b_ctrl  fresh directory, DATASET transcript   (the control)
#
# Arm 3 is the decisive one: identical config to arm 1 -- frozen prag reader,
# frames_16, 16 frames, transcript-limit 0 -- with the only difference being
# that no override map is passed, so resolve_transcript falls through to the
# dataset transcript for every video. Comparing arms 1 and 3 on the same 149
# videos isolates the fresh-Whisper transcript from the split.
#
#   bash scripts/duplex/zh_anomaly_diag.sh
#
# Log: results/testruns/logs/zh_diag.log

set -u
set -o pipefail

ROOT=/home/jehc223/Hate-follow-up
OUT=$ROOT/results/testruns/mhclip_zh
GPULOCK=$ROOT/results/testruns/gpu.lock
LOG=$ROOT/results/testruns/logs/zh_diag.log

cd "$ROOT" || exit 1
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC1091
source ~/venvs/SafetyContradiction/bin/activate

say() { echo "[$(date '+%F %T')] [zh_diag] $*"; }

say "starting; gpu: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"

judge() {
  local dir="$1" model="$2"; shift 2
  say "=== $dir ($model) ==="
  mkdir -p "$OUT/$dir"
  flock "$GPULOCK" python -u src/duplex/extract_duplex_readout.py \
    --dataset MHClip_ZH --split test \
    --model "$model" \
    --transcript-limit 0 \
    --out-dir "$OUT/$dir" "$@"
  say "$dir: $(wc -l < "$OUT/$dir/scores.jsonl") scored"
}

# arms 1 and 2: the C2 condition, resumed onto the repaired frame
judge judge_8b Qwen/Qwen3-VL-8B-Instruct \
  --transcript-override-json "$OUT/c2_overrides.json"
judge judge_2b Qwen/Qwen3-VL-2B-Instruct \
  --transcript-override-json "$OUT/c2_overrides.json"

# arm 3: the control. No override map -> the dataset transcript everywhere.
judge judge_8b_ctrl Qwen/Qwen3-VL-8B-Instruct

# arm 4: the 2B control, so the scale comparison is available under both inputs
judge judge_2b_ctrl Qwen/Qwen3-VL-2B-Instruct

say "=== DONE ==="
